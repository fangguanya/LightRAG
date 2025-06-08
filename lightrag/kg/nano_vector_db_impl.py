import asyncio
import os
from typing import Any, final
from dataclasses import dataclass
import numpy as np
import time

from lightrag.utils import (
    logger,
    compute_mdhash_id,
)
import pipmaster as pm
from lightrag.base import BaseVectorStorage

if not pm.is_installed("nano-vectordb"):
    pm.install("nano-vectordb")

from nano_vectordb import NanoVectorDB
from .shared_storage import (
    get_storage_lock,
    get_update_flag,
    set_all_update_flags,
)


@final
@dataclass
class NanoVectorDBStorage(BaseVectorStorage):
    def __post_init__(self):
        # Initialize basic attributes
        self._client = None
        self._storage_lock = None
        self.storage_updated = None

        # Use global config value if specified, otherwise use default
        kwargs = self.global_config.get("vector_db_storage_cls_kwargs", {})
        cosine_threshold = kwargs.get("cosine_better_than_threshold")
        if cosine_threshold is None:
            raise ValueError(
                "cosine_better_than_threshold must be specified in vector_db_storage_cls_kwargs"
            )
        self.cosine_better_than_threshold = cosine_threshold

        self._client_file_name = os.path.join(
            self.global_config["working_dir"], f"vdb_{self.namespace}.json"
        )
        self._max_batch_size = self.global_config["embedding_batch_num"]

        self._client = NanoVectorDB(
            self.embedding_func.embedding_dim,
            storage_file=self._client_file_name,
        )

    async def initialize(self):
        """Initialize storage data"""
        # Get the update flag for cross-process update notification
        self.storage_updated = await get_update_flag(self.namespace)
        # Get the storage lock for use in other methods
        self._storage_lock = get_storage_lock(enable_logging=False)

    async def _get_client(self):
        """Check if the storage should be reloaded"""
        # Acquire lock to prevent concurrent read and write
        async with self._storage_lock:
            # Check if data needs to be reloaded
            if self.storage_updated.value:
                logger.info(
                    f"Process {os.getpid()} reloading {self.namespace} due to update by another process"
                )
                # Reload data
                self._client = NanoVectorDB(
                    self.embedding_func.embedding_dim,
                    storage_file=self._client_file_name,
                )
                # Reset update flag
                self.storage_updated.value = False

            return self._client

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """
        Importance notes:
        1. Changes will be persisted to disk during the next index_done_callback
        2. Only one process should updating the storage at a time before index_done_callback,
           KG-storage-log should be used to avoid data corruption
        """
        import time
        start_time = time.time()
        
        logger.info(f"===== NanoVectorDBStorage.upsert 开始 =====")
        logger.info(f"命名空间: {self.namespace}")
        logger.info(f"插入数据量: {len(data)} 条")
        
        if not data:
            logger.info("数据为空，跳过插入")
            return

        # 分析数据内容
        total_content_length = sum(len(v.get('content', '')) for v in data.values())
        avg_content_length = total_content_length / len(data) if data else 0
        logger.info(f"平均内容长度: {avg_content_length:.0f} 字符，总长度: {total_content_length:,} 字符")

        current_time = int(time.time())
        logger.info(f"准备构建list_data...")
        list_data = [
            {
                "__id__": k,
                "__created_at__": current_time,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            for k, v in data.items()
        ]
        logger.info(f"list_data构建完成，包含 {len(list_data)} 条记录")
        
        logger.info(f"提取content字段...")
        contents = [v["content"] for v in data.values()]
        logger.info(f"content提取完成，包含 {len(contents)} 个文本")
        
        logger.info(f"准备分批，批大小: {self._max_batch_size}")
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]
        logger.info(f"分批完成，共 {len(batches)} 个批次")
        
        # 详细监控每个批次的embedding计算
        logger.info(f"开始执行embedding计算...")
        embedding_start_time = time.time()
        
        # 加载重试配置
        try:
            from ..retry_config import get_retry_config
            retry_config = get_retry_config()
            logger.info(f"已加载重试配置: {retry_config}")
        except ImportError:
            logger.warning("未找到retry_config模块，使用默认重试配置")
            retry_config = {
                "basic": {"max_retries": 3, "retry_delay": 2, "max_circuit_failures": 2},
                "extended": {"enabled": True, "delay": 30, "max_retries": 10},
                "task": {"max_retries": 2, "timeout": 60},
                "overall": {"timeout": 60},
                "fast_fail": {"max_total_runtime": 1800, "max_circuit_breaker_count": 100}
            }
        
        # 添加重试机制和熔断机制的embedding计算 - 支持延迟等待后持续重试
        max_retries = retry_config["basic"]["max_retries"]
        retry_delay = retry_config["basic"]["retry_delay"]
        circuit_breaker_failures = 0  # 熔断器故障计数
        max_circuit_failures = retry_config["basic"]["max_circuit_failures"]
        
        # 延迟等待后持续重试的配置
        extended_retry_enabled = retry_config["extended"]["enabled"]
        extended_retry_delay = retry_config["extended"]["delay"]
        extended_max_retries = retry_config["extended"]["max_retries"]
        
        # 添加系统性故障快速终止机制
        max_total_runtime = retry_config["fast_fail"]["max_total_runtime"]
        max_circuit_breaker_count = retry_config["fast_fail"]["max_circuit_breaker_count"]
        start_runtime = time.time()
        logger.info(f"快速终止配置: 最大运行时间={max_total_runtime/60:.1f}分钟, 最大熔断计数={max_circuit_breaker_count}")
        
        retry_attempt = 0
        while True:
            # 检查总运行时间限制
            current_runtime = time.time() - start_runtime
            if current_runtime > max_total_runtime:
                logger.error(f"embedding计算总运行时间超过限制 ({max_total_runtime/60:.1f} 分钟)，强制终止")
                logger.error(f"当前运行时间: {current_runtime:.1f} 秒，熔断器故障计数: {circuit_breaker_failures}")
                raise Exception(f"embedding计算运行时间超限强制终止 (运行时间: {current_runtime:.1f}秒，熔断器: {circuit_breaker_failures} 次失败)")
            
            # 检查熔断器故障计数限制 - 防止系统性故障无限重试
            if circuit_breaker_failures > max_circuit_breaker_count:
                logger.error(f"熔断器故障计数超过限制 ({max_circuit_breaker_count})，判定为系统性故障，强制终止")
                logger.error(f"当前故障计数: {circuit_breaker_failures}，运行时间: {current_runtime:.1f} 秒")
                logger.error("建议检查: 1) API密钥是否有效 2) 网络连接是否正常 3) embedding服务是否可用")
                raise Exception(f"embedding计算系统性故障强制终止 (熔断器: {circuit_breaker_failures} 次失败，运行时间: {current_runtime:.1f}秒)")
            try:
                if retry_attempt > 0:
                    if retry_attempt <= max_retries:
                        logger.info(f"embedding计算重试 {retry_attempt}/{max_retries}...")
                        
                        # 如果连续失败次数过多，增加延迟
                        if circuit_breaker_failures >= max_circuit_failures:
                            extended_delay = retry_delay * retry_attempt * 2
                            logger.warning(f"检测到连续失败，启用熔断延迟: {extended_delay} 秒")
                            await asyncio.sleep(extended_delay)
                        else:
                            await asyncio.sleep(retry_delay * retry_attempt)  # 递增延迟
                    else:
                        # 超过基本重试次数，进入延迟等待后持续重试模式
                        if extended_retry_enabled and retry_attempt <= max_retries + extended_max_retries:
                            extended_retry_num = retry_attempt - max_retries
                            logger.warning(f"基本重试已耗尽，启用延迟等待后持续重试 {extended_retry_num}/{extended_max_retries}...")
                            logger.warning(f"延迟等待 {extended_retry_delay} 秒后继续重试...")
                            await asyncio.sleep(extended_retry_delay)
                        else:
                            # 所有重试机会都用完了
                            logger.error("embedding计算已达最大重试次数（包括延迟等待重试），放弃处理")
                            logger.error(f"最终熔断器状态: {circuit_breaker_failures} 次失败")
                            
                            # 提供降级策略建议
                            if circuit_breaker_failures >= max_circuit_failures:
                                logger.error("建议检查embedding服务状态或网络连接")
                                logger.error("可能的解决方案: 1) 检查API密钥 2) 检查网络 3) 减少批处理大小")
                            
                            raise Exception(f"embedding计算彻底失败 (熔断器: {circuit_breaker_failures} 次失败，总重试: {retry_attempt} 次): embedding计算总体超时，已达最大重试次数")
                
                retry_attempt += 1
                
                # Execute embedding outside of lock to avoid long lock times
                embedding_tasks = []
                for i, batch in enumerate(batches):
                    logger.info(f"创建embedding任务 {i+1}/{len(batches)}，包含 {len(batch)} 个文本")
                    
                    # 为每个embedding任务添加单独的重试和超时
                    async def embedding_task_with_retry(batch_content, task_id=i+1):
                        task_max_retries = retry_config["task"]["max_retries"]
                        task_timeout = retry_config["task"]["timeout"]
                        for task_retry in range(task_max_retries + 1):
                            try:
                                if task_retry > 0:
                                    logger.warning(f"embedding任务 {task_id} 重试 {task_retry}/{task_max_retries}")
                                    await asyncio.sleep(1)
                                
                                # 为单个embedding任务设置超时
                                result = await asyncio.wait_for(
                                    self.embedding_func(batch_content),
                                    timeout=task_timeout
                                )
                                logger.info(f"embedding任务 {task_id} 完成，返回 {len(result)} 个向量")
                                return result
                                
                            except asyncio.TimeoutError as te:
                                logger.error(f"embedding任务 {task_id} 超时 (尝试 {task_retry + 1}/{task_max_retries + 1})")
                                if task_retry == task_max_retries:
                                    logger.error(f"embedding任务 {task_id} 超时失败，已达最大重试次数")
                                    raise
                                continue
                                
                            except Exception as task_e:
                                logger.error(f"embedding任务 {task_id} 异常 (尝试 {task_retry + 1}/{task_max_retries + 1}): {str(task_e)}")
                                logger.error(f"embedding任务 {task_id} 异常类型: {type(task_e).__name__}")
                                if task_retry == task_max_retries:
                                    logger.error(f"embedding任务 {task_id} 异常失败，已达最大重试次数")
                                    raise
                                continue
                        
                        raise Exception(f"embedding任务 {task_id} 重试耗尽")
                    
                    task = embedding_task_with_retry(batch, i+1)
                    embedding_tasks.append(task)
                
                logger.info(f"所有embedding任务已创建，开始并行执行...")
                logger.info(f"等待 {len(embedding_tasks)} 个embedding任务完成...")
                
                # 设置总体超时，避免无限等待
                total_embedding_timeout = retry_config["overall"]["timeout"]
                try:
                    embeddings_list = await asyncio.wait_for(
                        asyncio.gather(*embedding_tasks, return_exceptions=True),
                        timeout=total_embedding_timeout
                    )
                    
                    # 检查是否有任务返回异常
                    failed_tasks = []
                    successful_results = []
                    for i, result in enumerate(embeddings_list):
                        if isinstance(result, Exception):
                            failed_tasks.append((i+1, result))
                            logger.error(f"embedding任务 {i+1} 最终失败: {str(result)}")
                        else:
                            successful_results.append(result)
                    
                    if failed_tasks:
                        raise Exception(f"有 {len(failed_tasks)} 个embedding任务失败: {[f'任务{task_id}' for task_id, _ in failed_tasks]}")
                    
                    embeddings_list = successful_results
                    
                except asyncio.TimeoutError:
                    logger.error(f"embedding计算总体超时 ({total_embedding_timeout} 秒)")
                    if retry_attempt < max_retries:
                        logger.info(f"将在 {retry_delay * (retry_attempt + 1)} 秒后重试...")
                        continue
                    else:
                        raise Exception(f"embedding计算总体超时，已达最大重试次数")
                
                embedding_elapsed = time.time() - embedding_start_time
                logger.info(f"embedding计算完成，耗时: {embedding_elapsed:.1f} 秒")
                logger.info(f"获得 {len(embeddings_list)} 个embedding批次结果")
                
                # 验证结果完整性
                if len(embeddings_list) != len(batches):
                    raise Exception(f"embedding结果数量不匹配: 期望 {len(batches)}, 实际 {len(embeddings_list)}")
                
                # 重置熔断器，因为成功了
                circuit_breaker_failures = 0
                logger.info("embedding计算成功，重置熔断器计数")
                
                break  # 成功完成，退出重试循环
                
            except Exception as e:
                circuit_breaker_failures += 1
                embedding_elapsed = time.time() - embedding_start_time
                logger.error(f"embedding计算失败 (尝试 {retry_attempt + 1}/{max_retries + 1})，已运行: {embedding_elapsed:.1f} 秒")
                logger.error(f"embedding错误: {str(e)}")
                logger.error(f"错误类型: {type(e).__name__}")
                logger.error(f"熔断器故障计数: {circuit_breaker_failures}")
                
                # 继续重试循环，不退出
                continue
        
        # 如果到达这里说明重试成功了
        logger.info("embedding计算重试成功！")
        
        logger.info(f"开始合并embedding结果...")
        embeddings = np.concatenate(embeddings_list)
        logger.info(f"embedding合并完成，形状: {embeddings.shape}")
        
        if len(embeddings) == len(list_data):
            logger.info(f"embedding数量匹配，开始构建最终数据...")
            for i, d in enumerate(list_data):
                d["__vector__"] = embeddings[i]
            
            logger.info(f"获取数据库客户端...")
            client = await self._get_client()
            
            logger.info(f"开始执行数据库upsert操作...")
            db_upsert_start = time.time()
            try:
                results = client.upsert(datas=list_data)
                db_upsert_elapsed = time.time() - db_upsert_start
                logger.info(f"数据库upsert完成，耗时: {db_upsert_elapsed:.1f} 秒")
                
                total_elapsed = time.time() - start_time
                logger.info(f"===== NanoVectorDBStorage.upsert 完成，总耗时: {total_elapsed:.1f} 秒 =====")
                return results
            except Exception as e:
                db_upsert_elapsed = time.time() - db_upsert_start
                logger.error(f"数据库upsert失败，已运行: {db_upsert_elapsed:.1f} 秒")
                logger.error(f"数据库错误: {str(e)}")
                raise
        else:
            # sometimes the embedding is not returned correctly. just log it.
            total_elapsed = time.time() - start_time
            logger.error(f"embedding数量不匹配: {len(embeddings)} != {len(list_data)}")
            logger.error(f"===== NanoVectorDBStorage.upsert 失败，总耗时: {total_elapsed:.1f} 秒 =====")
            logger.error(
                f"embedding is not 1-1 with data, {len(embeddings)} != {len(list_data)}"
            )

    async def query(
        self, query: str, top_k: int, ids: list[str] | None = None
    ) -> list[dict[str, Any]]:
        # Execute embedding outside of lock to avoid improve cocurrent
        embedding = await self.embedding_func(
            [query], _priority=5
        )  # higher priority for query
        embedding = embedding[0]

        client = await self._get_client()
        results = client.query(
            query=embedding,
            top_k=top_k,
            better_than_threshold=self.cosine_better_than_threshold,
        )
        results = [
            {
                **dp,
                "id": dp["__id__"],
                "distance": dp["__metrics__"],
                "created_at": dp.get("__created_at__"),
            }
            for dp in results
        ]
        return results

    @property
    async def client_storage(self):
        client = await self._get_client()
        return getattr(client, "_NanoVectorDB__storage")

    async def delete(self, ids: list[str]):
        """Delete vectors with specified IDs

        Importance notes:
        1. Changes will be persisted to disk during the next index_done_callback
        2. Only one process should updating the storage at a time before index_done_callback,
           KG-storage-log should be used to avoid data corruption

        Args:
            ids: List of vector IDs to be deleted
        """
        try:
            client = await self._get_client()
            client.delete(ids)
            logger.debug(
                f"Successfully deleted {len(ids)} vectors from {self.namespace}"
            )
        except Exception as e:
            logger.error(f"Error while deleting vectors from {self.namespace}: {e}")

    async def delete_entity(self, entity_name: str) -> None:
        """
        Importance notes:
        1. Changes will be persisted to disk during the next index_done_callback
        2. Only one process should updating the storage at a time before index_done_callback,
           KG-storage-log should be used to avoid data corruption
        """

        try:
            entity_id = compute_mdhash_id(entity_name, prefix="ent-")
            logger.debug(
                f"Attempting to delete entity {entity_name} with ID {entity_id}"
            )

            # Check if the entity exists
            client = await self._get_client()
            if client.get([entity_id]):
                client.delete([entity_id])
                logger.debug(f"Successfully deleted entity {entity_name}")
            else:
                logger.debug(f"Entity {entity_name} not found in storage")
        except Exception as e:
            logger.error(f"Error deleting entity {entity_name}: {e}")

    async def delete_entity_relation(self, entity_name: str) -> None:
        """
        Importance notes:
        1. Changes will be persisted to disk during the next index_done_callback
        2. Only one process should updating the storage at a time before index_done_callback,
           KG-storage-log should be used to avoid data corruption
        """

        try:
            client = await self._get_client()
            storage = getattr(client, "_NanoVectorDB__storage")
            relations = [
                dp
                for dp in storage["data"]
                if dp["src_id"] == entity_name or dp["tgt_id"] == entity_name
            ]
            logger.debug(f"Found {len(relations)} relations for entity {entity_name}")
            ids_to_delete = [relation["__id__"] for relation in relations]

            if ids_to_delete:
                client = await self._get_client()
                client.delete(ids_to_delete)
                logger.debug(
                    f"Deleted {len(ids_to_delete)} relations for {entity_name}"
                )
            else:
                logger.debug(f"No relations found for entity {entity_name}")
        except Exception as e:
            logger.error(f"Error deleting relations for {entity_name}: {e}")

    async def index_done_callback(self) -> bool:
        """Save data to disk"""
        async with self._storage_lock:
            # Check if storage was updated by another process
            if self.storage_updated.value:
                # Storage was updated by another process, reload data instead of saving
                logger.warning(
                    f"Storage for {self.namespace} was updated by another process, reloading..."
                )
                self._client = NanoVectorDB(
                    self.embedding_func.embedding_dim,
                    storage_file=self._client_file_name,
                )
                # Reset update flag
                self.storage_updated.value = False
                return False  # Return error

        # Acquire lock and perform persistence
        async with self._storage_lock:
            try:
                # Save data to disk
                self._client.save()
                # Notify other processes that data has been updated
                await set_all_update_flags(self.namespace)
                # Reset own update flag to avoid self-reloading
                self.storage_updated.value = False
                return True  # Return success
            except Exception as e:
                logger.error(f"Error saving data for {self.namespace}: {e}")
                return False  # Return error

        return True  # Return success

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """Get vector data by its ID

        Args:
            id: The unique identifier of the vector

        Returns:
            The vector data if found, or None if not found
        """
        client = await self._get_client()
        result = client.get([id])
        if result:
            dp = result[0]
            return {
                **dp,
                "id": dp.get("__id__"),
                "created_at": dp.get("__created_at__"),
            }
        return None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get multiple vector data by their IDs

        Args:
            ids: List of unique identifiers

        Returns:
            List of vector data objects that were found
        """
        if not ids:
            return []

        client = await self._get_client()
        results = client.get(ids)
        return [
            {
                **dp,
                "id": dp.get("__id__"),
                "created_at": dp.get("__created_at__"),
            }
            for dp in results
        ]

    async def drop(self) -> dict[str, str]:
        """Drop all vector data from storage and clean up resources

        This method will:
        1. Remove the vector database storage file if it exists
        2. Reinitialize the vector database client
        3. Update flags to notify other processes
        4. Changes is persisted to disk immediately

        This method is intended for use in scenarios where all data needs to be removed,

        Returns:
            dict[str, str]: Operation status and message
            - On success: {"status": "success", "message": "data dropped"}
            - On failure: {"status": "error", "message": "<error details>"}
        """
        try:
            async with self._storage_lock:
                # delete _client_file_name
                if os.path.exists(self._client_file_name):
                    os.remove(self._client_file_name)

                self._client = NanoVectorDB(
                    self.embedding_func.embedding_dim,
                    storage_file=self._client_file_name,
                )

                # Notify other processes that data has been updated
                await set_all_update_flags(self.namespace)
                # Reset own update flag to avoid self-reloading
                self.storage_updated.value = False

                logger.info(
                    f"Process {os.getpid()} drop {self.namespace}(file:{self._client_file_name})"
                )
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            logger.error(f"Error dropping {self.namespace}: {e}")
            return {"status": "error", "message": str(e)}
