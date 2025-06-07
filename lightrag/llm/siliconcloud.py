import sys

if sys.version_info < (3, 9):
    pass
else:
    pass
import pipmaster as pm  # Pipmaster for dynamic library install

# install specific modules
if not pm.is_installed("lmdeploy"):
    pm.install("lmdeploy")

from openai import (
    APIConnectionError,
    RateLimitError,
    APITimeoutError,
)
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)


import numpy as np
import aiohttp
import base64
import struct
import logging
import asyncio
import time
from typing import List, Optional
import math

# 设置日志记录器
logger = logging.getLogger(__name__)

# 全局请求频率控制器
class RateLimiter:
    def __init__(self, rpm: int = 5000, tpm: int = 1000000):
        self.rpm = rpm  # 每分钟请求数限制
        self.tpm = tpm  # 每分钟token数限制
        self.request_times = []  # 请求时间记录
        self.token_usage = []   # token使用记录
        self.lock = asyncio.Lock()
    
    def estimate_tokens(self, texts: List[str]) -> int:
        """预估文本的token数量（简单估算：1个token约等于4个字符）"""
        total_chars = sum(len(text) for text in texts)
        return max(1, total_chars // 4)
    
    async def acquire(self, texts: List[str]):
        """获取请求许可，确保不超过频率限制"""
        async with self.lock:
            current_time = time.time()
            estimated_tokens = self.estimate_tokens(texts)
            
            # 清理1分钟前的记录
            cutoff_time = current_time - 60
            self.request_times = [t for t in self.request_times if t > cutoff_time]
            self.token_usage = [(t, tokens) for t, tokens in self.token_usage if t > cutoff_time]
            
            # 检查RPM限制
            if len(self.request_times) >= self.rpm:
                wait_time = 60 - (current_time - self.request_times[0])
                if wait_time > 0:
                    #logger.info(f"RPM限制，等待 {wait_time:.2f} 秒")
                    await asyncio.sleep(wait_time)
                    return await self.acquire(texts)
            
            # 检查TPM限制
            current_tokens = sum(tokens for _, tokens in self.token_usage)
            if current_tokens + estimated_tokens > self.tpm:
                if self.token_usage:
                    wait_time = 60 - (current_time - self.token_usage[0][0])
                    if wait_time > 0:
                        #logger.info(f"TPM限制，当前tokens: {current_tokens}, 预估需要: {estimated_tokens}, 等待 {wait_time:.2f} 秒")
                        await asyncio.sleep(wait_time)
                        return await self.acquire(texts)
            
            # 记录本次请求
            self.request_times.append(current_time)
            self.token_usage.append((current_time, estimated_tokens))
            
            # 添加基础间隔，避免请求过于密集
            await asyncio.sleep(0.1)

# 全局频率限制器实例
rate_limiter = RateLimiter(rpm=4000, tpm=800000)  # 设置为限制的80%，留出安全边际


@retry(
    stop=stop_after_attempt(5),  # 增加重试次数
    wait=wait_exponential(multiplier=2, min=4, max=120),  # 增加最大等待时间
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError)
    ),
)
async def siliconcloud_embedding(
    texts: list[str],
    model: str = "BAAI/bge-m3",
    base_url: str = "https://api.siliconflow.cn/v1/embeddings",
    max_token_size: int = 8192,
    api_key: str = None,
    batch_size: Optional[int] = None,
) -> np.ndarray:
    if not texts:
        return np.array([])
    
    # 自动确定批次大小
    if batch_size is None:
        # 根据文本长度和token限制动态调整批次大小
        avg_text_length = sum(len(text) for text in texts) / len(texts)
        estimated_tokens_per_text = max(1, avg_text_length // 4)
        # 确保单批次不超过TPM限制的10%
        max_batch_tokens = rate_limiter.tpm // 10
        batch_size = max(1, min(50, max_batch_tokens // estimated_tokens_per_text))
        #logger.info(f"🔧 自动确定批次大小: {batch_size}, 预估每文本token数: {estimated_tokens_per_text}")
    
    # 分批处理大量文本
    if len(texts) > batch_size:
        total_batches = math.ceil(len(texts) / batch_size)
        #logger.info(f"📦 文本数量 {len(texts)} 超过批次大小 {batch_size}，将分为 {total_batches} 个批次处理")
        
        all_embeddings = []
        start_time = time.time()
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            current_batch = i // batch_size + 1
            
            # 计算总体进度
            overall_progress = (current_batch - 1) / total_batches * 100
            #logger.info(f"🚀 开始处理批次 {current_batch}/{total_batches} ({overall_progress:.1f}%) - 包含 {len(batch_texts)} 个文本")
            
            batch_start_time = time.time()
            batch_embeddings = await siliconcloud_embedding(
                batch_texts, model, base_url, max_token_size, api_key, batch_size
            )
            batch_end_time = time.time()
            
            all_embeddings.extend(batch_embeddings)
            
            # 计算完成进度和剩余时间估算
            completed_progress = current_batch / total_batches * 100
            elapsed_time = time.time() - start_time
            if current_batch > 1:
                avg_time_per_batch = elapsed_time / current_batch
                remaining_batches = total_batches - current_batch
                estimated_remaining_time = avg_time_per_batch * remaining_batches
                #logger.info(f"✅ 批次 {current_batch} 完成 ({completed_progress:.1f}%) - 耗时 {batch_end_time - batch_start_time:.2f}s, 预计剩余时间 {estimated_remaining_time:.1f}s")
            # else:
                #logger.info(f"✅ 批次 {current_batch} 完成 ({completed_progress:.1f}%) - 耗时 {batch_end_time - batch_start_time:.2f}s")
        
        total_time = time.time() - start_time
        #logger.info(f"🎉 所有批次处理完成！总计处理 {len(texts)} 个文本，共 {total_batches} 个批次，总耗时 {total_time:.2f}s")
        return np.array(all_embeddings)
    
    # 请求频率控制
    await rate_limiter.acquire(texts)
    
    if api_key and not api_key.startswith("Bearer "):
        api_key = "Bearer " + api_key

    headers = {"Authorization": api_key, "Content-Type": "application/json"}

    truncate_texts = [text[0:max_token_size] for text in texts]

    payload = {"model": model, "input": truncate_texts, "encoding_format": "base64"}
    
    #logger.info(f"📡 发送API请求 - 处理 {len(texts)} 个文本")

    base64_strings = []
    async with aiohttp.ClientSession() as session:
        async with session.post(base_url, headers=headers, json=payload) as response:
            # 特殊处理429错误
            if response.status == 429:
                response_text = await response.text()
                logger.warning(f"⚠️ 遇到429限流错误: {response_text}")
                # 解析响应获取建议等待时间
                try:
                    error_content = await response.json() if response.content_type == 'application/json' else {}
                    if 'retry_after' in error_content:
                        wait_time = int(error_content['retry_after'])
                    else:
                        wait_time = 60  # 默认等待60秒
                except:
                    wait_time = 60
                
                #logger.info(f"⏳ 由于429错误，等待 {wait_time} 秒后重试")
                await asyncio.sleep(wait_time)
                raise RateLimitError(f"Rate limit exceeded: {response_text}")
            
            # 检查其他HTTP错误状态码
            if response.status != 200:
                response_text = await response.text()
                error_msg = f"API请求失败，状态码: {response.status}, 响应内容: {response_text}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            try:
                content = await response.json()
            except Exception as e:
                response_text = await response.text()
                error_msg = f"解析JSON响应失败: {str(e)}, 原始响应: {response_text}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # 记录完整的响应内容用于调试
            #logger.debug(f"API响应内容: {content}")
            
            # 检查API返回的错误码
            if "code" in content:
                error_msg = f"API返回错误码: {content}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # 检查data字段是否存在且不为None
            if "data" not in content:
                error_msg = f"API响应中缺少'data'字段，完整响应: {content}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            if content["data"] is None:
                error_msg = f"API响应中'data'字段为None，完整响应: {content}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            if not isinstance(content["data"], list):
                error_msg = f"API响应中'data'字段不是列表格式，类型: {type(content['data'])}, 完整响应: {content}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            if len(content["data"]) == 0:
                error_msg = f"API响应中'data'字段为空列表，完整响应: {content}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # 验证每个数据项是否包含embedding字段
            for i, item in enumerate(content["data"]):
                if not isinstance(item, dict):
                    error_msg = f"data[{i}]不是字典格式，类型: {type(item)}, 内容: {item}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                if "embedding" not in item:
                    error_msg = f"data[{i}]缺少'embedding'字段，内容: {item}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
            
            base64_strings = [item["embedding"] for item in content["data"]]
    
    #logger.info(f"📥 收到API响应，开始解码 {len(base64_strings)} 个embedding")

    embeddings = []
    total_embeddings = len(base64_strings)
    decode_start_time = time.time()
    
    for i, string in enumerate(base64_strings):
        try:
            decode_bytes = base64.b64decode(string)
            n = len(decode_bytes) // 4
            float_array = struct.unpack("<" + "f" * n, decode_bytes)
            embeddings.append(float_array)
            
            # 每处理10%或者每10个显示一次进度（取较小值）
            progress_interval = max(1, min(10, total_embeddings // 10))
            if (i + 1) % progress_interval == 0 or i == total_embeddings - 1:
                progress = (i + 1) / total_embeddings * 100
                elapsed = time.time() - decode_start_time
                if i > 0:
                    estimated_total = elapsed * total_embeddings / (i + 1)
                    remaining = estimated_total - elapsed
                    #logger.info(f"🔄 解码进度: {i + 1}/{total_embeddings} ({progress:.1f}%) - 已用时 {elapsed:.2f}s, 预计剩余 {remaining:.1f}s")
                # else:
                    #logger.info(f"🔄 解码进度: {i + 1}/{total_embeddings} ({progress:.1f}%)")
                    
        except Exception as e:
            error_msg = f"解码第{i}个embedding时失败: {str(e)}, base64字符串: {string[:100]}..."
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    decode_total_time = time.time() - decode_start_time
    #logger.info(f"✨ 成功处理 {len(embeddings)} 个文本的embedding - 解码总耗时 {decode_total_time:.2f}s")
    return np.array(embeddings)
