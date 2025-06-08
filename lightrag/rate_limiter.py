# -*- coding: utf-8 -*-
"""
LightRAG 通用频率限制控制器

支持RPM（每分钟请求数）和TPM（每分钟Token数）限制，
可用于LLM和embedding API请求的频率控制。
"""

import asyncio
import time
import logging
from typing import List, Dict, Any, Union, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """频率限制配置"""
    rpm: int = 1000                    # 每分钟请求数限制
    tpm: int = 1000000                 # 每分钟Token数限制
    enable_rpm: bool = True            # 是否启用RPM限制
    enable_tpm: bool = True            # 是否启用TPM限制
    safety_margin: float = 0.8         # 安全边际（使用限制的80%）
    base_interval: float = 0.1         # 基础请求间隔（秒）
    burst_threshold: int = 10          # 突发请求阈值
    adaptive_wait: bool = True         # 是否启用自适应等待
    log_wait_events: bool = True       # 是否记录等待事件


class UniversalRateLimiter:
    """通用频率限制器
    
    支持RPM和TPM双重限制，适用于各种API调用场景
    """
    
    def __init__(self, config: RateLimitConfig = None):
        self.config = config or RateLimitConfig()
        self.request_times = []           # 请求时间记录
        self.token_usage = []            # token使用记录
        self.lock = asyncio.Lock()       # 异步锁
        
        # 应用安全边际
        self.effective_rpm = int(self.config.rpm * self.config.safety_margin)
        self.effective_tpm = int(self.config.tpm * self.config.safety_margin)
        
        logger.info(f"初始化频率限制器: RPM={self.effective_rpm}, TPM={self.effective_tpm}")
    
    def estimate_tokens_from_texts(self, texts: List[str]) -> int:
        """从文本列表预估token数量"""
        total_chars = sum(len(text) for text in texts)
        return max(1, total_chars // 4)  # 简单估算：1 token ≈ 4 字符
    
    def estimate_tokens_from_messages(self, messages: List[Dict[str, Any]]) -> int:
        """从消息列表预估token数量"""
        total_chars = 0
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                # 处理多模态内容
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        total_chars += len(item.get("text", ""))
        return max(1, total_chars // 4)
    
    def estimate_tokens(self, content: Union[List[str], List[Dict[str, Any]], str, int]) -> int:
        """智能预估token数量"""
        if isinstance(content, int):
            return content  # 直接提供的token数
        elif isinstance(content, str):
            return max(1, len(content) // 4)
        elif isinstance(content, list):
            if not content:
                return 1
            if isinstance(content[0], str):
                return self.estimate_tokens_from_texts(content)
            elif isinstance(content[0], dict):
                return self.estimate_tokens_from_messages(content)
        return 1
    
    async def _clean_old_records(self, current_time: float):
        """清理1分钟前的记录"""
        cutoff_time = current_time - 60
        self.request_times = [t for t in self.request_times if t > cutoff_time]
        self.token_usage = [(t, tokens) for t, tokens in self.token_usage if t > cutoff_time]
    
    async def _check_rpm_limit(self, current_time: float) -> Optional[float]:
        """检查RPM限制，返回需要等待的时间"""
        if not self.config.enable_rpm:
            return None
            
        if len(self.request_times) >= self.effective_rpm:
            wait_time = 60 - (current_time - self.request_times[0])
            if wait_time > 0:
                return wait_time
        return None
    
    async def _check_tpm_limit(self, current_time: float, estimated_tokens: int) -> Optional[float]:
        """检查TPM限制，返回需要等待的时间"""
        if not self.config.enable_tpm:
            return None
            
        current_tokens = sum(tokens for _, tokens in self.token_usage)
        if current_tokens + estimated_tokens > self.effective_tpm:
            if self.token_usage:
                wait_time = 60 - (current_time - self.token_usage[0][0])
                if wait_time > 0:
                    return wait_time
        return None
    
    async def _adaptive_wait(self, wait_time: float, reason: str):
        """自适应等待策略"""
        if self.config.adaptive_wait:
            # 如果等待时间过长，分段等待并提供进度信息
            if wait_time > 10:
                segments = min(10, int(wait_time / 2))
                segment_time = wait_time / segments
                for i in range(segments):
                    remaining = wait_time - (i * segment_time)
                    if self.config.log_wait_events:
                        logger.info(f"⏰ {reason} - 剩余等待时间: {remaining:.1f}秒")
                    await asyncio.sleep(segment_time)
            else:
                if self.config.log_wait_events:
                    logger.info(f"⏰ {reason} - 等待 {wait_time:.2f}秒")
                await asyncio.sleep(wait_time)
        else:
            if self.config.log_wait_events:
                logger.info(f"⏰ {reason} - 等待 {wait_time:.2f}秒")
            await asyncio.sleep(wait_time)
    
    async def acquire(self, content: Union[List[str], List[Dict[str, Any]], str, int] = None):
        """获取请求许可，确保不超过频率限制
        
        Args:
            content: 要处理的内容，用于估算token数量
                   - List[str]: 文本列表（用于embedding）
                   - List[Dict]: 消息列表（用于LLM）
                   - str: 单个文本
                   - int: 直接提供的token数量
        """
        async with self.lock:
            current_time = time.time()
            estimated_tokens = self.estimate_tokens(content) if content is not None else 1
            
            # 清理过期记录
            await self._clean_old_records(current_time)
            
            # 检查RPM限制
            rpm_wait = await self._check_rpm_limit(current_time)
            if rpm_wait:
                reason = f"RPM限制({len(self.request_times)}/{self.effective_rpm})"
                await self._adaptive_wait(rpm_wait, reason)
                return await self.acquire(content)  # 递归重试
            
            # 检查TPM限制
            tpm_wait = await self._check_tpm_limit(current_time, estimated_tokens)
            if tpm_wait:
                current_tokens = sum(tokens for _, tokens in self.token_usage)
                reason = f"TPM限制(当前:{current_tokens}, 需要:{estimated_tokens}, 限制:{self.effective_tpm})"
                await self._adaptive_wait(tpm_wait, reason)
                return await self.acquire(content)  # 递归重试
            
            # 记录本次请求
            self.request_times.append(current_time)
            self.token_usage.append((current_time, estimated_tokens))
            
            # 基础间隔，避免请求过于密集
            if self.config.base_interval > 0:
                await asyncio.sleep(self.config.base_interval)
    
    async def acquire_with_actual_tokens(self, actual_tokens: int):
        """在API调用完成后，用实际token数量更新记录
        
        Args:
            actual_tokens: API返回的实际token使用量
        """
        async with self.lock:
            if self.token_usage:
                # 更新最后一次请求的token数量
                last_time, _ = self.token_usage[-1]
                self.token_usage[-1] = (last_time, actual_tokens)
                
                if self.config.log_wait_events:
                    logger.debug(f"更新实际token使用量: {actual_tokens}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取当前统计信息"""
        current_time = time.time()
        cutoff_time = current_time - 60
        
        recent_requests = [t for t in self.request_times if t > cutoff_time]
        recent_tokens = [(t, tokens) for t, tokens in self.token_usage if t > cutoff_time]
        
        current_rpm = len(recent_requests)
        current_tpm = sum(tokens for _, tokens in recent_tokens)
        
        return {
            "current_rpm": current_rpm,
            "max_rpm": self.effective_rpm,
            "rpm_usage_percent": (current_rpm / self.effective_rpm * 100) if self.effective_rpm > 0 else 0,
            "current_tpm": current_tpm,
            "max_tpm": self.effective_tpm,
            "tpm_usage_percent": (current_tpm / self.effective_tpm * 100) if self.effective_tpm > 0 else 0,
            "config": self.config
        }
    
    def reset(self):
        """重置所有记录"""
        self.request_times.clear()
        self.token_usage.clear()
        logger.info("频率限制器记录已重置")


# 预定义的配置模板
RATE_LIMIT_PRESETS = {
    "openai_free": RateLimitConfig(
        rpm=200, tpm=40000, 
        safety_margin=0.8,
        log_wait_events=True
    ),
    "openai_tier1": RateLimitConfig(
        rpm=500, tpm=150000,
        safety_margin=0.8,
        log_wait_events=True
    ),
    "openai_tier2": RateLimitConfig(
        rpm=5000, tpm=1500000,
        safety_margin=0.8,
        log_wait_events=True
    ),
    "azure_openai": RateLimitConfig(
        rpm=1000, tpm=300000,
        safety_margin=0.8,
        log_wait_events=True
    ),
    "siliconflow": RateLimitConfig(
        rpm=4000, tpm=800000,
        safety_margin=0.8,
        log_wait_events=True
    ),
    "claude": RateLimitConfig(
        rpm=1000, tpm=200000,
        safety_margin=0.8,
        log_wait_events=True
    ),
    "gemini": RateLimitConfig(
        rpm=2000, tpm=1000000,
        safety_margin=0.8,
        log_wait_events=True
    ),
    "zhipu": RateLimitConfig(
        rpm=1000, tpm=500000,
        safety_margin=0.8,
        log_wait_events=True
    ),
    "conservative": RateLimitConfig(
        rpm=100, tpm=50000,
        safety_margin=0.7,
        log_wait_events=True
    ),
    "aggressive": RateLimitConfig(
        rpm=10000, tpm=5000000,
        safety_margin=0.9,
        log_wait_events=False
    ),
    "unlimited": RateLimitConfig(
        rpm=999999, tpm=999999999,
        enable_rpm=False,
        enable_tpm=False,
        log_wait_events=False
    )
}


def create_rate_limiter(preset: str = "openai_tier1", **override_params) -> UniversalRateLimiter:
    """创建频率限制器
    
    Args:
        preset: 预设配置名称
        **override_params: 覆盖的配置参数
    
    Returns:
        配置好的频率限制器实例
    """
    if preset not in RATE_LIMIT_PRESETS:
        logger.warning(f"未知预设'{preset}'，使用默认配置")
        config = RateLimitConfig()
    else:
        config = RATE_LIMIT_PRESETS[preset]
    
    # 应用覆盖参数
    if override_params:
        for key, value in override_params.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                logger.warning(f"忽略未知配置参数: {key}")
    
    return UniversalRateLimiter(config)


# 全局默认限制器实例
default_llm_limiter = create_rate_limiter("openai_tier1")
default_embedding_limiter = create_rate_limiter("openai_tier1") 