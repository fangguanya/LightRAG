# -*- coding: utf-8 -*-
"""
LightRAG embedding计算重试配置

此配置文件允许用户自定义embedding计算失败后的重试策略，
包括基本重试和延迟等待后的持续重试机制。
"""

# 基本重试配置
BASIC_RETRY_CONFIG = {
    "max_retries": 3,           # 基本重试次数
    "retry_delay": 2,           # 基本重试间隔（秒）
    "max_circuit_failures": 2,  # 熔断器触发的最大连续失败次数
}

# 延迟等待后持续重试配置
EXTENDED_RETRY_CONFIG = {
    "enabled": True,            # 是否启用延迟等待后持续重试
    "delay": 30,               # 延迟等待时间（秒）
    "max_retries": 10,         # 延迟等待后的最大重试次数
}

# embedding任务级别重试配置
TASK_RETRY_CONFIG = {
    "max_retries": 2,          # 单个embedding任务最大重试次数
    "timeout": 60,             # 单个embedding任务超时时间（秒）
}

# 总体超时配置
OVERALL_TIMEOUT_CONFIG = {
    "timeout": 60,             # embedding计算总体超时时间（秒）
}

def get_retry_config():
    """获取完整的重试配置"""
    return {
        "basic": BASIC_RETRY_CONFIG,
        "extended": EXTENDED_RETRY_CONFIG,
        "task": TASK_RETRY_CONFIG,
        "overall": OVERALL_TIMEOUT_CONFIG,
    }

def update_retry_config(**kwargs):
    """更新重试配置
    
    Args:
        **kwargs: 配置参数，支持的键：
            - basic_max_retries: 基本重试次数
            - basic_retry_delay: 基本重试间隔（秒）
            - basic_max_circuit_failures: 熔断器触发阈值
            - extended_enabled: 是否启用延迟等待后持续重试
            - extended_delay: 延迟等待时间（秒）
            - extended_max_retries: 延迟等待后的最大重试次数
            - task_max_retries: 单个任务最大重试次数
            - task_timeout: 单个任务超时时间（秒）
            - overall_timeout: 总体超时时间（秒）
    """
    # 更新基本重试配置
    if "basic_max_retries" in kwargs:
        BASIC_RETRY_CONFIG["max_retries"] = kwargs["basic_max_retries"]
    if "basic_retry_delay" in kwargs:
        BASIC_RETRY_CONFIG["retry_delay"] = kwargs["basic_retry_delay"]
    if "basic_max_circuit_failures" in kwargs:
        BASIC_RETRY_CONFIG["max_circuit_failures"] = kwargs["basic_max_circuit_failures"]
    
    # 更新延迟等待后持续重试配置
    if "extended_enabled" in kwargs:
        EXTENDED_RETRY_CONFIG["enabled"] = kwargs["extended_enabled"]
    if "extended_delay" in kwargs:
        EXTENDED_RETRY_CONFIG["delay"] = kwargs["extended_delay"]
    if "extended_max_retries" in kwargs:
        EXTENDED_RETRY_CONFIG["max_retries"] = kwargs["extended_max_retries"]
    
    # 更新任务级别配置
    if "task_max_retries" in kwargs:
        TASK_RETRY_CONFIG["max_retries"] = kwargs["task_max_retries"]
    if "task_timeout" in kwargs:
        TASK_RETRY_CONFIG["timeout"] = kwargs["task_timeout"]
    
    # 更新总体超时配置
    if "overall_timeout" in kwargs:
        OVERALL_TIMEOUT_CONFIG["timeout"] = kwargs["overall_timeout"]

# 预设配置模板
PRESET_CONFIGS = {
    "conservative": {
        # 保守策略：少量重试，短延迟
        "basic_max_retries": 2,
        "basic_retry_delay": 1,
        "extended_enabled": True,
        "extended_delay": 15,
        "extended_max_retries": 5,
        "task_max_retries": 1,
        "task_timeout": 30,
        "overall_timeout": 30,
    },
    "aggressive": {
        # 激进策略：大量重试，长延迟
        "basic_max_retries": 5,
        "basic_retry_delay": 3,
        "extended_enabled": True,
        "extended_delay": 60,
        "extended_max_retries": 20,
        "task_max_retries": 3,
        "task_timeout": 120,
        "overall_timeout": 120,
    },
    "balanced": {
        # 平衡策略：中等重试，适中延迟（默认）
        "basic_max_retries": 3,
        "basic_retry_delay": 2,
        "extended_enabled": True,
        "extended_delay": 30,
        "extended_max_retries": 10,
        "task_max_retries": 2,
        "task_timeout": 60,
        "overall_timeout": 60,
    },
    "no_extended_retry": {
        # 禁用延迟等待后持续重试
        "basic_max_retries": 3,
        "basic_retry_delay": 2,
        "extended_enabled": False,
        "extended_delay": 0,
        "extended_max_retries": 0,
        "task_max_retries": 2,
        "task_timeout": 60,
        "overall_timeout": 60,
    }
}

def apply_preset_config(preset_name: str):
    """应用预设配置
    
    Args:
        preset_name: 预设配置名称，可选值：
            - "conservative": 保守策略
            - "aggressive": 激进策略  
            - "balanced": 平衡策略（默认）
            - "no_extended_retry": 禁用延迟等待后持续重试
    """
    if preset_name not in PRESET_CONFIGS:
        raise ValueError(f"未知的预设配置: {preset_name}，可选值: {list(PRESET_CONFIGS.keys())}")
    
    config = PRESET_CONFIGS[preset_name]
    update_retry_config(**config)
    return get_retry_config() 