"""
لایه زیرساخت (Infrastructure Layer) هسته RAG سلسله‌مراتبی آفلاین.

این بسته شامل درگاه‌ها و آداپتورهای اجرای مدل‌های کوچک آفلاین (`LocalOllamaAdapter`
و `DeterministicMockModelAdapter`) منطبق بر الگوی Ports & Adapters است.
"""

from src.infrastructure.offline_model_adapter import (
    ALLOWED_LOCAL_HOSTS,
    DeterministicMockModelAdapter,
    LocalOllamaAdapter,
    OfflineModelAdapterPort,
)

__all__ = [
    "ALLOWED_LOCAL_HOSTS",
    "OfflineModelAdapterPort",
    "LocalOllamaAdapter",
    "DeterministicMockModelAdapter",
]
