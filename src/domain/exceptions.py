"""
استثناهای اختصاصی دامنه (Custom Domain Exceptions) برای هسته RAG سلسله‌مراتبی آفلاین.

این ماژول خطاهای سطح دامنه و نقض ناوردایی‌های کسب‌وکار (Business Invariants) را بدون وابستگی
به لایه‌های زیرساخت یا فریم‌ورک‌های خارجی مدل‌سازی می‌کند.
"""

from typing import Optional


class DomainError(Exception):
    """کلاس پایه برای تمامی خطاهای منطقی و کسب‌وکاری دامنه."""

    def __init__(self, message: str, code: Optional[str] = None) -> None:
        super().__init__(message)
        self.message: str = message
        self.code: str = code or self.__class__.__name__


class DomainValidationError(DomainError):
    """
    خطای نقض قواعد اعتبارسنجی و ناوردایی‌های (Invariants) اشیاء مقداری و موجودیت‌های دامنه.
    """

    def __init__(self, message: str, field_name: Optional[str] = None) -> None:
        super().__init__(message, code="DOMAIN_VALIDATION_ERROR")
        self.field_name: Optional[str] = field_name


class InvalidLayerHierarchyError(DomainValidationError):
    """
    خطای نقض قوانین سلسله‌مراتب لایه‌ها و روابط والد-فرزند میان گره‌های سازمانی.

    موارد بروز:
    - تعریف والد برای گره لایه ۰ (لایه راهبردی).
    - عدم تعریف والد برای گره‌های لایه‌های پایین‌تر (لایه‌های ۱ تا ۴).
    - قرار داشتن گره والد در لایه‌ای هم‌سطح یا پایین‌تر از لایه گره فرزند.
    - عدم تطابق لایه ثبت‌شده گره با لایه هدف در رجیستری.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message, field_name="layer_hierarchy")
        self.code = "INVALID_LAYER_HIERARCHY_ERROR"


class LeafNodeBoundaryViolationError(DomainValidationError):
    """
    خطای نقض مرز گره نهایی (Leaf Node Boundary Violation).

    زمانی رخ می‌دهد که یک گره سازمانی با ویژگی `is_leaf_boundary=True` به عنوان مرز مدل‌سازی
    علامت‌گذاری شده باشد (زیرساختار داخلی آن خارج از مرز سامانه تلقی گردد) و تلاش شود
    زیرمجموعه یا گره فرزندی برای آن ثبت گردد.
    """

    def __init__(self, parent_node_id: str, child_node_id: str) -> None:
        message = (
            f"نقض مرز گره نهایی: گره والد '{parent_node_id}' به عنوان مرز نهایی مدل‌سازی "
            f"(is_leaf_boundary=True) علامت‌گذاری شده است و امکان ثبت گره فرزند "
            f"'{child_node_id}' برای آن وجود ندارد."
        )
        super().__init__(message, field_name="is_leaf_boundary")
        self.code = "LEAF_NODE_BOUNDARY_VIOLATION_ERROR"
        self.parent_node_id: str = parent_node_id
        self.child_node_id: str = child_node_id


class RegistryNotFoundError(DomainError):
    """
    خطای عدم وجود موجودیت (گره یا لایه) در رجیستری سلسله‌مراتب سازمانی.
    """

    def __init__(self, entity_type: str, identifier: str) -> None:
        message = f"موجودیت از نوع '{entity_type}' با شناسه '{identifier}' در رجیستری یافت نشد."
        super().__init__(message, code="REGISTRY_NOT_FOUND_ERROR")
        self.entity_type: str = entity_type
        self.identifier: str = identifier


class JudgeThresholdNotMetError(DomainError):
    """
    خطای عدم احراز حد آستانه امتیاز داور (LLM-as-a-Judge).

    زمانی رخ می‌دهد که امتیاز کل اکتسابی یک بسته انتقال یا خروجی دانشی کمتر از
    حد قابل قبول تعریف‌شده در `JudgeThreshold` (به‌طور پیش‌فرض ۱۸ از ۲۰) باشد.
    """

    def __init__(self, achieved_score: int, required_threshold: int) -> None:
        message = (
            f"امتیاز داوری ({achieved_score} از ۲۰) کمتر از حد آستانه مورد نیاز "
            f"({required_threshold} از ۲۰) است."
        )
        super().__init__(message, code="JUDGE_THRESHOLD_NOT_MET_ERROR")
        self.achieved_score: int = achieved_score
        self.required_threshold: int = required_threshold


__all__ = [
    "DomainError",
    "DomainValidationError",
    "InvalidLayerHierarchyError",
    "LeafNodeBoundaryViolationError",
    "RegistryNotFoundError",
    "JudgeThresholdNotMetError",
]
