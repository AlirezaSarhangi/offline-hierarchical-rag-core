"""
رجیستری مدل‌های کوچک آفلاین (`ModelRegistry`) طبق بخش ۸.۲، جدول ۱۹.۱ و فاز ۲ پروپوزال.

این ماژول وظیفه ثبت، دسته‌بندی، نگاشت مدل به وظیفه، ثبت محدودیت‌های عملکردی مدل‌های آفلاین
و نگهداری یادداشت‌های ارزیابی برای سه خانواده مدل‌های محلی را بر عهده دارد:
1. `EMBEDDING` (نمایش برداری، نمایه‌سازی معنایی و تشخیص تکرار/هم‌پوشانی)
2. `LLM` (خلاصه‌سازی سلسله‌مراتبی، استدلال، تجزیه مسئله، تولید RFP/بازخورد و داوری)
3. `LVM` (مدل بینایی-زبانی برای تفسیر جدول، تصویر و چیدمان پیچیده اسناد PDF/DOC)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

from src.domain.exceptions import (
    DomainValidationError,
    IncompatibleModelTaskError,
    RegistryNotFoundError,
)


class ModelCategory(str, Enum):
    """دسته‌بندی اصلی مدل‌های کوچک آفلاین (بخش ۸.۲ پروپوزال)."""

    EMBEDDING = "EMBEDDING"
    LLM = "LLM"
    LVM = "LVM"

    @property
    def label_fa(self) -> str:
        """عنوان فارسی دسته مدل."""
        labels = {
            ModelCategory.EMBEDDING: "مدل تعبیه برداری و بازیابی معنایی (Embedding)",
            ModelCategory.LLM: "مدل زبانی کوچک محلی (Small Local LLM)",
            ModelCategory.LVM: "مدل بینایی-زبانی تحلیل اسناد و چیدمان (Local LVM)",
        }
        return labels[self]


class ModelTaskType(str, Enum):
    """
    انواع وظایف تخصصی قابل نگاشت به مدل‌های کوچک آفلاین در هسته RAG سلسله‌مراتبی.
    """

    SEMANTIC_INDEXING = "SEMANTIC_INDEXING"
    QA_ANSWERING = "QA_ANSWERING"
    TECHNICAL_SUMMARIZATION = "TECHNICAL_SUMMARIZATION"
    EXECUTIVE_MANAGERIAL_SUMMARIZATION = "EXECUTIVE_MANAGERIAL_SUMMARIZATION"
    ISSUE_DECOMPOSITION = "ISSUE_DECOMPOSITION"
    EXECUTOR_EXPERTISE_ANALYSIS = "EXECUTOR_EXPERTISE_ANALYSIS"
    RFP_AND_FEEDBACK_GENERATION = "RFP_AND_FEEDBACK_GENERATION"
    LLM_AS_JUDGE_EVALUATION = "LLM_AS_JUDGE_EVALUATION"
    DOCUMENT_LAYOUT_ANALYSIS = "DOCUMENT_LAYOUT_ANALYSIS"

    @property
    def label_fa(self) -> str:
        """عنوان فارسی وظیفه مدل."""
        labels = {
            ModelTaskType.SEMANTIC_INDEXING: "نمایه‌سازی معنایی و محاسبه شباهت برداری",
            ModelTaskType.QA_ANSWERING: "پرسش و پاسخ مبتنی بر منبع",
            ModelTaskType.TECHNICAL_SUMMARIZATION: "خلاصه‌سازی فنی",
            ModelTaskType.EXECUTIVE_MANAGERIAL_SUMMARIZATION: "خلاصه‌سازی مدیریتی و اجرایی",
            ModelTaskType.ISSUE_DECOMPOSITION: "تجزیه مسئله به زیرمسئله و تحلیل شکاف",
            ModelTaskType.EXECUTOR_EXPERTISE_ANALYSIS: "تحلیل تخصص مجری و نگاشت IKE",
            ModelTaskType.RFP_AND_FEEDBACK_GENERATION: "تولید RFP، ارزیابی پیشنهاد و بازخورد",
            ModelTaskType.LLM_AS_JUDGE_EVALUATION: "داوری خروجی (LLM-as-a-Judge)",
            ModelTaskType.DOCUMENT_LAYOUT_ANALYSIS: "تحلیل بصری جدول، تصویر و چیدمان سند",
        }
        return labels[self]

    @property
    def required_category(self) -> ModelCategory:
        """دسته مدل مورد نیاز برای اجرای این وظیفه."""
        if self == ModelTaskType.SEMANTIC_INDEXING:
            return ModelCategory.EMBEDDING
        if self == ModelTaskType.DOCUMENT_LAYOUT_ANALYSIS:
            return ModelCategory.LVM
        return ModelCategory.LLM


@dataclass(frozen=True)
class OfflineModelSpecification:
    """
    مشخصات تغییرناپذیر یک مدل کوچک آفلاین در رجیستری مدل (جدول ۱۹.۱ و فاز ۲ پروپوزال).
    """

    model_id: str
    model_name: str
    category: ModelCategory
    quantization_format: str
    context_window_size: int
    supported_tasks: Tuple[ModelTaskType, ...]
    default_inference_params: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType(
            {"temperature": 0.1, "top_p": 0.9, "max_tokens": 1024}
        )
    )
    known_limitations: Tuple[str, ...] = ()
    evaluation_notes: str = ""
    is_primary_for_category: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise DomainValidationError(
                "شناسه مدل (model_id) در OfflineModelSpecification نمی‌تواند خالی باشد.",
                field_name="model_id",
            )
        object.__setattr__(self, "model_id", self.model_id.strip())

        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise DomainValidationError(
                "نام مدل (model_name) نمی‌تواند خالی باشد.",
                field_name="model_name",
            )
        object.__setattr__(self, "model_name", self.model_name.strip())

        if isinstance(self.category, str) and not isinstance(
            self.category, ModelCategory
        ):
            try:
                object.__setattr__(self, "category", ModelCategory(self.category))
            except ValueError as exc:
                raise DomainValidationError(
                    f"دسته مدل '{self.category}' نامعتبر است.",
                    field_name="category",
                ) from exc
        elif not isinstance(self.category, ModelCategory):
            raise DomainValidationError(
                "فیلد category باید از نوع ModelCategory باشد.",
                field_name="category",
            )

        if (
            not isinstance(self.quantization_format, str)
            or not self.quantization_format.strip()
        ):
            raise DomainValidationError(
                "فرمت کوانتیزاسیون (quantization_format) نمی‌تواند خالی باشد.",
                field_name="quantization_format",
            )
        object.__setattr__(
            self, "quantization_format", self.quantization_format.strip()
        )

        if (
            isinstance(self.context_window_size, bool)
            or not isinstance(self.context_window_size, int)
            or self.context_window_size < 128
        ):
            raise DomainValidationError(
                f"اندازه پنجره زمینه (context_window_size) باید عدد صحیح حداقل ۱۲۸ باشد؛ مقدار دریافتی: {self.context_window_size}.",
                field_name="context_window_size",
            )

        if isinstance(self.supported_tasks, (str, bytes)) or not isinstance(
            self.supported_tasks, Iterable
        ):
            raise DomainValidationError(
                "فهرست وظایف مجاز (supported_tasks) باید مجموعه‌ای از ModelTaskType باشد.",
                field_name="supported_tasks",
            )
        normalized_tasks: List[ModelTaskType] = []
        for task in self.supported_tasks:
            task_enum = (
                ModelTaskType(task)
                if isinstance(task, str) and not isinstance(task, ModelTaskType)
                else task
            )
            if not isinstance(task_enum, ModelTaskType):
                raise DomainValidationError(
                    f"مقدار وظیفه '{task}' از نوع ModelTaskType نیست.",
                    field_name="supported_tasks",
                )
            if task_enum.required_category != self.category:
                raise DomainValidationError(
                    f"وظیفه '{task_enum.value}' نیازمند دسته '{task_enum.required_category.value}' است و "
                    f"با دسته مدل '{self.category.value}' سازگار نیست.",
                    field_name="supported_tasks",
                )
            if task_enum not in normalized_tasks:
                normalized_tasks.append(task_enum)

        if not normalized_tasks:
            raise DomainValidationError(
                "هر مدل آفلاین باید حداقل یک وظیفه مجاز (supported_tasks) داشته باشد.",
                field_name="supported_tasks",
            )
        object.__setattr__(self, "supported_tasks", tuple(normalized_tasks))

        if not isinstance(self.default_inference_params, Mapping):
            raise DomainValidationError(
                "پارامترهای استنتاج (default_inference_params) باید از نوع دیکشنری/نگاشت باشد.",
                field_name="default_inference_params",
            )
        params_copy = dict(self.default_inference_params)
        for required_key in ("temperature", "top_p", "max_tokens"):
            if required_key not in params_copy:
                raise DomainValidationError(
                    f"پارامتر استنتاج پیش‌فرض '{required_key}' در default_inference_params الزامی است.",
                    field_name="default_inference_params",
                )
        temp = float(params_copy["temperature"])
        top_p = float(params_copy["top_p"])
        max_tok = int(params_copy["max_tokens"])
        if not (0.0 <= temp <= 2.0):
            raise DomainValidationError(
                "پارامتر temperature باید در بازه [0.0, 2.0] باشد.",
                field_name="default_inference_params",
            )
        if not (0.0 < top_p <= 1.0):
            raise DomainValidationError(
                "پارامتر top_p باید در بازه (0.0, 1.0] باشد.",
                field_name="default_inference_params",
            )
        if max_tok < 1:
            raise DomainValidationError(
                "پارامتر max_tokens باید حداقل ۱ باشد.",
                field_name="default_inference_params",
            )
        object.__setattr__(
            self, "default_inference_params", MappingProxyType(params_copy)
        )

        if isinstance(self.known_limitations, str) or not isinstance(
            self.known_limitations, Iterable
        ):
            raise DomainValidationError(
                "فیلد known_limitations باید مجموعه‌ای از رشته‌ها باشد.",
                field_name="known_limitations",
            )
        cleaned_limits = tuple(
            str(lim).strip() for lim in self.known_limitations if str(lim).strip()
        )
        if not cleaned_limits:
            raise DomainValidationError(
                "ثبت حداقل یک محدودیت عملکردی (known_limitations) برای مدل آفلاین طبق فاز ۲ پروپوزال الزامی است.",
                field_name="known_limitations",
            )
        object.__setattr__(self, "known_limitations", cleaned_limits)

        if not isinstance(self.evaluation_notes, str) or not self.evaluation_notes.strip():
            raise DomainValidationError(
                "یادداشت ارزیابی مدل (evaluation_notes) نمی‌تواند خالی باشد.",
                field_name="evaluation_notes",
            )
        object.__setattr__(self, "evaluation_notes", self.evaluation_notes.strip())

    def supports_task(self, task: Union[str, ModelTaskType]) -> bool:
        """بررسی اینکه آیا این مدل از وظیفه مورد نظر پشتیبانی می‌کند یا خیر."""
        task_enum = (
            ModelTaskType(task)
            if isinstance(task, str) and not isinstance(task, ModelTaskType)
            else task
        )
        return task_enum in self.supported_tasks

    def to_dict(self) -> Dict[str, Any]:
        """تبدیل مشخصات مدل به دیکشنری استاندارد."""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "category": self.category.value,
            "quantization_format": self.quantization_format,
            "context_window_size": self.context_window_size,
            "supported_tasks": [t.value for t in self.supported_tasks],
            "default_inference_params": dict(self.default_inference_params),
            "known_limitations": list(self.known_limitations),
            "evaluation_notes": self.evaluation_notes,
            "is_primary_for_category": self.is_primary_for_category,
        }


DEFAULT_OFFLINE_MODELS: Tuple[OfflineModelSpecification, ...] = (
    # ۱. مدل‌های Embedding آفلاین
    OfflineModelSpecification(
        model_id="bge-m3",
        model_name="BAAI/bge-m3 (Multilingual Dense & Sparse Embedding)",
        category=ModelCategory.EMBEDDING,
        quantization_format="GGUF_F16",
        context_window_size=8192,
        supported_tasks=(ModelTaskType.SEMANTIC_INDEXING,),
        default_inference_params={
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 512,
            "embedding_dim": 1024,
        },
        known_limitations=(
            "مصرف حافظه بالاتر نسبت به MiniLM در اسناد بسیار طولانی",
            "نیاز به نرمال‌سازی بردار خروجی برای شباهت کسینوسی دقیق در متون ترکیبی فارسی-انگلیسی",
        ),
        evaluation_notes=(
            "مدل منتخب اصلی (Primary) برای نمایه‌سازی معنایی فارسی و تشخیص تکرار/هم‌پوشانی قطعات دانش با دقت بالا."
        ),
        is_primary_for_category=True,
    ),
    OfflineModelSpecification(
        model_id="paraphrase-multilingual-MiniLM-L12-v2",
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        category=ModelCategory.EMBEDDING,
        quantization_format="ONNX_INT8",
        context_window_size=512,
        supported_tasks=(ModelTaskType.SEMANTIC_INDEXING,),
        default_inference_params={
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 384,
            "embedding_dim": 384,
        },
        known_limitations=(
            "محدودیت پنجره زمینه به ۵۱۲ توکن که نیازمند خردسازی ریزدانه‌تر اسناد است",
            "کاهش دقت بازیابی در اصطلاحات تخصصی نادر نسبت به bge-m3",
        ),
        evaluation_notes=(
            "مدل سبک و سریع جایگزین (Fallback) برای محیط‌های سخت‌افزاری بسیار محدود و اجرای فقط روی CPU."
        ),
        is_primary_for_category=False,
    ),
    # ۲. مدل‌های زبانی کوچک (LLM) آفلاین
    OfflineModelSpecification(
        model_id="qwen2.5:3b-instruct-q4_K_M",
        model_name="Qwen 2.5 3B Instruct (4-bit Quantized)",
        category=ModelCategory.LLM,
        quantization_format="GGUF_Q4_K_M",
        context_window_size=32768,
        supported_tasks=(
            ModelTaskType.QA_ANSWERING,
            ModelTaskType.TECHNICAL_SUMMARIZATION,
            ModelTaskType.EXECUTIVE_MANAGERIAL_SUMMARIZATION,
            ModelTaskType.ISSUE_DECOMPOSITION,
            ModelTaskType.EXECUTOR_EXPERTISE_ANALYSIS,
            ModelTaskType.RFP_AND_FEEDBACK_GENERATION,
            ModelTaskType.LLM_AS_JUDGE_EVALUATION,
        ),
        default_inference_params={
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 1536,
        },
        known_limitations=(
            "احتمال خطای قالب‌بندی JSON در صورتی که پرامپت فاقد Schema صریح و مثال ساخت‌یافته باشد",
            "حساسیت به زمینه بسیار طولانی بالای ۸ هزار توکن در نسخه ۳ میلیارد پارامتری کوانتیزه",
        ),
        evaluation_notes=(
            "مدل منتخب اصلی LLM به دلیل پشتیبانی قوی از زبان فارسی، رعایت ساختار JSON و دقت بالا در داوری ۲۰ نمره‌ای."
        ),
        is_primary_for_category=True,
    ),
    OfflineModelSpecification(
        model_id="llama3.2:3b-instruct-q4_K_M",
        model_name="Meta Llama 3.2 3B Instruct (4-bit Quantized)",
        category=ModelCategory.LLM,
        quantization_format="GGUF_Q4_K_M",
        context_window_size=16384,
        supported_tasks=(
            ModelTaskType.QA_ANSWERING,
            ModelTaskType.TECHNICAL_SUMMARIZATION,
            ModelTaskType.EXECUTIVE_MANAGERIAL_SUMMARIZATION,
            ModelTaskType.ISSUE_DECOMPOSITION,
            ModelTaskType.EXECUTOR_EXPERTISE_ANALYSIS,
            ModelTaskType.RFP_AND_FEEDBACK_GENERATION,
        ),
        default_inference_params={
            "temperature": 0.15,
            "top_p": 0.9,
            "max_tokens": 1024,
        },
        known_limitations=(
            "تسلط واژگانی کمتر در متون اداری و رسمی فارسی نسبت به Qwen2.5-3B",
            "عدم توصیه به عنوان داور اصلی (LLM-as-a-Judge) به دلیل نوسان جزئی در امتیازدهی معیارهای فارسی",
        ),
        evaluation_notes=(
            "مدل نامزد دوم برای خلاصه‌سازی فنی، تجزیه مسئله و مقایسه عملکرد با Qwen2.5 در فازهای ۲ و ۱۴."
        ),
        is_primary_for_category=False,
    ),
    # ۳. مدل‌های بینایی-زبانی (LVM) آفلاین
    OfflineModelSpecification(
        model_id="qwen2.5-vl:3b",
        model_name="Qwen 2.5 VL 3B (Vision-Language Model)",
        category=ModelCategory.LVM,
        quantization_format="GGUF_Q4_K_M",
        context_window_size=8192,
        supported_tasks=(ModelTaskType.DOCUMENT_LAYOUT_ANALYSIS,),
        default_inference_params={
            "temperature": 0.05,
            "top_p": 0.85,
            "max_tokens": 1024,
        },
        known_limitations=(
            "زمان استنتاج بالاتر روی صفحات اسکن‌شده با وضوح بیش از 300 DPI در حالت بدون GPU",
            "نیاز به ارسال تک‌صفحه‌ای برای جلوگیری از سرریز حافظه در جداول بسیار متراکم",
        ),
        evaluation_notes=(
            "مدل منتخب اصلی LVM برای تفسیر جداول پیچیده فارسی، نمودارها و چیدمان اسناد PDF/DOC در خط لوله ورود دانش."
        ),
        is_primary_for_category=True,
    ),
    OfflineModelSpecification(
        model_id="moondream2",
        model_name="Moondream2 1.8B Vision-Language Model",
        category=ModelCategory.LVM,
        quantization_format="GGUF_Q4_K_M",
        context_window_size=2048,
        supported_tasks=(ModelTaskType.DOCUMENT_LAYOUT_ANALYSIS,),
        default_inference_params={
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 512,
        },
        known_limitations=(
            "دقت پایین‌تر در خوانش متون ریز فارسی داخل جداول اسکن‌شده (OCR فارسی محدود)",
            "مناسب صرفاً برای تشخیص ساختار کلی چیدمان صفحه و توصیف تصاویر/نمودارها",
        ),
        evaluation_notes=(
            "مدل فوق‌سبک بینایی-زبانی برای تشخیص اولیه صفحات نیازمند پردازش بصری در سخت‌افزارهای ضعیف."
        ),
        is_primary_for_category=False,
    ),
)


class ModelRegistry:
    """
    رجیستری مدیریت و نگاشت مدل‌های کوچک آفلاین به وظایف سامانه (بخش ۸.۲ و جدول ۱۹.۱ پروپوزال).
    """

    def __init__(self, initialize_defaults: bool = True) -> None:
        self._models: Dict[str, OfflineModelSpecification] = {}
        if initialize_defaults:
            self.initialize_default_models()

    def initialize_default_models(self) -> None:
        """ثبت ۶ مدل نامزد پیش‌فرض آفلاین (۲ مدل Embedding، ۲ مدل LLM و ۲ مدل LVM)."""
        for spec in DEFAULT_OFFLINE_MODELS:
            self._models[spec.model_id] = spec

    def register_model(
        self, spec: OfflineModelSpecification, overwrite: bool = False
    ) -> OfflineModelSpecification:
        """ثبت یک مشخصه مدل آفلاین جدید در رجیستری."""
        if not isinstance(spec, OfflineModelSpecification):
            raise DomainValidationError(
                "ورودی register_model باید از نوع OfflineModelSpecification باشد.",
                field_name="spec",
            )
        if spec.model_id in self._models and not overwrite:
            raise DomainValidationError(
                f"مدل با شناسه '{spec.model_id}' قبلاً در ModelRegistry ثبت شده است.",
                field_name="model_id",
            )
        self._models[spec.model_id] = spec
        return spec

    def get_model(self, model_id: str) -> OfflineModelSpecification:
        """بازیابی مشخصات یک مدل بر اساس شناسه آن."""
        cleaned_id = model_id.strip() if isinstance(model_id, str) else ""
        if cleaned_id not in self._models:
            raise RegistryNotFoundError(
                entity_type="OfflineModelSpecification", identifier=cleaned_id
            )
        return self._models[cleaned_id]

    def get_models_by_category(
        self, category: Union[str, ModelCategory]
    ) -> List[OfflineModelSpecification]:
        """دریافت تمامی مدل‌های ثبت‌شده در یک دسته مشخص (`EMBEDDING`, `LLM`, `LVM`)."""
        cat_enum = (
            ModelCategory(category)
            if isinstance(category, str) and not isinstance(category, ModelCategory)
            else category
        )
        return [m for m in self._models.values() if m.category == cat_enum]

    def get_models_for_task(
        self, task: Union[str, ModelTaskType]
    ) -> List[OfflineModelSpecification]:
        """دریافت فهرست تمامی مدل‌های سازگار با یک وظیفه مشخص."""
        task_enum = (
            ModelTaskType(task)
            if isinstance(task, str) and not isinstance(task, ModelTaskType)
            else task
        )
        return [m for m in self._models.values() if m.supports_task(task_enum)]

    def get_primary_model_for_task(
        self, task: Union[str, ModelTaskType]
    ) -> OfflineModelSpecification:
        """
        انتخاب مدل اصلی (Primary) پیشنهادی برای یک وظیفه مشخص.
        اگر مدلی با پرچم `is_primary_for_category=True` برای آن وظیفه موجود باشد، اولویت دارد.
        """
        candidates = self.get_models_for_task(task)
        if not candidates:
            raise RegistryNotFoundError(
                entity_type="ModelForTask", identifier=str(task)
            )
        for candidate in candidates:
            if candidate.is_primary_for_category:
                return candidate
        return candidates[0]

    def validate_model_for_task(
        self, model_id: str, task: Union[str, ModelTaskType]
    ) -> OfflineModelSpecification:
        """
        بررسی سازگاری مدل انتخاب‌شده با وظیفه درخواستی.
        در صورت عدم پشتیبانی، استثنای `IncompatibleModelTaskError` صادر می‌شود.
        """
        spec = self.get_model(model_id)
        task_enum = (
            ModelTaskType(task)
            if isinstance(task, str) and not isinstance(task, ModelTaskType)
            else task
        )
        if not spec.supports_task(task_enum):
            raise IncompatibleModelTaskError(
                model_id=spec.model_id, task_name=task_enum.value
            )
        return spec

    @property
    def models(self) -> List[OfflineModelSpecification]:
        """فهرست تمامی مدل‌های ثبت‌شده در رجیستری."""
        return list(self._models.values())

    def to_dict(self) -> Dict[str, Any]:
        """تبدیل رجیستری مدل‌ها به دیکشنری استاندارد."""
        return {
            "total_models": len(self._models),
            "models": [m.to_dict() for m in self.models],
        }


__all__ = [
    "ModelCategory",
    "ModelTaskType",
    "OfflineModelSpecification",
    "DEFAULT_OFFLINE_MODELS",
    "ModelRegistry",
]
