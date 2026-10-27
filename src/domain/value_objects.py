"""
اشیاء مقداری (Value Objects) هسته دامنه طبق جدول ۵.۳، جدول ۶.۴، بخش ۷.۵ و جدول ۱۶.۱ پروپوزال.

تمامی کلاس‌های این ماژول به صورت تغییرناپذیر (`@dataclass(frozen=True)`) پیاده‌سازی شده‌اند
و ناوردایی‌های دامنه (Domain Invariants) را به شکل سخت‌گیرانه در متد `__post_init__` اعتبارسنجی می‌کنند.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, Optional, Tuple, Union

from src.domain.exceptions import DomainValidationError, JudgeThresholdNotMetError

_VERSION_TAG_PATTERN = re.compile(r"^v\d+\.\d+(\.\d+)?(-[a-zA-Z0-9._-]+)?$")

ALLOWED_DOCUMENT_TYPES: Tuple[str, ...] = ("DOC", "DOCX", "PDF")
ALLOWED_CONFIDENTIALITY_LEVELS: Tuple[str, ...] = (
    "PUBLIC",
    "INTERNAL",
    "CONFIDENTIAL",
    "RESTRICTED",
)
ALLOWED_PRIORITY_LEVELS: Tuple[str, ...] = ("LOW", "MEDIUM", "HIGH", "CRITICAL")


class TransferDetailLevel(str, Enum):
    """
    سطوح جزئیات انتقال دانش میان لایه‌های سازمانی (جدول ۶.۴ پروپوزال).
    """

    EXECUTIVE_HIGH = "EXECUTIVE_HIGH"
    MANAGERIAL = "MANAGERIAL"
    TECHNICAL_SUMMARY = "TECHNICAL_SUMMARY"
    TECHNICAL_DETAILED = "TECHNICAL_DETAILED"

    @property
    def label_fa(self) -> str:
        """عنوان فارسی سطح جزئیات انتقال."""
        labels = {
            TransferDetailLevel.EXECUTIVE_HIGH: "اجرایی عالی",
            TransferDetailLevel.MANAGERIAL: "مدیریتی",
            TransferDetailLevel.TECHNICAL_SUMMARY: "خلاصه فنی",
            TransferDetailLevel.TECHNICAL_DETAILED: "فنی تفصیلی",
        }
        return labels[self]

    @property
    def target_audience_fa(self) -> str:
        """مخاطب هدف در سلسله‌مراتب سازمانی طبق جدول ۶.۴."""
        audiences = {
            TransferDetailLevel.EXECUTIVE_HIGH: "تصمیم‌گیران ارشد",
            TransferDetailLevel.MANAGERIAL: "مدیران سازمان مادر",
            TransferDetailLevel.TECHNICAL_SUMMARY: "رهبران فنی و ناظران برنامه",
            TransferDetailLevel.TECHNICAL_DETAILED: "متخصصان و تیم‌های لایه پایین‌تر",
        }
        return audiences[self]

    @property
    def typical_output_fa(self) -> str:
        """خروجی معمول سطح جزئیات طبق جدول ۶.۴."""
        outputs = {
            TransferDetailLevel.EXECUTIVE_HIGH: (
                "دستاوردهای کلیدی، ریسک‌های اصلی، مسائل حل‌نشده، وابستگی‌های بحرانی و پیشنهادهای راهبردی"
            ),
            TransferDetailLevel.MANAGERIAL: (
                "وضعیت پروژه، وضعیت مسئله، شکاف پژوهشی، تصمیمات مورد نیاز، نیاز منابع و وضعیت RFP یا پیشنهادها"
            ),
            TransferDetailLevel.TECHNICAL_SUMMARY: (
                "روش‌ها، نتایج پژوهشی، اسناد مرتبط، محدودیت‌ها، گزینه‌های راه‌حل و توانمندی مجریان"
            ),
            TransferDetailLevel.TECHNICAL_DETAILED: (
                "جزئیات کامل مسئله، ارجاع اسناد، شکست زیرمسئله‌ها، نگاشت مجری، شواهد منبع و یادداشت‌های آزمون مدل و پرامپت"
            ),
        }
        return outputs[self]

    @classmethod
    def for_layer(cls, layer_index: int) -> TransferDetailLevel:
        """پیشنهاد سطح جزئیات پیش‌فرض بر اساس شماره لایه هدف."""
        mapping = {
            0: cls.EXECUTIVE_HIGH,
            1: cls.MANAGERIAL,
            2: cls.TECHNICAL_SUMMARY,
            3: cls.TECHNICAL_DETAILED,
            4: cls.TECHNICAL_DETAILED,
        }
        if layer_index not in mapping:
            raise DomainValidationError(
                f"شماره لایه '{layer_index}' برای تعیین سطح جزئیات انتقال معتبر نیست.",
                field_name="layer_index",
            )
        return mapping[layer_index]


@dataclass(frozen=True)
class NodeIdentifier:
    """
    شناسه پایدار و غیرخالی گره سازمانی (جدول ۵.۳).
    """

    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str):
            raise DomainValidationError(
                "شناسه گره (NodeIdentifier) باید از نوع رشته (str) باشد.",
                field_name="value",
            )
        cleaned = self.value.strip()
        if not cleaned:
            raise DomainValidationError(
                "شناسه گره (NodeIdentifier) نمی‌تواند خالی یا صرفاً فاصله باشد.",
                field_name="value",
            )
        if any(ch.isspace() for ch in cleaned):
            raise DomainValidationError(
                f"شناسه گره '{cleaned}' نباید شامل کاراکترهای فاصله در میانه رشته باشد.",
                field_name="value",
            )
        object.__setattr__(self, "value", cleaned)

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, order=True)
class LayerIndex:
    """
    جایگاه لایه در سلسله‌مراتب سازمانی (از لایه ۰ تا لایه ۴ طبق بخش ۶.۲ پروپوزال).

    - لایه ۰: لایه راهبردی (مرجع عالی تصمیم‌گیری یا مالک کلان ساختار سازمانی)
    - لایه ۱: سازمان‌های مادر (شرکت‌های اصلی، سازمان‌های اصلی، ستادها یا معاونت‌های راهبردی)
    - لایه ۲: زیرمجموعه‌ها و واحدهای تخصصی (شرکت‌های تابعه، مراکز پژوهشی، واحدهای مهندسی/عملیاتی)
    - لایه ۳: پروژه‌ها، تیم‌ها و گره‌های اجرایی (تیم‌های پروژه، واحدهای پیاده‌سازی، گروه‌های پژوهشی)
    - لایه ۴: منابع داخلی دانش اختیاری (کتابخانه‌ها، مجموعه اسناد، سوابق پرسنلی، آرشیو پروژه)
    """

    value: int

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, int):
            raise DomainValidationError(
                "شماره لایه (LayerIndex) باید یک عدد صحیح (int) باشد.",
                field_name="value",
            )
        if not (0 <= self.value <= 4):
            raise DomainValidationError(
                f"شماره لایه باید بین ۰ تا ۴ باشد؛ مقدار دریافتی: {self.value}.",
                field_name="value",
            )

    @property
    def name_fa(self) -> str:
        """نام رسمی لایه طبق جدول ۶.۲ پروپوزال."""
        names = {
            0: "لایه صفر: لایه راهبردی",
            1: "لایه یک: سازمان‌های مادر",
            2: "لایه دو: زیرمجموعه‌ها و واحدهای تخصصی",
            3: "لایه سه: پروژه‌ها، تیم‌ها و گره‌های اجرایی",
            4: "لایه چهار: منابع داخلی دانش اختیاری",
        }
        return names[self.value]

    @property
    def is_strategic_root(self) -> bool:
        """بررسی اینکه آیا این لایه، بالاترین سطح سلسله‌مراتب (لایه صفر راهبردی) است یا خیر."""
        return self.value == 0

    def is_higher_authority_than(self, other: LayerIndex) -> bool:
        """
        بررسی اینکه آیا این لایه در سلسله‌مراتب سازمانی بالاتر از لایه دیگر است یا خیر.
        در مدل لایه‌ای پروژه، لایه ۰ بالاترین سطح و لایه ۴ پایین‌ترین سطح است.
        """
        return self.value < other.value


@dataclass(frozen=True)
class KnowledgeScope:
    """
    دامنه، سطح محرمانگی و محدوده استفاده از یک قلم دانش (جدول ۵.۳ و بخش ۲۰ پروپوزال).
    """

    domain: str
    confidentiality_level: str = "INTERNAL"
    allowed_node_ids: Tuple[str, ...] = ()
    max_target_layer: LayerIndex = field(default_factory=lambda: LayerIndex(0))

    def __post_init__(self) -> None:
        if not isinstance(self.domain, str) or not self.domain.strip():
            raise DomainValidationError(
                "دامنه دانش (domain) در KnowledgeScope نمی‌تواند خالی باشد.",
                field_name="domain",
            )
        object.__setattr__(self, "domain", self.domain.strip())

        if not isinstance(self.confidentiality_level, str) or not self.confidentiality_level.strip():
            raise DomainValidationError(
                "سطح محرمانگی (confidentiality_level) نمی‌تواند خالی باشد.",
                field_name="confidentiality_level",
            )
        normalized_conf = self.confidentiality_level.strip().upper()
        if normalized_conf not in ALLOWED_CONFIDENTIALITY_LEVELS:
            raise DomainValidationError(
                f"سطح محرمانگی '{self.confidentiality_level}' نامعتبر است. مقادیر مجاز: {ALLOWED_CONFIDENTIALITY_LEVELS}",
                field_name="confidentiality_level",
            )
        object.__setattr__(self, "confidentiality_level", normalized_conf)

        if isinstance(self.allowed_node_ids, str) or not isinstance(self.allowed_node_ids, Iterable):
            raise DomainValidationError(
                "فهرست گره‌های مجاز (allowed_node_ids) باید مجموعه‌ای از شناسه‌ها باشد.",
                field_name="allowed_node_ids",
            )
        cleaned_nodes = tuple(str(nid).strip() for nid in self.allowed_node_ids if str(nid).strip())
        object.__setattr__(self, "allowed_node_ids", cleaned_nodes)

        if isinstance(self.max_target_layer, int) and not isinstance(self.max_target_layer, bool):
            object.__setattr__(self, "max_target_layer", LayerIndex(self.max_target_layer))
        elif not isinstance(self.max_target_layer, LayerIndex):
            raise DomainValidationError(
                "فیلد max_target_layer باید از نوع LayerIndex یا عدد صحیح معتبر باشد.",
                field_name="max_target_layer",
            )

    def is_accessible_by_node(self, node_id: Union[str, NodeIdentifier]) -> bool:
        """بررسی مجوز دسترسی یک گره به این قلم دانش بر اساس محدوده تعریف‌شده."""
        raw_id = str(node_id).strip()
        if not self.allowed_node_ids:
            return True
        return raw_id in self.allowed_node_ids


@dataclass(frozen=True)
class VersionTag:
    """
    شناسه تغییرناپذیر نسخه قلم دانش یا بسته انتقال (جدول ۵.۳ و بخش ۷.۶ پروپوزال).
    """

    tag: str
    sequence_number: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.tag, str) or not self.tag.strip():
            raise DomainValidationError(
                "برچسب نسخه (tag) در VersionTag نمی‌تواند خالی باشد.",
                field_name="tag",
            )
        cleaned_tag = self.tag.strip()
        if not _VERSION_TAG_PATTERN.match(cleaned_tag):
            raise DomainValidationError(
                f"فرمت برچسب نسخه '{cleaned_tag}' نامعتبر است. فرمت استاندارد مانند 'v1.0.0' یا 'v1.0' الزامی است.",
                field_name="tag",
            )
        object.__setattr__(self, "tag", cleaned_tag)

        if isinstance(self.sequence_number, bool) or not isinstance(self.sequence_number, int):
            raise DomainValidationError(
                "شماره توالی نسخه (sequence_number) باید عدد صحیح باشد.",
                field_name="sequence_number",
            )
        if self.sequence_number < 1:
            raise DomainValidationError(
                f"شماره توالی نسخه باید حداقل ۱ باشد؛ مقدار دریافتی: {self.sequence_number}.",
                field_name="sequence_number",
            )

    def next_version(self, new_tag: str) -> VersionTag:
        """تولید شیء مقداری نسخه بعدی با افزایش خودکار شماره توالی."""
        return VersionTag(tag=new_tag, sequence_number=self.sequence_number + 1)

    def __str__(self) -> str:
        return self.tag


@dataclass(frozen=True, order=True)
class ProgressPercentage:
    """
    مقدار پیشرفت معتبر بین ۰ تا ۱۰۰ (جدول ۵.۳ و بخش ۱۴.۱ پروپوزال).
    """

    value: float

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise DomainValidationError(
                "مقدار پیشرفت (ProgressPercentage) باید عدد (int یا float) باشد.",
                field_name="value",
            )
        numeric_val = float(self.value)
        if not (0.0 <= numeric_val <= 100.0):
            raise DomainValidationError(
                f"درصد پیشرفت باید بین 0.0 و 100.0 باشد؛ مقدار دریافتی: {self.value}.",
                field_name="value",
            )
        object.__setattr__(self, "value", round(numeric_val, 2))

    @property
    def is_completed(self) -> bool:
        """آیا پیشرفت به ۱۰۰ درصد رسیده است؟"""
        return self.value == 100.0


@dataclass(frozen=True)
class JudgeRubric:
    """
    چارچوب امتیازدهی ۲۰ نمره‌ای سازوکار LLM به‌عنوان داور (جدول ۵.۳ و جدول ۱۶.۱ پروپوزال).

    سقف امتیاز هر معیار طبق جدول ۱۶.۱:
    1. وفاداری به دانش منبع (source_fidelity): حداکثر ۵ امتیاز
    2. کامل بودن نسبت به درخواست کاربر (completeness): حداکثر ۴ امتیاز
    3. تناسب سطح جزئیات با لایه هدف (detail_level_fit): حداکثر ۳ امتیاز
    4. مدیریت درست عدم قطعیت و شکاف‌ها (uncertainty_and_gap_handling): حداکثر ۳ امتیاز
    5. وضوح و ساختار حرفه‌ای (clarity_and_structure): حداکثر ۲ امتیاز
    6. قابلیت ارجاع به منبع (source_attribution): حداکثر ۲ امتیاز
    7. نبود ادعای بدون پشتوانه (no_unsupported_claims): حداکثر ۱ امتیاز
    مجموع کل: ۲۰ امتیاز
    """

    source_fidelity: int
    completeness: int
    detail_level_fit: int
    uncertainty_and_gap_handling: int
    clarity_and_structure: int
    source_attribution: int
    no_unsupported_claims: int

    MAX_SCORES: Dict[str, int] = field(
        default_factory=lambda: {
            "source_fidelity": 5,
            "completeness": 4,
            "detail_level_fit": 3,
            "uncertainty_and_gap_handling": 3,
            "clarity_and_structure": 2,
            "source_attribution": 2,
            "no_unsupported_claims": 1,
        },
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        criteria_limits = {
            "source_fidelity": ("وفاداری به دانش منبع", 5),
            "completeness": ("کامل بودن نسبت به درخواست کاربر", 4),
            "detail_level_fit": ("تناسب سطح جزئیات با لایه هدف", 3),
            "uncertainty_and_gap_handling": ("مدیریت درست عدم قطعیت و شکاف‌ها", 3),
            "clarity_and_structure": ("وضوح و ساختار حرفه‌ای", 2),
            "source_attribution": ("قابلیت ارجاع به منبع", 2),
            "no_unsupported_claims": ("نبود ادعای بدون پشتوانه", 1),
        }
        for field_name, (label_fa, max_allowed) in criteria_limits.items():
            score_val = getattr(self, field_name)
            if isinstance(score_val, bool) or not isinstance(score_val, int):
                raise DomainValidationError(
                    f"امتیاز معیار '{label_fa}' ({field_name}) باید عدد صحیح باشد.",
                    field_name=field_name,
                )
            if not (0 <= score_val <= max_allowed):
                raise DomainValidationError(
                    f"امتیاز معیار '{label_fa}' ({field_name}) باید بین ۰ تا {max_allowed} باشد؛ "
                    f"مقدار دریافتی: {score_val}.",
                    field_name=field_name,
                )

    def total_score(self) -> int:
        """محاسبه مجموع امتیاز اکتسابی از ۲۰ نمره."""
        return (
            self.source_fidelity
            + self.completeness
            + self.detail_level_fit
            + self.uncertainty_and_gap_handling
            + self.clarity_and_structure
            + self.source_attribution
            + self.no_unsupported_claims
        )

    @property
    def max_possible_score(self) -> int:
        """سقف امتیاز چارچوب داوری (۲۰ نمره)."""
        return 20

    def to_dict(self) -> Dict[str, int]:
        """تبدیل جزئیات امتیازدهی و مجموع آن به دیکشنری استاندارد."""
        return {
            "source_fidelity": self.source_fidelity,
            "completeness": self.completeness,
            "detail_level_fit": self.detail_level_fit,
            "uncertainty_and_gap_handling": self.uncertainty_and_gap_handling,
            "clarity_and_structure": self.clarity_and_structure,
            "source_attribution": self.source_attribution,
            "no_unsupported_claims": self.no_unsupported_claims,
            "total_score": self.total_score(),
        }


@dataclass(frozen=True)
class JudgeThreshold:
    """
    حداقل امتیاز قابل قبول در سازوکار داوری (جدول ۵.۳ و بخش ۱۶.۱ پروپوزال: پیش‌فرض ۱۸ از ۲۰).
    """

    minimum_score: int = 18
    max_score: int = 20

    def __post_init__(self) -> None:
        if isinstance(self.minimum_score, bool) or not isinstance(self.minimum_score, int):
            raise DomainValidationError(
                "حداقل امتیاز قابل قبول (minimum_score) باید عدد صحیح باشد.",
                field_name="minimum_score",
            )
        if isinstance(self.max_score, bool) or not isinstance(self.max_score, int):
            raise DomainValidationError(
                "سقف امتیاز (max_score) باید عدد صحیح باشد.",
                field_name="max_score",
            )
        if self.max_score != 20:
            raise DomainValidationError(
                f"سقف امتیاز چارچوب داوری در این پروژه باید ۲۰ باشد؛ مقدار دریافتی: {self.max_score}.",
                field_name="max_score",
            )
        if not (1 <= self.minimum_score <= self.max_score):
            raise DomainValidationError(
                f"حد آستانه داوری باید بین ۱ تا {self.max_score} باشد؛ مقدار دریافتی: {self.minimum_score}.",
                field_name="minimum_score",
            )

    def is_satisfied_by(self, rubric_or_score: Union[JudgeRubric, int]) -> bool:
        """بررسی اینکه آیا امتیاز داده شده حد آستانه را احراز می‌کند یا خیر."""
        score = (
            rubric_or_score.total_score()
            if isinstance(rubric_or_score, JudgeRubric)
            else rubric_or_score
        )
        return score >= self.minimum_score

    def enforce(self, rubric_or_score: Union[JudgeRubric, int]) -> None:
        """اعتبارسنجی امتیاز و پرتاب استثنای `JudgeThresholdNotMetError` در صورت عدم احراز آستانه."""
        score = (
            rubric_or_score.total_score()
            if isinstance(rubric_or_score, JudgeRubric)
            else rubric_or_score
        )
        if score < self.minimum_score:
            raise JudgeThresholdNotMetError(
                achieved_score=score,
                required_threshold=self.minimum_score,
            )


@dataclass(frozen=True)
class RetrievalQuery:
    """
    نمایش نرمال‌شده پرسش کاربر یا سامانه جهت بازیابی از هسته RAG (جدول ۵.۳ و بخش ۸.۱ پروپوزال).
    """

    raw_query: str
    target_node_id: Optional[NodeIdentifier] = None
    normalized_text: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.raw_query, str) or not self.raw_query.strip():
            raise DomainValidationError(
                "متن پرسش بازیابی (raw_query) نمی‌تواند خالی باشد.",
                field_name="raw_query",
            )
        # نرمال‌سازی حروف عربی به فارسی و حذف فاصله‌های تکراری
        normalized = (
            self.raw_query.strip()
            .replace("ي", "ی")
            .replace("ك", "ک")
            .replace("\u200c", " ")
        )
        normalized = re.sub(r"\s+", " ", normalized).strip()
        if not normalized:
            raise DomainValidationError(
                "متن نرمال‌شده پرسش نمی‌تواند خالی باشد.",
                field_name="raw_query",
            )
        object.__setattr__(self, "normalized_text", normalized)

        if isinstance(self.target_node_id, str):
            object.__setattr__(self, "target_node_id", NodeIdentifier(self.target_node_id))
        elif self.target_node_id is not None and not isinstance(self.target_node_id, NodeIdentifier):
            raise DomainValidationError(
                "شناسه گره هدف (target_node_id) باید از نوع NodeIdentifier یا str معتبر باشد.",
                field_name="target_node_id",
            )


@dataclass(frozen=True, order=True)
class SimilarityScore:
    """
    امتیاز شباهت در بازیابی برداری، ترکیبی یا تشخیص تکرار و هم‌پوشانی (بین 0.0 تا 1.0 طبق جدول ۵.۳).
    """

    value: float

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise DomainValidationError(
                "امتیاز شباهت (SimilarityScore) باید عدد (int یا float) باشد.",
                field_name="value",
            )
        numeric_val = float(self.value)
        if not (0.0 <= numeric_val <= 1.0):
            raise DomainValidationError(
                f"امتیاز شباهت باید در بازه [0.0, 1.0] قرار داشته باشد؛ مقدار دریافتی: {self.value}.",
                field_name="value",
            )
        object.__setattr__(self, "value", round(numeric_val, 6))

    def exceeds_threshold(self, threshold: float) -> bool:
        """بررسی عبور امتیاز شباهت از یک آستانه مشخص."""
        return self.value >= threshold


@dataclass(frozen=True)
class MetadataRecord:
    """
    فراداده ساخت‌یافته متصل به مسئله یا قلم دانش شامل تمام ۱۸ فیلد بخش ۷.۵ پروپوزال:

    1. حوزه (`domain`)
    2. زیرحوزه (`sub_domain`)
    3. سازمان یا گره (`organization_or_node`)
    4. پروژه (`project`)
    5. نویسنده یا مالک (`author_or_owner`)
    6. نوع سند (`document_type` - پشتیبانی از DOC، DOCX و PDF)
    7. سطح محرمانگی (`confidentiality_level`)
    8. شناسه نسخه اولیه (`initial_version_id`)
    9. شناسه نسخه فعال (`active_version_id`)
    10. مسئله مرتبط (`related_issue_id`)
    11. مجری مرتبط (`related_executor_id`)
    12. RFP مرتبط (`related_rfp_id`)
    13. وضعیت (`status`)
    14. اولویت (`priority`)
    15. برچسب‌ها (`tags`)
    16. قابلیت اعتماد منبع (`source_reliability`)
    17. لایه هدف انتقال (`target_transfer_layer`)
    18. سطح جزئیات پیشنهادی (`suggested_detail_level`)
    """

    domain: str
    sub_domain: str
    organization_or_node: str
    project: str
    author_or_owner: str
    document_type: str
    confidentiality_level: str
    initial_version_id: str
    active_version_id: str
    related_issue_id: Optional[str] = None
    related_executor_id: Optional[str] = None
    related_rfp_id: Optional[str] = None
    status: str = "ACTIVE"
    priority: str = "MEDIUM"
    tags: Tuple[str, ...] = ()
    source_reliability: float = 1.0
    target_transfer_layer: LayerIndex = field(default_factory=lambda: LayerIndex(1))
    suggested_detail_level: TransferDetailLevel = TransferDetailLevel.MANAGERIAL

    def __post_init__(self) -> None:
        required_str_fields = {
            "domain": "حوزه",
            "sub_domain": "زیرحوزه",
            "organization_or_node": "سازمان یا گره",
            "project": "پروژه",
            "author_or_owner": "نویسنده یا مالک",
            "document_type": "نوع سند",
            "confidentiality_level": "سطح محرمانگی",
            "initial_version_id": "شناسه نسخه اولیه",
            "active_version_id": "شناسه نسخه فعال",
            "status": "وضعیت",
            "priority": "اولویت",
        }
        for attr_name, label_fa in required_str_fields.items():
            raw_val = getattr(self, attr_name)
            if not isinstance(raw_val, str) or not raw_val.strip():
                raise DomainValidationError(
                    f"فیلد فراداده '{label_fa}' ({attr_name}) الزامی است و نمی‌تواند خالی باشد.",
                    field_name=attr_name,
                )
            object.__setattr__(self, attr_name, raw_val.strip())

        doc_type_upper = self.document_type.upper()
        if doc_type_upper not in ALLOWED_DOCUMENT_TYPES:
            raise DomainValidationError(
                f"نوع سند '{self.document_type}' پشتیبانی نمی‌شود. قالب‌های مجاز: {ALLOWED_DOCUMENT_TYPES}.",
                field_name="document_type",
            )
        object.__setattr__(self, "document_type", doc_type_upper)

        conf_upper = self.confidentiality_level.upper()
        if conf_upper not in ALLOWED_CONFIDENTIALITY_LEVELS:
            raise DomainValidationError(
                f"سطح محرمانگی '{self.confidentiality_level}' معتبر نیست. مقادیر مجاز: {ALLOWED_CONFIDENTIALITY_LEVELS}.",
                field_name="confidentiality_level",
            )
        object.__setattr__(self, "confidentiality_level", conf_upper)

        priority_upper = self.priority.upper()
        if priority_upper not in ALLOWED_PRIORITY_LEVELS:
            raise DomainValidationError(
                f"اولویت '{self.priority}' معتبر نیست. مقادیر مجاز: {ALLOWED_PRIORITY_LEVELS}.",
                field_name="priority",
            )
        object.__setattr__(self, "priority", priority_upper)

        if isinstance(self.tags, str) or not isinstance(self.tags, Iterable):
            raise DomainValidationError(
                "فیلد برچسب‌ها (tags) باید یک مجموعه یا تاپل از رشته‌ها باشد.",
                field_name="tags",
            )
        cleaned_tags = tuple(str(t).strip() for t in self.tags if str(t).strip())
        object.__setattr__(self, "tags", cleaned_tags)

        if isinstance(self.source_reliability, bool) or not isinstance(
            self.source_reliability, (int, float)
        ):
            raise DomainValidationError(
                "قابلیت اعتماد منبع (source_reliability) باید عدد بین 0.0 تا 1.0 باشد.",
                field_name="source_reliability",
            )
        rel_val = float(self.source_reliability)
        if not (0.0 <= rel_val <= 1.0):
            raise DomainValidationError(
                f"قابلیت اعتماد منبع باید در بازه [0.0, 1.0] باشد؛ مقدار دریافتی: {self.source_reliability}.",
                field_name="source_reliability",
            )
        object.__setattr__(self, "source_reliability", round(rel_val, 4))

        if isinstance(self.target_transfer_layer, int) and not isinstance(
            self.target_transfer_layer, bool
        ):
            object.__setattr__(
                self, "target_transfer_layer", LayerIndex(self.target_transfer_layer)
            )
        elif not isinstance(self.target_transfer_layer, LayerIndex):
            raise DomainValidationError(
                "لایه هدف انتقال (target_transfer_layer) باید از نوع LayerIndex یا عدد صحیح ۰ تا ۴ باشد.",
                field_name="target_transfer_layer",
            )

        if isinstance(self.suggested_detail_level, str) and not isinstance(
            self.suggested_detail_level, TransferDetailLevel
        ):
            try:
                object.__setattr__(
                    self,
                    "suggested_detail_level",
                    TransferDetailLevel(self.suggested_detail_level),
                )
            except ValueError as exc:
                raise DomainValidationError(
                    f"سطح جزئیات پیشنهادی '{self.suggested_detail_level}' معتبر نیست.",
                    field_name="suggested_detail_level",
                ) from exc
        elif not isinstance(self.suggested_detail_level, TransferDetailLevel):
            raise DomainValidationError(
                "فیلد suggested_detail_level باید از نوع TransferDetailLevel باشد.",
                field_name="suggested_detail_level",
            )

    def with_active_version(self, new_version_id: str) -> MetadataRecord:
        """ایجاد رکورد فراداده جدید با به‌روزرسانی شناسه نسخه فعال (حفظ تغییرناپذیری)."""
        return MetadataRecord(
            domain=self.domain,
            sub_domain=self.sub_domain,
            organization_or_node=self.organization_or_node,
            project=self.project,
            author_or_owner=self.author_or_owner,
            document_type=self.document_type,
            confidentiality_level=self.confidentiality_level,
            initial_version_id=self.initial_version_id,
            active_version_id=new_version_id,
            related_issue_id=self.related_issue_id,
            related_executor_id=self.related_executor_id,
            related_rfp_id=self.related_rfp_id,
            status=self.status,
            priority=self.priority,
            tags=self.tags,
            source_reliability=self.source_reliability,
            target_transfer_layer=self.target_transfer_layer,
            suggested_detail_level=self.suggested_detail_level,
        )

    def to_dict(self) -> Dict[str, Any]:
        """تبدیل رکورد فراداده ۱۸ فیلدی به دیکشنری استاندارد."""
        return {
            "domain": self.domain,
            "sub_domain": self.sub_domain,
            "organization_or_node": self.organization_or_node,
            "project": self.project,
            "author_or_owner": self.author_or_owner,
            "document_type": self.document_type,
            "confidentiality_level": self.confidentiality_level,
            "initial_version_id": self.initial_version_id,
            "active_version_id": self.active_version_id,
            "related_issue_id": self.related_issue_id,
            "related_executor_id": self.related_executor_id,
            "related_rfp_id": self.related_rfp_id,
            "status": self.status,
            "priority": self.priority,
            "tags": list(self.tags),
            "source_reliability": self.source_reliability,
            "target_transfer_layer": self.target_transfer_layer.value,
            "suggested_detail_level": self.suggested_detail_level.value,
        }


__all__ = [
    "ALLOWED_CONFIDENTIALITY_LEVELS",
    "ALLOWED_DOCUMENT_TYPES",
    "ALLOWED_PRIORITY_LEVELS",
    "NodeIdentifier",
    "LayerIndex",
    "KnowledgeScope",
    "VersionTag",
    "ProgressPercentage",
    "JudgeRubric",
    "JudgeThreshold",
    "RetrievalQuery",
    "SimilarityScore",
    "TransferDetailLevel",
    "MetadataRecord",
]
