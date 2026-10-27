"""
موجودیت‌های کلیدی دامنه (Domain Entities) طبق جدول ۵.۲ پروپوزال.

این ماژول ۱۳ موجودیت اصلی هسته RAG سلسله‌مراتبی آفلاین را همراه با قواعد کسب‌وکار پایه،
کنترل ناوردایی‌ها و متدهای تغییر وضعیت مدل‌سازی می‌کند:
1. `Node`
2. `Layer`
3. `KnowledgeItem`
4. `KnowledgeVersion`
5. `KnowledgeChunk`
6. `Issue`
7. `SubIssue`
8. `Executor`
9. `TransferPackage`
10. `JudgeScore`
11. `RFP`
12. `Proposal`
13. `FeedbackReport`
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from src.domain.exceptions import (
    DomainValidationError,
    InvalidLayerHierarchyError,
    LeafNodeBoundaryViolationError,
)
from src.domain.value_objects import (
    JudgeRubric,
    JudgeThreshold,
    KnowledgeScope,
    LayerIndex,
    MetadataRecord,
    NodeIdentifier,
    ProgressPercentage,
    TransferDetailLevel,
    VersionTag,
)


def _ensure_non_empty_str(value: Any, field_name: str, label_fa: str) -> str:
    """تابع کمکی داخلی برای اعتبارسنجی رشته‌های غیرخالی در موجودیت‌ها."""
    if not isinstance(value, str) or not value.strip():
        raise DomainValidationError(
            f"فیلد '{label_fa}' ({field_name}) باید یک رشته غیرخالی باشد.",
            field_name=field_name,
        )
    return value.strip()


def _to_str_tuple(items: Iterable[str], field_name: str) -> Tuple[str, ...]:
    """تبدیل مجموعه رشته‌ها به تاپل تغییرناپذیر و پاک‌سازی‌شده."""
    if isinstance(items, str) or not isinstance(items, Iterable):
        raise DomainValidationError(
            f"فیلد '{field_name}' باید مجموعه‌ای از رشته‌ها باشد.",
            field_name=field_name,
        )
    return tuple(str(item).strip() for item in items if str(item).strip())


@dataclass
class Node:
    """
    موجودیت گره سازمانی (جدول ۵.۲ و بخش ۶ پروپوزال).
    نماینده یک شرکت، سازمان، زیرمجموعه، واحد، دفتر پروژه یا مالک دانش عملیاتی.
    """

    node_id: NodeIdentifier
    name: str
    layer_index: LayerIndex
    domain_of_activity: str
    parent_node_id: Optional[NodeIdentifier] = None
    is_leaf_boundary: bool = False
    reporting_rules: Dict[str, Any] = field(default_factory=dict)
    required_transfer_detail_level: TransferDetailLevel = TransferDetailLevel.MANAGERIAL
    confidentiality_constraints: Tuple[str, ...] = ("INTERNAL",)
    child_node_ids: List[NodeIdentifier] = field(default_factory=list)

    def __post_init__(self) -> None:
        if isinstance(self.node_id, str):
            self.node_id = NodeIdentifier(self.node_id)
        elif not isinstance(self.node_id, NodeIdentifier):
            raise DomainValidationError(
                "شناسه گره (node_id) باید از نوع NodeIdentifier باشد.",
                field_name="node_id",
            )

        self.name = _ensure_non_empty_str(self.name, "name", "نام گره")
        self.domain_of_activity = _ensure_non_empty_str(
            self.domain_of_activity, "domain_of_activity", "حوزه فعالیت گره"
        )

        if isinstance(self.layer_index, int) and not isinstance(self.layer_index, bool):
            self.layer_index = LayerIndex(self.layer_index)
        elif not isinstance(self.layer_index, LayerIndex):
            raise DomainValidationError(
                "جایگاه لایه (layer_index) باید از نوع LayerIndex باشد.",
                field_name="layer_index",
            )

        if isinstance(self.parent_node_id, str):
            self.parent_node_id = NodeIdentifier(self.parent_node_id)
        elif self.parent_node_id is not None and not isinstance(
            self.parent_node_id, NodeIdentifier
        ):
            raise DomainValidationError(
                "شناسه گره والد (parent_node_id) باید از نوع NodeIdentifier یا None باشد.",
                field_name="parent_node_id",
            )

        if self.parent_node_id is not None and self.parent_node_id == self.node_id:
            raise InvalidLayerHierarchyError(
                f"گره '{self.node_id}' نمی‌تواند والدِ خودش باشد."
            )

        if self.layer_index.is_strategic_root and self.parent_node_id is not None:
            raise InvalidLayerHierarchyError(
                f"گره لایه صفر راهبردی ('{self.node_id}') نباید دارای گره والد باشد."
            )

        if not self.layer_index.is_strategic_root and self.parent_node_id is None:
            raise InvalidLayerHierarchyError(
                f"گره '{self.node_id}' در لایه {self.layer_index.value} باید دارای یک گره والد معتبر در لایه بالاتر باشد."
            )

        if isinstance(self.required_transfer_detail_level, str) and not isinstance(
            self.required_transfer_detail_level, TransferDetailLevel
        ):
            self.required_transfer_detail_level = TransferDetailLevel(
                self.required_transfer_detail_level
            )

        self.confidentiality_constraints = _to_str_tuple(
            self.confidentiality_constraints, "confidentiality_constraints"
        )

        normalized_children: List[NodeIdentifier] = []
        for cid in self.child_node_ids:
            cid_vo = NodeIdentifier(cid) if isinstance(cid, str) else cid
            if self.is_leaf_boundary:
                raise LeafNodeBoundaryViolationError(
                    parent_node_id=self.node_id.value,
                    child_node_id=cid_vo.value,
                )
            if cid_vo not in normalized_children:
                normalized_children.append(cid_vo)
        self.child_node_ids = normalized_children

    def can_accept_children(self) -> bool:
        """بررسی اینکه آیا این گره مجاز به داشتن زیرمجموعه/فرزند در مدل سامانه است یا خیر."""
        return not self.is_leaf_boundary

    def register_child(self, child_node_id: Union[str, NodeIdentifier]) -> None:
        """
        ثبت گره فرزند برای این گره با بررسی قانون مرز گره نهایی (`is_leaf_boundary`).
        """
        child_vo = (
            NodeIdentifier(child_node_id)
            if isinstance(child_node_id, str)
            else child_node_id
        )
        if self.is_leaf_boundary:
            raise LeafNodeBoundaryViolationError(
                parent_node_id=self.node_id.value,
                child_node_id=child_vo.value,
            )
        if child_vo == self.node_id:
            raise InvalidLayerHierarchyError(
                f"گره '{self.node_id}' نمی‌تواند فرزند خودش باشد."
            )
        if child_vo not in self.child_node_ids:
            self.child_node_ids.append(child_vo)

    def mark_as_leaf_boundary(self) -> None:
        """
        علامت‌گذاری گره به عنوان گره نهایی (خارج کردن زیرساختار داخلی از دامنه مدل‌سازی).
        اگر گره از قبل دارای فرزند ثبت‌شده باشد، خطای `LeafNodeBoundaryViolationError` صادر می‌شود.
        """
        if self.child_node_ids:
            raise LeafNodeBoundaryViolationError(
                parent_node_id=self.node_id.value,
                child_node_id=self.child_node_ids[0].value,
            )
        self.is_leaf_boundary = True

    def to_dict(self) -> Dict[str, Any]:
        """تبدیل اطلاعات گره به دیکشنری استاندارد جهت ذخیره‌سازی یا خروجی رجیستری."""
        return {
            "node_id": self.node_id.value,
            "name": self.name,
            "layer_index": self.layer_index.value,
            "domain_of_activity": self.domain_of_activity,
            "parent_node_id": self.parent_node_id.value if self.parent_node_id else None,
            "is_leaf_boundary": self.is_leaf_boundary,
            "reporting_rules": dict(self.reporting_rules),
            "required_transfer_detail_level": self.required_transfer_detail_level.value,
            "confidentiality_constraints": list(self.confidentiality_constraints),
            "child_node_ids": [cid.value for cid in self.child_node_ids],
        }


@dataclass
class Layer:
    """
    موجودیت لایه سازمانی (جدول ۵.۲ و جدول ۶.۲ پروپوزال).
    سطحی از سلسله‌مراتب که شامل یک یا چند گره سازمانی است.
    """

    layer_index: LayerIndex
    name: str
    role_description: str
    trackable_info_summary: str
    default_detail_level: TransferDetailLevel
    member_node_ids: List[NodeIdentifier] = field(default_factory=list)

    def __post_init__(self) -> None:
        if isinstance(self.layer_index, int) and not isinstance(self.layer_index, bool):
            self.layer_index = LayerIndex(self.layer_index)
        elif not isinstance(self.layer_index, LayerIndex):
            raise DomainValidationError(
                "فیلد layer_index باید از نوع LayerIndex باشد.",
                field_name="layer_index",
            )

        self.name = _ensure_non_empty_str(self.name, "name", "نام لایه")
        self.role_description = _ensure_non_empty_str(
            self.role_description, "role_description", "نقش لایه در ساختار"
        )
        self.trackable_info_summary = _ensure_non_empty_str(
            self.trackable_info_summary,
            "trackable_info_summary",
            "اطلاعات قابل ردیابی لایه",
        )

        if isinstance(self.default_detail_level, str) and not isinstance(
            self.default_detail_level, TransferDetailLevel
        ):
            self.default_detail_level = TransferDetailLevel(self.default_detail_level)

        normalized_members: List[NodeIdentifier] = []
        for mid in self.member_node_ids:
            mid_vo = NodeIdentifier(mid) if isinstance(mid, str) else mid
            if mid_vo not in normalized_members:
                normalized_members.append(mid_vo)
        self.member_node_ids = normalized_members

    def add_member_node(self, node_id: Union[str, NodeIdentifier]) -> None:
        """افزودن شناسه گره عضو به این لایه در صورت عدم تکرار."""
        node_vo = NodeIdentifier(node_id) if isinstance(node_id, str) else node_id
        if node_vo not in self.member_node_ids:
            self.member_node_ids.append(node_vo)

    def has_member(self, node_id: Union[str, NodeIdentifier]) -> bool:
        """بررسی عضویت یک گره در این لایه."""
        node_vo = NodeIdentifier(node_id) if isinstance(node_id, str) else node_id
        return node_vo in self.member_node_ids

    def to_dict(self) -> Dict[str, Any]:
        """تبدیل اطلاعات لایه به دیکشنری استاندارد."""
        return {
            "layer_index": self.layer_index.value,
            "name": self.name,
            "role_description": self.role_description,
            "trackable_info_summary": self.trackable_info_summary,
            "default_detail_level": self.default_detail_level.value,
            "member_node_ids": [mid.value for mid in self.member_node_ids],
        }


@dataclass
class KnowledgeVersion:
    """
    موجودیت نسخه مشخصی از یک قلم دانش (جدول ۵.۲ و بخش ۷.۶ پروپوزال).
    """

    version_id: str
    knowledge_item_id: str
    version_tag: VersionTag
    content_summary: str
    change_reason: str
    metadata: MetadataRecord
    previous_version_id: Optional[str] = None
    is_active: bool = True
    diff_summary: Optional[str] = None
    chunk_ids: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.version_id = _ensure_non_empty_str(
            self.version_id, "version_id", "شناسه نسخه"
        )
        self.knowledge_item_id = _ensure_non_empty_str(
            self.knowledge_item_id, "knowledge_item_id", "شناسه قلم دانش"
        )
        if not isinstance(self.version_tag, VersionTag):
            raise DomainValidationError(
                "فیلد version_tag باید از نوع VersionTag باشد.",
                field_name="version_tag",
            )
        self.content_summary = _ensure_non_empty_str(
            self.content_summary, "content_summary", "خلاصه محتوای نسخه"
        )
        self.change_reason = _ensure_non_empty_str(
            self.change_reason, "change_reason", "علت تغییر نسخه"
        )
        if not isinstance(self.metadata, MetadataRecord):
            raise DomainValidationError(
                "فیلد metadata باید از نوع MetadataRecord باشد.",
                field_name="metadata",
            )

    def deactivate(self) -> None:
        """غیرفعال کردن نسخه هنگام ثبت نسخه جدید (حفظ سابقه برای ممیزی)."""
        self.is_active = False

    def attach_chunk(self, chunk_id: str) -> None:
        """اتصال شناسه قطعه دانش (Chunk) به این نسخه."""
        cleaned = _ensure_non_empty_str(chunk_id, "chunk_id", "شناسه قطعه دانش")
        if cleaned not in self.chunk_ids:
            self.chunk_ids.append(cleaned)


@dataclass
class KnowledgeItem:
    """
    موجودیت قلم دانش کنترل‌شده (جدول ۵.۲ و بخش ۷ پروپوزال).
    منبع کنترل‌شده دانش مانند سند، پروفایل، گزارش پروژه، پاسخ مسئله یا ساختار مدل.
    """

    item_id: str
    title: str
    owner_node_id: NodeIdentifier
    scope: KnowledgeScope
    metadata: MetadataRecord
    versions: List[KnowledgeVersion] = field(default_factory=list)
    active_version_tag: VersionTag = field(
        default_factory=lambda: VersionTag("v1.0.0", 1)
    )

    def __post_init__(self) -> None:
        self.item_id = _ensure_non_empty_str(
            self.item_id, "item_id", "شناسه پایدار قلم دانش"
        )
        self.title = _ensure_non_empty_str(self.title, "title", "عنوان قلم دانش")
        if isinstance(self.owner_node_id, str):
            self.owner_node_id = NodeIdentifier(self.owner_node_id)
        elif not isinstance(self.owner_node_id, NodeIdentifier):
            raise DomainValidationError(
                "شناسه گره مالک (owner_node_id) باید از نوع NodeIdentifier باشد.",
                field_name="owner_node_id",
            )
        if not isinstance(self.scope, KnowledgeScope):
            raise DomainValidationError(
                "فیلد scope باید از نوع KnowledgeScope باشد.",
                field_name="scope",
            )
        if not isinstance(self.metadata, MetadataRecord):
            raise DomainValidationError(
                "فیلد metadata باید از نوع MetadataRecord باشد.",
                field_name="metadata",
            )

    def publish_new_version(
        self,
        version_id: str,
        new_tag_str: str,
        content_summary: str,
        change_reason: str,
        diff_summary: Optional[str] = None,
    ) -> KnowledgeVersion:
        """
        ایجاد و فعال‌سازی نسخه جدید از قلم دانش بدون حذف نسخه‌های پیشین (بخش ۷.۶ پروپوزال).
        """
        previous_active = self.get_active_version()
        prev_version_id: Optional[str] = None
        if previous_active is not None:
            prev_version_id = previous_active.version_id
            next_tag = previous_active.version_tag.next_version(new_tag_str)
        elif self.versions:
            last_ver = self.versions[-1]
            prev_version_id = last_ver.version_id
            next_tag = last_ver.version_tag.next_version(new_tag_str)
        else:
            next_tag = VersionTag(tag=new_tag_str, sequence_number=1)

        for ver in self.versions:
            ver.deactivate()

        updated_metadata = self.metadata.with_active_version(version_id)
        new_version = KnowledgeVersion(
            version_id=version_id,
            knowledge_item_id=self.item_id,
            version_tag=next_tag,
            content_summary=content_summary,
            change_reason=change_reason,
            metadata=updated_metadata,
            previous_version_id=prev_version_id,
            is_active=True,
            diff_summary=diff_summary,
        )
        self.versions.append(new_version)
        self.active_version_tag = next_tag
        self.metadata = updated_metadata
        return new_version

    def get_active_version(self) -> Optional[KnowledgeVersion]:
        """دریافت نسخه فعال فعلی قلم دانش."""
        for ver in reversed(self.versions):
            if ver.is_active:
                return ver
        return None


@dataclass
class KnowledgeChunk:
    """
    موجودیت قطعه دانش قابل بازیابی در RAG (جدول ۵.۲ و بخش ۷.۴ پروپوزال).
    حفظ‌کننده شناسه سند منبع، شناسه نسخه، ارجاع صفحه/بخش، مسیر عنوان‌ها، فراداده ارث‌بری‌شده و ترتیب قطعه.
    """

    chunk_id: str
    source_document_id: str
    version_tag: VersionTag
    sequence_order: int
    text_content: str
    inherited_metadata: MetadataRecord
    page_or_section_ref: Optional[str] = None
    heading_path: Tuple[str, ...] = ()
    context_boundary_note: Optional[str] = None

    def __post_init__(self) -> None:
        self.chunk_id = _ensure_non_empty_str(
            self.chunk_id, "chunk_id", "شناسه قطعه دانش"
        )
        self.source_document_id = _ensure_non_empty_str(
            self.source_document_id, "source_document_id", "شناسه سند منبع"
        )
        if not isinstance(self.version_tag, VersionTag):
            raise DomainValidationError(
                "فیلد version_tag در KnowledgeChunk باید از نوع VersionTag باشد.",
                field_name="version_tag",
            )
        if isinstance(self.sequence_order, bool) or not isinstance(
            self.sequence_order, int
        ) or self.sequence_order < 0:
            raise DomainValidationError(
                "ترتیب قطعه (sequence_order) باید عدد صحیح نامنفی باشد.",
                field_name="sequence_order",
            )
        self.text_content = _ensure_non_empty_str(
            self.text_content, "text_content", "متن قطعه دانش"
        )
        if not isinstance(self.inherited_metadata, MetadataRecord):
            raise DomainValidationError(
                "فراداده ارث‌بری‌شده (inherited_metadata) باید از نوع MetadataRecord باشد.",
                field_name="inherited_metadata",
            )
        self.heading_path = _to_str_tuple(self.heading_path, "heading_path")


@dataclass
class SubIssue:
    """
    موجودیت زیرمسئله مشتق‌شده از مسئله والد (جدول ۵.۲ و بخش ۱۲.۲ پروپوزال).
    دارای شناسه مستقل، رابطه با مسئله والد، جایگاه نسبی، فراداده، وضعیت، درصد پیشرفت، دانش و مجری مرتبط.
    """

    sub_issue_id: str
    parent_issue_id: str
    title: str
    description: str
    relative_order: int
    metadata: MetadataRecord
    status: str = "NEW"
    progress: ProgressPercentage = field(
        default_factory=lambda: ProgressPercentage(0.0)
    )
    related_knowledge_ids: List[str] = field(default_factory=list)
    candidate_executor_ids: List[str] = field(default_factory=list)
    proposed_solution: Optional[str] = None

    def __post_init__(self) -> None:
        self.sub_issue_id = _ensure_non_empty_str(
            self.sub_issue_id, "sub_issue_id", "شناسه زیرمسئله"
        )
        self.parent_issue_id = _ensure_non_empty_str(
            self.parent_issue_id, "parent_issue_id", "شناسه مسئله والد"
        )
        self.title = _ensure_non_empty_str(self.title, "title", "عنوان زیرمسئله")
        self.description = _ensure_non_empty_str(
            self.description, "description", "شرح زیرمسئله"
        )
        if isinstance(self.relative_order, bool) or not isinstance(
            self.relative_order, int
        ) or self.relative_order < 1:
            raise DomainValidationError(
                "جایگاه نسبی زیرمسئله (relative_order) باید عدد صحیح بزرگ‌تر یا مساوی ۱ باشد.",
                field_name="relative_order",
            )
        if not isinstance(self.metadata, MetadataRecord):
            raise DomainValidationError(
                "فیلد metadata در SubIssue باید از نوع MetadataRecord باشد.",
                field_name="metadata",
            )
        if isinstance(self.progress, (int, float)) and not isinstance(
            self.progress, bool
        ):
            self.progress = ProgressPercentage(self.progress)
        elif not isinstance(self.progress, ProgressPercentage):
            raise DomainValidationError(
                "فیلد progress باید از نوع ProgressPercentage باشد.",
                field_name="progress",
            )

    def update_progress(
        self,
        new_progress: Union[float, int, ProgressPercentage],
        new_status: Optional[str] = None,
    ) -> None:
        """به‌روزرسانی درصد پیشرفت و وضعیت زیرمسئله."""
        self.progress = (
            new_progress
            if isinstance(new_progress, ProgressPercentage)
            else ProgressPercentage(new_progress)
        )
        if new_status is not None:
            self.status = _ensure_non_empty_str(new_status, "status", "وضعیت زیرمسئله")
        elif self.progress.is_completed:
            self.status = "RESOLVED"


@dataclass
class Issue:
    """
    موجودیت مسئله، پرسش، نیاز، شکاف پژوهشی یا دغدغه عملیاتی مدل‌شده در یک گره (جدول ۵.۲ و بخش ۱۲ پروپوزال).
    """

    issue_id: str
    owner_node_id: NodeIdentifier
    title: str
    description: str
    metadata: MetadataRecord
    status: str = "NEW"
    priority: str = "MEDIUM"
    progress: ProgressPercentage = field(
        default_factory=lambda: ProgressPercentage(0.0)
    )
    sub_issues: List[SubIssue] = field(default_factory=list)
    related_knowledge_ids: List[str] = field(default_factory=list)
    assigned_executor_ids: List[str] = field(default_factory=list)
    has_research_gap: bool = False
    research_gap_reason: Optional[str] = None

    def __post_init__(self) -> None:
        self.issue_id = _ensure_non_empty_str(
            self.issue_id, "issue_id", "شناسه مسئله"
        )
        if isinstance(self.owner_node_id, str):
            self.owner_node_id = NodeIdentifier(self.owner_node_id)
        elif not isinstance(self.owner_node_id, NodeIdentifier):
            raise DomainValidationError(
                "شناسه گره مالک (owner_node_id) باید از نوع NodeIdentifier باشد.",
                field_name="owner_node_id",
            )
        self.title = _ensure_non_empty_str(self.title, "title", "عنوان مسئله")
        self.description = _ensure_non_empty_str(
            self.description, "description", "شرح مسئله"
        )
        if not isinstance(self.metadata, MetadataRecord):
            raise DomainValidationError(
                "فیلد metadata در Issue باید از نوع MetadataRecord باشد.",
                field_name="metadata",
            )
        if isinstance(self.progress, (int, float)) and not isinstance(
            self.progress, bool
        ):
            self.progress = ProgressPercentage(self.progress)
        elif not isinstance(self.progress, ProgressPercentage):
            raise DomainValidationError(
                "فیلد progress باید از نوع ProgressPercentage باشد.",
                field_name="progress",
            )

    def add_sub_issue(self, sub_issue: SubIssue) -> None:
        """افزودن زیرمسئله به مسئله والد پس از اعتبارسنجی تطابق شناسه والد."""
        if sub_issue.parent_issue_id != self.issue_id:
            raise DomainValidationError(
                f"شناسه والد زیرمسئله ('{sub_issue.parent_issue_id}') با شناسه مسئله ('{self.issue_id}') هم‌خوانی ندارد.",
                field_name="parent_issue_id",
            )
        self.sub_issues.append(sub_issue)
        self.recalculate_progress()

    def recalculate_progress(self) -> ProgressPercentage:
        """محاسبه قطعی درصد پیشرفت مسئله بر اساس میانگین پیشرفت زیرمسئله‌ها (بخش ۱۴.۱ پروپوزال)."""
        if not self.sub_issues:
            return self.progress
        avg_val = sum(si.progress.value for si in self.sub_issues) / len(
            self.sub_issues
        )
        self.progress = ProgressPercentage(round(avg_val, 2))
        if self.progress.is_completed:
            self.status = "RESOLVED"
        elif self.progress.value > 0:
            self.status = "IN_PROGRESS"
        return self.progress

    def flag_research_gap(self, reason: str) -> None:
        """علامت‌گذاری مسئله به عنوان دارای شکاف پژوهشی (بخش ۱۴.۳ پروپوزال)."""
        self.has_research_gap = True
        self.research_gap_reason = _ensure_non_empty_str(
            reason, "research_gap_reason", "علت شکاف پژوهشی"
        )


@dataclass
class Executor:
    """
    موجودیت مجری (جدول ۵.۲ و بخش ۱۳.۲ پروپوزال).
    فرد، تیم، واحد یا توانمندی سازمانی که می‌تواند در حل مسئله نقش داشته باشد.
    """

    executor_id: str
    name: str
    node_id: NodeIdentifier
    organizational_role: str
    expertise_domains: Tuple[str, ...] = ()
    capabilities: Tuple[str, ...] = ()
    past_project_ids: Tuple[str, ...] = ()
    authored_document_ids: Tuple[str, ...] = ()
    active_issue_ids: List[str] = field(default_factory=list)
    availability_status: str = "AVAILABLE"

    def __post_init__(self) -> None:
        self.executor_id = _ensure_non_empty_str(
            self.executor_id, "executor_id", "شناسه مجری"
        )
        self.name = _ensure_non_empty_str(self.name, "name", "نام مجری")
        if isinstance(self.node_id, str):
            self.node_id = NodeIdentifier(self.node_id)
        elif not isinstance(self.node_id, NodeIdentifier):
            raise DomainValidationError(
                "شناسه گره مجری (node_id) باید از نوع NodeIdentifier باشد.",
                field_name="node_id",
            )
        self.organizational_role = _ensure_non_empty_str(
            self.organizational_role, "organizational_role", "نقش سازمانی مجری"
        )
        self.expertise_domains = _to_str_tuple(
            self.expertise_domains, "expertise_domains"
        )
        self.capabilities = _to_str_tuple(self.capabilities, "capabilities")
        self.past_project_ids = _to_str_tuple(self.past_project_ids, "past_project_ids")
        self.authored_document_ids = _to_str_tuple(
            self.authored_document_ids, "authored_document_ids"
        )

    def has_expertise_in(self, domain_keyword: str) -> bool:
        """بررسی تطابق تخصص مجری با یک کلیدواژه حوزه تخصصی."""
        kw = domain_keyword.strip().lower()
        return any(kw in exp.lower() for exp in self.expertise_domains) or any(
            kw in cap.lower() for cap in self.capabilities
        )

    def assign_issue(self, issue_id: str) -> None:
        """تخصیص یک مسئله فعال به مجری."""
        cleaned = _ensure_non_empty_str(issue_id, "issue_id", "شناسه مسئله")
        if cleaned not in self.active_issue_ids:
            self.active_issue_ids.append(cleaned)


@dataclass
class JudgeScore:
    """
    موجودیت امتیاز ساخت‌یافته تولیدشده توسط سازوکار LLM به‌عنوان داور (جدول ۵.۲ و بخش ۱۶.۲ پروپوزال).
    """

    score_id: str
    target_entity_id: str
    rubric: JudgeRubric
    threshold: JudgeThreshold = field(default_factory=JudgeThreshold)
    explanation: str = ""
    missing_information: Tuple[str, ...] = ()
    unsupported_claims: Tuple[str, ...] = ()
    excessive_details: Tuple[str, ...] = ()
    insufficient_details: Tuple[str, ...] = ()
    improvement_suggestions: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.score_id = _ensure_non_empty_str(
            self.score_id, "score_id", "شناسه رکورد داوری"
        )
        self.target_entity_id = _ensure_non_empty_str(
            self.target_entity_id, "target_entity_id", "شناسه موجودیت هدف داوری"
        )
        if not isinstance(self.rubric, JudgeRubric):
            raise DomainValidationError(
                "فیلد rubric در JudgeScore باید از نوع JudgeRubric باشد.",
                field_name="rubric",
            )
        if not isinstance(self.threshold, JudgeThreshold):
            raise DomainValidationError(
                "فیلد threshold در JudgeScore باید از نوع JudgeThreshold باشد.",
                field_name="threshold",
            )
        self.missing_information = _to_str_tuple(
            self.missing_information, "missing_information"
        )
        self.unsupported_claims = _to_str_tuple(
            self.unsupported_claims, "unsupported_claims"
        )
        self.excessive_details = _to_str_tuple(
            self.excessive_details, "excessive_details"
        )
        self.insufficient_details = _to_str_tuple(
            self.insufficient_details, "insufficient_details"
        )
        self.improvement_suggestions = _to_str_tuple(
            self.improvement_suggestions, "improvement_suggestions"
        )

    @property
    def total_score(self) -> int:
        """امتیاز کل اکتسابی از ۲۰."""
        return self.rubric.total_score()

    @property
    def is_approved(self) -> bool:
        """آیا امتیاز کل داور حداقل امتیاز قابل قبول (`threshold`) را کسب کرده است؟"""
        return self.threshold.is_satisfied_by(self.rubric)

    def enforce_threshold(self) -> None:
        """اعمال سخت‌گیرانه حد آستانه داوری و صدور `JudgeThresholdNotMetError` در صورت رد شدن."""
        self.threshold.enforce(self.rubric)


@dataclass
class TransferPackage:
    """
    موجودیت بسته دانشی آمادهشده برای ارسال از لایه پایین‌تر به لایه بالاتر (جدول ۵.۲ و بخش ۱۰.۳ پروپوزال).
    """

    package_id: str
    source_node_id: NodeIdentifier
    target_layer_index: LayerIndex
    version_tag: VersionTag
    detail_level: TransferDetailLevel
    reporting_phase: str
    completed_projects_summary: Tuple[str, ...] = ()
    research_conducted_summary: Tuple[str, ...] = ()
    achievements: Tuple[str, ...] = ()
    active_issues_summary: Tuple[str, ...] = ()
    challenges: Tuple[str, ...] = ()
    unresolved_issues: Tuple[str, ...] = ()
    key_research_gaps: Tuple[str, ...] = ()
    executor_capability_notes: Tuple[str, ...] = ()
    rfp_and_proposal_status: Tuple[str, ...] = ()
    proposed_decisions_for_upper_layer: Tuple[str, ...] = ()
    source_citations: Tuple[str, ...] = ()
    judge_score: Optional[JudgeScore] = None
    is_accepted_for_delivery: bool = False

    def __post_init__(self) -> None:
        self.package_id = _ensure_non_empty_str(
            self.package_id, "package_id", "شناسه بسته انتقال"
        )
        if isinstance(self.source_node_id, str):
            self.source_node_id = NodeIdentifier(self.source_node_id)
        elif not isinstance(self.source_node_id, NodeIdentifier):
            raise DomainValidationError(
                "شناسه گره مبدأ (source_node_id) باید از نوع NodeIdentifier باشد.",
                field_name="source_node_id",
            )
        if isinstance(self.target_layer_index, int) and not isinstance(
            self.target_layer_index, bool
        ):
            self.target_layer_index = LayerIndex(self.target_layer_index)
        elif not isinstance(self.target_layer_index, LayerIndex):
            raise DomainValidationError(
                "لایه هدف انتقال (target_layer_index) باید از نوع LayerIndex باشد.",
                field_name="target_layer_index",
            )
        if not isinstance(self.version_tag, VersionTag):
            raise DomainValidationError(
                "فیلد version_tag باید از نوع VersionTag باشد.",
                field_name="version_tag",
            )
        if isinstance(self.detail_level, str) and not isinstance(
            self.detail_level, TransferDetailLevel
        ):
            self.detail_level = TransferDetailLevel(self.detail_level)
        self.reporting_phase = _ensure_non_empty_str(
            self.reporting_phase, "reporting_phase", "فاز گزارش‌دهی"
        )
        self.completed_projects_summary = _to_str_tuple(
            self.completed_projects_summary, "completed_projects_summary"
        )
        self.research_conducted_summary = _to_str_tuple(
            self.research_conducted_summary, "research_conducted_summary"
        )
        self.achievements = _to_str_tuple(self.achievements, "achievements")
        self.active_issues_summary = _to_str_tuple(
            self.active_issues_summary, "active_issues_summary"
        )
        self.challenges = _to_str_tuple(self.challenges, "challenges")
        self.unresolved_issues = _to_str_tuple(
            self.unresolved_issues, "unresolved_issues"
        )
        self.key_research_gaps = _to_str_tuple(
            self.key_research_gaps, "key_research_gaps"
        )
        self.executor_capability_notes = _to_str_tuple(
            self.executor_capability_notes, "executor_capability_notes"
        )
        self.rfp_and_proposal_status = _to_str_tuple(
            self.rfp_and_proposal_status, "rfp_and_proposal_status"
        )
        self.proposed_decisions_for_upper_layer = _to_str_tuple(
            self.proposed_decisions_for_upper_layer,
            "proposed_decisions_for_upper_layer",
        )
        self.source_citations = _to_str_tuple(self.source_citations, "source_citations")

        if self.judge_score is not None:
            self.is_accepted_for_delivery = self.judge_score.is_approved

    def evaluate_with_judge(
        self, judge_score: JudgeScore, require_threshold: bool = True
    ) -> bool:
        """
        اعمال نتیجه ارزیابی داور روی بسته انتقال.
        در صورت فعال بودن `require_threshold` و عدم احراز حد آستانه (مثلاً ۱۸ از ۲۰)،
        استثنای `JudgeThresholdNotMetError` صادر می‌گردد.
        """
        self.judge_score = judge_score
        self.is_accepted_for_delivery = judge_score.is_approved
        if require_threshold:
            judge_score.enforce_threshold()
        return self.is_accepted_for_delivery


@dataclass
class RFP:
    """
    موجودیت سند اعلام نیاز (Request for Proposal) تولیدشده توسط سامانه (جدول ۵.۲ و بخش ۱۵.۱ پروپوزال).
    """

    rfp_id: str
    origin_issue_id: str
    issuer_node_id: NodeIdentifier
    title: str
    problem_statement: str
    sub_issue_model_summary: Tuple[str, ...] = ()
    related_knowledge_ids: Tuple[str, ...] = ()
    research_gaps: Tuple[str, ...] = ()
    technical_requirements: Tuple[str, ...] = ()
    expected_deliverables: Tuple[str, ...] = ()
    evaluation_criteria: Tuple[str, ...] = ()
    constraints: Tuple[str, ...] = ()
    required_expertise: Tuple[str, ...] = ()
    expected_documentation: Tuple[str, ...] = ()
    status: str = "OPEN"

    def __post_init__(self) -> None:
        self.rfp_id = _ensure_non_empty_str(self.rfp_id, "rfp_id", "شناسه RFP")
        self.origin_issue_id = _ensure_non_empty_str(
            self.origin_issue_id, "origin_issue_id", "شناسه مسئله منشأ"
        )
        if isinstance(self.issuer_node_id, str):
            self.issuer_node_id = NodeIdentifier(self.issuer_node_id)
        elif not isinstance(self.issuer_node_id, NodeIdentifier):
            raise DomainValidationError(
                "شناسه گره صادرکننده (issuer_node_id) باید از نوع NodeIdentifier باشد.",
                field_name="issuer_node_id",
            )
        self.title = _ensure_non_empty_str(self.title, "title", "عنوان RFP")
        self.problem_statement = _ensure_non_empty_str(
            self.problem_statement, "problem_statement", "شرح مسئله در RFP"
        )
        self.sub_issue_model_summary = _to_str_tuple(
            self.sub_issue_model_summary, "sub_issue_model_summary"
        )
        self.related_knowledge_ids = _to_str_tuple(
            self.related_knowledge_ids, "related_knowledge_ids"
        )
        self.research_gaps = _to_str_tuple(self.research_gaps, "research_gaps")
        self.technical_requirements = _to_str_tuple(
            self.technical_requirements, "technical_requirements"
        )
        self.expected_deliverables = _to_str_tuple(
            self.expected_deliverables, "expected_deliverables"
        )
        self.evaluation_criteria = _to_str_tuple(
            self.evaluation_criteria, "evaluation_criteria"
        )
        self.constraints = _to_str_tuple(self.constraints, "constraints")
        self.required_expertise = _to_str_tuple(
            self.required_expertise, "required_expertise"
        )
        self.expected_documentation = _to_str_tuple(
            self.expected_documentation, "expected_documentation"
        )

    def has_all_mandatory_sections(self) -> bool:
        """بررسی کامل بودن بخش‌های اجباری RFP طبق جدول ۱۹.۲ پروپوزال."""
        return bool(
            self.problem_statement
            and self.technical_requirements
            and self.expected_deliverables
            and self.evaluation_criteria
            and self.constraints
            and self.required_expertise
        )


@dataclass
class Proposal:
    """
    موجودیت پیشنهادی که در پاسخ به RFP ثبت و ارزیابی می‌شود (جدول ۵.۲ و بخش ۱۵.۲ پروپوزال).
    """

    proposal_id: str
    rfp_id: str
    proposer_id: str
    title: str
    proposed_methodology: str
    technical_completeness_summary: str
    feasibility_plan: str
    risk_management_plan: str
    metadata: MetadataRecord
    evaluation_status: str = "SUBMITTED"
    evaluated_score: Optional[float] = None
    feedback_report_id: Optional[str] = None

    def __post_init__(self) -> None:
        self.proposal_id = _ensure_non_empty_str(
            self.proposal_id, "proposal_id", "شناسه پیشنهاد"
        )
        self.rfp_id = _ensure_non_empty_str(self.rfp_id, "rfp_id", "شناسه RFP مرتبط")
        self.proposer_id = _ensure_non_empty_str(
            self.proposer_id, "proposer_id", "شناسه ارائه‌دهنده پیشنهاد"
        )
        self.title = _ensure_non_empty_str(self.title, "title", "عنوان پیشنهاد")
        self.proposed_methodology = _ensure_non_empty_str(
            self.proposed_methodology, "proposed_methodology", "روش‌شناسی پیشنهادی"
        )
        self.technical_completeness_summary = _ensure_non_empty_str(
            self.technical_completeness_summary,
            "technical_completeness_summary",
            "شرح کامل بودن فنی",
        )
        self.feasibility_plan = _ensure_non_empty_str(
            self.feasibility_plan, "feasibility_plan", "برنامه امکان‌پذیری"
        )
        self.risk_management_plan = _ensure_non_empty_str(
            self.risk_management_plan, "risk_management_plan", "برنامه مدیریت ریسک"
        )
        if not isinstance(self.metadata, MetadataRecord):
            raise DomainValidationError(
                "فیلد metadata در Proposal باید از نوع MetadataRecord باشد.",
                field_name="metadata",
            )

    def record_evaluation(
        self, score: float, status: str, feedback_report_id: str
    ) -> None:
        """ثبت نتیجه ارزیابی و اتصال به گزارش بازخورد."""
        if not (0.0 <= float(score) <= 20.0):
            raise DomainValidationError(
                "امتیاز ارزیابی پیشنهاد باید بین 0.0 و 20.0 باشد.",
                field_name="evaluated_score",
            )
        self.evaluated_score = round(float(score), 2)
        self.evaluation_status = _ensure_non_empty_str(
            status, "evaluation_status", "وضعیت ارزیابی پیشنهاد"
        )
        self.feedback_report_id = _ensure_non_empty_str(
            feedback_report_id, "feedback_report_id", "شناسه گزارش بازخورد"
        )


@dataclass
class FeedbackReport:
    """
    موجودیت گزارش بازخورد ساخت‌یافته برای یک پیشنهاد یا پاسخ به RFP (جدول ۵.۲ و بخش ۱۵.۳ پروپوزال).
    """

    report_id: str
    rfp_id: str
    proposal_id: str
    strengths: Tuple[str, ...] = ()
    weaknesses: Tuple[str, ...] = ()
    missing_requirements: Tuple[str, ...] = ()
    ambiguous_assumptions: Tuple[str, ...] = ()
    technical_risks: Tuple[str, ...] = ()
    clarifications_needed: Tuple[str, ...] = ()
    suggested_score: float = 0.0
    recommendation_decision: str = "REVISE"
    criterion_references: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.report_id = _ensure_non_empty_str(
            self.report_id, "report_id", "شناسه گزارش بازخورد"
        )
        self.rfp_id = _ensure_non_empty_str(self.rfp_id, "rfp_id", "شناسه RFP")
        self.proposal_id = _ensure_non_empty_str(
            self.proposal_id, "proposal_id", "شناسه پیشنهاد"
        )
        self.strengths = _to_str_tuple(self.strengths, "strengths")
        self.weaknesses = _to_str_tuple(self.weaknesses, "weaknesses")
        self.missing_requirements = _to_str_tuple(
            self.missing_requirements, "missing_requirements"
        )
        self.ambiguous_assumptions = _to_str_tuple(
            self.ambiguous_assumptions, "ambiguous_assumptions"
        )
        self.technical_risks = _to_str_tuple(self.technical_risks, "technical_risks")
        self.clarifications_needed = _to_str_tuple(
            self.clarifications_needed, "clarifications_needed"
        )
        self.criterion_references = _to_str_tuple(
            self.criterion_references, "criterion_references"
        )
        if not (0.0 <= float(self.suggested_score) <= 20.0):
            raise DomainValidationError(
                "امتیاز پیشنهادی در گزارش بازخورد باید بین 0.0 و 20.0 باشد.",
                field_name="suggested_score",
            )
        self.suggested_score = round(float(self.suggested_score), 2)
        self.recommendation_decision = _ensure_non_empty_str(
            self.recommendation_decision,
            "recommendation_decision",
            "تصمیم پیشنهادی بازخورد",
        ).upper()
        if self.recommendation_decision not in ("SELECT", "REVISE", "REJECT"):
            raise DomainValidationError(
                f"تصمیم پیشنهادی '{self.recommendation_decision}' نامعتبر است. مقادیر مجاز: SELECT, REVISE, REJECT.",
                field_name="recommendation_decision",
            )


__all__ = [
    "Node",
    "Layer",
    "KnowledgeItem",
    "KnowledgeVersion",
    "KnowledgeChunk",
    "Issue",
    "SubIssue",
    "Executor",
    "TransferPackage",
    "JudgeScore",
    "RFP",
    "Proposal",
    "FeedbackReport",
]
