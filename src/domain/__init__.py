"""
لایه دامنه (Domain Layer) هسته RAG سلسله‌مراتبی آفلاین.

این بسته شامل استثناهای اختصاصی دامنه، اشیاء مقداری تغییرناپذیر (Value Objects)،
موجودیت‌های کلیدی کسب‌وکار (Entities) و رجیستری لایه و گره سازمانی (`LayerNodeRegistry`) است.
"""

from src.domain.entities import (
    Executor,
    FeedbackReport,
    Issue,
    JudgeScore,
    KnowledgeChunk,
    KnowledgeItem,
    KnowledgeVersion,
    Layer,
    Node,
    Proposal,
    RFP,
    SubIssue,
    TransferPackage,
)
from src.domain.exceptions import (
    DomainError,
    DomainValidationError,
    InvalidLayerHierarchyError,
    JudgeThresholdNotMetError,
    LeafNodeBoundaryViolationError,
    RegistryNotFoundError,
)
from src.domain.layer_node_registry import (
    DEFAULT_LAYER_DEFINITIONS,
    LayerNodeRegistry,
)
from src.domain.value_objects import (
    ALLOWED_CONFIDENTIALITY_LEVELS,
    ALLOWED_DOCUMENT_TYPES,
    ALLOWED_PRIORITY_LEVELS,
    JudgeRubric,
    JudgeThreshold,
    KnowledgeScope,
    LayerIndex,
    MetadataRecord,
    NodeIdentifier,
    ProgressPercentage,
    RetrievalQuery,
    SimilarityScore,
    TransferDetailLevel,
    VersionTag,
)

__all__ = [
    # Exceptions
    "DomainError",
    "DomainValidationError",
    "InvalidLayerHierarchyError",
    "LeafNodeBoundaryViolationError",
    "RegistryNotFoundError",
    "JudgeThresholdNotMetError",
    # Value Objects & Constants
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
    # Entities
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
    # Registry
    "DEFAULT_LAYER_DEFINITIONS",
    "LayerNodeRegistry",
]
