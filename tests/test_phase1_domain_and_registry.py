"""
آزمون‌های واحد جامع فاز ۱: هسته دامنه DDD، اشیاء مقداری، موجودیت‌ها و رجیستری لایه و گره.

این مجموعه تست، تمامی ۱۱ شیء مقداری (حالت‌های معتبر، نامعتبر و تغییرناپذیری)،
۱۳ موجودیت کلیدی، قواعد سلسله‌مراتبی `LayerNodeRegistry`، خطای مرز گره نهایی
(`LeafNodeBoundaryViolationError`) و چارچوب داوری ۲۰ نمره‌ای (`JudgeRubric` و `JudgeThreshold`)
را به صورت ۱۰۰٪ قطعی و آفلاین راستی‌آزمایی می‌کند.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from src.domain import (
    DomainValidationError,
    Executor,
    FeedbackReport,
    InvalidLayerHierarchyError,
    Issue,
    JudgeRubric,
    JudgeScore,
    JudgeThreshold,
    JudgeThresholdNotMetError,
    KnowledgeChunk,
    KnowledgeItem,
    KnowledgeScope,
    Layer,
    LayerIndex,
    LayerNodeRegistry,
    LeafNodeBoundaryViolationError,
    MetadataRecord,
    Node,
    NodeIdentifier,
    ProgressPercentage,
    Proposal,
    RegistryNotFoundError,
    RetrievalQuery,
    RFP,
    SimilarityScore,
    SubIssue,
    TransferDetailLevel,
    TransferPackage,
    VersionTag,
)


@pytest.fixture
def sample_metadata() -> MetadataRecord:
    """فیکسچر تولید رکورد فراداده استاندارد ۱۸ فیلدی برای استفاده در آزمون‌های واحد."""
    return MetadataRecord(
        domain="هوش مصنوعی آفلاین",
        sub_domain="بازیابی سلسله‌مراتبی دانش",
        organization_or_node="L3-RAG-TEAM",
        project="هسته RAG سلسله‌مراتبی",
        author_or_owner="تیم معماری دامنه",
        document_type="PDF",
        confidentiality_level="INTERNAL",
        initial_version_id="VER-001",
        active_version_id="VER-001",
        related_issue_id="ISS-001",
        related_executor_id="EXEC-001",
        related_rfp_id="RFP-001",
        status="ACTIVE",
        priority="HIGH",
        tags=("DDD", "RAG", "Offline"),
        source_reliability=0.95,
        target_transfer_layer=LayerIndex(2),
        suggested_detail_level=TransferDetailLevel.TECHNICAL_SUMMARY,
    )


def test_01_node_identifier_and_layer_index_validation_and_immutability() -> None:
    """آزمون ۱: اعتبارسنجی و تغییرناپذیری `NodeIdentifier` و `LayerIndex` (لایه‌های ۰ تا ۴)."""
    node_id = NodeIdentifier("  L0-ROOT-NODE  ")
    assert node_id.value == "L0-ROOT-NODE"
    assert str(node_id) == "L0-ROOT-NODE"

    with pytest.raises(FrozenInstanceError):
        node_id.value = "MODIFIED"  # type: ignore[misc]

    for invalid_id in ["", "   ", "NODE WITH SPACES"]:
        with pytest.raises(DomainValidationError):
            NodeIdentifier(invalid_id)

    for idx in range(5):
        layer_vo = LayerIndex(idx)
        assert layer_vo.value == idx
        assert "لایه" in layer_vo.name_fa

    assert LayerIndex(0).is_strategic_root is True
    assert LayerIndex(1).is_strategic_root is False
    assert LayerIndex(0).is_higher_authority_than(LayerIndex(1)) is True
    assert LayerIndex(3).is_higher_authority_than(LayerIndex(2)) is False

    for invalid_layer in [-1, 5, 10, True]:
        with pytest.raises(DomainValidationError):
            LayerIndex(invalid_layer)  # type: ignore[arg-type]


def test_02_knowledge_scope_and_version_tag_invariants() -> None:
    """آزمون ۲: اعتبارسنجی `KnowledgeScope` و `VersionTag` (فرمت نسخه و شماره توالی)."""
    scope = KnowledgeScope(
        domain="مدیریت دانش سازمانی",
        confidentiality_level="confidential",
        allowed_node_ids=["L1-ORG", "L2-UNIT"],  # type: ignore[arg-type]
        max_target_layer=LayerIndex(0),
    )
    assert scope.confidentiality_level == "CONFIDENTIAL"
    assert isinstance(scope.allowed_node_ids, tuple)
    assert scope.is_accessible_by_node("L1-ORG") is True
    assert scope.is_accessible_by_node("L3-OTHER") is False

    with pytest.raises(DomainValidationError):
        KnowledgeScope(domain="", confidentiality_level="INTERNAL")

    with pytest.raises(DomainValidationError):
        KnowledgeScope(domain="فنی", confidentiality_level="TOP_SECRET_INVALID")

    v1 = VersionTag(tag="v1.0.0", sequence_number=1)
    v2 = v1.next_version("v1.1.0")
    assert v2.tag == "v1.1.0"
    assert v2.sequence_number == 2

    for bad_tag in ["1.0.0", "ver1.0", "", "v1"]:
        with pytest.raises(DomainValidationError):
            VersionTag(tag=bad_tag, sequence_number=1)

    with pytest.raises(DomainValidationError):
        VersionTag(tag="v1.0.0", sequence_number=0)


def test_03_progress_similarity_query_and_transfer_detail_level() -> None:
    """آزمون ۳: اعتبارسنجی `ProgressPercentage`، `SimilarityScore`، `RetrievalQuery` و `TransferDetailLevel`."""
    p0 = ProgressPercentage(0)
    p100 = ProgressPercentage(100.0)
    assert p0.is_completed is False
    assert p100.is_completed is True

    for invalid_progress in [-0.1, 100.01, -50, 150, False]:
        with pytest.raises(DomainValidationError):
            ProgressPercentage(invalid_progress)  # type: ignore[arg-type]

    sim = SimilarityScore(0.875)
    assert sim.exceeds_threshold(0.80) is True
    assert sim.exceeds_threshold(0.90) is False

    for invalid_sim in [-0.01, 1.01, 2.0, True]:
        with pytest.raises(DomainValidationError):
            SimilarityScore(invalid_sim)  # type: ignore[arg-type]

    query = RetrievalQuery(
        raw_query="  چگونه   مسائل کلیدی   را در گره   تحليل كنيم؟  ",
        target_node_id="L2-NLP-UNIT",  # type: ignore[arg-type]
    )
    assert query.normalized_text == "چگونه مسائل کلیدی را در گره تحلیل کنیم؟"
    assert query.target_node_id == NodeIdentifier("L2-NLP-UNIT")

    with pytest.raises(DomainValidationError):
        RetrievalQuery(raw_query="   ")

    assert TransferDetailLevel.for_layer(0) == TransferDetailLevel.EXECUTIVE_HIGH
    assert TransferDetailLevel.for_layer(1) == TransferDetailLevel.MANAGERIAL
    assert TransferDetailLevel.for_layer(2) == TransferDetailLevel.TECHNICAL_SUMMARY
    assert TransferDetailLevel.for_layer(3) == TransferDetailLevel.TECHNICAL_DETAILED
    assert TransferDetailLevel.EXECUTIVE_HIGH.label_fa == "اجرایی عالی"


def test_04_judge_rubric_20_point_calculation_and_ceiling_validation() -> None:
    """آزمون ۴: بررسی دقیق ۷ معیار جدول ۱۶.۱ در `JudgeRubric` و سقف امتیاز ۲۰ نمره‌ای."""
    perfect_rubric = JudgeRubric(
        source_fidelity=5,
        completeness=4,
        detail_level_fit=3,
        uncertainty_and_gap_handling=3,
        clarity_and_structure=2,
        source_attribution=2,
        no_unsupported_claims=1,
    )
    assert perfect_rubric.total_score() == 20
    assert perfect_rubric.max_possible_score == 20
    assert perfect_rubric.to_dict()["total_score"] == 20

    partial_rubric = JudgeRubric(
        source_fidelity=4,
        completeness=4,
        detail_level_fit=3,
        uncertainty_and_gap_handling=3,
        clarity_and_structure=2,
        source_attribution=1,
        no_unsupported_claims=1,
    )
    assert partial_rubric.total_score() == 18

    # بررسی تجاوز از سقف امتیاز هر یک از ۷ معیار
    invalid_cases = [
        {"source_fidelity": 6},
        {"completeness": 5},
        {"detail_level_fit": 4},
        {"uncertainty_and_gap_handling": 4},
        {"clarity_and_structure": 3},
        {"source_attribution": 3},
        {"no_unsupported_claims": 2},
        {"source_fidelity": -1},
    ]
    base_kwargs = {
        "source_fidelity": 5,
        "completeness": 4,
        "detail_level_fit": 3,
        "uncertainty_and_gap_handling": 3,
        "clarity_and_structure": 2,
        "source_attribution": 2,
        "no_unsupported_claims": 1,
    }
    for override in invalid_cases:
        params = {**base_kwargs, **override}
        with pytest.raises(DomainValidationError):
            JudgeRubric(**params)


def test_05_judge_threshold_enforcement_and_custom_exception() -> None:
    """آزمون ۵: بررسی `JudgeThreshold` (حد قابل قبول پیش‌فرض ۱۸ از ۲۰) و خطای `JudgeThresholdNotMetError`."""
    threshold = JudgeThreshold()
    assert threshold.minimum_score == 18
    assert threshold.max_score == 20

    passing_rubric = JudgeRubric(5, 4, 3, 2, 2, 1, 1)  # مجموع = ۱۸
    failing_rubric = JudgeRubric(4, 3, 2, 2, 2, 1, 1)  # مجموع = ۱۵

    assert threshold.is_satisfied_by(passing_rubric) is True
    assert threshold.is_satisfied_by(failing_rubric) is False

    threshold.enforce(passing_rubric)
    with pytest.raises(JudgeThresholdNotMetError) as exc_info:
        threshold.enforce(failing_rubric)
    assert exc_info.value.achieved_score == 15
    assert exc_info.value.required_threshold == 18


def test_06_metadata_record_18_fields_validation(sample_metadata: MetadataRecord) -> None:
    """آزمون ۶: بررسی کامل ۱۸ فیلد `MetadataRecord` (بخش ۷.۵ پروپوزال) و پشتیبانی از DOC/DOCX/PDF."""
    meta_dict = sample_metadata.to_dict()
    assert len(meta_dict) == 18
    assert sample_metadata.document_type == "PDF"
    assert sample_metadata.target_transfer_layer == LayerIndex(2)
    assert sample_metadata.suggested_detail_level == TransferDetailLevel.TECHNICAL_SUMMARY

    updated_meta = sample_metadata.with_active_version("VER-002")
    assert updated_meta.active_version_id == "VER-002"
    assert sample_metadata.active_version_id == "VER-001"

    for doc_fmt in ["doc", "docx", "pdf"]:
        rec = MetadataRecord(
            domain="فنی",
            sub_domain="اسناد",
            organization_or_node="L3-NODE",
            project="پروژه ۱",
            author_or_owner="مالک ۱",
            document_type=doc_fmt,
            confidentiality_level="PUBLIC",
            initial_version_id="V1",
            active_version_id="V1",
        )
        assert rec.document_type == doc_fmt.upper()

    with pytest.raises(DomainValidationError):
        MetadataRecord(
            domain="فنی",
            sub_domain="اسناد",
            organization_or_node="L3-NODE",
            project="پروژه ۱",
            author_or_owner="مالک ۱",
            document_type="EXE",
            confidentiality_level="PUBLIC",
            initial_version_id="V1",
            active_version_id="V1",
        )


def test_07_knowledge_item_versioning_and_chunks_lifecycle(
    sample_metadata: MetadataRecord,
) -> None:
    """آزمون ۷: بررسی چرخه عمر `KnowledgeItem`، `KnowledgeVersion` و `KnowledgeChunk` بدون حذف نسخه‌های قبلی."""
    item = KnowledgeItem(
        item_id="DOC-100",
        title="سند استاندارد خردسازی اسناد فارسی",
        owner_node_id=NodeIdentifier("L3-RAG-TEAM"),
        scope=KnowledgeScope(domain="پردازش زبان طبیعی", confidentiality_level="INTERNAL"),
        metadata=sample_metadata,
    )

    v1 = item.publish_new_version(
        version_id="VER-100-1",
        new_tag_str="v1.0.0",
        content_summary="نسخه اولیه قواعد خردسازی",
        change_reason="ثبت اولیه سند",
    )
    chunk = KnowledgeChunk(
        chunk_id="CHK-100-1",
        source_document_id=item.item_id,
        version_tag=v1.version_tag,
        sequence_order=0,
        text_content="هر قطعه دانش شناسه سند منبع و شماره صفحه را حفظ می‌کند.",
        inherited_metadata=sample_metadata,
        page_or_section_ref="صفحه ۱۴",
        heading_path=("مدل دانش", "خردسازی دانش"),
    )
    v1.attach_chunk(chunk.chunk_id)

    v2 = item.publish_new_version(
        version_id="VER-100-2",
        new_tag_str="v1.1.0",
        content_summary="افزودن قواعد تشخیص تکرار با Hash و Embedding",
        change_reason="بازبینی فاز ۵",
        diff_summary="اضافه شدن بخش ۷.۷",
    )

    assert len(item.versions) == 2
    assert v1.is_active is False
    assert v2.is_active is True
    assert v2.previous_version_id == "VER-100-1"
    assert v2.version_tag.sequence_number == 2
    assert item.get_active_version() == v2
    assert item.metadata.active_version_id == "VER-100-2"


def test_08_issue_subissue_decomposition_and_progress_calculation(
    sample_metadata: MetadataRecord,
) -> None:
    """آزمون ۸: بررسی تجزیه `Issue` به `SubIssue`، محاسبه قطعی درصد پیشرفت و ثبت شکاف پژوهشی."""
    issue = Issue(
        issue_id="ISS-200",
        owner_node_id=NodeIdentifier("L3-RAG-TEAM"),
        title="کاهش زمان استنتاج مدل LLM محلی در خلاصه‌سازی مدیریتی",
        description="بررسی تکنیک‌های کوانتیزاسیون و کش کردن زمینه در محیط آفلاین.",
        metadata=sample_metadata,
    )

    sub_a = SubIssue(
        sub_issue_id="SUB-200-1",
        parent_issue_id="ISS-200",
        title="آزمون مدل ۴ بیتی GGUF",
        description="ارزیابی دقت و سرعت مدل کوانتیزه‌شده.",
        relative_order=1,
        metadata=sample_metadata,
        progress=ProgressPercentage(100.0),
    )
    sub_b = SubIssue(
        sub_issue_id="SUB-200-2",
        parent_issue_id="ISS-200",
        title="تست پایداری خروجی داور روی مدل کوانتیزه",
        description="بررسی واریانس امتیازات Rubric.",
        relative_order=2,
        metadata=sample_metadata,
        progress=ProgressPercentage(40.0),
    )

    issue.add_sub_issue(sub_a)
    issue.add_sub_issue(sub_b)
    assert issue.progress.value == 70.0
    assert issue.status == "IN_PROGRESS"

    sub_b.update_progress(100.0)
    issue.recalculate_progress()
    assert issue.progress.value == 100.0
    assert issue.status == "RESOLVED"

    issue.flag_research_gap("نیاز به بنچمارک مدل‌های زیر ۳ میلیارد پارامتر روی متون حقوقی فارسی.")
    assert issue.has_research_gap is True

    # بررسی خطای عدم تطابق شناسه مسئله والد
    mismatched_sub = SubIssue(
        sub_issue_id="SUB-999",
        parent_issue_id="ISS-OTHER",
        title="زیرمسئله نامرتبط",
        description="شرح",
        relative_order=1,
        metadata=sample_metadata,
    )
    with pytest.raises(DomainValidationError):
        issue.add_sub_issue(mismatched_sub)


def test_09_executor_transfer_package_rfp_proposal_and_feedback_entities(
    sample_metadata: MetadataRecord,
) -> None:
    """آزمون ۹: بررسی موجودیت‌های `Executor`, `TransferPackage`, `JudgeScore`, `RFP`, `Proposal` و `FeedbackReport`."""
    executor = Executor(
        executor_id="EXEC-10",
        name="مهندس رضا محمدی",
        node_id=NodeIdentifier("L3-RAG-TEAM"),
        organizational_role="متخصص بازیابی برداری",
        expertise_domains=("Embedding", "Vector Search", "Python"),
        capabilities=("طراحی نمایه محلی",),
    )
    assert executor.has_expertise_in("vector") is True
    executor.assign_issue("ISS-200")
    assert "ISS-200" in executor.active_issue_ids

    rubric_ok = JudgeRubric(5, 4, 3, 3, 2, 2, 0)  # مجموع = ۱۹
    jscore_ok = JudgeScore(
        score_id="JS-1",
        target_entity_id="PKG-1",
        rubric=rubric_ok,
        explanation="امتیاز ۱۹ از ۲۰ - مورد تأیید",
    )
    pkg = TransferPackage(
        package_id="PKG-1",
        source_node_id=NodeIdentifier("L3-RAG-TEAM"),
        target_layer_index=LayerIndex(2),
        version_tag=VersionTag("v1.0.0", 1),
        detail_level=TransferDetailLevel.TECHNICAL_SUMMARY,
        reporting_phase="فاز ۱",
        achievements=("تکمیل مدل دامنه",),
        source_citations=("DOC-100",),
    )
    assert pkg.evaluate_with_judge(jscore_ok, require_threshold=True) is True
    assert pkg.is_accepted_for_delivery is True

    rfp = RFP(
        rfp_id="RFP-10",
        origin_issue_id="ISS-200",
        issuer_node_id=NodeIdentifier("L2-CENTER"),
        title="بهینه‌سازی موتور بازیابی برداری آفلاین",
        problem_statement="نیاز به کاهش تأخیر بازیابی در مخازن بالای ۱۰۰ هزار قطعه.",
        technical_requirements=("پشتیبانی از فیلتر فراداده ۱۸ فیلدی",),
        expected_deliverables=("ماژول نمایه برداری محلی",),
        evaluation_criteria=("زمان بازیابی زیر ۵۰ میلی‌ثانیه",),
        constraints=("اجرای کاملاً آفلاین روی CPU",),
        required_expertise=("مهندسی بازیابی اطلاعات",),
    )
    assert rfp.has_all_mandatory_sections() is True

    proposal = Proposal(
        proposal_id="PROP-10",
        rfp_id=rfp.rfp_id,
        proposer_id="L3-RAG-TEAM",
        title="پیشنهاد نمایه‌سازی ترکیبی با پیش‌فیلتر فراداده",
        proposed_methodology="اعمال فیلتر فراداده پیش از محاسبه شباهت کسینوسی.",
        technical_completeness_summary="پوشش ۱۰۰٪ نیازمندی‌ها.",
        feasibility_plan="قابل اجرا در ۱ فاز.",
        risk_management_plan="تست حافظه روی داده‌های مصنوعی.",
        metadata=sample_metadata,
    )
    feedback = FeedbackReport(
        report_id="FDB-10",
        rfp_id=rfp.rfp_id,
        proposal_id=proposal.proposal_id,
        strengths=("معماری تمیز و بدون وابستگی ابری",),
        suggested_score=19.0,
        recommendation_decision="SELECT",
        criterion_references=("معیار سرعت بازیابی",),
    )
    proposal.record_evaluation(
        score=feedback.suggested_score,
        status="APPROVED",
        feedback_report_id=feedback.report_id,
    )
    assert proposal.evaluated_score == 19.0
    assert proposal.feedback_report_id == "FDB-10"


def test_10_layer_node_registry_5_layer_hierarchy_and_ancestry_chain() -> None:
    """آزمون ۱۰: بررسی ثبت سلسله‌مراتب ۵ لایه‌ای (۰ تا ۴) و زنجیره بالادستی در `LayerNodeRegistry`."""
    registry = LayerNodeRegistry(initialize_standard_layers=True)
    assert len(registry.layers) == 5

    n0 = Node("L0-BOARD", "هیئت‌مدیره راهبردی", LayerIndex(0), "راهبری کلان")
    n1 = Node("L1-HOLDING", "سازمان مادر فناوری", LayerIndex(1), "مدیریت فناوری", parent_node_id="L0-BOARD")  # type: ignore[arg-type]
    n2 = Node("L2-AI-LAB", "مرکز پژوهشی هوش مصنوعی", LayerIndex(2), "پژوهش AI", parent_node_id="L1-HOLDING")  # type: ignore[arg-type]
    n3 = Node("L3-RAG-DEV", "تیم اجرایی RAG", LayerIndex(3), "توسعه هسته", parent_node_id="L2-AI-LAB")  # type: ignore[arg-type]
    n4 = Node("L4-LOCAL-LIB", "کتابخانه اسناد فنی", LayerIndex(4), "مخزن اسناد", parent_node_id="L3-RAG-DEV", is_leaf_boundary=True)  # type: ignore[arg-type]

    for node in (n0, n1, n2, n3, n4):
        registry.register_node(node)

    assert registry.validate_integrity() is True
    chain = registry.get_ancestry_chain("L4-LOCAL-LIB")
    assert [n.node_id.value for n in chain] == [
        "L4-LOCAL-LIB",
        "L3-RAG-DEV",
        "L2-AI-LAB",
        "L1-HOLDING",
        "L0-BOARD",
    ]


def test_11_layer_node_registry_parent_child_hierarchy_violations() -> None:
    """آزمون ۱۱: بررسی خطاهای نقض سلسله‌مراتب والد و فرزند (`InvalidLayerHierarchyError`)."""
    registry = LayerNodeRegistry(initialize_standard_layers=True)
    registry.register_node(Node("L0-ROOT", "لایه راهبردی", LayerIndex(0), "راهبری"))
    registry.register_node(
        Node("L2-UNIT", "واحد تخصصی", LayerIndex(2), "مهندسی", parent_node_id=NodeIdentifier("L0-ROOT"))
    )

    # ۱. گره لایه ۰ نمی‌تواند والد داشته باشد
    with pytest.raises(InvalidLayerHierarchyError):
        Node(
            node_id=NodeIdentifier("L0-INVALID"),
            name="گره راهبردی نامعتبر",
            layer_index=LayerIndex(0),
            domain_of_activity="راهبری",
            parent_node_id=NodeIdentifier("L0-ROOT"),
        )

    # ۲. گره لایه ۱ تا ۴ نمی‌تواند بدون والد باشد
    with pytest.raises(InvalidLayerHierarchyError):
        Node(
            node_id=NodeIdentifier("L1-ORPHAN"),
            name="سازمان بدون والد",
            layer_index=LayerIndex(1),
            domain_of_activity="مدیریت",
            parent_node_id=None,
        )

    # ۳. گره فرزند در لایه ۱ نمی‌تواند والدی در لایه ۲ (پایین‌تر از خود) داشته باشد
    with pytest.raises(InvalidLayerHierarchyError):
        registry.register_node(
            Node(
                node_id=NodeIdentifier("L1-BAD-PARENT"),
                name="گره لایه ۱ با والد لایه ۲",
                layer_index=LayerIndex(1),
                domain_of_activity="مدیریت",
                parent_node_id=NodeIdentifier("L2-UNIT"),
            )
        )

    # ۴. یک گره نمی‌تواند در دو لایه متفاوت ثبت شود
    with pytest.raises(InvalidLayerHierarchyError):
        registry.register_node(
            Node(
                node_id=NodeIdentifier("L2-UNIT"),
                name="ثبت مجدد در لایه ۳",
                layer_index=LayerIndex(3),
                domain_of_activity="اجرایی",
                parent_node_id=NodeIdentifier("L0-ROOT"),
            )
        )


def test_12_leaf_node_boundary_violation_error() -> None:
    """آزمون ۱۲: بررسی سخت‌گیرانه مرز گره نهایی (`is_leaf_boundary=True`) و پرتاب `LeafNodeBoundaryViolationError`."""
    registry = LayerNodeRegistry(initialize_standard_layers=True)
    registry.register_node(Node("L0-ROOT", "مرجع راهبردی", LayerIndex(0), "راهبری"))
    leaf_org = Node(
        node_id=NodeIdentifier("L1-LEAF-COMPANY"),
        name="شرکت مستقل با مرز بسته",
        layer_index=LayerIndex(1),
        domain_of_activity="عملیات مستقل",
        parent_node_id=NodeIdentifier("L0-ROOT"),
        is_leaf_boundary=True,
    )
    registry.register_node(leaf_org)

    # تلاش برای ثبت زیرمجموعه برای گره‌ای که is_leaf_boundary=True دارد باید خطا بدهد
    with pytest.raises(LeafNodeBoundaryViolationError) as exc_info:
        registry.register_node(
            Node(
                node_id=NodeIdentifier("L2-FORBIDDEN-CHILD"),
                name="زیرمجموعه غیرمجاز",
                layer_index=LayerIndex(2),
                domain_of_activity="زیرواحد",
                parent_node_id=NodeIdentifier("L1-LEAF-COMPANY"),
            )
        )
    assert exc_info.value.parent_node_id == "L1-LEAF-COMPANY"
    assert exc_info.value.child_node_id == "L2-FORBIDDEN-CHILD"

    # همچنین گره‌ای که از قبل فرزند دارد (L0-ROOT) نمی‌تواند به مرز نهایی تبدیل شود
    with pytest.raises(LeafNodeBoundaryViolationError):
        registry.set_leaf_boundary("L0-ROOT", is_leaf_boundary=True)


def test_13_registry_serialization_and_case_study_json_verification(tmp_path: Path) -> None:
    """آزمون ۱۳: بررسی خروجی/ورودی JSON رجیستری و صحت فایل تولیدشده مطالعه موردی در `sample_docs/`."""
    registry = LayerNodeRegistry(initialize_standard_layers=True)
    registry.register_node(Node("L0-MAIN", "ستاد راهبردی", LayerIndex(0), "سیاست‌گذاری"))
    registry.register_node(
        Node(
            "L1-SUB",
            "سازمان مادر",
            LayerIndex(1),
            "مدیریت",
            parent_node_id=NodeIdentifier("L0-MAIN"),
            is_leaf_boundary=True,
        )
    )
    registry.update_reporting_rules(
        "L1-SUB",
        reporting_rules={"min_judge_score": 18, "format": "CSV"},
        required_detail_level=TransferDetailLevel.EXECUTIVE_HIGH,
    )

    json_file = tmp_path / "test_registry.json"
    registry.save_to_json_file(json_file)
    reloaded = LayerNodeRegistry.load_from_json_file(json_file)

    assert len(reloaded.layers) == 5
    assert len(reloaded.nodes) == 2
    sub_node = reloaded.get_node("L1-SUB")
    assert sub_node.is_leaf_boundary is True
    assert sub_node.reporting_rules["min_judge_score"] == 18
    assert sub_node.required_transfer_detail_level == TransferDetailLevel.EXECUTIVE_HIGH

    with pytest.raises(RegistryNotFoundError):
        reloaded.get_node("NON-EXISTENT-NODE")

    # بررسی فایل مطالعه موردی فاز ۱ در صورت وجود
    case_study_file = Path(__file__).resolve().parent.parent / "sample_docs" / "case_study_registry.json"
    if case_study_file.exists():
        cs_registry = LayerNodeRegistry.load_from_json_file(case_study_file)
        assert cs_registry.validate_integrity() is True
        assert len(cs_registry.layers) == 5
        assert len(cs_registry.nodes) >= 10
