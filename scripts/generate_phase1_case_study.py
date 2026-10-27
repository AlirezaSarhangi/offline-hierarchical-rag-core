"""
اسکریپت تولید داده‌های پایه، رجیستری ۵ لایه‌ای و اسناد نمونه مطالعه موردی فاز ۱.

این اسکریپت یک مطالعه موردی کامل ۵ لایه‌ای (از لایه ۰ راهبردی تا لایه ۴ منابع داخلی دانش) را
با استفاده از ماژول‌های هسته دامنه (`src.domain`) شبیه‌سازی و اعتبارسنجی کرده و خروجی‌های زیر را
در پوشه `sample_docs/` ذخیره می‌نماید:
1. `sample_docs/case_study_registry.json`: رجیستری کامل لایه‌ها و گره‌های سازمانی مطالعه موردی.
2. `sample_docs/01_technical_spec_rag_pipeline.md`: نمونه سند فنی معماری و خط لوله RAG آفلاین.
3. `sample_docs/02_executor_profiles_and_ike_sample.json`: نمونه داده ساختاریافته مدل Issue-Knowledge-Executor، فراداده ۱۸ فیلدی، بسته انتقال و امتیاز داور.
4. `sample_docs/03_project_report_and_research_gaps.md`: نمونه گزارش پروژه، دستاوردها، مسائل حل‌نشده و شکاف‌های پژوهشی.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.domain import (
    Executor,
    FeedbackReport,
    Issue,
    JudgeRubric,
    JudgeScore,
    JudgeThreshold,
    KnowledgeChunk,
    KnowledgeItem,
    KnowledgeScope,
    LayerIndex,
    LayerNodeRegistry,
    MetadataRecord,
    Node,
    NodeIdentifier,
    ProgressPercentage,
    Proposal,
    RFP,
    SubIssue,
    TransferDetailLevel,
    TransferPackage,
    VersionTag,
)


def build_case_study_registry() -> LayerNodeRegistry:
    """ساخت و اعتبارسنجی رجیستری ۵ لایه‌ای مطالعه موردی هلدینگ فناوری و هوش مصنوعی آفلاین."""
    registry = LayerNodeRegistry(initialize_standard_layers=True)

    # لایه ۰: لایه راهبردی
    registry.register_node(
        Node(
            node_id=NodeIdentifier("L0-STRATEGIC-BOARD"),
            name="شورای عالی راهبردی و هیئت‌مدیره هلدینگ فناوری",
            layer_index=LayerIndex(0),
            domain_of_activity="سیاست‌گذاری کلان، حکمرانی دانش، تصویب RFPهای راهبردی و تخصیص کلان منابع",
            parent_node_id=None,
            is_leaf_boundary=False,
            reporting_rules={
                "mandatory_topics": [
                    "strategic_achievements",
                    "critical_risks",
                    "unresolved_key_issues",
                    "rfp_status",
                    "inter_node_priorities",
                ],
                "excluded_topics": ["low_level_code_logs", "raw_chunk_traces"],
                "min_judge_score": 18,
                "max_package_length_words": 1200,
                "preferred_output_format": "CSV_AND_DB",
            },
            required_transfer_detail_level=TransferDetailLevel.EXECUTIVE_HIGH,
            confidentiality_constraints=("PUBLIC", "INTERNAL", "CONFIDENTIAL", "RESTRICTED"),
        )
    )

    # لایه ۱: سازمان‌های مادر
    registry.register_node(
        Node(
            node_id=NodeIdentifier("L1-AI-MOTHER-ORG"),
            name="سازمان مادر هوش مصنوعی و سامانه‌های دانش‌بنیان",
            layer_index=LayerIndex(1),
            domain_of_activity="مدیریت مراکز پژوهشی هوش مصنوعی، راهبری هسته RAG آفلاین و نظارت بر انتقال دانش",
            parent_node_id=NodeIdentifier("L0-STRATEGIC-BOARD"),
            is_leaf_boundary=False,
            reporting_rules={
                "mandatory_topics": [
                    "project_status",
                    "issue_resolution_rate",
                    "research_gaps",
                    "resource_needs",
                ],
                "excluded_topics": ["unit_test_stacktraces"],
                "min_judge_score": 18,
                "preferred_output_format": "CSV_AND_DB",
            },
            required_transfer_detail_level=TransferDetailLevel.EXECUTIVE_HIGH,
            confidentiality_constraints=("PUBLIC", "INTERNAL", "CONFIDENTIAL"),
        )
    )

    registry.register_node(
        Node(
            node_id=NodeIdentifier("L1-INFRA-MOTHER-ORG"),
            name="سازمان مادر زیرساخت، امنیت و شبکه ایزوله",
            layer_index=LayerIndex(1),
            domain_of_activity="تأمین زیرساخت سخت‌افزاری آفلاین، امنیت شبکه داخلی و پایداری پایگاه‌های داده محلی",
            parent_node_id=NodeIdentifier("L0-STRATEGIC-BOARD"),
            is_leaf_boundary=False,
            reporting_rules={
                "mandatory_topics": ["offline_compliance", "hardware_capacity", "security_risks"],
                "min_judge_score": 18,
                "preferred_output_format": "CSV_AND_DB",
            },
            required_transfer_detail_level=TransferDetailLevel.EXECUTIVE_HIGH,
            confidentiality_constraints=("INTERNAL", "CONFIDENTIAL", "RESTRICTED"),
        )
    )

    # نمونه گره نهایی در لایه ۱ که زیرساختار داخلی آن خارج از مرز سامانه است (is_leaf_boundary=True)
    registry.register_node(
        Node(
            node_id=NodeIdentifier("L1-EXTERNAL-AUDIT-ORG"),
            name="مؤسسه مستقل ممیزی و استانداردسازی برون‌سازمانی",
            layer_index=LayerIndex(1),
            domain_of_activity="ممیزی مستقل کیفیت و استانداردهای امنیتی (زیرساختار داخلی خارج از مرز مدل‌سازی)",
            parent_node_id=NodeIdentifier("L0-STRATEGIC-BOARD"),
            is_leaf_boundary=True,
            reporting_rules={
                "mandatory_topics": ["compliance_certificates", "audit_findings"],
                "min_judge_score": 18,
                "boundary_note": "گره نهایی - زیرساختار داخلی مدل نمی‌شود",
            },
            required_transfer_detail_level=TransferDetailLevel.EXECUTIVE_HIGH,
            confidentiality_constraints=("PUBLIC", "INTERNAL"),
        )
    )

    # لایه ۲: زیرمجموعه‌ها و واحدهای تخصصی
    registry.register_node(
        Node(
            node_id=NodeIdentifier("L2-NLP-RESEARCH-CENTER"),
            name="مرکز پژوهشی پردازش زبان طبیعی و مدل‌های زبانی کوچک (SLM)",
            layer_index=LayerIndex(2),
            domain_of_activity="پژوهش و توسعه مدل‌های کوچک محلی Embedding و LLM، مهندسی پرامپت و ارزیابی داور",
            parent_node_id=NodeIdentifier("L1-AI-MOTHER-ORG"),
            is_leaf_boundary=False,
            reporting_rules={
                "mandatory_topics": [
                    "methodology_summary",
                    "model_evaluation_metrics",
                    "unresolved_technical_issues",
                    "rfp_proposals_summary",
                ],
                "min_judge_score": 18,
                "preferred_output_format": "CSV_AND_DB",
            },
            required_transfer_detail_level=TransferDetailLevel.MANAGERIAL,
            confidentiality_constraints=("INTERNAL", "CONFIDENTIAL"),
        )
    )

    registry.register_node(
        Node(
            node_id=NodeIdentifier("L2-DOC-VISION-UNIT"),
            name="واحد مهندسی پردازش اسناد و مدل‌های بینایی-زبانی (LVM)",
            layer_index=LayerIndex(2),
            domain_of_activity="استخراج ساختاریافته DOC، DOCX و PDF و تفسیر چیدمان و جداول اسناد با LVM محلی",
            parent_node_id=NodeIdentifier("L1-AI-MOTHER-ORG"),
            is_leaf_boundary=False,
            reporting_rules={
                "mandatory_topics": ["extraction_accuracy", "supported_formats", "layout_challenges"],
                "min_judge_score": 18,
                "preferred_output_format": "CSV_AND_DB",
            },
            required_transfer_detail_level=TransferDetailLevel.MANAGERIAL,
            confidentiality_constraints=("INTERNAL", "CONFIDENTIAL"),
        )
    )

    registry.register_node(
        Node(
            node_id=NodeIdentifier("L2-OFFLINE-SEC-UNIT"),
            name="واحد عملیات ذخیره‌سازی محلی و کنترل قیود آفلاین",
            layer_index=LayerIndex(2),
            domain_of_activity="مدیریت پایگاه داده محلی، نمایه برداری آفلاین و کنترل عدم وابستگی به اینترنت",
            parent_node_id=NodeIdentifier("L1-INFRA-MOTHER-ORG"),
            is_leaf_boundary=True,
            reporting_rules={
                "mandatory_topics": ["storage_health", "airgap_verification"],
                "min_judge_score": 18,
                "boundary_note": "واحد عملیاتی با مرز نهایی مدل‌سازی در لایه ۲",
            },
            required_transfer_detail_level=TransferDetailLevel.MANAGERIAL,
            confidentiality_constraints=("INTERNAL", "CONFIDENTIAL", "RESTRICTED"),
        )
    )

    # لایه ۳: پروژه‌ها، تیم‌ها و گره‌های اجرایی
    registry.register_node(
        Node(
            node_id=NodeIdentifier("L3-RAG-CORE-TEAM"),
            name="تیم اجرایی توسعه هسته RAG سلسله‌مراتبی و مدل دامنه",
            layer_index=LayerIndex(3),
            domain_of_activity="پیاده‌سازی معماری DDD، چرخه عمر دانش، تشخیص تکرار/هم‌پوشانی و مدل Issue-Knowledge-Executor",
            parent_node_id=NodeIdentifier("L2-NLP-RESEARCH-CENTER"),
            is_leaf_boundary=False,
            reporting_rules={
                "mandatory_topics": [
                    "issue_decomposition",
                    "executor_mapping",
                    "chunking_and_dedup_metrics",
                    "unit_test_coverage",
                ],
                "min_judge_score": 18,
                "preferred_output_format": "CSV_AND_DB",
            },
            required_transfer_detail_level=TransferDetailLevel.TECHNICAL_SUMMARY,
            confidentiality_constraints=("INTERNAL", "CONFIDENTIAL"),
        )
    )

    registry.register_node(
        Node(
            node_id=NodeIdentifier("L3-JUDGE-CALIBRATION-TEAM"),
            name="گروه اجرایی کالیبراسیون داور LLM و قالب‌های پرامپت",
            layer_index=LayerIndex(3),
            domain_of_activity="طراحی Rubric بیست‌نمره‌ای، نسخه‌بندی پرامپت‌ها و ارزیابی وفاداری پاسخ به منبع",
            parent_node_id=NodeIdentifier("L2-NLP-RESEARCH-CENTER"),
            is_leaf_boundary=True,
            reporting_rules={
                "mandatory_topics": ["rubric_stability", "prompt_schema_stability"],
                "min_judge_score": 18,
            },
            required_transfer_detail_level=TransferDetailLevel.TECHNICAL_SUMMARY,
            confidentiality_constraints=("INTERNAL",),
        )
    )

    registry.register_node(
        Node(
            node_id=NodeIdentifier("L3-INGESTION-PIPELINE-TEAM"),
            name="تیم پیاده‌سازی مبدل‌های اسناد (DOC/DOCX/PDF) و مفسر کمکی LVM",
            layer_index=LayerIndex(3),
            domain_of_activity="پیاده‌سازی Adapterهای استخراج متن، حفظ ارجاع صفحه/بخش و علامت‌گذاری اسناد بصری",
            parent_node_id=NodeIdentifier("L2-DOC-VISION-UNIT"),
            is_leaf_boundary=True,
            reporting_rules={
                "mandatory_topics": ["doc_docx_pdf_pipeline", "lvm_adapter_status"],
                "min_judge_score": 18,
            },
            required_transfer_detail_level=TransferDetailLevel.TECHNICAL_SUMMARY,
            confidentiality_constraints=("INTERNAL",),
        )
    )

    # لایه ۴: منابع داخلی دانش اختیاری
    registry.register_node(
        Node(
            node_id=NodeIdentifier("L4-ARCH-DOCS-ARCHIVE"),
            name="مخزن محلی اسناد معماری، مقالات علمی و مستندات کتابخانه‌های نرم‌افزاری",
            layer_index=LayerIndex(4),
            domain_of_activity="نگهداری اسناد مرجع PDF/DOCX، استانداردهای طراحی دامنه‌محور و راهنماهای فنی آفلاین",
            parent_node_id=NodeIdentifier("L3-RAG-CORE-TEAM"),
            is_leaf_boundary=True,
            reporting_rules={
                "mandatory_topics": ["indexed_documents_count", "version_history"],
                "min_judge_score": 18,
            },
            required_transfer_detail_level=TransferDetailLevel.TECHNICAL_DETAILED,
            confidentiality_constraints=("PUBLIC", "INTERNAL"),
        )
    )

    registry.register_node(
        Node(
            node_id=NodeIdentifier("L4-PERSONNEL-PROFILES-REPO"),
            name="آرشیو سوابق پرسنلی، پروفایل تخصص مجریان و گزارش‌های تکمیل پروژه",
            layer_index=LayerIndex(4),
            domain_of_activity="نگهداری پروفایل‌های نیروی انسانی، برچسب‌های تخصص فنی و سوابق حل مسائل قبلی",
            parent_node_id=NodeIdentifier("L3-RAG-CORE-TEAM"),
            is_leaf_boundary=True,
            reporting_rules={
                "mandatory_topics": ["executor_profiles", "historical_issue_resolutions"],
                "min_judge_score": 18,
            },
            required_transfer_detail_level=TransferDetailLevel.TECHNICAL_DETAILED,
            confidentiality_constraints=("INTERNAL", "CONFIDENTIAL"),
        )
    )

    registry.validate_integrity()
    return registry


def build_sample_domain_objects_payload() -> dict:
    """ساخت نمونه کامل اشیاء مقداری و موجودیت‌های دامنه برای ذخیره در `sample_docs/`."""
    sample_metadata = MetadataRecord(
        domain="مدیریت دانش و هوش مصنوعی آفلاین",
        sub_domain="بازیابی سلسله‌مراتبی و استخراج اسناد",
        organization_or_node="L3-RAG-CORE-TEAM",
        project="هسته RAG سلسله‌مراتبی آفلاین - فاز ۱",
        author_or_owner="دکتر سارا احمدی (معمار ارشد دامنه)",
        document_type="PDF",
        confidentiality_level="INTERNAL",
        initial_version_id="VER-001",
        active_version_id="VER-001",
        related_issue_id="ISS-101",
        related_executor_id="EXEC-201",
        related_rfp_id="RFP-301",
        status="ACTIVE",
        priority="HIGH",
        tags=("DDD", "Offline-RAG", "Hierarchical-Transfer", "LLM-Judge"),
        source_reliability=0.96,
        target_transfer_layer=LayerIndex(2),
        suggested_detail_level=TransferDetailLevel.TECHNICAL_SUMMARY,
    )

    knowledge_item = KnowledgeItem(
        item_id="KI-001",
        title="مستند معماری مرجع خط لوله RAG آفلاین و قواعد خردسازی",
        owner_node_id=NodeIdentifier("L3-RAG-CORE-TEAM"),
        scope=KnowledgeScope(
            domain="مهندسی نرم‌افزار و هوش مصنوعی",
            confidentiality_level="INTERNAL",
            allowed_node_ids=("L3-RAG-CORE-TEAM", "L2-NLP-RESEARCH-CENTER", "L1-AI-MOTHER-ORG"),
            max_target_layer=LayerIndex(0),
        ),
        metadata=sample_metadata,
    )
    ver1 = knowledge_item.publish_new_version(
        version_id="VER-001",
        new_tag_str="v1.0.0",
        content_summary="نسخه پایه معماری خط لوله ۱۲ مرحله‌ای RAG و قواعد خردسازی اسناد فارسی.",
        change_reason="ثبت اولیه در فاز کشف دامنه",
    )
    chunk1 = KnowledgeChunk(
        chunk_id="CHK-001-01",
        source_document_id=knowledge_item.item_id,
        version_tag=ver1.version_tag,
        sequence_order=1,
        text_content=(
            "هسته RAG آفلاین اسناد محلی با قالب‌های DOC، DOCX و PDF را پردازش کرده و بدون وابستگی به "
            "سرویس‌های ابری، قطعات دانش را همراه با شناسه سند منبع، نسخه و شماره صفحه بازیابی می‌کند."
        ),
        inherited_metadata=sample_metadata,
        page_or_section_ref="صفحه ۱۴ - بخش ۷.۳ و ۷.۴",
        heading_path=("۷. مدل دانش", "۷.۴ خردسازی دانش"),
        context_boundary_note="مرز قطعه اول - تعریف قواعد نگهداری ارجاع منبع",
    )
    ver1.attach_chunk(chunk1.chunk_id)

    executor = Executor(
        executor_id="EXEC-201",
        name="مهندس علی کاظمی",
        node_id=NodeIdentifier("L3-RAG-CORE-TEAM"),
        organizational_role="مهندس ارشد بازیابی اطلاعات و مدل‌های زبانی آفلاین",
        expertise_domains=("پردازش زبان طبیعی فارسی", "مدل‌های Embedding محلی", "طراحی دامنه‌محور DDD"),
        capabilities=("کالیبراسیون داور LLM", "خردسازی آگاه از ساختار اسناد", "بهینه‌سازی بازیابی ترکیبی"),
        past_project_ids=("PRJ-OFFLINE-SEARCH-1403", "PRJ-DOC-PARSER-1404"),
        authored_document_ids=("KI-001",),
        availability_status="AVAILABLE",
    )

    parent_issue = Issue(
        issue_id="ISS-101",
        owner_node_id=NodeIdentifier("L3-RAG-CORE-TEAM"),
        title="بهینه‌سازی دقت استخراج جداول پیچیده فارسی در اسناد PDF اسکن‌شده آفلاین",
        description=(
            "در برخی گزارش‌های فنی قدیمی که به صورت PDF اسکن‌شده یا دارای جداول تودرتو هستند، "
            "استخراج متنی ساده باعث به هم ریختن ستون‌ها می‌شود و نیازمند راهبرد ترکیبی و مفسر کمکی LVM است."
        ),
        metadata=sample_metadata,
        status="IN_PROGRESS",
        priority="HIGH",
    )
    sub1 = SubIssue(
        sub_issue_id="SUB-101-1",
        parent_issue_id="ISS-101",
        title="طراحی Adapter تشخیص صفحات چیدمان‌محور و جدول‌دار",
        description="شناسایی خودکار صفحاتی که نیازمند ارسال به مدل بینایی-زبانی (LVM) محلی هستند.",
        relative_order=1,
        metadata=sample_metadata,
        status="RESOLVED",
        progress=ProgressPercentage(100.0),
        related_knowledge_ids=["KI-001"],
        candidate_executor_ids=["EXEC-201"],
        proposed_solution="استفاده از تحلیل تراکم خطوط جدول و ارسال انتخابی صفحه به Adapter مدل LVM.",
    )
    sub2 = SubIssue(
        sub_issue_id="SUB-101-2",
        parent_issue_id="ISS-101",
        title="ارزیابی مصرف حافظه و زمان پاسخ مدل LVM کوچک روی سخت‌افزار محلی",
        description="تست بنچمارک مدل‌های بینایی-زبانی کوانتیزه‌شده در محیط بدون اینترنت.",
        relative_order=2,
        metadata=sample_metadata,
        status="IN_PROGRESS",
        progress=ProgressPercentage(50.0),
        related_knowledge_ids=["KI-001"],
        candidate_executor_ids=["EXEC-201"],
    )
    parent_issue.add_sub_issue(sub1)
    parent_issue.add_sub_issue(sub2)
    parent_issue.flag_research_gap(
        "کمبود شواهد تجربی درباره دقت مدل‌های LVM زیر ۴ میلیارد پارامتر روی جداول مالی و مهندسی فارسی."
    )
    executor.assign_issue(parent_issue.issue_id)

    rubric = JudgeRubric(
        source_fidelity=5,
        completeness=4,
        detail_level_fit=3,
        uncertainty_and_gap_handling=3,
        clarity_and_structure=2,
        source_attribution=2,
        no_unsupported_claims=1,
    )
    judge_score = JudgeScore(
        score_id="JDG-501",
        target_entity_id="TPKG-401",
        rubric=rubric,
        threshold=JudgeThreshold(minimum_score=18),
        explanation="بسته انتقال کاملاً منطبق بر اسناد منبع گره L3-RAG-CORE-TEAM بوده و شکاف پژوهشی LVM را به‌درستی اعلام کرده است.",
    )

    transfer_pkg = TransferPackage(
        package_id="TPKG-401",
        source_node_id=NodeIdentifier("L3-RAG-CORE-TEAM"),
        target_layer_index=LayerIndex(2),
        version_tag=VersionTag("v1.0.0", 1),
        detail_level=TransferDetailLevel.TECHNICAL_SUMMARY,
        reporting_phase="فاز ۱ - کشف دامنه و خط پایه معماری",
        completed_projects_summary=("تکمیل مدل دامنه هسته DDD و رجیستری ۵ لایه‌ای سازمان",),
        research_conducted_summary=("بررسی معماری خط لوله ۱۲ مرحله‌ای RAG و چارچوب داور ۲۰ نمره‌ای",),
        achievements=("پیاده‌سازی کامل ۱۱ شیء مقداری، ۱۳ موجودیت و رجیستری سلسله‌مراتبی لایه و گره",),
        active_issues_summary=(f"{parent_issue.issue_id}: {parent_issue.title} (پیشرفت: {parent_issue.progress.value}%)",),
        challenges=("محدودیت توان پردازشی محلی برای اسناد تصویری سنگین",),
        unresolved_issues=("بهینه‌سازی نهایی مفسر کمکی LVM روی جداول پیچیده فارسی",),
        key_research_gaps=(parent_issue.research_gap_reason or "",),
        executor_capability_notes=(f"{executor.name} ({executor.executor_id}): آماده راهبری فنی فاز ورود اسناد",),
        rfp_and_proposal_status=("RFP-301 برای بهینه‌سازی استخراج جداول تصویری تدوین شد",),
        proposed_decisions_for_upper_layer=("تصویب تخصیص نمونه داده استاندارد PDF جدول‌دار در فاز ۲ و ۴",),
        source_citations=("KI-001 (VER-001, CHK-001-01)",),
    )
    transfer_pkg.evaluate_with_judge(judge_score, require_threshold=True)

    rfp = RFP(
        rfp_id="RFP-301",
        origin_issue_id=parent_issue.issue_id,
        issuer_node_id=NodeIdentifier("L2-NLP-RESEARCH-CENTER"),
        title="توسعه و کالیبراسیون ماژول استخراج جدول‌های پیچیده فارسی با مدل کمکی LVM آفلاین",
        problem_statement=parent_issue.description,
        sub_issue_model_summary=(sub1.title, sub2.title),
        related_knowledge_ids=("KI-001",),
        research_gaps=(parent_issue.research_gap_reason or "",),
        technical_requirements=(
            "اجرای ۱۰۰٪ آفلاین بدون فراخوانی ابری",
            "حفظ شناسه سند منبع، نسخه و شماره صفحه در تمامی قطعات خروجی",
        ),
        expected_deliverables=("Adapter استخراج جدول", "مجموعه آزمون واحد قطعی با Test Double"),
        evaluation_criteria=("دقت استخراج ساختار جدول بالای ۹۰٪", "احراز حداقل امتیاز ۱۸ از ۲۰ از داور LLM"),
        constraints=("حداکثر مصرف حافظه ۸ گیگابایت RAM/VRAM در محیط محلی",),
        required_expertise=("بینایی ماشین و پردازش اسناد", "مهندسی پایتون و معماری DDD"),
        expected_documentation=("مستند فنی طراحی Adapter", "گزارش ارزیابی خط پایه"),
    )

    proposal = Proposal(
        proposal_id="PROP-601",
        rfp_id=rfp.rfp_id,
        proposer_id="L3-INGESTION-PIPELINE-TEAM",
        title="پیشنهاد فنی خط لوله دو مرحله‌ای استخراج PDF با تشخیص چیدمان و مفسر LVM کوانتیزه",
        proposed_methodology="ترکیب استخراج متنی مبتنی بر مختصات با فراخوانی شرطی مدل LVM صرفاً برای صفحات دارای جدول پیچیده.",
        technical_completeness_summary="تمامی نیازمندی‌های حفظ فراداده، ارجاع صفحه و اجرای آفلاین پوشش داده شده‌اند.",
        feasibility_plan="قابل پیاده‌سازی در ۲ فاز اجرایی با استفاده از Adapterهای قطعی و قابل تست واحد.",
        risk_management_plan="تعریف راهبرد جایگزین (Fallback) به استخراج متنی استاندارد در صورت زمان‌بر شدن پردازش صفحه.",
        metadata=sample_metadata,
    )

    feedback = FeedbackReport(
        report_id="FDB-701",
        rfp_id=rfp.rfp_id,
        proposal_id=proposal.proposal_id,
        strengths=("رعایت کامل قید آفلاین", "طراحی ماژولار منطبق بر الگوی Adapter و قابلیت آزمون واحد"),
        weaknesses=("نیاز به تعیین دقیق‌تر حد آستانه تشخیص پیچیدگی چیدمان صفحه",),
        missing_requirements=(),
        ambiguous_assumptions=("فرضیات مربوط به وضوح اسکن (DPI) اسناد قدیمی باید شفاف شود",),
        technical_risks=("افزایش زمان پردازش چرخه‌ای در صورت زیاد بودن صفحات اسکن‌شده",),
        clarifications_needed=("تعیین حداکثر زمان مجاز پردازش هر صفحه در چرخه آفلاین",),
        suggested_score=18.5,
        recommendation_decision="SELECT",
        criterion_references=("نیازمندی فنی ۱: اجرای آفلاین", "معیار ارزیابی ۲: حفظ ارجاع صفحه"),
    )
    proposal.record_evaluation(
        score=feedback.suggested_score,
        status="APPROVED",
        feedback_report_id=feedback.report_id,
    )

    return {
        "metadata_record_18_fields": sample_metadata.to_dict(),
        "knowledge_item": {
            "item_id": knowledge_item.item_id,
            "title": knowledge_item.title,
            "owner_node_id": knowledge_item.owner_node_id.value,
            "active_version_tag": str(knowledge_item.active_version_tag),
            "chunks": [
                {
                    "chunk_id": chunk1.chunk_id,
                    "source_document_id": chunk1.source_document_id,
                    "version_tag": str(chunk1.version_tag),
                    "sequence_order": chunk1.sequence_order,
                    "page_or_section_ref": chunk1.page_or_section_ref,
                    "heading_path": list(chunk1.heading_path),
                    "text_content": chunk1.text_content,
                }
            ],
        },
        "issue_knowledge_executor_triangle": {
            "issue": {
                "issue_id": parent_issue.issue_id,
                "title": parent_issue.title,
                "status": parent_issue.status,
                "progress_percentage": parent_issue.progress.value,
                "has_research_gap": parent_issue.has_research_gap,
                "research_gap_reason": parent_issue.research_gap_reason,
                "sub_issues": [
                    {
                        "sub_issue_id": si.sub_issue_id,
                        "title": si.title,
                        "relative_order": si.relative_order,
                        "status": si.status,
                        "progress_percentage": si.progress.value,
                    }
                    for si in parent_issue.sub_issues
                ],
            },
            "executor": {
                "executor_id": executor.executor_id,
                "name": executor.name,
                "node_id": executor.node_id.value,
                "organizational_role": executor.organizational_role,
                "expertise_domains": list(executor.expertise_domains),
                "active_issue_ids": executor.active_issue_ids,
            },
            "supporting_knowledge_ids": ["KI-001"],
        },
        "transfer_package_and_judge": {
            "package_id": transfer_pkg.package_id,
            "source_node_id": transfer_pkg.source_node_id.value,
            "target_layer_index": transfer_pkg.target_layer_index.value,
            "detail_level": transfer_pkg.detail_level.value,
            "detail_level_label_fa": transfer_pkg.detail_level.label_fa,
            "is_accepted_for_delivery": transfer_pkg.is_accepted_for_delivery,
            "judge_score": {
                "score_id": judge_score.score_id,
                "total_score": judge_score.total_score,
                "threshold": judge_score.threshold.minimum_score,
                "is_approved": judge_score.is_approved,
                "rubric_breakdown": judge_score.rubric.to_dict(),
                "explanation": judge_score.explanation,
            },
        },
        "rfp_proposal_feedback_cycle": {
            "rfp_id": rfp.rfp_id,
            "rfp_title": rfp.title,
            "has_all_mandatory_sections": rfp.has_all_mandatory_sections(),
            "proposal_id": proposal.proposal_id,
            "proposal_status": proposal.evaluation_status,
            "evaluated_score": proposal.evaluated_score,
            "feedback_report_id": feedback.report_id,
            "recommendation_decision": feedback.recommendation_decision,
        },
    }


def write_sample_markdown_documents(sample_dir: Path) -> None:
    """ایجاد اسناد متنی نمونه در پوشه `sample_docs/` برای استفاده در فازهای بعدی و تست‌ها."""
    doc1_path = sample_dir / "01_technical_spec_rag_pipeline.md"
    doc1_content = """# مشخصات فنی خط لوله RAG سلسله‌مراتبی آفلاین (سند نمونه KI-001)

- **شناسه سند:** `KI-001`
- **گره مالک:** `L3-RAG-CORE-TEAM` (تیم اجرایی توسعه هسته RAG سلسله‌مراتبی و مدل دامنه)
- **نسخه فعال:** `v1.0.0`
- **سطح محرمانگی:** `INTERNAL`

## ۱. هدف و معماری خط لوله آفلاین
این سند قواعد فنی پردازش اسناد محلی (`DOC`، `DOCX` و `PDF`) و نحوه بازیابی مقید به مرز گره سازمانی را تعریف می‌کند. تمامی مدل‌های `Embedding`، `LLM` و `LVM` به صورت کاملاً محلی اجرا شده و هیچ‌گونه فراخوانی به سرویس‌های ابری انجام نمی‌شود.

## ۲. قواعد خردسازی (Chunking) و حفظ ارجاع منبع
هر قطعه دانش (`KnowledgeChunk`) که در فرآیند خردسازی تولید می‌شود باید ناوردایی‌های زیر را حفظ نماید:
1. شناسه یکتای سند منبع (`source_document_id`) و شناسه نسخه (`VersionTag`).
2. شماره صفحه یا ارجاع بخش (`page_or_section_ref`) جهت تضمین قابلیت ردیابی منبع.
3. مسیر سلسله‌مراتبی عنوان‌ها (`heading_path`) و فراداده ۱۸ فیلدی ارث‌بری‌شده از قلم دانش اصلی.

## ۳. کنترل کیفیت با سازوکار LLM به‌عنوان داور
پیش از ثبت نهایی بسته‌های انتقال دانش به لایه بالاتر، خروجی تولیدشده توسط `JudgeRubric` در ۷ معیار (مجموعاً ۲۰ امتیاز) ارزیابی می‌شود. حداقل امتیاز قابل قبول برای خروجی‌های مهم **۱۸ از ۲۰** است.
"""
    doc1_path.write_text(doc1_content, encoding="utf-8")

    doc3_path = sample_dir / "03_project_report_and_research_gaps.md"
    doc3_content = """# گزارش وضعیت پروژه، دستاوردها و شکاف‌های پژوهشی (گره L3-RAG-CORE-TEAM)

- **شناسه گزارش:** `KI-002`
- **گره گزارش‌دهنده:** `L3-RAG-CORE-TEAM` (لایه ۳)
- **لایه هدف انتقال:** لایه ۲ (`L2-NLP-RESEARCH-CENTER`) با سطح جزئیات `TECHNICAL_SUMMARY`
- **نسخه گزارش:** `v1.0.0`

## ۱. خلاصه پروژه‌های تکمیل‌شده و دستاوردها
- طراحی و پیاده‌سازی کامل مدل دامنه هسته مبتنی بر Domain-Driven Design (شامل ۱۰ زمینه مرزبندی‌شده، ۱۳ موجودیت و ۱۱ شیء مقداری تغییرناپذیر).
- پیاده‌سازی رجیستری لایه و گره (`LayerNodeRegistry`) با پشتیبانی از ۵ لایه سازمانی (لایه ۰ تا لایه ۴) و کنترل سخت‌گیرانه مرز گره نهایی (`is_leaf_boundary`).

## ۲. مسائل جاری و تجزیه به زیرمسئله
- **مسئله اصلی (`ISS-101`):** بهینه‌سازی دقت استخراج جداول پیچیده فارسی در اسناد PDF اسکن‌شده آفلاین (درصد پیشرفت: ۷۵٪).
  - **زیرمسئله ۱ (`SUB-101-1`):** طراحی Adapter تشخیص صفحات چیدمان‌محور (تکمیل‌شده - ۱۰۰٪).
  - **زیرمسئله ۲ (`SUB-101-2`):** ارزیابی مصرف حافظه و زمان پاسخ مدل LVM کوچک روی سخت‌افزار محلی (در حال اجرا - ۵۰٪).

## ۳. شکاف‌های پژوهشی شناسایی‌شده و وضعیت RFP
- **شکاف پژوهشی کلیدی:** کمبود شواهد تجربی درباره دقت مدل‌های بینایی-زبانی (LVM) کوچک زیر ۴ میلیارد پارامتر روی جداول مهندسی و مالی فارسی در محیط بدون GPU سنگین.
- **اقدام انجام‌شده:** صدور سند اعلام نیاز `RFP-301` و دریافت پیشنهاد فنی `PROP-601` از تیم خط لوله ورود اسناد (`L3-INGESTION-PIPELINE-TEAM`) که با امتیاز `18.5` از `20` مورد تأیید قرار گرفت.
"""
    doc3_path.write_text(doc3_content, encoding="utf-8")


def main() -> None:
    """اجرای کامل تولید مطالعه موردی فاز ۱ و ذخیره فایل‌ها در `sample_docs/`."""
    sample_dir = PROJECT_ROOT / "sample_docs"
    sample_dir.mkdir(parents=True, exist_ok=True)

    registry = build_case_study_registry()
    registry_file = registry.save_to_json_file(sample_dir / "case_study_registry.json")

    domain_payload = build_sample_domain_objects_payload()
    sample_json_file = sample_dir / "02_executor_profiles_and_ike_sample.json"
    sample_json_file.write_text(
        json.dumps(domain_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    write_sample_markdown_documents(sample_dir)

    print("=== Phase 1 Case Study Generation Completed Successfully ===")
    print(f"1. Registry JSON saved to: {registry_file}")
    print(f"   - Total Layers: {len(registry.layers)} (Layers 0 to 4)")
    print(f"   - Total Nodes : {len(registry.nodes)}")
    print(f"2. Domain & IKE Sample JSON saved to: {sample_json_file}")
    print(f"3. Sample Markdown Documents saved in: {sample_dir}")


if __name__ == "__main__":
    main()
