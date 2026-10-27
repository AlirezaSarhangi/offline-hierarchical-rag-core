"""
آزمون‌های واحد جامع فاز ۲: رجیستری مدل‌های آفلاین، رجیستری نسخه‌بندی پرامپت‌ها و آداپتورهای Ports & Adapters.

این مجموعه تست، تمامی نیازمندی‌های فاز ۲ (بخش‌های ۸.۲، ۸.۳، ۱۸.۲، ۱۹.۱ و ۱۹.۲ پروپوزال) را شامل
`ModelRegistry`، `OfflineModelSpecification`، هر ۱۲ دسته `PromptTaskCategory` در `PromptRegistry`،
مونتاژ پرامپت و خطای `PromptAssemblyError`، اعتبارسنجی Schema JSON و خطای `PromptOutputValidationError`،
قید شبکه محلی در `LocalOllamaAdapter`، تولید قطعی در `DeterministicMockModelAdapter` و گردش کار داوری
`LLM-as-a-Judge` با حد آستانه ۱۸ از ۲۰ راستی‌آزمایی می‌کند.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from src.domain import (
    DomainValidationError,
    IncompatibleModelTaskError,
    JudgeThreshold,
    JudgeThresholdNotMetError,
    ModelCategory,
    ModelRegistry,
    ModelTaskType,
    OfflineAdapterError,
    OfflineModelSpecification,
    PromptAssemblyError,
    PromptOutputValidationError,
    PromptRegistry,
    PromptTaskCategory,
    PromptTemplate,
    RegistryNotFoundError,
    VersionTag,
)
from src.infrastructure import (
    DeterministicMockModelAdapter,
    LocalOllamaAdapter,
)


def test_01_offline_model_specification_immutability_and_invariants() -> None:
    """آزمون ۱: بررسی تغییرناپذیری و اعتبارسنجی ناوردایی‌های `OfflineModelSpecification`."""
    spec = OfflineModelSpecification(
        model_id="custom-llm:3b-q4",
        model_name="Custom Local Persian LLM 3B",
        category=ModelCategory.LLM,
        quantization_format="GGUF_Q4_K_M",
        context_window_size=8192,
        supported_tasks=(ModelTaskType.QA_ANSWERING, ModelTaskType.TECHNICAL_SUMMARIZATION),
        default_inference_params={"temperature": 0.1, "top_p": 0.9, "max_tokens": 1024},
        known_limitations=("محدودیت دقت در متون بسیار طولانی بالای ۸ هزار توکن",),
        evaluation_notes="آزمایش‌شده در محیط ایزوله آزمایشگاه.",
    )
    assert spec.model_id == "custom-llm:3b-q4"
    assert spec.supports_task(ModelTaskType.QA_ANSWERING) is True
    assert spec.supports_task(ModelTaskType.SEMANTIC_INDEXING) is False

    with pytest.raises(FrozenInstanceError):
        spec.model_id = "mutated"  # type: ignore[misc]

    # عدم تطابق دسته مدل (EMBEDDING) با وظیفه مربوط به LLM
    with pytest.raises(DomainValidationError):
        OfflineModelSpecification(
            model_id="bad-emb",
            model_name="Bad Embedding Spec",
            category=ModelCategory.EMBEDDING,
            quantization_format="GGUF_F16",
            context_window_size=1024,
            supported_tasks=(ModelTaskType.QA_ANSWERING,),
            known_limitations=("محدودیت تست",),
            evaluation_notes="یادداشت تست",
        )

    # الزام ثبت حداقل یک محدودیت عملکردی شناخته شده (known_limitations)
    with pytest.raises(DomainValidationError):
        OfflineModelSpecification(
            model_id="no-limits-llm",
            model_name="LLM Without Limitations",
            category=ModelCategory.LLM,
            quantization_format="GGUF_Q4_K_M",
            context_window_size=4096,
            supported_tasks=(ModelTaskType.QA_ANSWERING,),
            known_limitations=(),
            evaluation_notes="یادداشت تست",
        )


def test_02_model_registry_default_candidates_and_task_routing() -> None:
    """آزمون ۲: بررسی ثبت ۶ مدل نامزد پیش‌فرض (Embedding, LLM, LVM) و بازیابی بر اساس وظیفه در `ModelRegistry`."""
    registry = ModelRegistry(initialize_defaults=True)
    assert len(registry.models) == 6

    emb_models = registry.get_models_by_category(ModelCategory.EMBEDDING)
    llm_models = registry.get_models_by_category(ModelCategory.LLM)
    lvm_models = registry.get_models_by_category(ModelCategory.LVM)
    assert {m.model_id for m in emb_models} == {
        "bge-m3",
        "paraphrase-multilingual-MiniLM-L12-v2",
    }
    assert {m.model_id for m in llm_models} == {
        "qwen2.5:3b-instruct-q4_K_M",
        "llama3.2:3b-instruct-q4_K_M",
    }
    assert {m.model_id for m in lvm_models} == {"qwen2.5-vl:3b", "moondream2"}

    assert (
        registry.get_primary_model_for_task(ModelTaskType.SEMANTIC_INDEXING).model_id
        == "bge-m3"
    )
    assert (
        registry.get_primary_model_for_task(ModelTaskType.LLM_AS_JUDGE_EVALUATION).model_id
        == "qwen2.5:3b-instruct-q4_K_M"
    )
    assert (
        registry.get_primary_model_for_task(ModelTaskType.DOCUMENT_LAYOUT_ANALYSIS).model_id
        == "qwen2.5-vl:3b"
    )


def test_03_model_registry_task_compatibility_validation_errors() -> None:
    """آزمون ۳: بررسی خطای `IncompatibleModelTaskError` و `RegistryNotFoundError` در `ModelRegistry`."""
    registry = ModelRegistry(initialize_defaults=True)

    # مدل bge-m3 از دسته EMBEDDING است و مجاز به اجرای QA_ANSWERING نیست
    with pytest.raises(IncompatibleModelTaskError) as exc_info:
        registry.validate_model_for_task("bge-m3", ModelTaskType.QA_ANSWERING)
    assert exc_info.value.model_id == "bge-m3"

    # مدل llama3.2 به عنوان داور اصلی تنظیم نشده است
    with pytest.raises(IncompatibleModelTaskError):
        registry.validate_model_for_task(
            "llama3.2:3b-instruct-q4_K_M", ModelTaskType.LLM_AS_JUDGE_EVALUATION
        )

    with pytest.raises(RegistryNotFoundError):
        registry.get_model("non-existent-model")


def test_04_prompt_registry_covers_all_12_proposal_categories() -> None:
    """آزمون ۴: بررسی پوشش کامل هر ۱۲ دسته پرامپت صفحه ۱۸ پروپوزال در `PromptRegistry`."""
    prompt_registry = PromptRegistry(initialize_defaults=True)
    all_categories = list(PromptTaskCategory)
    assert len(all_categories) == 12
    assert len(prompt_registry.active_templates) == 12

    for category in all_categories:
        tmpl = prompt_registry.get_active_template(category)
        assert tmpl.task_category == category
        assert tmpl.is_active is True
        assert tmpl.version_tag == VersionTag("v1.0.0", 1)
        assert "[DOC-ID]" in tmpl.citation_rule
        assert "INSUFFICIENT_KNOWLEDGE" in tmpl.uncertainty_rule
        assert tmpl.judge_threshold.minimum_score == 18


def test_05_prompt_template_versioning_preserves_history() -> None:
    """آزمون ۵: بررسی نسخه‌بندی قالب پرامپت در `PromptRegistry` و حفظ نسخه‌های قبلی بدون حذف."""
    prompt_registry = PromptRegistry(initialize_defaults=True)
    v1_tmpl = prompt_registry.get_active_template(PromptTaskCategory.QA_GROUNDED)
    assert v1_tmpl.version_tag.tag == "v1.0.0"

    v2_tmpl = PromptTemplate(
        template_id=v1_tmpl.template_id,
        version_tag=v1_tmpl.version_tag.next_version("v1.1.0"),
        task_category=PromptTaskCategory.QA_GROUNDED,
        system_instruction="نسخه ارتقایافته پاسخ‌گویی مقید به منبع با کنترل سخت‌گیرانه‌تر ارجاعات.",
        user_prompt_template=v1_tmpl.user_prompt_template,
        required_variables=v1_tmpl.required_variables,
        expected_json_keys=v1_tmpl.expected_json_keys,
    )
    prompt_registry.register_template(v2_tmpl)

    active = prompt_registry.get_active_template(PromptTaskCategory.QA_GROUNDED)
    assert active.version_tag.tag == "v1.1.0"
    assert active.version_tag.sequence_number == 2

    versions = prompt_registry.get_template_versions(v1_tmpl.template_id)
    assert len(versions) == 2
    assert versions[0].version_tag.tag == "v1.0.0"
    assert versions[0].is_active is False
    assert versions[1].version_tag.tag == "v1.1.0"
    assert versions[1].is_active is True


def test_06_prompt_assembly_success_and_missing_variable_error() -> None:
    """آزمون ۶: بررسی مونتاژ پرامپت (`assemble_prompt`) و پرتاب `PromptAssemblyError` در صورت کمبود متغیر."""
    prompt_registry = PromptRegistry(initialize_defaults=True)
    qa_template = prompt_registry.get_active_template(PromptTaskCategory.QA_GROUNDED)

    assembled = qa_template.assemble_prompt(
        node_id="L3-RAG-CORE-TEAM",
        context="[KI-001] مستندات معماری هسته RAG آفلاین.",
        query="چگونه ارجاع منبع حفظ می‌شود؟",
    )
    assert "L3-RAG-CORE-TEAM" in assembled
    assert "[DOC-ID]" in assembled
    assert "INSUFFICIENT_KNOWLEDGE" in assembled
    assert '"answer"' in assembled

    # حذف متغیر الزامی query باید خطای PromptAssemblyError صادر کند
    with pytest.raises(PromptAssemblyError) as exc_info:
        qa_template.assemble_prompt(
            node_id="L3-RAG-CORE-TEAM",
            context="[KI-001] مستندات.",
        )
    assert "query" in exc_info.value.missing_vars

    # ارسال رشته خالی برای متغیر الزامی نیز باید خطا بدهد
    with pytest.raises(PromptAssemblyError):
        qa_template.assemble_prompt(
            node_id="L3-RAG-CORE-TEAM",
            context="   ",
            query="پرسش تست",
        )


def test_07_prompt_output_json_parser_and_schema_stability_validation() -> None:
    """آزمون ۷: بررسی `parse_and_validate_output` در برابر JSON خام، بلوک Markdown و خروجی نامعتبر."""
    prompt_registry = PromptRegistry(initialize_defaults=True)
    qa_template = prompt_registry.get_active_template(PromptTaskCategory.QA_GROUNDED)

    # ۱. خروجی معتبر محصور در بلوک ```json
    fenced_json = """```json
    {
      "answer": "پاسخ مستند به [KI-001]",
      "citations": ["[KI-001]"],
      "uncertainty_status": "SUFFICIENT_EVIDENCE",
      "related_issues": ["ISS-101"],
      "suggested_executors": ["EXEC-201"]
    }
    ```"""
    parsed = qa_template.parse_and_validate_output(fenced_json)
    assert parsed["uncertainty_status"] == "SUFFICIENT_EVIDENCE"
    assert parsed["citations"] == ["[KI-001]"]

    # ۲. خروجی JSON با نحو شکسته
    with pytest.raises(PromptOutputValidationError):
        qa_template.parse_and_validate_output("{broken json:")

    # ۳. خروجی JSON معتبر اما فاقد کلیدهای الزامی Schema
    incomplete_json = '{"answer": "پاسخ ناقص بدون کلید citations"}'
    with pytest.raises(PromptOutputValidationError) as exc_info:
        qa_template.parse_and_validate_output(incomplete_json)
    assert "citations" in exc_info.value.reason


def test_08_local_ollama_adapter_airgap_loopback_security_constraint() -> None:
    """آزمون ۸: بررسی قید امنیتی آفلاین در `LocalOllamaAdapter` (جلوگیری از اتصال به اینترنت)."""
    local_adapter = LocalOllamaAdapter(base_url="http://localhost:11434")
    assert local_adapter.base_url == "http://localhost:11434"

    ip_adapter = LocalOllamaAdapter(base_url="http://127.0.0.1:11434/")
    assert ip_adapter.base_url == "http://127.0.0.1:11434"

    # تلاش برای استفاده از آدرس ابری یا غیراز localhost باید خطای OfflineAdapterError بدهد
    for remote_url in [
        "https://api.openai.com/v1",
        "http://192.168.1.50:11434",
        "https://my-cloud-llm.example.com",
    ]:
        with pytest.raises(OfflineAdapterError):
            LocalOllamaAdapter(base_url=remote_url)


def test_09_deterministic_mock_adapter_embeddings_and_lvm_layout() -> None:
    """آزمون ۹: بررسی تولید بردارهای Embedding قطعی و تحلیل بصری LVM در `DeterministicMockModelAdapter`."""
    adapter = DeterministicMockModelAdapter()

    vec_a = adapter.generate_embedding("مدیریت دانش سلسله‌مراتبی آفلاین", model_id="bge-m3", dimensions=16)
    vec_b = adapter.generate_embedding("مدیریت   دانش   سلسله‌مراتبی آفلاین", model_id="bge-m3", dimensions=16)
    vec_c = adapter.generate_embedding("متن کاملاً متفاوت درباره سخت‌افزار شبکه", model_id="bge-m3", dimensions=16)

    assert len(vec_a) == 16
    assert vec_a == vec_b
    assert vec_a != vec_c
    # بررسی نرمال بودن بردار (طول اقلیدسی برابر ۱)
    l2_norm = sum(x * x for x in vec_a) ** 0.5
    assert pytest.approx(l2_norm, rel=1e-3) == 1.0

    # بررسی تحلیل چیدمان LVM
    layout_info = adapter.analyze_document_layout(
        page_reference="DOC-100::Page-5",
        raw_content_hint="جدول مقایسه مدل‌های آفلاین",
        model_id="qwen2.5-vl:3b",
    )
    assert layout_info["has_complex_table"] is True
    assert layout_info["page_reference"] == "DOC-100::Page-5"


def test_10_deterministic_mock_adapter_all_12_prompt_categories_end_to_end() -> None:
    """آزمون ۱۰: اجرای سرتاسری مونتاژ، تولید خروجی و اعتبارسنجی Schema برای هر ۱۲ دسته پرامپت."""
    prompt_registry = PromptRegistry(initialize_defaults=True)
    adapter = DeterministicMockModelAdapter(prompt_registry=prompt_registry)

    for category in PromptTaskCategory:
        template = prompt_registry.get_active_template(category)
        dummy_vars = {var_name: f"مقدار نمونه برای {var_name} [KI-001]" for var_name in template.required_variables}
        assembled = template.assemble_prompt(dummy_vars)
        raw_output = adapter.generate_structured_text(
            assembled_prompt=assembled,
            task_category=category,
        )
        parsed = template.parse_and_validate_output(raw_output)
        for req_key in template.expected_json_keys:
            assert req_key in parsed


def test_11_insufficient_knowledge_uncertainty_handling_in_qa() -> None:
    """آزمون ۱۱: بررسی اعلام صریح عدم قطعیت (`INSUFFICIENT_KNOWLEDGE`) در صورت کمبود دانش در گره."""
    prompt_registry = PromptRegistry(initialize_defaults=True)
    adapter = DeterministicMockModelAdapter(prompt_registry=prompt_registry)
    qa_template = prompt_registry.get_active_template(PromptTaskCategory.QA_GROUNDED)

    assembled = qa_template.assemble_prompt(
        node_id="L3-RAG-CORE-TEAM",
        context="INSUFFICIENT_CONTEXT: هیچ سندی درباره بودجه سال ۱۴۱۰ در این گره وجود ندارد.",
        query="بودجه مصوب سال ۱۴۱۰ چقدر است؟",
    )
    raw_output = adapter.generate_structured_text(
        assembled_prompt=assembled,
        task_category=PromptTaskCategory.QA_GROUNDED,
    )
    parsed = qa_template.parse_and_validate_output(raw_output)
    assert parsed["uncertainty_status"] == "INSUFFICIENT_KNOWLEDGE"
    assert "INSUFFICIENT_KNOWLEDGE" in parsed["answer"]


def test_12_llm_as_judge_workflow_passing_and_failing_threshold_18_of_20() -> None:
    """آزمون ۱۲: بررسی کامل گردش کار داوری `LLM-as-a-Judge` در دو حالت قبولی (>=۱۸) و رد شدن (<۱۸)."""
    adapter = DeterministicMockModelAdapter()
    threshold = JudgeThreshold(minimum_score=18)

    # حالت اول: خروجی باکیفیت و مستند به منبع -> کسب امتیاز ۲۰ از ۲۰ و قبولی
    approved_score = adapter.evaluate_output_with_judge(
        score_id="JDG-PASS-01",
        target_entity_id="TPKG-01",
        source_knowledge="[KI-001] مستندات رسمی گره L3.",
        task_instruction="خلاصه‌سازی فنی با ارجاع منبع",
        generated_output="خلاصه فنی مستند به [KI-001] با رعایت کامل ساختار.",
        threshold=threshold,
    )
    assert approved_score.total_score >= 18
    assert approved_score.is_approved is True
    approved_score.enforce_threshold()

    # حالت دوم: خروجی ضعیف (حاوی نشانه LOW_QUALITY_OUTPUT) -> کسب امتیاز ۱۴ از ۲۰ و پرتاب خطا
    rejected_score = adapter.evaluate_output_with_judge(
        score_id="JDG-FAIL-01",
        target_entity_id="TPKG-02",
        source_knowledge="[KI-001] مستندات رسمی گره L3.",
        task_instruction="خلاصه‌سازی فنی با ارجاع منبع",
        generated_output="LOW_QUALITY_OUTPUT: ادعای بدون ارجاع و ناقص.",
        threshold=threshold,
    )
    assert rejected_score.total_score == 14
    assert rejected_score.is_approved is False
    with pytest.raises(JudgeThresholdNotMetError) as exc_info:
        rejected_score.enforce_threshold()
    assert exc_info.value.achieved_score == 14
    assert exc_info.value.required_threshold == 18


def test_13_phase2_experiment_results_artifact_verification() -> None:
    """آزمون ۱۳: بررسی صحت فایل نتایج آزمایش خط پایه فاز ۲ (`sample_docs/phase2_experiment_results.json`) در صورت وجود."""
    results_file = (
        Path(__file__).resolve().parent.parent
        / "sample_docs"
        / "phase2_experiment_results.json"
    )
    if results_file.exists():
        import json

        data = json.loads(results_file.read_text(encoding="utf-8"))
        assert data["model_registry_summary"]["total_registered_models"] == 6
        assert data["prompt_testing_summary"]["total_prompt_categories_tested"] == 12
        assert data["prompt_testing_summary"]["all_schemas_valid"] is True
        assert data["prompt_testing_summary"]["all_judge_thresholds_met"] is True
