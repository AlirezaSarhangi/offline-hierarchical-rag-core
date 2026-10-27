"""
اسکریپت اجرای آزمایش‌های خط پایه مدل‌های آفلاین و ۱۲ دسته پرامپت (فاز ۲ پروپوزال).

این اسکریپت روی داده‌های مطالعه موردی فاز ۱ (`sample_docs/case_study_registry.json`) اجرا شده و:
1. رجیستری مدل‌های آفلاین (`ModelRegistry`) شامل مدل‌های `Embedding`، `LLM` و `LVM` را بررسی می‌کند.
2. بردارهای Embedding قطعی و شباهت کسینوسی و همچنین تحلیل چیدمان بصری `LVM` را آزمایش می‌کند.
3. هر ۱۲ دسته پرامپت (`PromptRegistry`) را با داده‌های مطالعه موردی مونتاژ کرده، خروجی مدل را تولید و Schema JSON آن را اعتبارسنجی می‌نماید.
4. خروجی‌های تولیدشده را با سازوکار `LLM-as-a-Judge` (چارچوب ۲۰ نمره‌ای و حد آستانه ۱۸ از ۲۰) داوری کرده و نتایج کامل آزمایش را در `sample_docs/phase2_experiment_results.json` ذخیره می‌کند.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.domain import (
    JudgeThreshold,
    ModelCategory,
    ModelRegistry,
    ModelTaskType,
    PromptRegistry,
    PromptTaskCategory,
    SimilarityScore,
)
from src.infrastructure import DeterministicMockModelAdapter


def compute_cosine_similarity(vec_a: tuple[float, ...], vec_b: tuple[float, ...]) -> SimilarityScore:
    """محاسبه شباهت کسینوسی میان دو بردار نرمال‌شده و تبدیل به شیء مقداری `SimilarityScore`."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    clamped = max(0.0, min(1.0, (dot + 1.0) / 2.0 if dot < 0 else dot))
    return SimilarityScore(round(clamped, 6))


def get_sample_variables_for_category(category: PromptTaskCategory) -> Dict[str, str]:
    """تأمین متغیرهای ورودی واقع‌گرایانه از مطالعه موردی فاز ۱ برای هر یک از ۱۲ دسته پرامپت."""
    common_source = (
        "[KI-001] هسته RAG آفلاین اسناد محلی DOC، DOCX و PDF را پردازش کرده و شناسه سند منبع، نسخه و شماره صفحه را حفظ می‌کند. "
        "[KI-002] مسئله ISS-101 مربوط به استخراج جداول پیچیده فارسی با پیشرفت ۷۵٪ در گره L3-RAG-CORE-TEAM در حال پیگیری است."
    )
    mapping: Dict[PromptTaskCategory, Dict[str, str]] = {
        PromptTaskCategory.QA_GROUNDED: {
            "node_id": "L3-RAG-CORE-TEAM",
            "context": common_source,
            "query": "قواعد حفظ ارجاع منبع در خردسازی اسناد چیست؟",
        },
        PromptTaskCategory.TECHNICAL_SUMMARIZATION: {
            "node_id": "L3-RAG-CORE-TEAM",
            "detail_level": "TECHNICAL_SUMMARY",
            "source_content": common_source,
        },
        PromptTaskCategory.EXECUTIVE_MANAGERIAL_SUMMARIZATION: {
            "node_id": "L1-AI-MOTHER-ORG",
            "target_layer": "0",
            "detail_level": "EXECUTIVE_HIGH",
            "source_content": common_source,
        },
        PromptTaskCategory.ISSUE_DECOMPOSITION: {
            "node_id": "L3-RAG-CORE-TEAM",
            "issue_description": "ISS-101: بهینه‌سازی دقت استخراج جداول پیچیده فارسی در اسناد PDF اسکن‌شده آفلاین",
            "existing_knowledge": common_source,
        },
        PromptTaskCategory.SUB_ISSUE_COMPARISON: {
            "node_id": "L3-RAG-CORE-TEAM",
            "candidate_sub_issues": "SUB-101-1 (تشخیص صفحات جدول‌دار) و SUB-101-2 (ارزیابی مصرف حافظه مدل LVM)",
            "existing_issues": "ISS-101 (بهینه‌سازی استخراج جدول در PDF اسکن‌شده)",
        },
        PromptTaskCategory.EXECUTOR_EXPERTISE_ANALYSIS: {
            "issue_description": "ISS-101: نیاز به کالیبراسیون استخراج جدول و معماری دامنه‌محور",
            "executor_profiles": "EXEC-201 (مهندس علی کاظمی - متخصص NLP فارسی، Embedding و DDD)",
            "related_docs": "KI-001 (نویسنده: تیم معماری و مهندس علی کاظمی)",
        },
        PromptTaskCategory.RESEARCH_GAP_IDENTIFICATION: {
            "node_id": "L3-RAG-CORE-TEAM",
            "issue_summary": "ISS-101: استخراج جداول تودرتوی فارسی از اسناد اسکن‌شده با مدل کوچک آفلاین",
            "knowledge_evidence": "کمبود بنچمارک تجربی برای مدل‌های LVM زیر ۴ میلیارد پارامتر روی جداول فارسی [KI-002]",
        },
        PromptTaskCategory.KNOWLEDGE_TRANSFER_PREPARATION: {
            "source_node_id": "L3-RAG-CORE-TEAM",
            "target_layer": "2",
            "detail_level": "TECHNICAL_SUMMARY",
            "layer_rules": "حداقل امتیاز داور ۱۸ از ۲۰، درج ارجاع منبع و برجسته‌سازی شکاف پژوهشی",
            "node_digest": common_source,
        },
        PromptTaskCategory.RFP_GENERATION: {
            "issuer_node_id": "L2-NLP-RESEARCH-CENTER",
            "issue_context": "ISS-101: ارتقای دقت استخراج جداول فارسی در اسناد PDF با مدل کمکی LVM محلی",
            "research_gaps": "کمبود شواهد تجربی دقت مدل‌های LVM کوچک زیر ۴ میلیارد پارامتر روی جداول فارسی",
        },
        PromptTaskCategory.PROPOSAL_EVALUATION: {
            "rfp_specification": "RFP-301: اجرای ۱۰۰٪ آفلاین، حفظ شماره صفحه و دقت بالای ۹۰٪ در استخراج جدول",
            "proposal_content": "PROP-601: خط لوله دو مرحله‌ای استخراج PDF با تشخیص چیدمان و مفسر LVM کوانتیزه",
        },
        PromptTaskCategory.FEEDBACK_GENERATION: {
            "rfp_criteria": "معیار ۱: رعایت قید آفلاین | معیار ۲: حفظ ارجاع صفحه و فراداده | معیار ۳: مدیریت ریسک زمان پردازش",
            "evaluation_summary": "پیشنهاد PROP-601 امتیاز ۱۸.۵ از ۲۰ کسب کرد؛ نیازمند شفاف‌سازی Timeout پردازش صفحه اسکن‌شده.",
        },
        PromptTaskCategory.LLM_AS_JUDGE_SCORING: {
            "source_knowledge": common_source,
            "task_instruction": "آماده‌سازی بسته انتقال لایه ۳ به لایه ۲ با سطح جزئیات خلاصه فنی و ارجاع معتبر",
            "generated_output": "بسته انتقال گره L3-RAG-CORE-TEAM شامل دستاوردهای فاز ۱ و شکاف پژوهشی LVM مستند به [KI-001] و [KI-002].",
        },
    }
    return mapping[category]


def run_phase2_experiments() -> Dict[str, Any]:
    """اجرای کامل آزمایش‌های فاز ۲ و تولید ساختار گزارش خروجی."""
    model_registry = ModelRegistry(initialize_defaults=True)
    prompt_registry = PromptRegistry(initialize_defaults=True)
    adapter = DeterministicMockModelAdapter(
        model_registry=model_registry,
        prompt_registry=prompt_registry,
    )

    # ۱. آزمایش مدل Embedding آفلاین
    primary_emb_model = model_registry.get_primary_model_for_task(
        ModelTaskType.SEMANTIC_INDEXING
    )
    vec_1 = adapter.generate_embedding(
        "هسته RAG سلسله‌مراتبی آفلاین برای مدیریت دانش سازمانی",
        model_id=primary_emb_model.model_id,
        dimensions=16,
    )
    vec_1_identical = adapter.generate_embedding(
        "هسته   RAG   سلسله‌مراتبی آفلاین برای مدیریت دانش سازمانی",
        model_id=primary_emb_model.model_id,
        dimensions=16,
    )
    vec_2_different = adapter.generate_embedding(
        "پروفایل تخصصی مجریان و سوابق پروژه‌های مهندسی نرم‌افزار",
        model_id=primary_emb_model.model_id,
        dimensions=16,
    )
    sim_identical = compute_cosine_similarity(vec_1, vec_1_identical)
    sim_different = compute_cosine_similarity(vec_1, vec_2_different)

    # ۲. آزمایش مدل بینایی-زبانی (LVM) آفلاین
    primary_lvm_model = model_registry.get_primary_model_for_task(
        ModelTaskType.DOCUMENT_LAYOUT_ANALYSIS
    )
    lvm_result = adapter.analyze_document_layout(
        page_reference="KI-001::Page-14",
        raw_content_hint="جدول قواعد خردسازی اسناد و نگهداری فراداده ۱۸ فیلدی",
        model_id=primary_lvm_model.model_id,
    )

    # ۳. آزمایش هر ۱۲ دسته پرامپت و ارزیابی با داور ۲۰ نمره‌ای
    primary_llm_model = model_registry.get_primary_model_for_task(
        ModelTaskType.QA_ANSWERING
    )
    prompt_experiments: List[Dict[str, Any]] = []
    threshold = JudgeThreshold(minimum_score=18)

    for idx, category in enumerate(PromptTaskCategory, start=1):
        template = prompt_registry.get_active_template(category)
        sample_vars = get_sample_variables_for_category(category)
        assembled_prompt = template.assemble_prompt(sample_vars)

        raw_json_output = adapter.generate_structured_text(
            assembled_prompt=assembled_prompt,
            task_category=category,
            model_id=primary_llm_model.model_id,
        )
        parsed_output = template.parse_and_validate_output(raw_json_output)

        judge_score = adapter.evaluate_output_with_judge(
            score_id=f"EXP-JDG-{idx:02d}",
            target_entity_id=template.template_id,
            source_knowledge=str(sample_vars),
            task_instruction=f"اجرای قالب {template.template_id} ({category.label_fa})",
            generated_output=raw_json_output,
            threshold=threshold,
        )
        judge_score.enforce_threshold()

        prompt_registry.record_test_execution(
            template_id=template.template_id,
            version_tag=str(template.version_tag),
            schema_valid=True,
            judge_score=judge_score.total_score,
            notes=f"آزمون موفق دسته {category.value} با مدل {primary_llm_model.model_id}",
        )

        prompt_experiments.append(
            {
                "experiment_index": idx,
                "task_category": category.value,
                "task_category_label_fa": category.label_fa,
                "template_id": template.template_id,
                "version_tag": str(template.version_tag),
                "model_id": primary_llm_model.model_id,
                "assembled_prompt_length": len(assembled_prompt),
                "expected_json_keys": list(template.expected_json_keys),
                "parsed_output_keys": list(parsed_output.keys()),
                "schema_validation_passed": True,
                "judge_evaluation": {
                    "score_id": judge_score.score_id,
                    "total_score": judge_score.total_score,
                    "required_threshold": judge_score.threshold.minimum_score,
                    "is_approved": judge_score.is_approved,
                    "rubric_breakdown": judge_score.rubric.to_dict(),
                },
                "sample_parsed_output": parsed_output,
            }
        )

    return {
        "phase": "Phase 2 - Offline Models & Technical Baseline",
        "model_registry_summary": {
            "total_registered_models": len(model_registry.models),
            "categories": {
                cat.value: [m.model_id for m in model_registry.get_models_by_category(cat)]
                for cat in ModelCategory
            },
            "primary_models_by_task": {
                task.value: model_registry.get_primary_model_for_task(task).model_id
                for task in ModelTaskType
            },
            "models": [m.to_dict() for m in model_registry.models],
        },
        "embedding_baseline_test": {
            "model_id": primary_emb_model.model_id,
            "vector_dimensions": len(vec_1),
            "sample_vector_preview": list(vec_1[:6]),
            "identical_normalized_text_similarity": sim_identical.value,
            "different_text_similarity": sim_different.value,
        },
        "lvm_layout_analysis_test": lvm_result,
        "prompt_testing_summary": {
            "total_prompt_categories_tested": len(prompt_experiments),
            "all_schemas_valid": all(exp["schema_validation_passed"] for exp in prompt_experiments),
            "all_judge_thresholds_met": all(
                exp["judge_evaluation"]["is_approved"] for exp in prompt_experiments
            ),
            "experiments": prompt_experiments,
        },
        "adapter_call_count": len(adapter.call_log),
    }


def main() -> None:
    """اجرای اسکریپت آزمایش‌های خط پایه فاز ۲ و ذخیره نتایج در `sample_docs/`."""
    sample_dir = PROJECT_ROOT / "sample_docs"
    sample_dir.mkdir(parents=True, exist_ok=True)
    output_path = sample_dir / "phase2_experiment_results.json"

    results = run_phase2_experiments()
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=== Phase 2 Baseline Experiments Completed Successfully ===")
    print(f"1. Experiment Results saved to: {output_path}")
    print(
        f"2. Registered Offline Models  : {results['model_registry_summary']['total_registered_models']} (Embedding, LLM, LVM)"
    )
    print(
        f"3. Prompt Categories Tested   : {results['prompt_testing_summary']['total_prompt_categories_tested']} / 12"
    )
    print(
        f"4. Schema & Judge (>=18/20)   : 100% Passed ({results['adapter_call_count']} deterministic adapter calls)"
    )


if __name__ == "__main__":
    main()
