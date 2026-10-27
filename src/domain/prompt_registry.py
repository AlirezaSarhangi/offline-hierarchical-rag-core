"""
رجیستری و نسخه‌بندی قالب‌های پرامپت (`PromptRegistry`) طبق بخش ۸.۳، صفحه ۱۸ و جدول ۱۹.۲ پروپوزال.

این ماژول ۱۲ دسته اصلی پرامپت‌های سامانه را همراه با نسخه‌بندی (`VersionTag`)، قواعد مونتاژ قابل آزمون،
الزام ارجاع به منبع (`[DOC-ID]`)، قانون اعلام صریح عدم قطعیت (`INSUFFICIENT_KNOWLEDGE`)،
اتصال به چارچوب داوری ۲۰ نمره‌ای (`JudgeRubric`) و اعتبارسنجی Schema خروجی JSON پیاده‌سازی می‌کند.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

from src.domain.exceptions import (
    DomainValidationError,
    PromptAssemblyError,
    PromptOutputValidationError,
    RegistryNotFoundError,
)
from src.domain.value_objects import JudgeRubric, JudgeThreshold, VersionTag

_CODE_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_PLACEHOLDER_PATTERN = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


class PromptTaskCategory(str, Enum):
    """
    ۱۲ دسته پرامپت رسمی پروژه طبق بخش ۸.۳ (صفحه ۱۸ پروپوزال):
    1. پرسش و پاسخ (`QA_GROUNDED`)
    2. خلاصه‌سازی فنی (`TECHNICAL_SUMMARIZATION`)
    3. خلاصه‌سازی مدیریتی و اجرایی (`EXECUTIVE_MANAGERIAL_SUMMARIZATION`)
    4. تجزیه مسئله (`ISSUE_DECOMPOSITION`)
    5. مقایسه زیرمسئله‌ها (`SUB_ISSUE_COMPARISON`)
    6. تحلیل تخصص مجریان (`EXECUTOR_EXPERTISE_ANALYSIS`)
    7. شناسایی شکاف پژوهشی (`RESEARCH_GAP_IDENTIFICATION`)
    8. آماده‌سازی انتقال دانش (`KNOWLEDGE_TRANSFER_PREPARATION`)
    9. تولید RFP (`RFP_GENERATION`)
    10. ارزیابی پیشنهاد (`PROPOSAL_EVALUATION`)
    11. تولید بازخورد (`FEEDBACK_GENERATION`)
    12. امتیازدهی LLM به‌عنوان داور (`LLM_AS_JUDGE_SCORING`)
    """

    QA_GROUNDED = "QA_GROUNDED"
    TECHNICAL_SUMMARIZATION = "TECHNICAL_SUMMARIZATION"
    EXECUTIVE_MANAGERIAL_SUMMARIZATION = "EXECUTIVE_MANAGERIAL_SUMMARIZATION"
    ISSUE_DECOMPOSITION = "ISSUE_DECOMPOSITION"
    SUB_ISSUE_COMPARISON = "SUB_ISSUE_COMPARISON"
    EXECUTOR_EXPERTISE_ANALYSIS = "EXECUTOR_EXPERTISE_ANALYSIS"
    RESEARCH_GAP_IDENTIFICATION = "RESEARCH_GAP_IDENTIFICATION"
    KNOWLEDGE_TRANSFER_PREPARATION = "KNOWLEDGE_TRANSFER_PREPARATION"
    RFP_GENERATION = "RFP_GENERATION"
    PROPOSAL_EVALUATION = "PROPOSAL_EVALUATION"
    FEEDBACK_GENERATION = "FEEDBACK_GENERATION"
    LLM_AS_JUDGE_SCORING = "LLM_AS_JUDGE_SCORING"

    @property
    def label_fa(self) -> str:
        """عنوان رسمی فارسی دسته پرامپت طبق صفحه ۱۸ پروپوزال."""
        labels = {
            PromptTaskCategory.QA_GROUNDED: "پرسش و پاسخ مبتنی بر منبع",
            PromptTaskCategory.TECHNICAL_SUMMARIZATION: "خلاصه‌سازی فنی",
            PromptTaskCategory.EXECUTIVE_MANAGERIAL_SUMMARIZATION: "خلاصه‌سازی مدیریتی و اجرایی",
            PromptTaskCategory.ISSUE_DECOMPOSITION: "تجزیه مسئله",
            PromptTaskCategory.SUB_ISSUE_COMPARISON: "مقایسه زیرمسئله‌ها",
            PromptTaskCategory.EXECUTOR_EXPERTISE_ANALYSIS: "تحلیل تخصص مجریان",
            PromptTaskCategory.RESEARCH_GAP_IDENTIFICATION: "شناسایی شکاف پژوهشی",
            PromptTaskCategory.KNOWLEDGE_TRANSFER_PREPARATION: "آماده‌سازی انتقال دانش",
            PromptTaskCategory.RFP_GENERATION: "تولید RFP",
            PromptTaskCategory.PROPOSAL_EVALUATION: "ارزیابی پیشنهاد",
            PromptTaskCategory.FEEDBACK_GENERATION: "تولید بازخورد",
            PromptTaskCategory.LLM_AS_JUDGE_SCORING: "امتیازدهی LLM به‌عنوان داور",
        }
        return labels[self]


@dataclass(frozen=True)
class PromptTemplate:
    """
    قالب نسخه‌بندی‌شده پرامپت همراه با قواعد مونتاژ، ارجاع به منبع، اعلام عدم قطعیت و Parser خروجی.
    """

    template_id: str
    version_tag: VersionTag
    task_category: PromptTaskCategory
    system_instruction: str
    user_prompt_template: str
    required_variables: Tuple[str, ...]
    expected_json_keys: Tuple[str, ...]
    citation_rule: str = (
        "الزام ارجاع به منبع: تمامی ادعاها و یافته‌ها باید با فرمت [DOC-ID] به شناسه سند منبع ارجاع دهند."
    )
    uncertainty_rule: str = (
        "قانون عدم قطعیت: در صورت کافی نبودن دانش منبع، مقدار INSUFFICIENT_KNOWLEDGE را صریحاً اعلام کن و از گمانه‌زنی خودداری نما."
    )
    judge_threshold: JudgeThreshold = field(default_factory=JudgeThreshold)
    rubric_criteria_names: Tuple[str, ...] = (
        "source_fidelity",
        "completeness",
        "detail_level_fit",
        "uncertainty_and_gap_handling",
        "clarity_and_structure",
        "source_attribution",
        "no_unsupported_claims",
    )
    is_active: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.template_id, str) or not self.template_id.strip():
            raise DomainValidationError(
                "شناسه قالب پرامپت (template_id) نمی‌تواند خالی باشد.",
                field_name="template_id",
            )
        object.__setattr__(self, "template_id", self.template_id.strip())

        if isinstance(self.version_tag, str):
            object.__setattr__(self, "version_tag", VersionTag(self.version_tag, 1))
        elif not isinstance(self.version_tag, VersionTag):
            raise DomainValidationError(
                "فیلد version_tag باید از نوع VersionTag باشد.",
                field_name="version_tag",
            )

        if isinstance(self.task_category, str) and not isinstance(
            self.task_category, PromptTaskCategory
        ):
            object.__setattr__(
                self, "task_category", PromptTaskCategory(self.task_category)
            )
        elif not isinstance(self.task_category, PromptTaskCategory):
            raise DomainValidationError(
                "فیلد task_category باید از نوع PromptTaskCategory باشد.",
                field_name="task_category",
            )

        if (
            not isinstance(self.system_instruction, str)
            or not self.system_instruction.strip()
        ):
            raise DomainValidationError(
                "دستورالعمل سیستمی (system_instruction) نمی‌تواند خالی باشد.",
                field_name="system_instruction",
            )
        if (
            not isinstance(self.user_prompt_template, str)
            or not self.user_prompt_template.strip()
        ):
            raise DomainValidationError(
                "قالب پرامپت کاربر (user_prompt_template) نمی‌تواند خالی باشد.",
                field_name="user_prompt_template",
            )

        if isinstance(self.required_variables, str) or not isinstance(
            self.required_variables, Iterable
        ):
            raise DomainValidationError(
                "متغیرهای الزامی (required_variables) باید تاپل یا لیست از رشته‌ها باشد.",
                field_name="required_variables",
            )
        cleaned_vars = tuple(
            str(v).strip() for v in self.required_variables if str(v).strip()
        )
        if not cleaned_vars:
            raise DomainValidationError(
                "هر قالب پرامپت باید حداقل یک متغیر ورودی الزامی داشته باشد.",
                field_name="required_variables",
            )
        object.__setattr__(self, "required_variables", cleaned_vars)

        # بررسی حضور متغیرهای الزامی در متن قالب
        placeholders_in_template = set(
            _PLACEHOLDER_PATTERN.findall(self.user_prompt_template)
        )
        for req_var in cleaned_vars:
            if req_var not in placeholders_in_template:
                raise DomainValidationError(
                    f"متغیر الزامی '{{{req_var}}}' در متن قالب پرامپت '{self.template_id}' یافت نشد.",
                    field_name="user_prompt_template",
                )

        if isinstance(self.expected_json_keys, str) or not isinstance(
            self.expected_json_keys, Iterable
        ):
            raise DomainValidationError(
                "کلیدهای مورد انتظار خروجی (expected_json_keys) باید مجموعه‌ای از رشته‌ها باشد.",
                field_name="expected_json_keys",
            )
        cleaned_keys = tuple(
            str(k).strip() for k in self.expected_json_keys if str(k).strip()
        )
        if not cleaned_keys:
            raise DomainValidationError(
                "هر قالب پرامپت باید کلیدهای الزامی Schema خروجی (expected_json_keys) را مشخص کند.",
                field_name="expected_json_keys",
            )
        object.__setattr__(self, "expected_json_keys", cleaned_keys)

        if "[DOC-ID]" not in self.citation_rule:
            raise DomainValidationError(
                "قانون ارجاع به منبع (citation_rule) باید شامل الگوی '[DOC-ID]' باشد.",
                field_name="citation_rule",
            )
        if "INSUFFICIENT_KNOWLEDGE" not in self.uncertainty_rule:
            raise DomainValidationError(
                "قانون عدم قطعیت (uncertainty_rule) باید شامل عبارت 'INSUFFICIENT_KNOWLEDGE' باشد.",
                field_name="uncertainty_rule",
            )

    def assemble_prompt(
        self,
        variables: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> str:
        """
        مونتاژ نهایی پرامپت با اعتبارسنجی حضور و غیرخالی بودن تمامی متغیرهای الزامی.
        در صورت کمبود هر متغیر، خطای `PromptAssemblyError` پرتاب می‌شود.
        """
        merged_vars: Dict[str, Any] = {}
        if variables is not None:
            if not isinstance(variables, Mapping):
                raise DomainValidationError(
                    "ورودی variables در assemble_prompt باید یک دیکشنری باشد.",
                    field_name="variables",
                )
            merged_vars.update(variables)
        merged_vars.update(kwargs)

        missing: List[str] = []
        for req_var in self.required_variables:
            if req_var not in merged_vars or merged_vars[req_var] is None:
                missing.append(req_var)
            elif isinstance(merged_vars[req_var], str) and not merged_vars[req_var].strip():
                missing.append(req_var)

        if missing:
            raise PromptAssemblyError(
                template_id=self.template_id, missing_vars=tuple(missing)
            )

        rendered_user = self.user_prompt_template.format(**merged_vars)
        expected_schema_str = ", ".join(f'"{k}"' for k in self.expected_json_keys)

        assembled = (
            f"### [SYSTEM INSTRUCTION | {self.template_id} ({self.version_tag})]\n"
            f"{self.system_instruction}\n\n"
            f"### [MANDATORY GOVERNANCE RULES]\n"
            f"- {self.citation_rule}\n"
            f"- {self.uncertainty_rule}\n"
            f"- حداقل امتیاز داوری مورد انتظار (JudgeThreshold): {self.judge_threshold.minimum_score} از {self.judge_threshold.max_score}\n"
            f"- خروجی باید صرفاً یک شیء JSON معتبر شامل کلیدهای الزامی زیر باشد: [{expected_schema_str}]\n\n"
            f"### [TASK INPUT]\n"
            f"{rendered_user}"
        )
        return assembled

    def parse_and_validate_output(self, raw_output: str) -> Dict[str, Any]:
        """
        پارس کردن رشته JSON خروجی مدل و اعتبارسنجی حضور تمام کلیدهای الزامی (`expected_json_keys`).
        در صورت نامعتبر بودن JSON یا نبود کلید الزامی، خطای `PromptOutputValidationError` صادر می‌شود.
        """
        if not isinstance(raw_output, str) or not raw_output.strip():
            raise PromptOutputValidationError(
                template_id=self.template_id,
                reason="خروجی مدل خالی است.",
            )

        cleaned = raw_output.strip()
        fence_match = _CODE_FENCE_PATTERN.search(cleaned)
        if fence_match:
            cleaned = fence_match.group(1).strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise PromptOutputValidationError(
                template_id=self.template_id,
                reason=f"ساختار JSON معتبر نیست: {exc}",
            ) from exc

        if not isinstance(parsed, dict):
            raise PromptOutputValidationError(
                template_id=self.template_id,
                reason="ریشه خروجی JSON باید یک شیء (Object/Dict) باشد.",
            )

        missing_keys = [k for k in self.expected_json_keys if k not in parsed]
        if missing_keys:
            raise PromptOutputValidationError(
                template_id=self.template_id,
                reason=f"کلیدهای الزامی Schema در خروجی JSON وجود ندارند: {missing_keys}",
            )

        return parsed

    def to_dict(self) -> Dict[str, Any]:
        """تبدیل قالب پرامپت به دیکشنری استاندارد."""
        return {
            "template_id": self.template_id,
            "version_tag": str(self.version_tag),
            "sequence_number": self.version_tag.sequence_number,
            "task_category": self.task_category.value,
            "task_category_label_fa": self.task_category.label_fa,
            "required_variables": list(self.required_variables),
            "expected_json_keys": list(self.expected_json_keys),
            "judge_threshold": self.judge_threshold.minimum_score,
            "is_active": self.is_active,
        }


DEFAULT_PROMPT_TEMPLATES: Tuple[PromptTemplate, ...] = (
    # ۱. پرسش و پاسخ مبتنی بر منبع
    PromptTemplate(
        template_id="PRM-01-QA-GROUNDED",
        version_tag=VersionTag("v1.0.0", 1),
        task_category=PromptTaskCategory.QA_GROUNDED,
        system_instruction=(
            "تو دستیار تخصصی پاسخ‌گویی مبتنی بر دانش گره سازمانی هستی. صرفاً بر اساس قطعات دانش مجاز گره پاسخ بده."
        ),
        user_prompt_template=(
            "گره هدف: {node_id}\nقطعات دانش بازیابی‌شده:\n{context}\n\nپرسش کاربر:\n{query}"
        ),
        required_variables=("node_id", "context", "query"),
        expected_json_keys=(
            "answer",
            "citations",
            "uncertainty_status",
            "related_issues",
            "suggested_executors",
        ),
    ),
    # ۲. خلاصه‌سازی فنی
    PromptTemplate(
        template_id="PRM-02-TECH-SUMMARY",
        version_tag=VersionTag("v1.0.0", 1),
        task_category=PromptTaskCategory.TECHNICAL_SUMMARIZATION,
        system_instruction=(
            "تو متخصص خلاصه‌سازی فنی برای رهبران فنی و ناظران برنامه (لایه ۲ و ۳) هستی. روش‌ها، آزمایش‌ها و محدودیت‌ها را دقیق خلاصه کن."
        ),
        user_prompt_template=(
            "گره مبدأ: {node_id}\nسطح جزئیات: {detail_level}\nمحتوای فنی منبع:\n{source_content}"
        ),
        required_variables=("node_id", "detail_level", "source_content"),
        expected_json_keys=(
            "technical_summary",
            "methods_and_evidence",
            "technical_limitations",
            "citations",
            "uncertainty_status",
        ),
    ),
    # ۳. خلاصه‌سازی مدیریتی و اجرایی
    PromptTemplate(
        template_id="PRM-03-EXEC-MANAGERIAL-SUMMARY",
        version_tag=VersionTag("v1.0.0", 1),
        task_category=PromptTaskCategory.EXECUTIVE_MANAGERIAL_SUMMARIZATION,
        system_instruction=(
            "تو مشاور ارشد انتقال دانش به لایه مدیریتی (لایه ۱) و راهبردی (لایه ۰) هستی. جزئیات کدنویسی را حذف و دستاوردها، ریسک‌ها و تصمیمات لازم را برجسته کن."
        ),
        user_prompt_template=(
            "گره گزارش‌دهنده: {node_id}\nلایه هدف: {target_layer}\nسطح جزئیات: {detail_level}\nگزارش منبع:\n{source_content}"
        ),
        required_variables=("node_id", "target_layer", "detail_level", "source_content"),
        expected_json_keys=(
            "executive_summary",
            "key_achievements",
            "critical_risks",
            "decisions_required",
            "citations",
            "uncertainty_status",
        ),
    ),
    # ۴. تجزیه مسئله
    PromptTemplate(
        template_id="PRM-04-ISSUE-DECOMPOSITION",
        version_tag=VersionTag("v1.0.0", 1),
        task_category=PromptTaskCategory.ISSUE_DECOMPOSITION,
        system_instruction=(
            "تو معمار تحلیل مسئله هستی. مسئله اصلی گره را به زیرمسئله‌های منسجم، مستقل و قابل اقدام همراه با جایگاه نسبی تجزیه کن."
        ),
        user_prompt_template=(
            "گره مالک: {node_id}\nعنوان و شرح مسئله: {issue_description}\nدانش مرتبط موجود:\n{existing_knowledge}"
        ),
        required_variables=("node_id", "issue_description", "existing_knowledge"),
        expected_json_keys=(
            "parent_issue_title",
            "suggested_priority",
            "sub_issues",
            "citations",
            "uncertainty_status",
        ),
    ),
    # ۵. مقایسه زیرمسئله‌ها
    PromptTemplate(
        template_id="PRM-05-SUB-ISSUE-COMPARISON",
        version_tag=VersionTag("v1.0.0", 1),
        task_category=PromptTaskCategory.SUB_ISSUE_COMPARISON,
        system_instruction=(
            "تو تحلیل‌گر هم‌پوشانی و وابستگی مسائل هستی. زیرمسئله‌های جدید را با مسائل و زیرمسئله‌های موجود مقایسه و موارد تکراری یا مسدودکننده را مشخص کن."
        ),
        user_prompt_template=(
            "گره: {node_id}\nزیرمسئله‌های نامزد:\n{candidate_sub_issues}\nمسائل موجود در گره:\n{existing_issues}"
        ),
        required_variables=("node_id", "candidate_sub_issues", "existing_issues"),
        expected_json_keys=(
            "overlapping_pairs",
            "dependency_order",
            "merge_recommendations",
            "citations",
        ),
    ),
    # ۶. تحلیل تخصص مجریان
    PromptTemplate(
        template_id="PRM-06-EXECUTOR-ANALYSIS",
        version_tag=VersionTag("v1.0.0", 1),
        task_category=PromptTaskCategory.EXECUTOR_EXPERTISE_ANALYSIS,
        system_instruction=(
            "تو تحلیل‌گر توانمندی مجریان در مدل Issue-Knowledge-Executor هستی. بر اساس پروفایل‌ها و سوابق سند/پروژه، بهترین مجریان را پیشنهاد بده."
        ),
        user_prompt_template=(
            "مسئله هدف: {issue_description}\nپروفایل مجریان و سوابق:\n{executor_profiles}\nاسناد مرتبط:\n{related_docs}"
        ),
        required_variables=("issue_description", "executor_profiles", "related_docs"),
        expected_json_keys=(
            "recommended_executors",
            "matching_rationale",
            "expertise_gaps",
            "citations",
        ),
    ),
    # ۷. شناسایی شکاف پژوهشی
    PromptTemplate(
        template_id="PRM-07-RESEARCH-GAP",
        version_tag=VersionTag("v1.0.0", 1),
        task_category=PromptTaskCategory.RESEARCH_GAP_IDENTIFICATION,
        system_instruction=(
            "تو تحلیل‌گر شکاف‌های پژوهشی (بخش ۱۴.۳ پروپوزال) هستی. کمبود شواهد دانشی، دانش قدیمی یا تعارض منابع را برای مسئله شناسایی کن."
        ),
        user_prompt_template=(
            "گره: {node_id}\nمسئله و زیرمسئله‌ها:\n{issue_summary}\nشواهد دانشی موجود:\n{knowledge_evidence}"
        ),
        required_variables=("node_id", "issue_summary", "knowledge_evidence"),
        expected_json_keys=(
            "has_research_gap",
            "identified_gaps",
            "gap_severity",
            "requires_rfp",
            "citations",
            "uncertainty_status",
        ),
    ),
    # ۸. آماده‌سازی انتقال دانش
    PromptTemplate(
        template_id="PRM-08-TRANSFER-PREPARATION",
        version_tag=VersionTag("v1.0.0", 1),
        task_category=PromptTaskCategory.KNOWLEDGE_TRANSFER_PREPARATION,
        system_instruction=(
            "تو مسئول مونتاژ بسته انتقال دانش چرخه‌ای (TransferPackage) به لایه بالاتر هستی و پیشنهادهای اصلاحی سطح جزئیات را ارائه می‌دهی."
        ),
        user_prompt_template=(
            "گره مبدأ: {source_node_id}\nلایه هدف: {target_layer}\nسطح جزئیات مورد نیاز: {detail_level}\n"
            "قواعد انتقال لایه: {layer_rules}\nداده‌های تجمیعی گره:\n{node_digest}"
        ),
        required_variables=(
            "source_node_id",
            "target_layer",
            "detail_level",
            "layer_rules",
            "node_digest",
        ),
        expected_json_keys=(
            "transfer_package_summary",
            "ai_improvement_suggestions",
            "highlighted_risks_and_gaps",
            "decisions_for_upper_layer",
            "citations",
            "uncertainty_status",
        ),
    ),
    # ۹. تولید RFP
    PromptTemplate(
        template_id="PRM-09-RFP-GENERATION",
        version_tag=VersionTag("v1.0.0", 1),
        task_category=PromptTaskCategory.RFP_GENERATION,
        system_instruction=(
            "تو مولد خودکار سند اعلام نیاز (RFP) طبق بخش ۱۵.۱ پروپوزال هستی. تمامی بخش‌های اجباری RFP را بر اساس مسئله و شکاف پژوهشی تولید کن."
        ),
        user_prompt_template=(
            "گره صادرکننده: {issuer_node_id}\nشرح مسئله و زیرمسئله‌ها:\n{issue_context}\nشکاف پژوهشی:\n{research_gaps}"
        ),
        required_variables=("issuer_node_id", "issue_context", "research_gaps"),
        expected_json_keys=(
            "rfp_title",
            "problem_statement",
            "technical_requirements",
            "expected_deliverables",
            "evaluation_criteria",
            "constraints",
            "required_expertise",
            "citations",
        ),
    ),
    # ۱۰. ارزیابی پیشنهاد
    PromptTemplate(
        template_id="PRM-10-PROPOSAL-EVALUATION",
        version_tag=VersionTag("v1.0.0", 1),
        task_category=PromptTaskCategory.PROPOSAL_EVALUATION,
        system_instruction=(
            "تو ارزیاب فنی پیشنهادهای دریافتی (Proposal) در برابر معیارهای RFP طبق بخش ۱۵.۲ پروپوزال هستی."
        ),
        user_prompt_template=(
            "مشخصات و معیارهای RFP:\n{rfp_specification}\nمتن پیشنهاد دریافتی:\n{proposal_content}"
        ),
        required_variables=("rfp_specification", "proposal_content"),
        expected_json_keys=(
            "evaluated_score",
            "criterion_scores",
            "technical_completeness_assessment",
            "feasibility_assessment",
            "risk_assessment",
            "citations",
        ),
    ),
    # ۱۱. تولید بازخورد
    PromptTemplate(
        template_id="PRM-11-FEEDBACK-GENERATION",
        version_tag=VersionTag("v1.0.0", 1),
        task_category=PromptTaskCategory.FEEDBACK_GENERATION,
        system_instruction=(
            "تو مسئول تولید گزارش بازخورد خودکار (FeedbackReport) طبق بخش ۱۵.۳ پروپوزال هستی. تمام نقدها را به معیارهای RFP ارجاع بده."
        ),
        user_prompt_template=(
            "معیارهای RFP:\n{rfp_criteria}\nنتیجه ارزیابی پیشنهاد:\n{evaluation_summary}"
        ),
        required_variables=("rfp_criteria", "evaluation_summary"),
        expected_json_keys=(
            "strengths",
            "weaknesses",
            "missing_requirements",
            "technical_risks",
            "clarifications_needed",
            "suggested_score",
            "recommendation_decision",
            "criterion_references",
        ),
    ),
    # ۱۲. امتیازدهی LLM به‌عنوان داور
    PromptTemplate(
        template_id="PRM-12-LLM-AS-JUDGE",
        version_tag=VersionTag("v1.0.0", 1),
        task_category=PromptTaskCategory.LLM_AS_JUDGE_SCORING,
        system_instruction=(
            "تو داور کنترل کیفیت (LLM-as-a-Judge) طبق بخش ۱۶ پروپوزال هستی. خروجی تولیدشده را دقیقاً بر اساس ۷ معیار جدول ۱۶.۱ (مجموع ۲۰ امتیاز) داوری کن."
        ),
        user_prompt_template=(
            "دانش اصلی منبع:\n{source_knowledge}\nدرخواست کاربر یا قاعده انتقال:\n{task_instruction}\nخروجی تولیدشده جهت داوری:\n{generated_output}"
        ),
        required_variables=("source_knowledge", "task_instruction", "generated_output"),
        expected_json_keys=(
            "source_fidelity",
            "completeness",
            "detail_level_fit",
            "uncertainty_and_gap_handling",
            "clarity_and_structure",
            "source_attribution",
            "no_unsupported_claims",
            "total_score",
            "is_approved",
            "explanation",
            "improvement_suggestions",
        ),
    ),
)


class PromptRegistry:
    """
    رجیستری مرکزی قالب‌های پرامپت با پشتیبانی از نسخه‌بندی، بازیابی بر اساس دسته وظیفه
    و نگهداری سوابق آزمون پرامپت (بخش ۸.۳ پروپوزال).
    """

    def __init__(self, initialize_defaults: bool = True) -> None:
        self._templates_by_id: Dict[str, List[PromptTemplate]] = {}
        self._active_by_category: Dict[PromptTaskCategory, PromptTemplate] = {}
        self._test_history: List[Dict[str, Any]] = []
        if initialize_defaults:
            self.initialize_default_templates()

    def initialize_default_templates(self) -> None:
        """بارگذاری ۱۲ قالب استاندارد پیش‌فرض برای هر ۱۲ دسته پرامپت صفحه ۱۸ پروپوزال."""
        for tmpl in DEFAULT_PROMPT_TEMPLATES:
            self.register_template(tmpl)

    def register_template(self, template: PromptTemplate) -> PromptTemplate:
        """
        ثبت یک قالب پرامپت جدید یا نسخه جدیدی از یک قالب موجود بدون حذف نسخه‌های پیشین.
        """
        if not isinstance(template, PromptTemplate):
            raise DomainValidationError(
                "ورودی register_template باید از نوع PromptTemplate باشد.",
                field_name="template",
            )

        history = self._templates_by_id.setdefault(template.template_id, [])
        for existing in history:
            if existing.version_tag == template.version_tag:
                raise DomainValidationError(
                    f"قالب پرامپت '{template.template_id}' با نسخه '{template.version_tag}' قبلاً ثبت شده است.",
                    field_name="version_tag",
                )

        # غیرفعال کردن نسخه‌های قبلی همان شناسه در صورت فعال بودن نسخه جدید
        if template.is_active and history:
            updated_history: List[PromptTemplate] = []
            for old_t in history:
                updated_history.append(
                    PromptTemplate(
                        template_id=old_t.template_id,
                        version_tag=old_t.version_tag,
                        task_category=old_t.task_category,
                        system_instruction=old_t.system_instruction,
                        user_prompt_template=old_t.user_prompt_template,
                        required_variables=old_t.required_variables,
                        expected_json_keys=old_t.expected_json_keys,
                        citation_rule=old_t.citation_rule,
                        uncertainty_rule=old_t.uncertainty_rule,
                        judge_threshold=old_t.judge_threshold,
                        rubric_criteria_names=old_t.rubric_criteria_names,
                        is_active=False,
                    )
                )
            self._templates_by_id[template.template_id] = updated_history

        self._templates_by_id[template.template_id].append(template)
        if template.is_active:
            self._active_by_category[template.task_category] = template
        return template

    def get_active_template(
        self, category: Union[str, PromptTaskCategory]
    ) -> PromptTemplate:
        """دریافت قالب پرامپت فعال برای یکی از ۱۲ دسته وظیفه."""
        cat_enum = (
            PromptTaskCategory(category)
            if isinstance(category, str) and not isinstance(category, PromptTaskCategory)
            else category
        )
        if cat_enum not in self._active_by_category:
            raise RegistryNotFoundError(
                entity_type="PromptTemplateForCategory", identifier=str(cat_enum)
            )
        return self._active_by_category[cat_enum]

    def get_template_by_id(
        self, template_id: str, version_tag_str: Optional[str] = None
    ) -> PromptTemplate:
        """بازیابی یک قالب بر اساس شناسه و نسخه (یا آخرین نسخه فعال در صورت عدم تعیین نسخه)."""
        cleaned_id = template_id.strip() if isinstance(template_id, str) else ""
        if cleaned_id not in self._templates_by_id:
            raise RegistryNotFoundError(
                entity_type="PromptTemplate", identifier=cleaned_id
            )
        versions = self._templates_by_id[cleaned_id]
        if version_tag_str is not None:
            for ver in versions:
                if ver.version_tag.tag == version_tag_str.strip():
                    return ver
            raise RegistryNotFoundError(
                entity_type="PromptTemplateVersion",
                identifier=f"{cleaned_id}@{version_tag_str}",
            )
        for ver in reversed(versions):
            if ver.is_active:
                return ver
        return versions[-1]

    def get_template_versions(self, template_id: str) -> List[PromptTemplate]:
        """دریافت تاریخچه کامل نسخه‌های یک قالب پرامپت."""
        cleaned_id = template_id.strip() if isinstance(template_id, str) else ""
        if cleaned_id not in self._templates_by_id:
            raise RegistryNotFoundError(
                entity_type="PromptTemplate", identifier=cleaned_id
            )
        return list(self._templates_by_id[cleaned_id])

    def record_test_execution(
        self,
        template_id: str,
        version_tag: str,
        schema_valid: bool,
        judge_score: Optional[int] = None,
        notes: str = "",
    ) -> None:
        """ثبت سابقه آزمون پرامپت طبق بخش ۸.۳ پروپوزال."""
        self._test_history.append(
            {
                "template_id": template_id,
                "version_tag": version_tag,
                "schema_valid": schema_valid,
                "judge_score": judge_score,
                "notes": notes,
            }
        )

    @property
    def active_templates(self) -> List[PromptTemplate]:
        """فهرست تمامی قالب‌های فعال در ۱۲ دسته وظیفه."""
        return list(self._active_by_category.values())

    @property
    def test_history(self) -> List[Dict[str, Any]]:
        """سوابق ثبت‌شده آزمون پرامپت‌ها."""
        return list(self._test_history)

    def to_dict(self) -> Dict[str, Any]:
        """تبدیل رجیستری پرامپت به دیکشنری استاندارد."""
        return {
            "total_categories_covered": len(self._active_by_category),
            "active_templates": [t.to_dict() for t in self.active_templates],
            "test_history_count": len(self._test_history),
        }


__all__ = [
    "PromptTaskCategory",
    "PromptTemplate",
    "DEFAULT_PROMPT_TEMPLATES",
    "PromptRegistry",
]
