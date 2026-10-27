"""
الگوی Ports & Adapters برای اجرای مدل‌های کوچک آفلاین (`Embedding`, `LLM`, `LVM`) طبق بخش ۸.۲، ۱۸.۲ و ۲۰ پروپوزال.

این ماژول شامل:
1. `OfflineModelAdapterPort`: درگاه (Port) انتزاعی برای تعامل لایه دامنه/کاربرد با مدل‌های هوش مصنوعی آفلاین.
2. `LocalOllamaAdapter`: آداپتور اتصال کاملاً آفلاین به سرویس محلی Ollama (`http://localhost:11434`) با کتابخانه استاندارد `urllib.request` و کنترل سخت‌گیرانه قید شبکه محلی (Loopback Only).
3. `DeterministicMockModelAdapter`: آداپتور قطعی (Test Double) طبق صفحه ۳۲ پروپوزال جهت اجرای سریع و ۱۰۰٪ تکرارپذیر آزمون‌های واحد برای بردارهای Embedding، هر ۱۲ دسته پرامپت، داوری ۲۰ نمره‌ای و تحلیل بصری LVM.
"""

from __future__ import annotations

import hashlib
import json
import math
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from src.domain.entities import JudgeScore
from src.domain.exceptions import (
    DomainValidationError,
    OfflineAdapterError,
)
from src.domain.model_registry import (
    ModelCategory,
    ModelRegistry,
    ModelTaskType,
)
from src.domain.prompt_registry import (
    PromptRegistry,
    PromptTaskCategory,
)
from src.domain.value_objects import JudgeRubric, JudgeThreshold

ALLOWED_LOCAL_HOSTS: Tuple[str, ...] = ("localhost", "127.0.0.1", "::1")


class OfflineModelAdapterPort(ABC):
    """
    درگاه انتزاعی (Port) برای اجرای مدل‌های کوچک آفلاین بدون وابستگی به پیاده‌سازی زیرساخت.
    """

    @abstractmethod
    def generate_embedding(
        self,
        text: str,
        model_id: str = "bge-m3",
        dimensions: int = 16,
    ) -> Tuple[float, ...]:
        """تولید بردار تعبیه (Embedding) نرمال‌شده برای متن ورودی."""

    @abstractmethod
    def generate_structured_text(
        self,
        assembled_prompt: str,
        task_category: Union[str, PromptTaskCategory],
        model_id: str = "qwen2.5:3b-instruct-q4_K_M",
        inference_params: Optional[Mapping[str, Any]] = None,
    ) -> str:
        """تولید خروجی متنی ساختاریافته (JSON) توسط مدل زبانی کوچک محلی (LLM)."""

    @abstractmethod
    def analyze_document_layout(
        self,
        page_reference: str,
        raw_content_hint: str,
        model_id: str = "qwen2.5-vl:3b",
    ) -> Dict[str, Any]:
        """تفسیر بصری جدول، تصویر یا چیدمان پیچیده صفحه سند توسط مدل بینایی-زبانی محلی (LVM)."""


class LocalOllamaAdapter(OfflineModelAdapterPort):
    """
    آداپتور اجرای محلی مدل‌ها از طریق سرویس آفلاین Ollama با استفاده از کتابخانه استاندارد `urllib.request`.

    قید امنیتی آفلاین (بخش ۲۰ پروپوزال):
    آدرس سرویس فقط می‌تواند روی میزبان محلی (`localhost` یا `127.0.0.1`) قرار داشته باشد و هرگونه
    تلاش برای اتصال به آدرس‌های اینترنتی خارجی با خطای `OfflineAdapterError` مسدود می‌گردد.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout_seconds: float = 30.0,
        model_registry: Optional[ModelRegistry] = None,
    ) -> None:
        parsed_url = urllib.parse.urlparse(base_url)
        hostname = (parsed_url.hostname or "").lower()
        if hostname not in ALLOWED_LOCAL_HOSTS:
            raise OfflineAdapterError(
                f"نقض قید اجرای آفلاین: آدرس '{base_url}' یک میزبان محلی مجاز ({ALLOWED_LOCAL_HOSTS}) نیست. "
                "فراخوانی سرویس‌های ابری یا اینترنتی در این سامانه ممنوع است."
            )
        self.base_url: str = base_url.rstrip("/")
        self.timeout_seconds: float = float(timeout_seconds)
        self.model_registry: ModelRegistry = model_registry or ModelRegistry()

    def _post_json(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """ارسال درخواست HTTP POST محلی با `urllib.request` و مدیریت خطاهای ارتباطی."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url=url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as resp:
                raw_resp = resp.read().decode("utf-8")
                return json.loads(raw_resp)
        except urllib.error.URLError as exc:
            raise OfflineAdapterError(
                f"امکان برقراری ارتباط با سرویس محلی Ollama در آدرس '{url}' وجود ندارد: {exc}"
            ) from exc
        except (TimeoutError, json.JSONDecodeError) as exc:
            raise OfflineAdapterError(
                f"خطا در دریافت یا پارس پاسخ از سرویس محلی Ollama در '{url}': {exc}"
            ) from exc

    def generate_embedding(
        self,
        text: str,
        model_id: str = "bge-m3",
        dimensions: int = 16,
    ) -> Tuple[float, ...]:
        """درخواست تولید بردار Embedding از سرویس محلی."""
        if not isinstance(text, str) or not text.strip():
            raise DomainValidationError(
                "متن ورودی برای تولید Embedding نمی‌تواند خالی باشد.",
                field_name="text",
            )
        self.model_registry.validate_model_for_task(
            model_id, ModelTaskType.SEMANTIC_INDEXING
        )
        response = self._post_json(
            "/api/embeddings",
            {"model": model_id, "prompt": text.strip()},
        )
        embedding = response.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise OfflineAdapterError(
                f"پاسخ سرویس محلی برای مدل '{model_id}' فاقد بردار embedding معتبر است."
            )
        sliced = [round(float(x), 6) for x in embedding[:dimensions]]
        return tuple(sliced)

    def generate_structured_text(
        self,
        assembled_prompt: str,
        task_category: Union[str, PromptTaskCategory],
        model_id: str = "qwen2.5:3b-instruct-q4_K_M",
        inference_params: Optional[Mapping[str, Any]] = None,
    ) -> str:
        """ارسال پرامپت مونتاژشده به مدل زبانی محلی با فرمت خروجی JSON."""
        if not isinstance(assembled_prompt, str) or not assembled_prompt.strip():
            raise DomainValidationError(
                "پرامپت مونتاژشده (assembled_prompt) نمی‌تواند خالی باشد.",
                field_name="assembled_prompt",
            )
        spec = self.model_registry.get_model(model_id)
        if spec.category != ModelCategory.LLM:
            raise OfflineAdapterError(
                f"مدل '{model_id}' از دسته LLM نیست و نمی‌تواند برای تولید متن ساختاریافته استفاده شود."
            )
        options = dict(spec.default_inference_params)
        if inference_params:
            options.update(inference_params)

        response = self._post_json(
            "/api/generate",
            {
                "model": model_id,
                "prompt": assembled_prompt,
                "format": "json",
                "stream": False,
                "options": options,
            },
        )
        output_text = response.get("response")
        if not isinstance(output_text, str) or not output_text.strip():
            raise OfflineAdapterError(
                f"خروجی متنی معتبری از مدل محلی '{model_id}' دریافت نشد."
            )
        return output_text.strip()

    def analyze_document_layout(
        self,
        page_reference: str,
        raw_content_hint: str,
        model_id: str = "qwen2.5-vl:3b",
    ) -> Dict[str, Any]:
        """تحلیل چیدمان و جدول صفحه سند توسط مدل بینایی-زبانی (LVM) محلی."""
        if not isinstance(page_reference, str) or not page_reference.strip():
            raise DomainValidationError(
                "مرجع صفحه (page_reference) برای تحلیل LVM الزامی است.",
                field_name="page_reference",
            )
        self.model_registry.validate_model_for_task(
            model_id, ModelTaskType.DOCUMENT_LAYOUT_ANALYSIS
        )
        prompt = (
            f"تحلیل ساختار بصری و جداول صفحه: {page_reference}\n"
            f"راهنمای محتوا: {raw_content_hint}\n"
            "خروجی را به صورت JSON شامل کلیدهای page_reference, has_complex_table, extracted_markdown_table برگردان."
        )
        response = self._post_json(
            "/api/generate",
            {
                "model": model_id,
                "prompt": prompt,
                "format": "json",
                "stream": False,
            },
        )
        raw_json = response.get("response", "{}")
        try:
            return json.loads(raw_json)
        except json.JSONDecodeError as exc:
            raise OfflineAdapterError(
                f"خروجی LVM محلی در قالب JSON معتبر نیست: {exc}"
            ) from exc


class DeterministicMockModelAdapter(OfflineModelAdapterPort):
    """
    آداپتور قطعی (Test Double / Deterministic Mock) طبق بخش ۱۸.۲ (صفحه ۳۲ پروپوزال).

    ویژگی‌ها:
    - تولید بردارهای Embedding نرمال‌شده (Unit L2 Norm) کاملاً قطعی بر اساس SHA-256 متن.
    - تولید خروجی‌های JSON معتبر و منطبق بر Schema برای تمام ۱۲ دسته پرامپت `PromptTaskCategory`.
    - پشتیبانی از شبیه‌سازی کمبود دانش (`INSUFFICIENT_KNOWLEDGE`) و داوری ۲۰ نمره‌ای (`JudgeRubric`).
    - ثبت شمارنده فراخوانی‌ها (`call_log`) جهت راستی‌آزمایی رفتار سرویس‌ها در آزمون‌های واحد.
    """

    def __init__(
        self,
        model_registry: Optional[ModelRegistry] = None,
        prompt_registry: Optional[PromptRegistry] = None,
    ) -> None:
        self.model_registry: ModelRegistry = model_registry or ModelRegistry()
        self.prompt_registry: PromptRegistry = prompt_registry or PromptRegistry()
        self.call_log: List[Dict[str, Any]] = []

    def generate_embedding(
        self,
        text: str,
        model_id: str = "bge-m3",
        dimensions: int = 16,
    ) -> Tuple[float, ...]:
        """تولید بردار Embedding قطعی و نرمال‌شده (L2=1.0) با ابعاد مشخص."""
        if not isinstance(text, str) or not text.strip():
            raise DomainValidationError(
                "متن ورودی برای تولید Embedding نمی‌تواند خالی باشد.",
                field_name="text",
            )
        if dimensions < 2:
            raise DomainValidationError(
                "ابعاد بردار Embedding باید حداقل ۲ باشد.",
                field_name="dimensions",
            )
        self.model_registry.validate_model_for_task(
            model_id, ModelTaskType.SEMANTIC_INDEXING
        )

        normalized_text = " ".join(text.strip().split())
        raw_values: List[float] = []
        counter = 0
        while len(raw_values) < dimensions:
            digest = hashlib.sha256(
                f"{model_id}::{counter}::{normalized_text}".encode("utf-8")
            ).digest()
            for idx in range(0, len(digest), 2):
                if len(raw_values) >= dimensions:
                    break
                uint16_val = int.from_bytes(digest[idx : idx + 2], byteorder="big")
                mapped = (uint16_val / 32767.5) - 1.0
                raw_values.append(mapped)
            counter += 1

        norm = math.sqrt(sum(v * v for v in raw_values)) or 1.0
        unit_vector = tuple(round(v / norm, 6) for v in raw_values)

        self.call_log.append(
            {
                "method": "generate_embedding",
                "model_id": model_id,
                "dimensions": dimensions,
                "text_length": len(normalized_text),
            }
        )
        return unit_vector

    def generate_structured_text(
        self,
        assembled_prompt: str,
        task_category: Union[str, PromptTaskCategory],
        model_id: str = "qwen2.5:3b-instruct-q4_K_M",
        inference_params: Optional[Mapping[str, Any]] = None,
    ) -> str:
        """
        تولید خروجی JSON قطعی و منطبق با کلیدهای الزامی برای هر یک از ۱۲ دسته پرامپت.
        """
        if not isinstance(assembled_prompt, str) or not assembled_prompt.strip():
            raise DomainValidationError(
                "پرامپت مونتاژشده (assembled_prompt) نمی‌تواند خالی باشد.",
                field_name="assembled_prompt",
            )
        cat_enum = (
            PromptTaskCategory(task_category)
            if isinstance(task_category, str)
            and not isinstance(task_category, PromptTaskCategory)
            else task_category
        )
        if not isinstance(cat_enum, PromptTaskCategory):
            raise DomainValidationError(
                "فیلد task_category باید از نوع PromptTaskCategory باشد.",
                field_name="task_category",
            )

        spec = self.model_registry.get_model(model_id)
        if spec.category != ModelCategory.LLM:
            raise OfflineAdapterError(
                f"مدل '{model_id}' از دسته LLM نیست و نمی‌تواند پرامپت متنی را اجرا کند."
            )

        is_insufficient = "INSUFFICIENT_CONTEXT" in assembled_prompt
        uncertainty_val = (
            "INSUFFICIENT_KNOWLEDGE" if is_insufficient else "SUFFICIENT_EVIDENCE"
        )

        payload = self._build_deterministic_payload(
            cat_enum, assembled_prompt, uncertainty_val
        )
        json_str = json.dumps(payload, ensure_ascii=False, indent=2)

        self.call_log.append(
            {
                "method": "generate_structured_text",
                "model_id": model_id,
                "task_category": cat_enum.value,
                "prompt_length": len(assembled_prompt),
            }
        )
        return json_str

    def analyze_document_layout(
        self,
        page_reference: str,
        raw_content_hint: str,
        model_id: str = "qwen2.5-vl:3b",
    ) -> Dict[str, Any]:
        """شبیه‌سازی قطعی تحلیل بصری صفحه سند و استخراج جدول توسط مدل LVM محلی."""
        if not isinstance(page_reference, str) or not page_reference.strip():
            raise DomainValidationError(
                "مرجع صفحه (page_reference) نمی‌تواند خالی باشد.",
                field_name="page_reference",
            )
        self.model_registry.validate_model_for_task(
            model_id, ModelTaskType.DOCUMENT_LAYOUT_ANALYSIS
        )

        result = {
            "model_id": model_id,
            "page_reference": page_reference.strip(),
            "layout_type": "COMPLEX_TABLE_AND_TEXT",
            "has_complex_table": True,
            "extracted_markdown_table": (
                "| شاخص | مقدار هدف | وضعیت |\n| --- | --- | --- |\n| امتیاز داور | >= 18/20 | تایید شده [KI-001] |"
            ),
            "visual_confidence": 0.94,
            "notes": f"تفسیر بصری چیدمان برای '{raw_content_hint[:50]}' با موفقیت انجام شد.",
        }
        self.call_log.append(
            {
                "method": "analyze_document_layout",
                "model_id": model_id,
                "page_reference": page_reference.strip(),
            }
        )
        return result

    def evaluate_output_with_judge(
        self,
        score_id: str,
        target_entity_id: str,
        source_knowledge: str,
        task_instruction: str,
        generated_output: str,
        threshold: Optional[JudgeThreshold] = None,
    ) -> JudgeScore:
        """
        اجرای کامل زنجیره داوری (مونتاژ پرامپت دسته ۱۲، تولید خروجی داور، پارس JSON و ساخت `JudgeScore`).
        """
        judge_template = self.prompt_registry.get_active_template(
            PromptTaskCategory.LLM_AS_JUDGE_SCORING
        )
        assembled = judge_template.assemble_prompt(
            source_knowledge=source_knowledge,
            task_instruction=task_instruction,
            generated_output=generated_output,
        )
        raw_judge_json = self.generate_structured_text(
            assembled_prompt=assembled,
            task_category=PromptTaskCategory.LLM_AS_JUDGE_SCORING,
        )
        parsed = judge_template.parse_and_validate_output(raw_judge_json)

        rubric = JudgeRubric(
            source_fidelity=int(parsed["source_fidelity"]),
            completeness=int(parsed["completeness"]),
            detail_level_fit=int(parsed["detail_level_fit"]),
            uncertainty_and_gap_handling=int(parsed["uncertainty_and_gap_handling"]),
            clarity_and_structure=int(parsed["clarity_and_structure"]),
            source_attribution=int(parsed["source_attribution"]),
            no_unsupported_claims=int(parsed["no_unsupported_claims"]),
        )
        return JudgeScore(
            score_id=score_id,
            target_entity_id=target_entity_id,
            rubric=rubric,
            threshold=threshold or judge_template.judge_threshold,
            explanation=str(parsed["explanation"]),
            improvement_suggestions=tuple(parsed.get("improvement_suggestions", [])),
        )

    @staticmethod
    def _build_deterministic_payload(
        category: PromptTaskCategory,
        assembled_prompt: str,
        uncertainty_val: str,
    ) -> Dict[str, Any]:
        """تولید دیکشنری استاندارد منطبق با کلیدهای الزامی هر یک از ۱۲ دسته پرامپت."""
        if category == PromptTaskCategory.QA_GROUNDED:
            return {
                "answer": (
                    "بر اساس مستندات گره [KI-001]، هسته RAG آفلاین اسناد DOC، DOCX و PDF را به صورت محلی پردازش کرده "
                    "و هر قطعه دانش شناسه سند منبع، نسخه و شماره صفحه را حفظ می‌کند."
                    if uncertainty_val == "SUFFICIENT_EVIDENCE"
                    else "INSUFFICIENT_KNOWLEDGE: دانش کافی در اسناد مجاز این گره برای پاسخ قطعی یافت نشد."
                ),
                "citations": ["[KI-001]", "[VER-001]"],
                "uncertainty_status": uncertainty_val,
                "related_issues": ["ISS-101"],
                "suggested_executors": ["EXEC-201"],
            }

        if category == PromptTaskCategory.TECHNICAL_SUMMARIZATION:
            return {
                "technical_summary": (
                    "خط لوله ورود اسناد با استفاده از خردسازی آگاه از ساختار و مدل‌های کوچک آفلاین (bge-m3 و Qwen2.5-3B) "
                    "اجرا شده و ارجاعات منبع [KI-001] در سطح هر قطعه حفظ می‌گردد."
                ),
                "methods_and_evidence": [
                    "خردسازی با حفظ مسیر عنوان و شماره صفحه [KI-001]",
                    "تفسیر جداول پیچیده با مفسر کمکی LVM محلی [KI-002]",
                ],
                "technical_limitations": [
                    "محدودیت مصرف حافظه روی اسناد اسکن‌شده با وضوح بالا در حالت CPU"
                ],
                "citations": ["[KI-001]", "[KI-002]"],
                "uncertainty_status": uncertainty_val,
            }

        if category == PromptTaskCategory.EXECUTIVE_MANAGERIAL_SUMMARIZATION:
            return {
                "executive_summary": (
                    "فاز کشف دامنه و خط پایه مدل‌های آفلاین تکمیل شده و هسته RAG آماده انتقال بسته‌های دانشی تاییدشده "
                    "با امتیاز بالای ۱۸ از ۲۰ به لایه راهبردی است [KI-002]."
                ),
                "key_achievements": [
                    "استقرار کامل رجیستری ۵ لایه‌ای و کنترل مرز گره‌ها [KI-001]",
                    "تدوین RFP-301 جهت رفع شکاف پژوهشی استخراج جداول تصویری [KI-002]",
                ],
                "critical_risks": [
                    "نیاز به بهینه‌سازی سرعت مدل بینایی-زبانی LVM روی سرورهای محلی بدون GPU"
                ],
                "decisions_required": [
                    "تصویب تخصیص منابع سخت‌افزاری محلی برای اجرای مدل کوانتیزه Qwen2.5-VL"
                ],
                "citations": ["[KI-001]", "[KI-002]"],
                "uncertainty_status": uncertainty_val,
            }

        if category == PromptTaskCategory.ISSUE_DECOMPOSITION:
            return {
                "parent_issue_title": "بهینه‌سازی دقت استخراج جداول پیچیده فارسی در اسناد PDF آفلاین",
                "suggested_priority": "HIGH",
                "sub_issues": [
                    {
                        "relative_order": 1,
                        "title": "طراحی تشخیص‌دهنده صفحات جدول‌دار و چیدمان‌محور [KI-001]",
                        "status": "RESOLVED",
                    },
                    {
                        "relative_order": 2,
                        "title": "کالیبراسیون مدل LVM کوانتیزه برای استخراج ساختار جدول فارسی [KI-002]",
                        "status": "IN_PROGRESS",
                    },
                ],
                "citations": ["[KI-001]", "[KI-002]"],
                "uncertainty_status": uncertainty_val,
            }

        if category == PromptTaskCategory.SUB_ISSUE_COMPARISON:
            return {
                "overlapping_pairs": [
                    {
                        "sub_issue_a": "SUB-101-1",
                        "sub_issue_b": "SUB-101-2",
                        "similarity_score": 0.42,
                        "relation": "SEQUENTIAL_DEPENDENCY",
                    }
                ],
                "dependency_order": ["SUB-101-1", "SUB-101-2"],
                "merge_recommendations": [
                    "زیرمسئله‌ها مستقل بوده و نیازی به ادغام ندارند؛ SUB-101-1 پیش‌نیاز SUB-101-2 است [KI-002]."
                ],
                "citations": ["[KI-002]"],
            }

        if category == PromptTaskCategory.EXECUTOR_EXPERTISE_ANALYSIS:
            return {
                "recommended_executors": [
                    {
                        "executor_id": "EXEC-201",
                        "name": "مهندس علی کاظمی",
                        "match_score": 0.95,
                    }
                ],
                "matching_rationale": (
                    "تطابق مستقیم تخصص پردازش زبان طبیعی فارسی و طراحی دامنه‌محور و نویسندگی سند مرجع [KI-001]."
                ),
                "expertise_gaps": [
                    "نیاز به همکار مکمل در حوزه بهینه‌سازی کرنل‌های بینایی ماشین (LVM)"
                ],
                "citations": ["[KI-001]", "[EXEC-201]"],
            }

        if category == PromptTaskCategory.RESEARCH_GAP_IDENTIFICATION:
            return {
                "has_research_gap": True,
                "identified_gaps": [
                    "کمبود شواهد تجربی درباره دقت مدل‌های LVM زیر ۴ میلیارد پارامتر روی جداول مالی و مهندسی فارسی [KI-002]."
                ],
                "gap_severity": "HIGH",
                "requires_rfp": True,
                "citations": ["[KI-002]"],
                "uncertainty_status": "INSUFFICIENT_KNOWLEDGE",
            }

        if category == PromptTaskCategory.KNOWLEDGE_TRANSFER_PREPARATION:
            return {
                "transfer_package_summary": (
                    "بسته انتقال گره L3-RAG-CORE-TEAM برای لایه ۲ با سطح جزئیات خلاصه فنی (TECHNICAL_SUMMARY) آماده شد [KI-001]."
                ),
                "ai_improvement_suggestions": [
                    "حذف لاگ‌های سطح قطعه و برجسته‌سازی شکاف پژوهشی LVM و وضعیت RFP-301 برای مدیران لایه بالاتر."
                ],
                "highlighted_risks_and_gaps": [
                    "شکاف پژوهشی دقت جدول‌خوانی فارسی در اسناد اسکن‌شده قدیمی [KI-002]"
                ],
                "decisions_for_upper_layer": [
                    "تأیید پیشنهاد فنی PROP-601 برای پیاده‌سازی خط لوله دو مرحله‌ای PDF"
                ],
                "citations": ["[KI-001]", "[KI-002]"],
                "uncertainty_status": uncertainty_val,
            }

        if category == PromptTaskCategory.RFP_GENERATION:
            return {
                "rfp_title": "توسعه ماژول استخراج جداول پیچیده فارسی با مفسر کمکی LVM آفلاین",
                "problem_statement": "ارتقای دقت استخراج جداول تودرتو در گزارش‌های PDF اسکن‌شده بدون اتصال به اینترنت [KI-002].",
                "technical_requirements": [
                    "اجرای ۱۰۰٪ آفلاین روی سخت‌افزار محلی",
                    "حفظ شناسه سند منبع، نسخه و شماره صفحه در قطعات خروجی [KI-001]",
                ],
                "expected_deliverables": [
                    "آداپتور استخراج دو مرحله‌ای PDF/LVM",
                    "مجموعه آزمون واحد قطعی",
                ],
                "evaluation_criteria": [
                    "دقت استخراج ساختار جدول بالای ۹۰٪",
                    "کسب حداقل امتیاز ۱۸ از ۲۰ از داور LLM",
                ],
                "constraints": ["حداکثر مصرف حافظه ۸ گیگابایت"],
                "required_expertise": ["بینایی ماشین", "معماری دامنه‌محور پایتون"],
                "citations": ["[KI-001]", "[KI-002]"],
            }

        if category == PromptTaskCategory.PROPOSAL_EVALUATION:
            return {
                "evaluated_score": 18.5,
                "criterion_scores": {
                    "rfp_alignment": 5.0,
                    "technical_completeness": 4.5,
                    "methodology_and_feasibility": 5.0,
                    "risk_management": 4.0,
                },
                "technical_completeness_assessment": "تمامی الزامات اجرای آفلاین و حفظ فراداده پوشش داده شده‌اند [KI-001].",
                "feasibility_assessment": "رویکرد دو مرحله‌ای در محیط محلی کاملاً عملیاتی و قابل آزمون واحد است.",
                "risk_assessment": "ریسک زمان پردازش با مکانیزم Fallback کنترل شده است.",
                "citations": ["[RFP-301]", "[PROP-601]"],
            }

        if category == PromptTaskCategory.FEEDBACK_GENERATION:
            return {
                "strengths": [
                    "رعایت کامل قید آفلاین و معماری Ports & Adapters [RFP-301]"
                ],
                "weaknesses": [
                    "حد آستانه تشخیص پیچیدگی چیدمان صفحه نیاز به کالیبراسیون عددی دارد"
                ],
                "missing_requirements": [],
                "technical_risks": [
                    "افزایش زمان پردازش چرخه‌ای در اسناد بالای ۲۰۰ صفحه اسکن‌شده"
                ],
                "clarifications_needed": [
                    "تعیین سقف زمان مجاز (Timeout) برای پردازش هر صفحه تصویری"
                ],
                "suggested_score": 18.5,
                "recommendation_decision": "SELECT",
                "criterion_references": [
                    "نیازمندی فنی ۱: اجرای آفلاین [RFP-301]",
                    "معیار ارزیابی ۲: حفظ ارجاع صفحه [KI-001]",
                ],
            }

        # دسته ۱۲: LLM_AS_JUDGE_SCORING
        is_low_quality = "LOW_QUALITY_OUTPUT" in assembled_prompt
        if is_low_quality:
            return {
                "source_fidelity": 3,
                "completeness": 3,
                "detail_level_fit": 2,
                "uncertainty_and_gap_handling": 2,
                "clarity_and_structure": 2,
                "source_attribution": 1,
                "no_unsupported_claims": 1,
                "total_score": 14,
                "is_approved": False,
                "explanation": "خروجی فاقد ارجاع کامل به منبع بوده و امتیاز ۱۴ از ۲۰ کمتر از حد آستانه ۱۸ است.",
                "improvement_suggestions": [
                    "افزودن ارجاعات دقیق [DOC-ID] به تمامی ادعاها",
                    "اعلام صریح شکاف‌های پژوهشی",
                ],
            }

        return {
            "source_fidelity": 5,
            "completeness": 4,
            "detail_level_fit": 3,
            "uncertainty_and_gap_handling": 3,
            "clarity_and_structure": 2,
            "source_attribution": 2,
            "no_unsupported_claims": 0 if "MINOR_CLAIM_WARNING" in assembled_prompt else 1,
            "total_score": 19 if "MINOR_CLAIM_WARNING" in assembled_prompt else 20,
            "is_approved": True,
            "explanation": "خروجی کاملاً منطبق با دانش منبع [KI-001]، دارای ساختار حرفه‌ای و بالاتر از حد آستانه ۱۸ از ۲۰ است.",
            "improvement_suggestions": [],
        }


__all__ = [
    "ALLOWED_LOCAL_HOSTS",
    "OfflineModelAdapterPort",
    "LocalOllamaAdapter",
    "DeterministicMockModelAdapter",
]
