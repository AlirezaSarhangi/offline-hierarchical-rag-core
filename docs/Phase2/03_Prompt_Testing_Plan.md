# برنامه آزمون و مدیریت پرامپت (Prompt Testing Plan) — فاز ۲

**ماژول مرجع در کد:** [`src/domain/prompt_registry.py`](file:///d:/offline-hierarchical-rag-core/src/domain/prompt_registry.py)  
**تست‌های واحد مرتبط:** [`tests/test_phase2_models_and_prompts.py`](file:///d:/offline-hierarchical-rag-core/tests/test_phase2_models_and_prompts.py)  
**نتایج اجرای آزمایش‌های پایه:** [`sample_docs/phase2_experiment_results.json`](file:///d:/offline-hierarchical-rag-core/sample_docs/phase2_experiment_results.json)

---

## ۱. اصول مهندسی و نسخه‌بندی پرامپت در هسته RAG آفلاین
طبق بخش ۸.۳ پروپوزال (صفحات ۱۷ و ۱۸)، پرامپت‌ها به‌عنوان بخشی از کدهای دامنه پروژه ثبت، نسخه‌بندی و آزمایش می‌شوند. کلاس تغییرناپذیر [`PromptTemplate`](file:///d:/offline-hierarchical-rag-core/src/domain/prompt_registry.py#L85-L279) و رجیستری [`PromptRegistry`](file:///d:/offline-hierarchical-rag-core/src/domain/prompt_registry.py#L483-L607) تضمین می‌کنند که:
1. هر قالب پرامپت دارای شناسه یکتا (`template_id`) و نسخه تغییرناپذیر ([`VersionTag`](file:///d:/offline-hierarchical-rag-core/src/domain/value_objects.py#L243-L283)) است.
2. هنگام ثبت نسخه جدیدی از یک قالب (مثلاً ارتقا از `v1.0.0` به `v1.1.0`)، نسخه جدید به عنوان نسخه فعال (`is_active=True`) تنظیم شده و نسخه‌های پیشین بدون حذف شدن، برای ممیزی و مقایسه حفظ می‌شوند.
3. مونتاژ پرامپت از طریق متد [`assemble_prompt`](file:///d:/offline-hierarchical-rag-core/src/domain/prompt_registry.py#L186-L223) تمامی متغیرهای ورودی الزامی (`required_variables`) را پیش از ارسال به مدل بررسی کرده و در صورت کمبود یا خالی بودن هر متغیر، خطای [`PromptAssemblyError`](file:///d:/offline-hierarchical-rag-core/src/domain/exceptions.py#L109-L117) صادر می‌کند.

---

## ۲. جدول کامل ۱۲ دسته پرامپت رسمی سامانه (بخش ۸.۳ — صفحه ۱۸ پروپوزال)

| ردیف | کد دسته (`PromptTaskCategory`) | شناسه قالب پیش‌فرض | عنوان فارسی دسته | متغیرهای ورودی الزامی (`required_variables`) | کلیدهای الزامی خروجی JSON (`expected_json_keys`) |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **۱** | `QA_GROUNDED` | `PRM-01-QA-GROUNDED` | پرسش و پاسخ مبتنی بر منبع | `node_id`, `context`, `query` | `answer`, `citations`, `uncertainty_status`, `related_issues`, `suggested_executors` |
| **۲** | `TECHNICAL_SUMMARIZATION` | `PRM-02-TECH-SUMMARY` | خلاصه‌سازی فنی | `node_id`, `detail_level`, `source_content` | `technical_summary`, `methods_and_evidence`, `technical_limitations`, `citations`, `uncertainty_status` |
| **۳** | `EXECUTIVE_MANAGERIAL_SUMMARIZATION` | `PRM-03-EXEC-MANAGERIAL-SUMMARY` | خلاصه‌سازی مدیریتی و اجرایی | `node_id`, `target_layer`, `detail_level`, `source_content` | `executive_summary`, `key_achievements`, `critical_risks`, `decisions_required`, `citations`, `uncertainty_status` |
| **۴** | `ISSUE_DECOMPOSITION` | `PRM-04-ISSUE-DECOMPOSITION` | تجزیه مسئله | `node_id`, `issue_description`, `existing_knowledge` | `parent_issue_title`, `suggested_priority`, `sub_issues`, `citations`, `uncertainty_status` |
| **۵** | `SUB_ISSUE_COMPARISON` | `PRM-05-SUB-ISSUE-COMPARISON` | مقایسه زیرمسئله‌ها | `node_id`, `candidate_sub_issues`, `existing_issues` | `overlapping_pairs`, `dependency_order`, `merge_recommendations`, `citations` |
| **۶** | `EXECUTOR_EXPERTISE_ANALYSIS` | `PRM-06-EXECUTOR-ANALYSIS` | تحلیل تخصص مجریان | `issue_description`, `executor_profiles`, `related_docs` | `recommended_executors`, `matching_rationale`, `expertise_gaps`, `citations` |
| **۷** | `RESEARCH_GAP_IDENTIFICATION` | `PRM-07-RESEARCH-GAP` | شناسایی شکاف پژوهشی | `node_id`, `issue_summary`, `knowledge_evidence` | `has_research_gap`, `identified_gaps`, `gap_severity`, `requires_rfp`, `citations`, `uncertainty_status` |
| **۸** | `KNOWLEDGE_TRANSFER_PREPARATION` | `PRM-08-TRANSFER-PREPARATION` | آماده‌سازی انتقال دانش | `source_node_id`, `target_layer`, `detail_level`, `layer_rules`, `node_digest` | `transfer_package_summary`, `ai_improvement_suggestions`, `highlighted_risks_and_gaps`, `decisions_for_upper_layer`, `citations`, `uncertainty_status` |
| **۹** | `RFP_GENERATION` | `PRM-09-RFP-GENERATION` | تولید RFP | `issuer_node_id`, `issue_context`, `research_gaps` | `rfp_title`, `problem_statement`, `technical_requirements`, `expected_deliverables`, `evaluation_criteria`, `constraints`, `required_expertise`, `citations` |
| **۱۰** | `PROPOSAL_EVALUATION` | `PRM-10-PROPOSAL-EVALUATION` | ارزیابی پیشنهاد | `rfp_specification`, `proposal_content` | `evaluated_score`, `criterion_scores`, `technical_completeness_assessment`, `feasibility_assessment`, `risk_assessment`, `citations` |
| **۱۱** | `FEEDBACK_GENERATION` | `PRM-11-FEEDBACK-GENERATION` | تولید بازخورد | `rfp_criteria`, `evaluation_summary` | `strengths`, `weaknesses`, `missing_requirements`, `technical_risks`, `clarifications_needed`, `suggested_score`, `recommendation_decision`, `criterion_references` |
| **۱۲** | `LLM_AS_JUDGE_SCORING` | `PRM-12-LLM-AS-JUDGE` | امتیازدهی LLM به‌عنوان داور | `source_knowledge`, `task_instruction`, `generated_output` | `source_fidelity`, `completeness`, `detail_level_fit`, `uncertainty_and_gap_handling`, `clarity_and_structure`, `source_attribution`, `no_unsupported_claims`, `total_score`, `is_approved`, `explanation`, `improvement_suggestions` |

---

## ۳. مکانیزم‌های حکمرانی پرامپت: ارجاع به منبع و مدیریت عدم قطعیت

در تمامی قالب‌های [`PromptTemplate`](file:///d:/offline-hierarchical-rag-core/src/domain/prompt_registry.py#L85-L279)، دو قانون حکمرانی اجباری در متد `__post_init__` اعتبارسنجی شده و در بخش `### [MANDATORY GOVERNANCE RULES]` به پرامپت مونتاژشده تزریق می‌گردند:
1. **قانون ارجاع به منبع (`citation_rule`):**
   هر قالب ملزم به داشتن الگوی `[DOC-ID]` است تا مدل در تمام خروجی‌ها شناسه یکتای سند یا قطعه دانش (مانند `[KI-001]`) را درج نماید.
2. **قانون اعلام صریح عدم قطعیت (`uncertainty_rule`):**
   هر قالب ملزم به داشتن توکن استاندارد `INSUFFICIENT_KNOWLEDGE` است. هنگامی که دانش موجود در گره برای پاسخ‌گویی یا حل مسئله کافی نباشد، مدل موظف است به‌جای حدس‌زدن یا تولید ادعای بدون پشتوانه، وضعیت `INSUFFICIENT_KNOWLEDGE` را بازگرداند.

---

## ۴. اتصال به چارچوب داوری ۲۰ نمره‌ای (`JudgeRubric`) و آزمون واحد پایداری Schema

طبق جدول ۱۹.۲ پروپوزال (ردیف پایداری پرامپت):
> *«ورودی ساخت‌یافته مشابه، خروجی با Schema پایدار تولید می‌کند. آزمون‌های واحد مونتاژ پرامپت و رفتار Parser خروجی را تأیید می‌کنند.»*

### ۴.۱ روش آزمون واحد Parser خروجی (`parse_and_validate_output`)
متد [`PromptTemplate.parse_and_validate_output`](file:///d:/offline-hierarchical-rag-core/src/domain/prompt_registry.py#L225-L265) در آزمون‌های واحد در برابر سناریوهای زیر آزمایش می‌شود:
- **خروجی JSON خام و معتبر:** پارس موفق و بازگرداندن دیکشنری شامل تمام کلیدهای `expected_json_keys`.
- **خروجی JSON محصور در بلوک Markdown (` ```json ... ``` `):** استخراج خودکار بدنه JSON با عبارت منظم و اعتبارسنجی موفق کلیدها.
- **خروجی دارای نحو JSON شکسته یا غیر دیکشنری:** پرتاب استثنای [`PromptOutputValidationError`](file:///d:/offline-hierarchical-rag-core/src/domain/exceptions.py#L120-L128).
- **خروجی JSON معتبر اما فاقد یکی از کلیدهای الزامی `expected_json_keys`:** پرتاب استثنای `PromptOutputValidationError` با ذکر نام کلیدهای مفقود.

### ۴.۲ چرخه ارزیابی با داور (`PRM-12-LLM-AS-JUDGE`)
هر خروجی تولیدشده توسط دسته‌های ۱ تا ۱۱ می‌تواند به قالب دسته ۱۲ (`PRM-12-LLM-AS-JUDGE`) ارسال شود. خروجی داور پس از عبور از `parse_and_validate_output` به شیء مقداری [`JudgeRubric`](file:///d:/offline-hierarchical-rag-core/src/domain/value_objects.py#L314-L406) و موجودیت [`JudgeScore`](file:///d:/offline-hierarchical-rag-core/src/domain/entities.py#L681-L746) تبدیل شده و در برابر حد آستانه پیش‌فرض **۱۸ از ۲۰** ([`JudgeThreshold`](file:///d:/offline-hierarchical-rag-core/src/domain/value_objects.py#L409-L462)) سنجیده می‌شود.
