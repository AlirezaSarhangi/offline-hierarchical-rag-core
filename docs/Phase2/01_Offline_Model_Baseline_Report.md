# گزارش خط پایه مدل‌های آفلاین (Offline Model Baseline Report) — فاز ۲

**ماژول مرجع در کد:** [`src/domain/model_registry.py`](file:///d:/offline-hierarchical-rag-core/src/domain/model_registry.py)  
**خروجی آزمایش خط پایه:** [`sample_docs/phase2_experiment_results.json`](file:///d:/offline-hierarchical-rag-core/sample_docs/phase2_experiment_results.json)  
**وضعیت سند:** نسخه نهایی و مصوب فاز ۲ (`v0.2-phase2`)

---

## ۱. مقدمه و فلسفه انتخاب مدل‌های کوچک آفلاین
طبق بخش ۸.۲ پروپوزال، یکی از قیود بنیادین پروژه **اجرای ۱۰۰٪ محلی و آفلاین** تمامی گردش‌کارهای هوش مصنوعی بدون هرگونه وابستگی به اینترنت یا سرویس‌های ابری است. از آنجا که مدل‌های کوچک آفلاین (Small Local Models) نسبت به مدل‌های بزرگ ابری دارای محدودیت‌های محاسباتی و ظرافت‌های زبانی هستند، پروژه فرض نمی‌کند که یک مدل واحد برای تمامی وظایف بهترین انتخاب است. در عوض، مدل‌های محلی نامزد در سه خانواده تخصصی (`Embedding`، `LLM` و `LVM`) بررسی شده و از طریق [`ModelRegistry`](file:///d:/offline-hierarchical-rag-core/src/domain/model_registry.py) به وظایف نُه‌گانه سامانه نگاشت شده‌اند.

---

## ۲. بررسی و مقایسه مدل‌های نامزد آفلاین (Embedding، LLM و LVM)

| شناسه مدل (`model_id`) | دسته (`Category`) | فرمت کوانتیزاسیون | پنجره زمینه (توکن) | مصرف تقریبی حافظه (RAM/VRAM) | کیفیت زبان فارسی | پایداری خروجی JSON | نقش در معماری سامانه |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`bge-m3`** | `EMBEDDING` | `GGUF_F16` | `8,192` | `~2.2 GB` | عالی | — (خروجی برداری ۱۰۲۴ بعدی) | **مدل اصلی (Primary)** نمایه‌سازی معنایی، بازیابی برداری و تشخیص تکرار/هم‌پوشانی. |
| **`paraphrase-multilingual-MiniLM-L12-v2`** | `EMBEDDING` | `ONNX_INT8` | `512` | `~0.5 GB` | خوب | — (خروجی برداری ۳۸۴ بعدی) | **مدل جایگزین سبک (Fallback)** برای محیط‌های فقط CPU با منابع بسیار محدود. |
| **`qwen2.5:3b-instruct-q4_K_M`** | `LLM` | `GGUF_Q4_K_M` | `32,768` | `~2.4 GB` | بسیار عالی | بسیار بالا (منطبق بر Schema) | **مدل اصلی (Primary)** برای هر ۱۲ دسته پرامپت، خلاصه‌سازی سلسله‌مراتبی، RFP و داوری ۲۰ نمره‌ای. |
| **`llama3.2:3b-instruct-q4_K_M`** | `LLM` | `GGUF_Q4_K_M` | `16,384` | `~2.3 GB` | متوسط رو به خوب | بالا | **مدل نامزد دوم** برای خلاصه‌سازی فنی، تجزیه مسئله و ارزیابی مقایسه‌ای در فاز ۱۴. |
| **`qwen2.5-vl:3b`** | `LVM` | `GGUF_Q4_K_M` | `8,192` | `~3.1 GB` | عالی | بالا | **مدل اصلی (Primary)** بینایی-زبانی برای تفسیر جداول پیچیده فارسی و چیدمان اسناد `PDF`/`DOC`. |
| **`moondream2`** | `LVM` | `GGUF_Q4_K_M` | `2,048` | `~1.6 GB` | محدود در متون ریز | متوسط | **مدل سبک کمکی** برای تشخیص اولیه صفحات تصویری و چیدمان کلی سند. |

---

## ۳. جدول نگاشت وظایف سامانه به مدل‌های آفلاین (`Task-to-Model Mapping`)

در کلاس [`ModelRegistry`](file:///d:/offline-hierarchical-rag-core/src/domain/model_registry.py#L316-L426)، هر نوع وظیفه ([`ModelTaskType`](file:///d:/offline-hierarchical-rag-core/src/domain/model_registry.py#L43-L82)) به مدل اصلی و جایگزین خود نگاشت شده است:

| ردیف | کد وظیفه (`ModelTaskType`) | شرح وظیفه در هسته RAG | دسته مدل مورد نیاز | مدل اصلی منتخب (`Primary`) | مدل جایگزین (`Fallback`) |
| :---: | :--- | :--- | :---: | :--- | :--- |
| **۱** | `SEMANTIC_INDEXING` | نمایه‌سازی معنایی و محاسبه شباهت برداری | `EMBEDDING` | `bge-m3` | `paraphrase-multilingual-MiniLM-L12-v2` |
| **۲** | `QA_ANSWERING` | پرسش و پاسخ مبتنی بر منبع در مرز گره | `LLM` | `qwen2.5:3b-instruct-q4_K_M` | `llama3.2:3b-instruct-q4_K_M` |
| **۳** | `TECHNICAL_SUMMARIZATION` | خلاصه‌سازی فنی برای لایه‌های ۲ و ۳ | `LLM` | `qwen2.5:3b-instruct-q4_K_M` | `llama3.2:3b-instruct-q4_K_M` |
| **۴** | `EXECUTIVE_MANAGERIAL_SUMMARIZATION` | خلاصه‌سازی مدیریتی و اجرایی عالی (لایه ۰ و ۱) | `LLM` | `qwen2.5:3b-instruct-q4_K_M` | `llama3.2:3b-instruct-q4_K_M` |
| **۵** | `ISSUE_DECOMPOSITION` | تجزیه مسئله به زیرمسئله و تحلیل شکاف | `LLM` | `qwen2.5:3b-instruct-q4_K_M` | `llama3.2:3b-instruct-q4_K_M` |
| **۶** | `EXECUTOR_EXPERTISE_ANALYSIS` | تحلیل تخصص مجریان و مثلث `IKE` | `LLM` | `qwen2.5:3b-instruct-q4_K_M` | `llama3.2:3b-instruct-q4_K_M` |
| **۷** | `RFP_AND_FEEDBACK_GENERATION` | تولید RFP، ارزیابی پیشنهاد و گزارش بازخورد | `LLM` | `qwen2.5:3b-instruct-q4_K_M` | `llama3.2:3b-instruct-q4_K_M` |
| **۸** | `LLM_AS_JUDGE_EVALUATION` | امتیازدهی داور ۲۰ نمره‌ای (`JudgeRubric`) | `LLM` | `qwen2.5:3b-instruct-q4_K_M` | — (صرفاً مدل با دقت فارسی بالا) |
| **۹** | `DOCUMENT_LAYOUT_ANALYSIS` | تحلیل بصری جدول، تصویر و چیدمان سند | `LVM` | `qwen2.5-vl:3b` | `moondream2` |

---

## ۴. تحلیل عمیق محدودیت‌های عملکردی مدل‌های آفلاین و راهکارهای کاهش ریسک (بخش ۲۳ — صفحه ۴۵ پروپوزال)

بر اساس آزمایش‌های خط پایه فاز ۲ و جدول چالش‌های احتمالی پروپوزال (بخش ۲۳)، محدودیت‌های ذاتی مدل‌های کوچک آفلاین و راهکارهای معماری پیاده‌سازی‌شده در سامانه به شرح زیر است:

| ردیف | محدودیت یا ریسک شناسایی‌شده در مدل آفلاین | سطح اثر | راهکار کاهش ریسک در معماری هسته (Mitigation Strategy) |
| :---: | :--- | :---: | :--- |
| **۱** | **ظرفیت استدلال محدودتر مدل‌های ۳ میلیارد پارامتری نسبت به مدل‌های بزرگ ابری** | متوسط تا زیاد | شکستن گردش‌کارهای پیچیده به **۱۲ دسته پرامپت تخصصی و کوچک**، اتکای قوی به بازیابی دقیق قطعات، و کنترل کیفیت نهایی با سازوکار `LLM-as-a-Judge`. |
| **۲** | **خطای احتمالی در تولید ساختار JSON (شکستن Schema خروجی)** | متوسط | اجبار `format="json"` در آداپتور، درج صریح کلیدهای الزامی (`expected_json_keys`) در بخش حکمرانی پرامپت، و پیاده‌سازی متد [`PromptTemplate.parse_and_validate_output`](file:///d:/offline-hierarchical-rag-core/src/domain/prompt_registry.py#L225-L265) با قابلیت استخراج از بلوک‌های کد و پرتاب استثنای [`PromptOutputValidationError`](file:///d:/offline-hierarchical-rag-core/src/domain/exceptions.py#L120-L128). |
| **۳** | **توهم (Hallucination) در مواجهه با کمبود دانش در گره** | زیاد | تزریق خودکار قانون `INSUFFICIENT_KNOWLEDGE` و الزام ارجاع `[DOC-ID]` در تمامی قالب‌های پرامپت + اختصاص معیارهای «وفاداری به دانش منبع (۵ نمره)»، «مدیریت عدم قطعیت (۳ نمره)» و «نبود ادعای بدون پشتوانه (۱ نمره)» در [`JudgeRubric`](file:///d:/offline-hierarchical-rag-core/src/domain/value_objects.py#L314-L406). |
| **۴** | **ناپایداری امتیازدهی داور در دماهای بالا (`temperature > 0.2`)** | متوسط | تثبیت پارامتر استنتاج `temperature = 0.1` در [`OfflineModelSpecification`](file:///d:/offline-hierarchical-rag-core/src/domain/model_registry.py#L85-L216)، تعریف کلیدهای عددی ثابت برای ۷ معیار جدول ۱۶.۱ و اعمال حد آستانه ۱۸ از ۲۰ ([`JudgeThreshold`](file:///d:/offline-hierarchical-rag-core/src/domain/value_objects.py#L409-L462)). |
| **۵** | **کندی پردازش مدل‌های بینایی-زبانی (`LVM`) روی اسناد PDF حجیم** | متوسط | استفاده انتخابی از `LVM`: استخراج متنی استاندارد برای صفحات معمولی و فراخوانی آداپتور `LVM` صرفاً برای صفحاتی که دارای جدول پیچیده، نمودار یا اسکن تصویری علامت‌گذاری شده‌اند. |
| **۶** | **محدودیت پنجره زمینه در مدل‌های Embedding سبک (`512` توکن در MiniLM)** | متوسط | طراحی موجودیت [`KnowledgeChunk`](file:///d:/offline-hierarchical-rag-core/src/domain/entities.py#L410-L462) با اندازه قطعه کنترل‌شده و حفظ مسیر عنوان‌ها (`heading_path`) و شماره صفحه در هر قطعه. |
