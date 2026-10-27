# هسته RAG سلسله‌مراتبی آفلاین (Offline Hierarchical RAG Core)

**طراحی و پیاده‌سازی سامانه آفلاین، دامنه‌محور (DDD) و سلسله‌مراتبی مبتنی بر Retrieval-Augmented Generation برای مدیریت دانش سازمانی، مدیریت مسئله، نگاشت مجری و انتقال دانش مدیریتی**

---

## ۱. معرفی پروژه و اصول معماری

پروژه **هسته RAG سلسله‌مراتبی آفلاین** یک هسته نرم‌افزاری ماژولار (Core Library) مبتنی بر **Python 3.11+** است که با هدف حل چالش مدیریت و انتقال دانش میان لایه‌های مختلف سازمانی (از تیم‌های فنی و پژوهشی تا مدیران سازمان مادر و هیئت‌مدیره راهبردی) در **محیط‌های کاملاً آفلاین و ایزوله (Air-Gapped)** توسعه یافته است.

### ویژگی‌ها و قیود کلیدی معماری
- **معماری دامنه‌محور (Domain-Driven Design - DDD) و Clean Architecture:**
  - تفکیک دامنه به **۱۰ زمینه مرزبندی‌شده (Bounded Contexts)**، **۱۳ موجودیت کلیدی (Entities)** و **۱۱ شیء مقداری تغییرناپذیر (`@dataclass(frozen=True)` Value Objects)** با اعتبارسنجی سخت‌گیرانه ناوردایی‌ها در `__post_init__`.
  - جداسازی کامل منطق کسب‌وکار دامنه (`src/domain`) از جزئیات زیرساخت و اجرای مدل‌ها (`src/infrastructure`) با الگوی **Ports & Adapters**.
- **بدون بک‌اند و فرانت‌اند مجزا:** تمرکز ۱۰۰٪ پروژه بر ساخت هسته نرم‌افزاری RAG، منطق دامنه و مجموعه آزمون‌های واحد (Unit Tests) قطعی است.
- **اجرای ۱۰۰٪ محلی و آفلاین:** استفاده از مدل‌های کوچک محلی در سه خانواده تخصصی:
  1. **`Embedding`** (مانند `bge-m3` و `paraphrase-multilingual-MiniLM-L12-v2`) برای نمایه‌سازی معنایی و تشخیص تکرار/هم‌پوشانی.
  2. **`LLM`** (مانند `qwen2.5:3b-instruct-q4_K_M` و `llama3.2:3b-instruct-q4_K_M`) برای خلاصه‌سازی سلسله‌مراتبی، پرسش و پاسخ مقید به منبع، تجزیه مسئله، تولید RFP/بازخورد و داوری خروجی.
  3. **`LVM`** (مانند `qwen2.5-vl:3b` و `moondream2`) برای تفسیر بصری جداول پیچیده و چیدمان اسناد `DOC`، `DOCX` و `PDF`.
- **مدل سه‌ضلعی `Issue-Knowledge-Executor` (مدل `IKE`):** پیوند ساختاریافته میان مسائل و زیرمسئله‌ها، شواهد دانشی پشتیبان/محدودکننده و توانمندی مجریان که خودِ این مدل نیز به‌عنوان بخشی از دانش هر گره قابل خلاصه‌سازی و انتقال به لایه بالاتر است.
- **کنترل کیفیت با سازوکار `LLM-as-a-Judge`:** ارزیابی تمام خروجی‌ها و بسته‌های انتقال دانش بر اساس چارچوب ۲۰ نمره‌ای (`JudgeRubric` شامل ۷ معیار جدول ۱۶.۱ پروپوزال) با حد آستانه قبولی پیش‌فرض **۱۸ از ۲۰** (`JudgeThreshold`).

---

## ۲. ساختار درختی پوشه‌ها و ماژول‌های پروژه

```text
offline-hierarchical-rag-core/
├── pyproject.toml                                  # تنظیمات پکیج پایتون و پیکربندی pytest
├── .gitignore                                      # قواعد نادیده‌گرفتن فایل‌های محیطی و کش‌ها
├── persian-project-proposal.pdf                    # سند رسمی پروپوزال ۱۶ فازی پروژه
├── README.md                                       # مستند راهنمای جامع و نقشه راه پروژه
│
├── src/                                            # کدهای منبع هسته نرم‌افزاری
│   ├── __init__.py
│   ├── domain/                                     # لایه هسته دامنه (DDD Core Domain)
│   │   ├── __init__.py
│   │   ├── exceptions.py                           # استثناهای اختصاصی دامنه (Custom Domain Exceptions)
│   │   ├── value_objects.py                        # ۱۱ شیء مقداری تغییرناپذیر (جدول ۵.۳، ۶.۴، ۷.۵ و ۱۶.۱)
│   │   ├── entities.py                             # ۱۳ موجودیت کلیدی دامنه (جدول ۵.۲ پروپوزال)
│   │   ├── layer_node_registry.py                  # رجیستری ۵ لایه‌ای گره و لایه و مرز گره نهایی (is_leaf_boundary)
│   │   ├── model_registry.py                       # رجیستری مدل‌های کوچک آفلاین (Embedding, LLM, LVM)
│   │   └── prompt_registry.py                      # رجیستری و نسخه‌بندی ۱۲ دسته پرامپت و Parser خروجی
│   │
│   └── infrastructure/                             # لایه زیرساخت (Ports & Adapters)
│       ├── __init__.py
│       └── offline_model_adapter.py                # درگاه مدل آفلاین، LocalOllamaAdapter و DeterministicMockModelAdapter
│
├── tests/                                          # مجموعه آزمون‌های واحد (Unit Tests)
│   ├── test_phase1_domain_and_registry.py          # ۱۳ تست واحد جامع فاز ۱ (دامنه، موجودیت‌ها و رجیستری لایه/گره)
│   └── test_phase2_models_and_prompts.py           # ۱۳ تست واحد جامع فاز ۲ (رجیستری مدل، ۱۲ پرامپت، آداپتور و داور)
│
├── scripts/                                        # اسکریپت‌های شبیه‌سازی مطالعه موردی و آزمایش خط پایه
│   ├── generate_phase1_case_study.py               # تولید رجیستری ۵ لایه‌ای و داده‌های نمونه فاز ۱
│   └── run_phase2_baseline_experiments.py          # اجرای آزمایش‌های خط پایه مدل‌ها و ۱۲ دسته پرامپت فاز ۲
│
├── sample_docs/                                    # اسناد نمونه و خروجی‌های JSON مطالعه موردی
│   ├── case_study_registry.json                    # خروجی JSON رجیستری ۵ لایه‌ای (۱۲ گره سازمانی)
│   ├── 01_technical_spec_rag_pipeline.md           # سند فنی نمونه خط لوله RAG و خردسازی
│   ├── 02_executor_profiles_and_ike_sample.json    # نمونه داده ساختاریافته مدل IKE، فراداده ۱۸ فیلدی و RFP
│   ├── 03_project_report_and_research_gaps.md      # گزارش نمونه پروژه، دستاوردها و شکاف‌های پژوهشی
│   └── phase2_experiment_results.json              # نتایج اجرای آزمایش‌های خط پایه فاز ۲
│
└── docs/                                           # مستندات رسمی و آکادمیک فازهای پروژه
    ├── Phase1/
    │   ├── 01_Domain_Discovery_Report.md           # گزارش کشف دامنه
    │   ├── 02_Ubiquitous_Language_Glossary.md      # واژه‌نامه زبان مشترک DDD
    │   ├── 03_Node_and_Layer_Registry_Template.md  # قالب اولیه رجیستری گره و لایه
    │   └── 04_Evaluation_Metrics_Baseline.md       # خط پایه شاخص‌ها و معیارهای ارزیابی
    └── Phase2/
        ├── 01_Offline_Model_Baseline_Report.md     # گزارش خط پایه مدل‌های آفلاین
        ├── 02_Model_Usage_Procedure.md             # روش استفاده از مدل‌های آفلاین و آداپتورها
        └── 03_Prompt_Testing_Plan.md               # برنامه آزمون و نسخه‌بندی ۱۲ دسته پرامپت
```

---

## ۳. نقشه راه و وضعیت پیشرفت ۱۶ فاز پروژه (بخش ۲۱ پروپوزال)

| فاز | عنوان فاز طبق سند پروپوزال | وضعیت | ماژول‌های کد و تست مرتبط | مستندات رسمی تحویلی |
| :---: | :--- | :---: | :--- | :--- |
| **فاز ۱** | **آغاز پروژه و کشف دامنه** | ✅ **تکمیل شده** (`v0.1-phase1`) | [`exceptions.py`](src/domain/exceptions.py)، [`value_objects.py`](src/domain/value_objects.py)، [`entities.py`](src/domain/entities.py)، [`layer_node_registry.py`](src/domain/layer_node_registry.py)، [`test_phase1_domain_and_registry.py`](tests/test_phase1_domain_and_registry.py) | 1. [گزارش کشف دامنه](docs/Phase1/01_Domain_Discovery_Report.md)<br>2. [واژه‌نامه زبان مشترک DDD](docs/Phase1/02_Ubiquitous_Language_Glossary.md)<br>3. [قالب رجیستری گره و لایه](docs/Phase1/03_Node_and_Layer_Registry_Template.md)<br>4. [خط پایه شاخص‌های ارزیابی](docs/Phase1/04_Evaluation_Metrics_Baseline.md) |
| **فاز ۲** | **آشنایی با مدل‌های آفلاین و خط پایه فنی** | ✅ **تکمیل شده** (`v0.2-phase2`) | [`model_registry.py`](src/domain/model_registry.py)، [`prompt_registry.py`](src/domain/prompt_registry.py)، [`offline_model_adapter.py`](src/infrastructure/offline_model_adapter.py)، [`test_phase2_models_and_prompts.py`](tests/test_phase2_models_and_prompts.py) | 5. [گزارش خط پایه مدل آفلاین](docs/Phase2/01_Offline_Model_Baseline_Report.md)<br>6. [روش استفاده از مدل](docs/Phase2/02_Model_Usage_Procedure.md)<br>7. [برنامه آزمون پرامپت](docs/Phase2/03_Prompt_Testing_Plan.md) |
| **فاز ۳** | معماری DDD و مدل دامنه اصلی (Aggregateها، Repositoryها و سرویس‌های دامنه) | ⏳ در نوبت اجرا | پایه دامنه در `src/domain/` آماده شده است | سند معماری DDD، مشخصات مدل دامنه، برنامه آزمون واحد |
| **فاز ۴** | زیربنای ورود سند (`DOC`, `DOCX`, `PDF` و Adapterهای استخراج متن) | ⏳ در نوبت اجرا | — | نمونه اولیه ورود دانش، Schema فراداده، مشخصات خردسازی |
| **فاز ۵** | نسخه‌بندی، خردسازی و تشخیص تکرار و هم‌پوشانی دانش | ⏳ در نوبت اجرا | — | ماژول چرخه عمر دانش، ماژول تشخیص تکرار و هم‌پوشانی |
| **فاز ۶** | ماژول گره، لایه و سلسله‌مراتب سازمانی پیشرفته | ⏳ در نوبت اجرا | هسته اولیه در [`layer_node_registry.py`](src/domain/layer_node_registry.py) | گردش‌کارهای عضویت لایه و نیازمندی‌های انتقال ویژه هر لایه |
| **فاز ۷** | مدل‌سازی مسئله، مقایسه مسئله و تجزیه به زیرمسئله | ⏳ در نوبت اجرا | هسته اولیه در [`entities.py`](src/domain/entities.py) | ماژول مدیریت مسئله، مدل زیرمسئله و منطق مقایسه مسئله |
| **فاز ۸** | هسته بازیابی RAG، مونتاژ زمینه و پرسش و پاسخ مقید به گره | ⏳ در نوبت اجرا | — | هسته RAG آفلاین، گردش‌کار پرسش و پاسخ |
| **فاز ۹** | تحلیل مجری و روابط سه‌جانبه `Issue-Knowledge-Executor` | ⏳ در نوبت اجرا | — | ماژول مدیریت مجری، ماژول رابطه `IKE` و تولید خلاصه روابط |
| **فاز ۱۰** | سازوکار `LLM-as-a-Judge` و کنترل کیفیت ۲۰ نمره‌ای | ⏳ در نوبت اجرا | پایه در [`value_objects.py`](src/domain/value_objects.py) و [`prompt_registry.py`](src/domain/prompt_registry.py) | ماژول امتیازدهی داور، Parser امتیاز و مدیریت حد آستانه ۱۸ از ۲۰ |
| **فاز ۱۱** | پردازش چرخه‌ای گره، تنظیم سطح جزئیات و انتقال آفلاین به بالا (`CSV`/DB) | ⏳ در نوبت اجرا | — | ماژول پردازش گره، بسته انتقال، خروجی `CSV` و تحویل آخرین نسخه |
| **فاز ۱۲** | ردیابی پروژه، پژوهش، دستاورد، اولویت‌بندی مسئله و شکاف پژوهشی | ⏳ در نوبت اجرا | — | ماژول دانش پروژه، ماژول شکاف پژوهشی و اولویت‌بندی مسائل کلیدی |
| **فاز ۱۳** | تولید خودکار `RFP`، مدیریت و ارزیابی پیشنهاد (`Proposal`) و تولید بازخورد | ⏳ در نوبت اجرا | هسته اولیه در [`entities.py`](src/domain/entities.py) | ماژول تولید RFP، ارزیابی پیشنهاد و تولید گزارش بازخورد |
| **فاز ۱۴** | آزمون پرامپت، تنظیم مدل و ارزیابی مجموعه داده نمونه | ⏳ در نوبت اجرا | — | گزارش ارزیابی مدل و پرامپت، مجموعه پرامپت کالیبره‌شده |
| **فاز ۱۵** | تکمیل آزمون واحد و سخت‌سازی فنی (Hardening) | ⏳ در نوبت اجرا | — | گزارش کامل آزمون واحد، نسخه آزمایشی نهایی هسته نرم‌افزاری |
| **فاز ۱۶** | جمع‌بندی نهایی، مستندسازی جامع و ارائه | ⏳ در نوبت اجرا | — | نمونه نهایی هسته نرم‌افزاری، گزارش جمع‌بندی و محتوای آموزشی |

---

## ۴. راهنمای نصب و اجرای آزمون‌های واحد و اسکریپت‌ها

### ۴.۱ آماده‌سازی محیط مجازی (در صورت نیاز)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

### ۴.۲ اجرای اسکریپت‌های مطالعه موردی فاز ۱ و آزمایش‌های خط پایه فاز ۲
1. **تولید رجیستری ۵ لایه‌ای و اسناد نمونه مطالعه موردی فاز ۱:**
   ```powershell
   python scripts/generate_phase1_case_study.py
   ```
   *(خروجی‌ها در پوشه [`sample_docs/`](sample_docs/) شامل [`case_study_registry.json`](sample_docs/case_study_registry.json) و [`02_executor_profiles_and_ike_sample.json`](sample_docs/02_executor_profiles_and_ike_sample.json) ذخیره می‌شوند).*

2. **اجرای آزمایش‌های خط پایه مدل‌های آفلاین و ۱۲ دسته پرامپت فاز ۲:**
   ```powershell
   python scripts/run_phase2_baseline_experiments.py
   ```
   *(نتایج کامل آزمایش در [`sample_docs/phase2_experiment_results.json`](sample_docs/phase2_experiment_results.json) ذخیره می‌گردد).*

### ۴.۳ اجرای کامل مجموعه آزمون‌های واحد (`pytest`)
```powershell
pytest -v
```
تمامی **۲۶ آزمون واحد** (۱۳ آزمون فاز ۱ در [`tests/test_phase1_domain_and_registry.py`](tests/test_phase1_domain_and_registry.py) و ۱۳ آزمون فاز ۲ در [`tests/test_phase2_models_and_prompts.py`](tests/test_phase2_models_and_prompts.py)) به صورت ۱۰۰٪ آفلاین و قطعی در کمتر از `0.2` ثانیه اجرا و پاس می‌شوند.
