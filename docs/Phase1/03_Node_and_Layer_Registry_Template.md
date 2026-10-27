# قالب اولیه رجیستری گره و لایه سازمانی (Node and Layer Registry Template) — فاز ۱

**ماژول مرجع در کد:** [`src/domain/layer_node_registry.py`](file:///d:/offline-hierarchical-rag-core/src/domain/layer_node_registry.py)  
**خروجی مطالعه موردی:** [`sample_docs/case_study_registry.json`](file:///d:/offline-hierarchical-rag-core/sample_docs/case_study_registry.json)

---

## ۱. هدف رجیستری لایه و گره
طبق بخش ۶.۱ پروپوزال، رجیستری لایه و گره (`LayerNodeRegistry`) وظیفه تعیین جایگاه هر شرکت، سازمان، زیرمجموعه، واحد یا تیم پروژه را در سلسله‌مراتب ۵ لایه‌ای سامانه بر عهده دارد. این رجیستری موارد زیر را به‌صورت قطعی تعریف و اعتبارسنجی می‌کند:
1. شماره و نام رسمی هر لایه (از لایه ۰ تا لایه ۴).
2. شرکت‌ها، سازمان‌ها، زیرمجموعه‌ها، واحدها، پروژه‌ها یا گره‌های عضو هر لایه.
3. روابط والد و فرزند میان گره‌ها.
4. حوزه فعالیت هر گره (`domain_of_activity`).
5. قواعد گزارش‌دهی هر گره (`reporting_rules`).
6. قالب و سطح جزئیات مورد نیاز برای انتقال دانش به لایه بالاتر (`required_transfer_detail_level`).
7. محدودیت‌های محرمانگی یا قابلیت مشاهده (`confidentiality_constraints`).
8. وضعیت اینکه آیا ساختار داخلی گره ردیابی می‌شود یا خارج از مرز سامانه تلقی می‌گردد (`is_leaf_boundary`).

---

## ۲. جدول لایه‌های پنج‌گانه پیشنهادی (بخش ۶.۲ پروپوزال)

| شماره لایه (`LayerIndex`) | نام رسمی لایه | نقش نمونه در ساختار سازمانی | اطلاعات قابل ردیابی و دریافت در این لایه | سطح جزئیات پیش‌فرض |
| :---: | :--- | :--- | :--- | :---: |
| **`0`** | **لایه صفر: لایه راهبردی** | مرجع عالی تصمیم‌گیری یا مالک کلان ساختار سازمانی در سناریوی نمونه. | دریافت خلاصه راهبردی دانش، مسائل کلیدی حل‌نشده، وضعیت RFP، شکاف‌های پژوهشی و اولویت‌های بین‌گرهی. | `EXECUTIVE_HIGH` (اجرایی عالی) |
| **`1`** | **لایه یک: سازمان‌های مادر** | شرکت‌های اصلی، سازمان‌های اصلی، ستادها یا معاونت‌های راهبردی. | دریافت خلاصه مدیریتی، ریسک‌ها و دستاوردهای زیرمجموعه‌ها. | `MANAGERIAL` (مدیریتی) |
| **`2`** | **لایه دو: زیرمجموعه‌ها و واحدهای تخصصی** | شرکت‌های تابعه، مراکز پژوهشی، واحدهای مهندسی، واحدهای عملیاتی یا دفاتر برنامه. | نگهداری دانش تفصیلی و آماده‌سازی گزارش‌های کنترل‌شده به لایه بالاتر. | `TECHNICAL_SUMMARY` (خلاصه فنی) |
| **`3`** | **لایه سه: پروژه‌ها، تیم‌ها و گره‌های اجرایی** | تیم‌های پروژه، واحدهای پیاده‌سازی، گروه‌های پژوهشی، تیم‌های میدانی و گروه‌های مجری. | نگهداری جزئی‌ترین دانش فنی، عملیاتی، نیروی انسانی و مسئله‌محور. | `TECHNICAL_DETAILED` (فنی تفصیلی) |
| **`4`** | **لایه چهار: منابع داخلی دانش اختیاری** | کتابخانه‌ها، مجموعه اسناد، سوابق پرسنلی، آرشیو پروژه یا مخازن محلی. | در صورت نیاز مطالعه موردی، به‌عنوان منبع دانش مستقل مدل می‌شوند. | `TECHNICAL_DETAILED` (فنی تفصیلی) |

---

## ۳. سطوح چهارگانه جزئیات انتقال دانش (جدول ۶.۴ پروپوزال)

| کد سطح (`TransferDetailLevel`) | عنوان فارسی | مخاطب هدف | خروجی معمول در بسته انتقال (`TransferPackage`) |
| :--- | :--- | :--- | :--- |
| `EXECUTIVE_HIGH` | **اجرایی عالی** | تصمیم‌گیران ارشد (لایه ۰) | دستاوردهای کلیدی، ریسک‌های اصلی، مسائل حل‌نشده، وابستگی‌های بحرانی و پیشنهادهای راهبردی. |
| `MANAGERIAL` | **مدیریتی** | مدیران سازمان مادر (لایه ۱) | وضعیت پروژه، وضعیت مسئله، شکاف پژوهشی، تصمیمات مورد نیاز، نیاز منابع و وضعیت RFP یا پیشنهادها. |
| `TECHNICAL_SUMMARY` | **خلاصه فنی** | رهبران فنی و ناظران برنامه (لایه ۲) | روش‌ها، نتایج پژوهشی، اسناد مرتبط، محدودیت‌ها، گزینه‌های راه‌حل و توانمندی مجریان. |
| `TECHNICAL_DETAILED` | **فنی تفصیلی** | متخصصان و تیم‌های لایه پایین‌تر (لایه ۳ و ۴) | جزئیات کامل مسئله، ارجاع اسناد، شکست زیرمسئله‌ها، نگاشت مجری، شواهد منبع و یادداشت‌های آزمون مدل و پرامپت. |

---

## ۴. قوانین ناوردایی (Domain Invariants) و مدیریت مرز گره در `LayerNodeRegistry`

1. **قاعده تعلق یکتا به لایه (Single Primary Layer Invariant):**
   هر گره پیکربندی‌شده دقیقاً یک لایه اصلی معتبر (`0 <= layer_index <= 4`) دارد. تلاش برای ثبت یک گره در دو لایه متفاوت بلافاصله با استثنای [`InvalidLayerHierarchyError`](file:///d:/offline-hierarchical-rag-core/src/domain/exceptions.py) متوقف می‌شود.
2. **قاعده والد و فرزند میان لایه‌ها (Parent-Child Hierarchy Rule):**
   - گره‌های **لایه ۰ (راهبردی)** در رأس هرم قرار دارند و **نباید** دارای والد باشند (`parent_node_id = None`).
   - گره‌های **لایه‌های ۱ تا ۴** **الزاماً** باید دارای یک گره والد ثبت‌شده در رجیستری باشند و شماره لایه والد باید اکیداً کوچک‌تر (یعنی در سلسله‌مراتب، بالاتر) از شماره لایه فرزند باشد (`parent.layer_index.value < child.layer_index.value`).
3. **قاعده مرز گره نهایی (`Leaf Node Boundary Rule`):**
   - اگر گره‌ای با `is_leaf_boundary = True` علامت‌گذاری شود، زیرساختار داخلی آن خارج از مرز سامانه تلقی می‌گردد.
   - هرگونه تلاش برای ثبت گره فرزند زیرمجموعه برای گره‌ای که `is_leaf_boundary = True` دارد، خطای اختصاصی [`LeafNodeBoundaryViolationError`](file:///d:/offline-hierarchical-rag-core/src/domain/exceptions.py) صادر می‌کند.
   - همچنین اگر گره‌ای از قبل دارای فرزند ثبت‌شده باشد، تغییر وضعیت آن به `is_leaf_boundary = True` ممنوع بوده و همان خطای `LeafNodeBoundaryViolationError` را صادر می‌نماید.

---

## ۵. قالب استاندارد JSON رجیستری گره و لایه

در ادامه، ساختار استاندارد JSON تولیدشده توسط متد `LayerNodeRegistry.to_json()` نمایش داده شده است:

```json
{
  "schema_version": "1.0.0",
  "total_layers": 5,
  "total_nodes": 12,
  "layers": [
    {
      "layer_index": 0,
      "name": "لایه صفر: لایه راهبردی",
      "role_description": "مرجع عالی تصمیم‌گیری یا مالک کلان ساختار سازمانی در سناریوی نمونه",
      "trackable_info_summary": "دریافت خلاصه راهبردی دانش، مسائل کلیدی حل‌نشده، وضعیت RFP، شکاف‌های پژوهشی و اولویت‌های بین‌گرهی",
      "default_detail_level": "EXECUTIVE_HIGH",
      "member_node_ids": [
        "L0-STRATEGIC-BOARD"
      ]
    },
    {
      "layer_index": 1,
      "name": "لایه یک: سازمان‌های مادر",
      "role_description": "شرکت‌های اصلی، سازمان‌های اصلی، ستادها یا معاونت‌های راهبردی",
      "trackable_info_summary": "دریافت خلاصه مدیریتی، ریسک‌ها و دستاوردهای زیرمجموعه‌ها",
      "default_detail_level": "MANAGERIAL",
      "member_node_ids": [
        "L1-AI-MOTHER-ORG",
        "L1-INFRA-MOTHER-ORG",
        "L1-EXTERNAL-AUDIT-ORG"
      ]
    }
  ],
  "nodes": [
    {
      "node_id": "L0-STRATEGIC-BOARD",
      "name": "شورای عالی راهبردی و هیئت‌مدیره هلدینگ فناوری",
      "layer_index": 0,
      "domain_of_activity": "سیاست‌گذاری کلان، حکمرانی دانش، تصویب RFPهای راهبردی و تخصیص کلان منابع",
      "parent_node_id": null,
      "is_leaf_boundary": false,
      "reporting_rules": {
        "mandatory_topics": [
          "strategic_achievements",
          "critical_risks",
          "unresolved_key_issues",
          "rfp_status",
          "inter_node_priorities"
        ],
        "excluded_topics": [
          "low_level_code_logs",
          "raw_chunk_traces"
        ],
        "min_judge_score": 18,
        "max_package_length_words": 1200,
        "preferred_output_format": "CSV_AND_DB"
      },
      "required_transfer_detail_level": "EXECUTIVE_HIGH",
      "confidentiality_constraints": [
        "PUBLIC",
        "INTERNAL",
        "CONFIDENTIAL",
        "RESTRICTED"
      ],
      "child_node_ids": [
        "L1-AI-MOTHER-ORG",
        "L1-INFRA-MOTHER-ORG",
        "L1-EXTERNAL-AUDIT-ORG"
      ]
    },
    {
      "node_id": "L1-EXTERNAL-AUDIT-ORG",
      "name": "مؤسسه مستقل ممیزی و استانداردسازی برون‌سازمانی",
      "layer_index": 1,
      "domain_of_activity": "ممیزی مستقل کیفیت و استانداردهای امنیتی (زیرساختار داخلی خارج از مرز مدل‌سازی)",
      "parent_node_id": "L0-STRATEGIC-BOARD",
      "is_leaf_boundary": true,
      "reporting_rules": {
        "mandatory_topics": [
          "compliance_certificates",
          "audit_findings"
        ],
        "min_judge_score": 18,
        "boundary_note": "گره نهایی - زیرساختار داخلی مدل نمی‌شود"
      },
      "required_transfer_detail_level": "EXECUTIVE_HIGH",
      "confidentiality_constraints": [
        "PUBLIC",
        "INTERNAL"
      ],
      "child_node_ids": []
    }
  ]
}
```
*(نسخه کامل با تمامی ۵ لایه و ۱۲ گره در فایل [`sample_docs/case_study_registry.json`](file:///d:/offline-hierarchical-rag-core/sample_docs/case_study_registry.json) تولید و ذخیره شده است).*
