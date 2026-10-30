# Explainable Hospital Readmission Prediction for Diabetic Patients

ทำนายการกลับเข้ารักษาซ้ำของผู้ป่วยเบาหวาน จากประวัติการเข้ารับการรักษา พร้อม Clinical Decision Support Dashboard

โปรเจกต์สัมมนา สาขาปัญญาประดิษฐ์ (AI) — 67070271 ยวิษฐา ทองเกลี้ยง · 67070295 อมลวรรณ แก้วสอาด
อาจารย์ที่ปรึกษา: ผศ.ดร. พัฒนพงษ์ ฉันทมิตรโอภาส

---

## สารบัญ
1. [โปรเจกต์นี้ทำอะไร](#1-โปรเจกต์นี้ทำอะไร)
2. [เริ่มใช้งานเร็วที่สุด: รัน dashboard](#2-เริ่มใช้งานเร็วที่สุด-รัน-dashboard)
3. [ติดตั้ง](#3-ติดตั้ง)
4. [โครงสร้างโฟลเดอร์](#4-โครงสร้างโฟลเดอร์)
5. [รันการทดลองซ้ำ (notebook 11–15)](#5-รันการทดลองซ้ำ-notebook-1115)
6. [ใช้โมเดลในโค้ดของคุณเอง](#6-ใช้โมเดลในโค้ดของคุณเอง)
7. [วิธีใช้ dashboard](#7-วิธีใช้-dashboard)
8. [ผลลัพธ์](#8-ผลลัพธ์)
9. [ข้อจำกัด](#9-ข้อจำกัด)
10. [แก้ปัญหาที่พบบ่อย](#10-แก้ปัญหาที่พบบ่อย)
11. [อ้างอิง](#11-อ้างอิง)

---

## 1. โปรเจกต์นี้ทำอะไร

- **ข้อมูล**: [Diabetes 130-US Hospitals for Years 1999–2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) (UCI) 101,766 encounter → **99,340** หลังทำความสะอาด
- **ทำนาย 2 นิยาม**
  - `30d` — กลับมานอนโรงพยาบาลภายใน 30 วัน (นิยามมาตรฐาน HRRP) base rate 11.4%
  - `any` — กลับมารักษาซ้ำเมื่อไรก็ได้ (`<30` หรือ `>30`) base rate 46.5%
- **โมเดลสุดท้าย**: Blend ของ **LSTM** (อ่านประวัติการ admit 5 ครั้งล่าสุด) กับ **XGBoost + ตัวแปรประวัติ** แล้ว calibrate ด้วย isotonic regression และแบ่งความเสี่ยง 3 ระดับ (สูง 20% / ปานกลาง 30% / ต่ำ 50%)
- **Dashboard** (Streamlit): ทะเบียนผู้ป่วย บันทึกการ admit ประเมินความเสี่ยงทั้ง 2 นิยาม อธิบายปัจจัยด้วย SHAP และรายชื่อผู้ป่วยที่ต้องติดตามหลังจำหน่าย

ข้อมูลทั้งหมดแบ่งตามผู้ป่วย (`patient_nbr`) ผู้ป่วย 1 คนอยู่ได้ชุดเดียว เพื่อกัน data leakage

---

## 2. เริ่มใช้งานเร็วที่สุด: รัน dashboard

repo นี้มีโมเดลที่เทรนแล้ว (`models/final_30d/`, `models/final_any/`) จึงรัน dashboard ได้ทันทีโดยไม่ต้องเทรนใหม่

```bash
git clone https://github.com/Yrolls/Readmission.git
cd Readmission
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r dashboard/requirements.txt
cd dashboard
streamlit run dashboard_app.py
```

เปิดเบราว์เซอร์ที่ `http://localhost:8501` แถบด้านซ้าย (sidebar) ต้องขึ้นกล่องสีเขียว **"โมเดลพร้อมใช้งาน"**

> Dashboard เริ่มต้นแบบว่าง ไม่มีข้อมูลผู้ป่วย ต้องลงทะเบียนผู้ป่วยเองก่อน (ดูหัวข้อ 7) ข้อมูลที่กรอกเก็บไว้ในไฟล์ `dashboard/patients.db` บนเครื่องของคุณเท่านั้น

---

## 3. ติดตั้ง

**ต้องมี**: Python 3.11 ขึ้นไป (พัฒนาและทดสอบบน Python 3.13.1, Windows)

| ใช้ทำอะไร | คำสั่ง |
|---|---|
| รัน dashboard อย่างเดียว | `pip install -r dashboard/requirements.txt` |
| รัน notebook ทั้งหมด | `pip install -r requirements.txt` |

`requirements.txt` ที่ root ล็อกเวอร์ชันทุก package ที่ใช้ตอนทำงาน (pip freeze) package หลัก:

| Package | เวอร์ชัน | ใช้ใน |
|---|---|---|
| numpy / pandas / scikit-learn | 2.4.6 / 2.3.3 / 1.9.0 | ทุกส่วน |
| xgboost | 3.3.0 | โมเดลหลัก, survival (AFT), SHAP |
| torch | 2.13.0 | LSTM |
| streamlit | 1.60.0 | dashboard |
| shap / lifelines / imbalanced-learn / sdv | 0.52.0 / 0.30.3 / 0.14.2 / 1.38.1 | notebook 06–12 |

> `requirements.txt` ถูกบันทึกจาก PowerShell เป็นไฟล์ UTF-16 ซึ่ง pip อ่านได้ปกติ แต่ถ้าเครื่องมืออื่นอ่านไม่ออก ให้แปลงเป็น UTF-8 ก่อน
> torch ขนาดใหญ่ ถ้าไม่มี GPU ติดตั้งรุ่น CPU ได้: `pip install torch --index-url https://download.pytorch.org/whl/cpu`

---

## 4. โครงสร้างโฟลเดอร์

```
Readmission/
├── data/
│   ├── diabetic_data.csv          ← ข้อมูลดิบจาก UCI (ใช้ไฟล์นี้ไฟล์เดียวใน notebook 11–15)
│   └── IDS_mapping.csv            ← ความหมายของรหัส admission/discharge
├── notebook/                      ← รัน notebook จากโฟลเดอร์นี้ (path ทั้งหมดเป็นแบบ ../)
│   ├── 01_… – 10_…ipynb           ← รุ่นแรก (Stacking) เก็บไว้เป็นประวัติการทดลอง
│   ├── 11_… – 15_…ipynb           ← pipeline ปัจจุบัน
│   ├── readmission_features.py    ← cleaning + feature engineering ที่ทุก notebook ใช้ร่วมกัน
│   └── export_dashboard_artifacts.py
├── outputs/                       ← ผลของแต่ละ notebook (ตาราง CSV, รูป, split_assignment.csv)
├── models/
│   ├── final_30d/                 ← โมเดลหลัก นิยาม 30 วัน
│   ├── final_any/                 ← โมเดลหลัก นิยาม any
│   └── dashboard_artifacts.json   ← ข้อมูลที่ dashboard ต้องใช้
└── dashboard/
    ├── dashboard_app.py           ← หน้าเว็บ (Streamlit)
    ├── readmission_registry.py    ← ฐานข้อมูล SQLite + RiskEngine (คำนวณความเสี่ยง)
    ├── readmission_features.py    ← สำเนาเดียวกับใน notebook/
    └── requirements.txt
```

ในแต่ละโฟลเดอร์ `models/final_*` มี:

| ไฟล์ | เนื้อหา |
|---|---|
| `xgb_history.json` | XGBoost + ตัวแปรประวัติ |
| `lstm_seed0.pt` … `lstm_seed4.pt` | LSTM 5 seed (นำค่าทำนายมาเฉลี่ยกัน) |
| `final_bundle.pkl` | น้ำหนัก blend, isotonic calibrator, จุดตัด risk tier, ค่า mean/SD สำหรับ normalize, รายชื่อ feature |

**ไฟล์ที่ไม่ได้อยู่ใน repo** (ใหญ่เกินไป ดู `.gitignore`): `data/*.pkl`, `models/*.pkl` ระดับบนสุด (โมเดลรุ่นเก่าจาก notebook 04–11) และ `dashboard/patients.db` — สร้างใหม่ได้โดยรัน notebook ตามหัวข้อ 5

> ⚠️ `readmission_features.py` มี 2 สำเนา (ใน `notebook/` และ `dashboard/`) ถ้าแก้ไฟล์หนึ่ง ต้องคัดลอกไปอีกที่ แล้วรัน `export_dashboard_artifacts.py` ใหม่

---

## 5. รันการทดลองซ้ำ (notebook 11–15)

รันจากโฟลเดอร์ `notebook/` ตามลำดับนี้ notebook ส่วนใหญ่มีตัวแปร `LABEL_MODE` ในเซลล์แรก

| ลำดับ | Notebook | `LABEL_MODE` | ได้อะไร | ใช้เวลาโดยประมาณ* |
|---|---|---|---|---|
| 1 | `11_relabel_any_readmission.ipynb` | `any` | `outputs/split_assignment.csv`, `outputs/11_any/`, `models/model_11_any.pkl` | 5–15 นาที |
| 2 | `11_relabel_30d_readmission.ipynb` | `30d` | `outputs/11_30d/`, `models/model_11_30d.pkl` | 5–15 นาที |
| 3 | `12_feature_analysis.ipynb` | `any` | `outputs/12_features_any/` (odds ratio, SHAP) | 10–20 นาที |
| 4 | `13_survival_models.ipynb` | — | `outputs/13_survival/` | 10–20 นาที |
| 5 | `13b_control_experiment.ipynb` | — | `outputs/13b_control/` | ไม่กี่นาที |
| 6 | `15_final_model_any.ipynb` / `15_final_model_30d.ipynb` | `any` / `30d` | `models/final_*/`, `outputs/15_final_*/` | 15–40 นาทีต่อไฟล์ |
| 7 | `14_lstm_sequence_any.ipynb` / `14_lstm_sequence_30d.ipynb` | `any` / `30d` | `outputs/14_lstm_*/` (ablation) | 20–60 นาทีต่อไฟล์ |
| 8 | `python export_dashboard_artifacts.py` | — | `models/dashboard_artifacts.json` | 1–2 นาที |

\*บน CPU ของโน้ตบุ๊ก ขึ้นกับเครื่อง

- **ข้อ 1 ต้องรันก่อนเสมอ** เพราะสร้างไฟล์ `split_assignment.csv` ที่ทุก notebook ใช้แบ่ง train/val/test ชุดเดียวกัน (ไฟล์นี้อยู่ใน repo แล้ว ถ้าไม่แก้การแบ่งข้อมูลก็ใช้ของเดิมได้)
- notebook 14 ตรวจว่าผลของตัวเองตรงกับ notebook 15 จึงควรรัน 15 ก่อน
- ถ้ารันข้อ 8 สำเร็จ ต้องเห็นข้อความ:
  ```
  ✓ featurize_raw ตรงกับ load_prepared ทุกคอลัมน์
  ✓ feature ของ XGBoost และ LSTM ตรงกับตอนเทรน (any)
  ✓ feature ของ XGBoost และ LSTM ตรงกับตอนเทรน (30d)
  ```
- ทุก notebook ตั้ง `RANDOM_STATE = 42` ผลควรตรงกับในตารางหัวข้อ 8 แต่ LSTM อาจต่างในทศนิยมตำแหน่งที่ 4 ถ้ารันบนเครื่องหรือเวอร์ชัน torch ต่างกัน

**Notebook 01–10** เป็นการทดลองรุ่นแรก (Stacking RF + XGBoost, เปรียบเทียบ SMOTE, Charlson/LACE, bootstrap CI) บางไฟล์ต้องใช้ไฟล์ `.pkl` ที่ไม่ได้อยู่ใน repo ต้องรันตามลำดับเลขเพื่อสร้างใหม่ ไม่จำเป็นสำหรับผลสุดท้าย

---

## 6. ใช้โมเดลในโค้ดของคุณเอง

ฟังก์ชันใน `readmission_features.py` ใช้ได้โดยตรง:

```python
from readmission_features import load_prepared, attach_split, build_history_features

# X = feature ที่ encode แล้ว, y = label, groups = patient_nbr, meta = encounter_id + label ดิบ
X, y, groups, meta, readable = load_prepared("../data/diabetic_data.csv", label_mode="30d")
split = attach_split(meta, "../outputs/split_assignment.csv")   # 'train' / 'val' / 'test' ต่อแถว
hist = build_history_features(meta)                              # ตัวแปรประวัติ 5 ตัว
```

ทำนายความเสี่ยงของผู้ป่วยรายใหม่ ให้ใช้ `RiskEngine` ใน `dashboard/readmission_registry.py` (โหลดโมเดลทั้ง 2 นิยาม คำนวณ feature จากประวัติ แล้วคืนค่าความน่าจะเป็น ระดับความเสี่ยง และปัจจัย SHAP) ดูตัวอย่างการเรียกใช้ใน `dashboard_app.py` ฟังก์ชัน `assess_and_save`

ข้อมูลที่ส่งเข้าต้องมีคอลัมน์แบบเดียวกับ `diabetic_data.csv` และใช้ได้เฉพาะรหัสที่โมเดลเคยเห็นตอนเทรน ค่าใหม่จะเกิด `ValueError`

---

## 7. วิธีใช้ dashboard

1. **ลงทะเบียนผู้ป่วย** (sidebar → ➕): HN, ชื่อ, ปีเกิด, เพศ, เชื้อชาติ
2. **เปิดการ admit**: กรอกข้อมูลแรกรับ (ประเภทการเข้ารักษา, แผนก, จำนวน lab/ยา/การวินิจฉัย, ประวัติ 12 เดือน, กลุ่มโรค, ผลตรวจ, ยาเบาหวาน, วันนอนที่คาดไว้) → ระบบแสดงผลการประเมินทันที
3. **อัปเดตอาการ** (หน้าผู้ป่วย → แท็บ 📝): แก้ข้อมูลเมื่อมีผลตรวจใหม่ ระบบประเมินใหม่และเก็บเป็นกราฟแนวโน้ม
4. **จำหน่าย** (แท็บ 🏁): ใส่วันที่และปลายทาง → ผลครั้งสุดท้ายใช้วางแผนติดตาม
5. **ผู้ป่วยกลับมาอีก**: เปิด admit ใหม่ ระบบคำนวณเองว่ากลับมากี่วันหลังจำหน่าย และใช้ประวัติทั้งหมดในการประเมิน

| หน้า | แสดงอะไร |
|---|---|
| 📋 รายชื่อผู้ป่วย | ผู้ป่วยทุกคน สถานะ ความเสี่ยงล่าสุด ค้นหาด้วย HN/ชื่อ |
| 🔔 ผู้ป่วยที่ต้องติดตาม | ผู้ที่จำหน่ายภายใน 30 วันและยังไม่กลับมา เรียงตามความเร่งด่วน |
| 👤 หน้าผู้ป่วย | การ admit ปัจจุบัน ประวัติทุกครั้ง กราฟแนวโน้ม แก้ไข/ลบข้อมูล |
| 📊 ผลการประเมิน | ความเสี่ยง 2 นิยาม, ปัจจัยที่มีผล (แดง = เพิ่ม, เขียว = ลด), แนวทางที่ควรพิจารณา (อ้างอิง AHRQ IDEAL, CMS, ADA) |

ล้างข้อมูลผู้ป่วยทั้งหมด: ปิด streamlit แล้วลบไฟล์ `dashboard/patients.db`

> Dashboard นี้เป็น proof-of-concept สำหรับการศึกษา **ไม่ใช่เครื่องมือวินิจฉัย** และห้ามใส่ข้อมูลผู้ป่วยจริง เพราะไม่มีระบบ login และไม่ได้ออกแบบตาม PDPA

---

## 8. ผลลัพธ์

ผลบน test set (19,773 encounter, ผู้ป่วยที่โมเดลไม่เคยเห็น) CI 95% จาก cluster bootstrap ตามผู้ป่วย

| | นิยาม 30 วัน | นิยาม any |
|---|---|---|
| ROC-AUC | **0.692** (0.679–0.704) | **0.703** (0.694–0.711) |
| PR-AUC (base rate) | 0.256 (0.114) | 0.673 (0.465) |
| ดีกว่า Stacking รุ่นแรก | +0.017 ROC-AUC | +0.023 ROC-AUC |
| กลุ่มเสี่ยงสูง (20% แรก) กลับมาจริง | 24.1% (2.1 เท่าของค่าเฉลี่ย) | 73.4% |
| สัดส่วนผู้ที่กลับมาทั้งหมดที่อยู่ในกลุ่มเสี่ยงสูง | 41% | 31% |
| น้ำหนัก LSTM ใน blend | 0.5 | 0.7 |

ข้อค้นพบหลัก:
- ทุกโมเดลอยู่ในช่วง ROC-AUC 0.67–0.70 ข้อมูลมีแต่สิ่งที่เกิดในโรงพยาบาล ความแม่นยำจึงมีเพดาน
- ปัจจัยที่สัมพันธ์มากที่สุดคือประวัติการใช้บริการ: เคย admit ต่อครั้ง OR 1.51, เคยมา ER ต่อครั้ง OR 1.29
- ปัจจัยระยะสั้นกับระยะยาวต่างกัน: ส่งต่อสถานพยาบาล HR 1.62 ใน 30 วันแรก แต่ 0.96 ระยะยาว
- ตัวแปรประวัติจากการมองข้อมูลเป็นลำดับ (LSTM) เป็นสิ่งเดียวที่ขยับผลได้อย่างมีนัยสำคัญ

ตารางผลทั้งหมดอยู่ใน `outputs/15_final_30d/` และ `outputs/15_final_any/`

---

## 9. ข้อจำกัด

- **ไม่มีข้อมูลหลังจำหน่าย** (การกินยา การมาตามนัด) ใช้จัดลำดับความสำคัญการติดตามได้ แต่ไม่แม่นพอจะใช้แทนการประเมินของแพทย์
- **ไม่มีวันที่จริง**: เรียงลำดับ encounter ด้วย `encounter_id` (สมมติว่าเรียงตามเวลา) และ label มีแค่ 3 ช่วง (`<30`, `>30`, `NO`)
- **ความเป็นธรรม**: ผู้ป่วยอายุ 90+ ได้ความแม่นยำต่ำสุด (ROC-AUC 0.553 ในนิยาม 30 วัน) dashboard จึงแสดงคำเตือนตั้งแต่อายุ 80 ปี
- **ยังไม่มี external validation**: เทรนและทดสอบด้วยข้อมูลโรงพยาบาลสหรัฐฯ ปี 1999–2008 (ICD-9) ชุดเดียว การใช้ในบริบทอื่นต้องทดสอบใหม่
- **คำอธิบาย SHAP เป็นค่าประมาณ**: คำนวณจากส่วน XGBoost ของ blend เท่านั้น และบอกว่าโมเดลใช้อะไร ไม่ได้บอกเหตุและผลทางการแพทย์
- **Dashboard**: ใช้บนคอมพิวเตอร์เท่านั้น (ยังไม่รองรับมือถือ), ฐานข้อมูลอยู่บนเครื่องเดียว, ไม่เชื่อมต่อระบบ HIS

---

## 10. แก้ปัญหาที่พบบ่อย

| อาการ | วิธีแก้ |
|---|---|
| sidebar แดง "โหลดโมเดลไม่สำเร็จ" | กด "รายละเอียด" ดู path ที่หาไม่เจอ ตรวจว่ามี `models/final_30d/`, `models/final_any/` และ `models/dashboard_artifacts.json` แล้ว Ctrl+C รัน streamlit ใหม่ |
| `File does not exist: dashboard_app.py` | ต้อง `cd dashboard` ก่อนรัน |
| ไม่เห็นเมนู | sidebar ถูกพับ กด **»** มุมซ้ายบน |
| `ModuleNotFoundError: readmission_features` | รัน notebook จากโฟลเดอร์ `notebook/` (ไฟล์ .py ต้องอยู่โฟลเดอร์เดียวกัน) |
| `encounter บางตัวไม่มีใน split_assignment.csv` | รัน `11_relabel_any_readmission.ipynb` ใหม่ |
| VS Code เปลี่ยน import เป็น `notebook.dashboard.readmission_features` | แก้กลับเป็น `from readmission_features import ...` |
| `ค่า {...} ในคอลัมน์ ... ไม่เคยพบตอนเทรน` | ข้อมูลที่ส่งเข้ามีรหัสที่โมเดลไม่รู้จัก ใช้ได้เฉพาะค่าที่อยู่ใน `dashboard_artifacts.json` |

---

## 11. อ้างอิง

- Strack, B. et al. (2014). Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records. *BioMed Research International*.
- Lu, H. & Uddin, S. (2022). Explainable stacking-based model for predicting hospital readmission for diabetic patients. *Information*, 13(9), 436.
- Prentice, R. L. & Gloeckler, L. A. (1978). Regression analysis of grouped survival data with application to breast cancer data. *Biometrics*, 34(1), 57–67.
- AHRQ. IDEAL Discharge Planning. Agency for Healthcare Research and Quality.
- American Diabetes Association. Standards of Care in Diabetes — Diabetes Self-Management Education and Support (DSMES).
