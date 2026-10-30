"""
ระบบประเมินความเสี่ยงการกลับเข้ารักษาซ้ำในโรงพยาบาล
================================================
Dashboard สำหรับ capstone project — ใช้โมเดลจาก 07_final_calibrated_pipeline.ipynb

วิธีรัน:
    streamlit run dashboard_app.py

ก่อนรัน ต้องมีไฟล์โมเดลที่ save จาก notebook (joblib.dump ที่ cell สุดท้าย) อยู่ในโฟลเดอร์เดียวกัน
หรือแก้ MODEL_PATH ด้านล่างให้ตรงกับตำแหน่งไฟล์จริง

*** สิ่งที่ต้องตรวจสอบ/ปรับก่อนใช้งานจริง (มี TODO กำกับไว้ในโค้ด) ***
1. [แก้แล้ว] TODO-1: รายการ medical_specialty top-10 อัปเดตให้ตรงกับ value_counts จริงจาก dataset แล้ว
2. TODO-2: ข้อความ "แนวทางที่ควรพิจารณา" ต้องมี citation จาก clinical guideline จริงก่อนใช้งานจริง
   [แก้แล้ว] ใส่ citation อ้างอิง AHRQ IDEAL Discharge Planning + CMS Care Coordination Best Practices แล้ว
3. TODO-3: ค่า default ของ feature ที่ dashboard ไม่ได้เก็บจาก user โดยตรง
   (เช่น visit_number, prior_change_count) ตั้งเป็นค่าสมมติสำหรับ "ผู้ป่วยรายใหม่" — ควรทวนกับอาจารย์/ข้อมูลจริง
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

# ============================================================
# CONFIG
# ============================================================
MODEL_PATH = "final_calibrated_model_v1.pkl"

# Base rate ของ dataset จริง (จาก 07_final_calibrated_pipeline.ipynb: df['readmitted'].value_counts(normalize=True))
DATASET_BASE_RATE = 0.1135

# Threshold ที่คำนวณจริงจาก 07_final_calibrated_pipeline.ipynb
THRESHOLD_LOW_MED = 0.084   # F2-optimal — เส้นแบ่ง Low / Medium
THRESHOLD_MED_HIGH = 0.150  # Cost-optimal — เส้นแบ่ง Medium / High (threshold หลักของโมเดล)

TIER_CONFIG = {
    "Low":    {"label": "ความเสี่ยงต่ำ",       "color": "#0F6E56", "bg": "#E1F5EE", "emoji": "🟢"},
    "Medium": {"label": "ความเสี่ยงปานกลาง",  "color": "#854F0B", "bg": "#FAEEDA", "emoji": "🟡"},
    "High":   {"label": "ความเสี่ยงสูง",       "color": "#A32D2D", "bg": "#FCEBEB", "emoji": "🔴"},
}

# แปลชื่อ feature (ตัวแปรทางเทคนิค) เป็นภาษาไทยอ่านง่าย สำหรับแสดงในกราฟ "ปัจจัยที่มีผลต่อการทำนาย"
# ครอบคลุม feature ทั้งหมดใน feature_names ของโมเดล — ถ้าเจอชื่อไม่อยู่ใน dict นี้ โค้ดจะโชว์ชื่อเดิม
FEATURE_LABELS_TH = {
    "admission_type_id": "ประเภทการเข้ารับการรักษา",
    "discharge_disposition_id": "ลักษณะการจำหน่ายผู้ป่วย",
    "admission_source_id": "แหล่งที่มาของการเข้ารักษา",
    "time_in_hospital": "ระยะเวลานอนโรงพยาบาล",
    "num_lab_procedures": "จำนวนการตรวจ lab",
    "num_procedures": "จำนวนหัตถการ",
    "num_medications": "จำนวนรายการยา",
    "number_outpatient": "จำนวนครั้ง outpatient (ปีที่ผ่านมา)",
    "number_emergency": "จำนวนครั้ง emergency (ปีที่ผ่านมา)",
    "number_inpatient": "จำนวนครั้ง inpatient (ปีที่ผ่านมา)",
    "number_diagnoses": "จำนวนการวินิจฉัย",
    "race": "เชื้อชาติ",
    "gender": "เพศ",
    "medical_specialty": "แผนกที่รักษา",
    "max_glu_serum": "ผลตรวจ glucose serum",
    "A1Cresult": "ผลตรวจ A1C",
    "diabetesMed": "การได้รับยาเบาหวาน",
    "change": "การปรับยาระหว่างนอน รพ.",
    "insulin": "การใช้ insulin",
    "diag_1": "กลุ่มโรควินิจฉัยหลัก",
    "diag_2": "กลุ่มโรควินิจฉัยรอง",
    "diag_3": "กลุ่มโรควินิจฉัยที่ 3",
    "medication_change_count": "จำนวนครั้งที่ปรับขนาดยา",
    "num_meds_active": "จำนวนยาที่ใช้อยู่",
    "tested_glucose": "มีการตรวจ glucose หรือไม่",
    "tested_A1C": "มีการตรวจ A1C หรือไม่",
    "visit_number": "ลำดับครั้งที่เข้ารักษา",
    "prior_change_count": "จำนวนครั้งที่เคยปรับยามาก่อน",
    "total_prior_visits": "จำนวนครั้งเข้ารักษารวมในอดีต",
    "had_prior_inpatient": "เคยนอน รพ. มาก่อนหรือไม่",
    "comorbidity_count": "จำนวนกลุ่มโรคร่วม",
    "age_ord": "อายุ",
    "age_x_medications": "อายุ × จำนวนยา",
    "hospital_stay_x_inpatient": "ระยะเวลานอน × จำนวนครั้ง inpatient",
}

# ใช้เฉพาะกับกราฟ matplotlib (ปัจจัยที่มีผลต่อการทำนาย) เพราะ font default ของ matplotlib
# ไม่รองรับภาษาไทย (แสดงเป็นกล่องสี่เหลี่ยม) — ส่วนอื่นของ dashboard ยังเป็นภาษาไทยตามปกติ
FEATURE_LABELS_EN = {
    "admission_type_id": "Admission type",
    "discharge_disposition_id": "Discharge disposition",
    "admission_source_id": "Admission source",
    "time_in_hospital": "Time in hospital",
    "num_lab_procedures": "Number of lab procedures",
    "num_procedures": "Number of procedures",
    "num_medications": "Number of medications",
    "number_outpatient": "Prior outpatient visits",
    "number_emergency": "Prior emergency visits",
    "number_inpatient": "Prior inpatient visits",
    "number_diagnoses": "Number of diagnoses",
    "race": "Race",
    "gender": "Gender",
    "medical_specialty": "Medical specialty",
    "max_glu_serum": "Max glucose serum",
    "A1Cresult": "A1C result",
    "diabetesMed": "On diabetes medication",
    "change": "Medication changed during stay",
    "insulin": "Insulin usage",
    "diag_1": "Primary diagnosis group",
    "diag_2": "Secondary diagnosis group",
    "diag_3": "Tertiary diagnosis group",
    "medication_change_count": "Medication change count",
    "num_meds_active": "Active medications count",
    "tested_glucose": "Glucose tested",
    "tested_A1C": "A1C tested",
    "visit_number": "Visit number",
    "prior_change_count": "Prior medication changes",
    "total_prior_visits": "Total prior visits",
    "had_prior_inpatient": "Had prior inpatient stay",
    "comorbidity_count": "Comorbidity count",
    "age_ord": "Age",
    "age_x_medications": "Age x medications",
    "hospital_stay_x_inpatient": "Stay length x inpatient visits",
}

# Mapping ID -> คำอธิบาย จาก IDS_mapping.csv (ตัวเลขในวงเล็บคือ ID จริงที่ส่งเข้าโมเดล)
ADMISSION_TYPE_MAP = {
    1: "Emergency", 2: "Urgent", 3: "Elective", 4: "Newborn",
    5: "Not Available", 6: "NULL", 7: "Trauma Center", 8: "Not Mapped",
}

DISCHARGE_DISPOSITION_MAP = {
    1: "Discharged to home",
    2: "Discharged/transferred to another short term hospital",
    3: "Discharged/transferred to SNF",
    4: "Discharged/transferred to ICF",
    5: "Discharged/transferred to another type of inpatient care institution",
    6: "Discharged/transferred to home with home health service",
    7: "Left AMA",
    8: "Discharged/transferred to home under care of Home IV provider",
    9: "Admitted as an inpatient to this hospital",
    10: "Neonate discharged to another hospital for neonatal aftercare",
    12: "Still patient or expected to return for outpatient services",
    15: "Discharged/transferred within this institution to Medicare approved swing bed",
    16: "Discharged/transferred/referred another institution for outpatient services",
    17: "Discharged/transferred/referred to this institution for outpatient services",
    18: "NULL",
    22: "Discharged/transferred to another rehab facility",
    23: "Discharged/transferred to a long term care hospital",
    24: "Discharged/transferred to a nursing facility (Medicaid only)",
    25: "Not Mapped",
    26: "Unknown/Invalid",
    27: "Discharged/transferred to a federal health care facility",
    28: "Discharged/transferred/referred to a psychiatric hospital",
    29: "Discharged/transferred to a Critical Access Hospital (CAH)",
    30: "Discharged/transferred to another Type of Health Care Institution",
    # หมายเหตุ: รหัส 11, 13, 14, 19, 20, 21 (เสียชีวิต/hospice) ถูกกรองออกจาก training data แล้ว
    # จึงไม่ใส่เป็นตัวเลือกใน dashboard (โมเดลไม่เคยเห็นค่าพวกนี้ตอน train)
}

ADMISSION_SOURCE_MAP = {
    1: "Physician Referral", 2: "Clinic Referral", 3: "HMO Referral",
    4: "Transfer from a hospital", 5: "Transfer from a Skilled Nursing Facility (SNF)",
    6: "Transfer from another health care facility", 7: "Emergency Room",
    8: "Court/Law Enforcement", 9: "Not Available",
    10: "Transfer from critical access hospital", 11: "Normal Delivery",
    12: "Premature Delivery", 13: "Sick Baby", 14: "Extramural Birth",
    15: "Not Available", 17: "NULL", 18: "Transfer From Another Home Health Agency",
    19: "Readmission to Same Home Health Agency", 20: "Not Mapped",
    21: "Unknown/Invalid", 22: "Transfer from hospital inpt (separate claim)",
    23: "Born inside this hospital", 24: "Born outside this hospital",
    25: "Transfer from Ambulatory Surgery Center", 26: "Transfer from Hospice",
}

# อ้างอิง: AHRQ IDEAL Discharge Planning framework (Include, Discuss, Educate, Assess, Listen)
# + CMS 5 best practices for discharge coordination + Kripalani et al. systematic review
# (multi-component transitional care programs — medication reconciliation, education,
#  scheduled follow-up, post-discharge contact — ลด readmission ได้จริงเมื่อทำร่วมกันหลายข้อ)
GUIDELINE_CITATION = "อ้างอิงจาก AHRQ IDEAL Discharge Planning Framework และ CMS Care Coordination Best Practices"

RECOMMENDATIONS_BY_TIER = {
    "High": [
        "พิจารณาทบทวนรายการยาก่อนจำหน่าย (medication reconciliation)",
        "พิจารณาวางแผนการติดตามหลังจำหน่ายภายใน 7 วัน และนัดหมาย follow-up ก่อนออกจากโรงพยาบาล",
        "พิจารณาประเมินความเข้าใจของผู้ป่วยเกี่ยวกับการดูแลตนเองด้วยวิธี teach-back",
        "พิจารณาส่งต่อทีม care coordination หากมีประวัติ inpatient ซ้ำ",
    ],
    "Medium": [
        "พิจารณาวางแผนการติดตามหลังจำหน่ายตามความเหมาะสม",
        "พิจารณาให้ความรู้เกี่ยวกับการดูแลโรคเบาหวานเพิ่มเติมด้วยภาษาที่เข้าใจง่าย",
    ],
    "Low": [
        "ดูแลตามมาตรฐานปกติ ไม่มีข้อบ่งชี้พิเศษเพิ่มเติม",
    ],
}

st.set_page_config(page_title="ระบบประเมินความเสี่ยงการกลับเข้ารักษาซ้ำ", layout="centered")


@st.cache_resource
def load_model_bundle(path):
    return joblib.load(path)


def get_risk_tier(proba):
    if proba >= THRESHOLD_MED_HIGH:
        return "High"
    elif proba >= THRESHOLD_LOW_MED:
        return "Medium"
    else:
        return "Low"


def build_feature_row(inputs, label_encoders, feature_names):
    """
    สร้าง 1 แถวของ feature ให้ตรงกับ feature_names ที่โมเดลคาดหวัง
    ตาม feature engineering ใน 07_final_calibrated_pipeline.ipynb

    TODO-3: ค่า default ด้านล่างสำหรับ feature ที่ไม่ได้เก็บจาก UI โดยตรง
    เป็นค่าสมมติสำหรับ "ผู้ป่วยเข้ารับการรักษาครั้งแรกที่บันทึกในระบบ" ควรทวนความสมเหตุสมผลอีกครั้ง
    """
    row = {}

    # --- ค่าที่รับจาก UI ตรงๆ ---
    row["admission_type_id"] = inputs["admission_type_id"]
    row["discharge_disposition_id"] = inputs["discharge_disposition_id"]
    row["admission_source_id"] = inputs["admission_source_id"]
    row["time_in_hospital"] = inputs["time_in_hospital"]
    row["num_lab_procedures"] = inputs["num_lab_procedures"]
    row["num_procedures"] = inputs["num_procedures"]
    row["num_medications"] = inputs["num_medications"]
    row["number_outpatient"] = inputs["number_outpatient"]
    row["number_emergency"] = inputs["number_emergency"]
    row["number_inpatient"] = inputs["number_inpatient"]
    row["number_diagnoses"] = inputs["number_diagnoses"]
    row["race"] = inputs["race"]
    row["gender"] = inputs["gender"]
    row["medical_specialty"] = inputs["medical_specialty"]
    row["max_glu_serum"] = inputs["max_glu_serum"]
    row["A1Cresult"] = inputs["A1Cresult"]
    row["diabetesMed"] = inputs["diabetesMed"]
    row["change"] = inputs["change"]
    row["insulin"] = inputs["insulin"]
    row["diag_1"] = inputs["diag_1_group"]
    row["diag_2"] = inputs["diag_2_group"]
    row["diag_3"] = inputs["diag_3_group"]

    # --- ยาตัวอื่นๆ ที่ไม่ได้ทำ input แยก (default = "No" เพื่อความง่าย) ---
    other_meds = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
        "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
        "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
        "glyburide-metformin", "glipizide-metformin",
        "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
    ]
    for m in other_meds:
        row.setdefault(m, "No")

    # --- Engineered features (ตาม cell 6-7 ของ notebook) ---
    med_cols_all = other_meds + ["insulin"]
    row["medication_change_count"] = sum(1 for m in med_cols_all if row.get(m) in ("Up", "Down"))
    row["num_meds_active"] = sum(1 for m in med_cols_all if row.get(m) != "No")
    row["tested_glucose"] = int(row["max_glu_serum"] != "None")
    row["tested_A1C"] = int(row["A1Cresult"] != "None")

    # TODO-3: ผู้ป่วยใหม่ในระบบ dashboard สมมติว่าเป็น visit แรกที่บันทึก
    row["visit_number"] = 1
    row["prior_change_count"] = 0
    row["total_prior_visits"] = (
        inputs["number_outpatient"] + inputs["number_emergency"] + inputs["number_inpatient"]
    )
    row["had_prior_inpatient"] = int(inputs["number_inpatient"] > 0)

    diag_categories = [row["diag_1"], row["diag_2"], row["diag_3"]]
    row["comorbidity_count"] = len(set(c for c in diag_categories if c != "Missing"))

    age_map = {
        "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
        "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75,
        "[80-90)": 85, "[90-100)": 95,
    }
    row["age_ord"] = age_map[inputs["age_group"]]
    row["age_x_medications"] = row["age_ord"] * row["num_medications"]
    row["hospital_stay_x_inpatient"] = row["time_in_hospital"] * inputs["number_inpatient"]

    df_row = pd.DataFrame([row])

    # Label-encode ตาม encoder ที่ train ไว้จริง
    for col, le in label_encoders.items():
        if col in df_row.columns:
            val = str(df_row.at[0, col])
            if val in le.classes_:
                df_row[col] = le.transform([val])[0]
            else:
                # ค่าที่ไม่เคยเจอตอน train -> fallback เป็น class แรก พร้อมเตือนผู้ใช้
                df_row[col] = le.transform([le.classes_[0]])[0]
                st.warning(f"ค่า '{val}' ใน {col} ไม่พบตอน train — ใช้ค่า default แทน")

    # เรียงคอลัมน์ให้ตรงกับตอน train เป๊ะ
    df_row = df_row.reindex(columns=feature_names, fill_value=0)
    return df_row


@st.cache_resource
def get_shap_explainer(_bundle):
    """Cache SHAP explainer แยกจาก _bundle (ขึ้นต้น _ กัน streamlit hash object ทั้งก้อน)"""
    stacking_model = _bundle["calibrated_stacking_model"]
    base_stacking = stacking_model.calibrated_classifiers_[0].estimator
    rf_model = dict(base_stacking.named_estimators_)["rf"]
    return shap.TreeExplainer(rf_model)


def get_top_factors(model_bundle, X_row, n=5):
    """
    Local (per-patient) explanation ด้วย SHAP TreeExplainer บน RF base model
    (ใช้ RF แทน full stacking classifier — เป็น proxy ที่ตกลงไว้ เพราะ explain
    StackingClassifier ทั้งก้อนซับซ้อนกว่ามาก แต่ตอนนี้เป็น per-patient แล้ว
    ไม่ใช่ global importance เหมือนเดิม)
    """
    try:
        explainer = get_shap_explainer(model_bundle)
        shap_values = explainer.shap_values(X_row)

        # รองรับทั้ง output แบบ list (shap เวอร์ชันเก่า) และ ndarray 3 มิติ (เวอร์ชันใหม่)
        if isinstance(shap_values, list):
            vals = shap_values[1][0]  # class "Readmitted"
        elif shap_values.ndim == 3:
            vals = shap_values[0, :, 1]
        else:
            vals = shap_values[0]

        feature_names = model_bundle["feature_names"]
        top_idx = np.argsort(np.abs(vals))[::-1][:n]
        return [(feature_names[i], vals[i]) for i in top_idx]
    except Exception as e:
        st.warning(f"คำนวณ SHAP ไม่สำเร็จ: {e}")
        return []


# ============================================================
# UI
# ============================================================
st.title("ระบบประเมินความเสี่ยงการกลับเข้ารักษาซ้ำ")
st.caption("Proof-of-concept — สำหรับสนับสนุนการตัดสินใจทางคลินิก ไม่ใช่เครื่องมือวินิจฉัย")

try:
    bundle = load_model_bundle(MODEL_PATH)
except FileNotFoundError:
    st.error(f"ไม่พบไฟล์โมเดลที่ {MODEL_PATH} — กรุณาวางไฟล์ .pkl ที่ save จาก notebook ไว้ในโฟลเดอร์เดียวกัน")
    st.stop()

patient_form = st.container(border=True)
patient_form.subheader("ข้อมูลผู้ป่วย")

col1, col2 = patient_form.columns(2)
with col1:
    age_group = st.selectbox("ช่วงอายุ", [
        "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
        "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)",
    ], index=6)
    gender = st.selectbox("เพศ", ["Male", "Female"])
    race = st.selectbox("เชื้อชาติ", ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "Missing"])
    time_in_hospital = st.number_input("ระยะเวลานอนโรงพยาบาล (วัน)", 1, 14, 4)
    number_diagnoses = st.number_input("จำนวนการวินิจฉัย", 1, 16, 7)
    number_inpatient = st.number_input("จำนวนครั้งที่เคยนอน รพ. (ปีที่ผ่านมา)", 0, 20, 0)

with col2:
    number_outpatient = st.number_input("จำนวนครั้ง outpatient (ปีที่ผ่านมา)", 0, 20, 0)
    number_emergency = st.number_input("จำนวนครั้ง emergency (ปีที่ผ่านมา)", 0, 20, 0)
    num_medications = st.number_input("จำนวนรายการยา", 1, 40, 12)
    num_lab_procedures = st.number_input("จำนวนการตรวจ lab", 0, 120, 40)
    num_procedures = st.number_input("จำนวนหัตถการ", 0, 6, 1)
    insulin = st.selectbox("การใช้ insulin", ["No", "Steady", "Up", "Down"])

with patient_form.expander("ข้อมูลเพิ่มเติม (ผลตรวจ / การวินิจฉัย)"):
    max_glu_serum = st.selectbox("ผล max glucose serum", ["None", "Norm", ">200", ">300"])
    A1Cresult = st.selectbox("ผล A1C", ["None", "Norm", ">7", ">8"])
    diabetesMed = st.selectbox("ได้รับยาเบาหวานหรือไม่", ["Yes", "No"])
    change = st.selectbox("มีการปรับยาระหว่างนอน รพ. หรือไม่", ["Ch", "No"])
    admission_type_id = st.selectbox(
        "ประเภทการเข้ารับการรักษา", options=list(ADMISSION_TYPE_MAP.keys()),
        format_func=lambda x: ADMISSION_TYPE_MAP[x], index=0,
    )
    discharge_disposition_id = st.selectbox(
        "ลักษณะการจำหน่ายผู้ป่วย (คาดการณ์)", options=list(DISCHARGE_DISPOSITION_MAP.keys()),
        format_func=lambda x: DISCHARGE_DISPOSITION_MAP[x], index=0,
    )
    admission_source_id = st.selectbox(
        "แหล่งที่มาของการเข้ารักษา", options=list(ADMISSION_SOURCE_MAP.keys()),
        format_func=lambda x: ADMISSION_SOURCE_MAP[x], index=6,  # default = Emergency Room
    )
    diag_1_group = st.selectbox("กลุ่มโรควินิจฉัยหลัก (diag_1)", [
        "Circulatory", "Respiratory", "Digestive", "Diabetes", "Injury",
        "Musculoskeletal", "Genitourinary", "Neoplasms", "Other", "Missing",
    ])
    diag_2_group = st.selectbox("กลุ่มโรควินิจฉัยรอง (diag_2)", [
        "Circulatory", "Respiratory", "Digestive", "Diabetes", "Injury",
        "Musculoskeletal", "Genitourinary", "Neoplasms", "Other", "Missing",
    ], index=9)
    diag_3_group = st.selectbox("กลุ่มโรควินิจฉัยที่ 3 (diag_3)", [
        "Circulatory", "Respiratory", "Digestive", "Diabetes", "Injury",
        "Musculoskeletal", "Genitourinary", "Neoplasms", "Other", "Missing",
    ], index=9)
    # TODO-1 แก้แล้ว: รายชื่อนี้คือ top-10 จริงจาก df['medical_specialty'].value_counts().nlargest(10)
    # หลัง preprocessing เดียวกับ notebook (ตัด Unknown/Invalid gender และ death/hospice disposition)
    medical_specialty = st.selectbox("แผนกที่รักษา", [
        "InternalMedicine", "Emergency/Trauma", "Family/GeneralPractice", "Cardiology",
        "Surgery-General", "Nephrology", "Orthopedics", "Orthopedics-Reconstructive",
        "Radiologist", "Pulmonology", "Other", "Missing",
    ])

inputs = dict(
    age_group=age_group, gender=gender, race=race, time_in_hospital=time_in_hospital,
    number_diagnoses=number_diagnoses, number_inpatient=number_inpatient,
    number_outpatient=number_outpatient, number_emergency=number_emergency,
    num_medications=num_medications, num_lab_procedures=num_lab_procedures,
    num_procedures=num_procedures, insulin=insulin, max_glu_serum=max_glu_serum,
    A1Cresult=A1Cresult, diabetesMed=diabetesMed, change=change,
    admission_type_id=admission_type_id, discharge_disposition_id=discharge_disposition_id,
    admission_source_id=admission_source_id, diag_1_group=diag_1_group,
    diag_2_group=diag_2_group, diag_3_group=diag_3_group, medical_specialty=medical_specialty,
)

evaluate_clicked = patient_form.button("ประเมินความเสี่ยง", type="primary", use_container_width=True)

if evaluate_clicked:
    X_row = build_feature_row(inputs, bundle["label_encoders"], bundle["feature_names"])
    proba = bundle["calibrated_stacking_model"].predict_proba(X_row)[0, 1]
    tier = get_risk_tier(proba)
    cfg = TIER_CONFIG[tier]
    if tier == "Low":
        primary_threshold, threshold_label = THRESHOLD_LOW_MED, "เกณฑ์การประเมิน (F2-optimal, เส้นแบ่ง Low/Medium)"
    else:
        primary_threshold, threshold_label = THRESHOLD_MED_HIGH, "เกณฑ์การประเมิน (cost-optimal, เส้นแบ่ง Medium/High)"

    # --- ผลการประเมิน ---
    with st.container(border=True):
        st.subheader("ผลการประเมิน")
        st.markdown(
            f"<div style='text-align:center;padding:8px 0'>"
            f"<span style='font-size:32px'>{cfg['emoji']}</span> "
            f"<span style='font-size:24px;font-weight:600;color:{cfg['color']}'>{cfg['label']}</span>"
            f"</div>", unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        c1.metric("ความน่าจะเป็นของการกลับเข้ารักษาซ้ำ", f"{proba*100:.1f}%")
        c2.metric(threshold_label, f"{primary_threshold:.3f}")
        st.caption(
            f"ช่วงเกณฑ์ทั้งหมด — Low: < {THRESHOLD_LOW_MED:.3f} · "
            f"Medium: {THRESHOLD_LOW_MED:.3f}–{THRESHOLD_MED_HIGH:.3f} · "
            f"High: ≥ {THRESHOLD_MED_HIGH:.3f}"
        )
        with st.expander("ทำไมตัวเลขนี้ถึงถูกจัดว่าเป็นความเสี่ยงระดับนี้?"):
            lift = proba / DATASET_BASE_RATE
            st.markdown(
                f"- ความน่าจะเป็นเฉลี่ยของการ readmit ในผู้ป่วยทั้งหมด (base rate) อยู่ที่ "
                f"**{DATASET_BASE_RATE*100:.1f}%** ผู้ป่วยรายนี้มีความน่าจะเป็น **{proba*100:.1f}%** "
                f"ซึ่งสูงกว่าค่าเฉลี่ยประมาณ **{lift:.1f} เท่า**\n"
                f"- เกณฑ์การจัด tier **ไม่ได้ใช้ 50%** เป็นเส้นแบ่ง เพราะข้อมูลมีความไม่สมดุลสูง "
                f"(คนที่ readmit จริงมีแค่ ~11% ของทั้งหมด) การรอจนกว่าความน่าจะเป็นเกิน 50% "
                f"จะทำให้พลาดเคสที่เสี่ยงจริงไปเกือบทั้งหมด\n"
                f"- เกณฑ์ 0.150 คำนวณจาก **ต้นทุนที่คาดหวัง (expected cost)** โดยตั้งว่าการพลาดผู้ป่วยที่จะ "
                f"readmit จริง (False Negative) มีต้นทุนแพงกว่าการเตือนเกินความจำเป็น (False Positive) "
                f"ประมาณ 5 เท่า — จึงยอมให้มี False Positive มากขึ้นเพื่อจับเคสจริงให้ได้มากที่สุด\n"
                f"- สรุป: **\"ความเสี่ยงสูง\" ในระบบนี้หมายถึง \"เสี่ยงกว่าค่าเฉลี่ยของผู้ป่วยทั่วไปอย่างมีนัยสำคัญ "
                f"ถึงจุดที่คุ้มค่าต่อการเฝ้าระวังเพิ่มเติม\" ไม่ได้แปลว่ามีโอกาสเกิน 50%**"
            )

    # --- ปัจจัยที่มีผลต่อการทำนาย ---
    with st.container(border=True):
        st.subheader("ปัจจัยที่มีผลต่อการทำนาย")
        top_factors = get_top_factors(bundle, X_row)
        if top_factors:
            factor_df = pd.DataFrame(top_factors, columns=["ปัจจัย", "SHAP value"])
            factor_df["ปัจจัย"] = factor_df["ปัจจัย"].map(lambda f: FEATURE_LABELS_TH.get(f, f))
            st.bar_chart(factor_df.set_index("ปัจจัย"))
        else:
            st.info("ไม่สามารถคำนวณปัจจัยได้ในขณะนี้")
        st.caption(
            "แสดงปัจจัยที่มีอิทธิพลต่อการทำนายของโมเดลสำหรับผู้ป่วยรายนี้โดยเฉพาะ (SHAP value) "
            "ค่าบวก = ดันความเสี่ยงให้สูงขึ้น, ค่าลบ = ดันความเสี่ยงให้ต่ำลง "
            "ไม่ได้หมายถึงสาเหตุทางการแพทย์โดยตรง"
        )

    # --- แนวทางที่ควรพิจารณา ---
    with st.container(border=True):
        st.subheader("แนวทางที่ควรพิจารณา")
        for rec in RECOMMENDATIONS_BY_TIER[tier]:
            st.markdown(f"- {rec}")
        st.caption(GUIDELINE_CITATION)

    st.caption(
        "หมายเหตุ: ผลการประเมินนี้ใช้เพื่อสนับสนุนการตัดสินใจทางคลินิก "
        "ไม่ใช่เพื่อวินิจฉัยหรือกำหนดแนวทางการรักษา "
        "ผู้ป่วยควรได้รับการประเมินโดยบุคลากรทางการแพทย์"
    )
