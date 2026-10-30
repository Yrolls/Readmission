"""
ระบบติดตามผู้ป่วยและประเมินความเสี่ยงการกลับเข้ารักษาซ้ำ (Readmission Risk Registry)
======================================================================================
Dashboard สำหรับ capstone project — ใช้โมเดลหลักจาก 15_final_model.ipynb
(Blend LSTM + XGBoost, ทั้ง label "30 วัน" และ "กลับมารักษาซ้ำ")

การทำงาน
    1. เริ่มต้นระบบว่าง → แพทย์ลงทะเบียนผู้ป่วยใหม่
    2. เปิดการ admit → บันทึกข้อมูลแรกรับ → ระบบประเมินความเสี่ยง
    3. อัปเดตอาการ/ผลตรวจ/ยาระหว่างนอน รพ. ได้เรื่อยๆ → ประเมินใหม่ทุกครั้ง เห็นแนวโน้มความเสี่ยง
    4. จำหน่ายผู้ป่วย → ประเมินครั้งสุดท้าย
    5. ถ้าผู้ป่วยกลับมา admit อีก ระบบรู้เองว่ากลับมาภายในกี่วัน และใช้ประวัติทั้งหมดในการประเมินครั้งใหม่

วิธีรัน:
    streamlit run dashboard_app.py

ไฟล์ที่ต้องมีในโฟลเดอร์เดียวกัน:
    readmission_features.py, readmission_registry.py
    models/dashboard_artifacts.json          (จาก export_dashboard_artifacts.py)
    models/final_any/  และ  models/final_30d/  (จาก notebook 15)
ข้อมูลผู้ป่วยถูกเก็บใน patients.db (SQLite) — ลบไฟล์นี้เพื่อเริ่มระบบใหม่
"""
import datetime as dt
import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from readmission_features import MED_COLS, DIAG_GROUPS
from readmission_registry import (Registry, RiskEngine, derive_outcomes, length_of_stay, los_for_model,
                                  age_group, to_date, MAX_STAY_DAYS)

# ============================================================
# CONFIG
# ============================================================
APP_DIR = Path(__file__).resolve().parent
DB_PATH = str(APP_DIR / "patients.db")
MODEL_ROOT = str(APP_DIR.parent / "models")

# ผลจริงบน test set จาก 15_final_model.ipynb (ใช้อธิบายความหมายของแต่ละ tier)
LABEL_INFO = {
    "30d": {"title": "ความเสี่ยงกลับมาภายใน 30 วัน", "short": "30 วัน", "base": 0.1135,
            "tier_rate": {"High": 0.241, "Medium": 0.118, "Low": 0.062}, "capture_high": 0.41,
            "roc_auc": 0.692},
    "any": {"title": "ความเสี่ยงกลับมารักษาซ้ำ (ทุกช่วงเวลา)", "short": "รักษาซ้ำ", "base": 0.465,
            "tier_rate": {"High": 0.734, "Medium": 0.533, "Low": 0.322}, "capture_high": 0.31,
            "roc_auc": 0.703},
}

TIER_CONFIG = {
    "Low":    {"label": "ความเสี่ยงต่ำ",      "color": "#0F6E56", "bg": "#E1F5EE", "emoji": "🟢"},
    "Medium": {"label": "ความเสี่ยงปานกลาง", "color": "#854F0B", "bg": "#FAEEDA", "emoji": "🟡"},
    "High":   {"label": "ความเสี่ยงสูง",      "color": "#A32D2D", "bg": "#FCEBEB", "emoji": "🔴"},
}

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
    "race": "เชื้อชาติ", "gender": "เพศ", "medical_specialty": "แผนกที่รักษา",
    "max_glu_serum": "ผลตรวจ glucose serum", "A1Cresult": "ผลตรวจ A1C",
    "diabetesMed": "การได้รับยาเบาหวาน", "change": "การปรับยาระหว่างนอน รพ.",
    "insulin": "การใช้ insulin", "metformin": "การใช้ metformin",
    "diag_1": "กลุ่มโรควินิจฉัยหลัก", "diag_2": "กลุ่มโรควินิจฉัยรอง", "diag_3": "กลุ่มโรควินิจฉัยที่ 3",
    "medication_change_count": "จำนวนยาที่ปรับขนาด", "num_meds_active": "จำนวนยาเบาหวานที่ใช้อยู่",
    "tested_glucose": "มีการตรวจ glucose หรือไม่", "tested_A1C": "มีการตรวจ A1C หรือไม่",
    "visit_number": "ลำดับครั้งที่ admit ในระบบ", "prior_change_count": "จำนวนครั้งที่เคยปรับยามาก่อน",
    "total_prior_visits": "จำนวนครั้งเข้ารักษารวมในอดีต", "had_prior_inpatient": "เคยนอน รพ. มาก่อนหรือไม่",
    "comorbidity_count": "จำนวนกลุ่มโรคร่วม", "age_ord": "อายุ",
    "age_x_medications": "อายุ × จำนวนยา", "hospital_stay_x_inpatient": "ระยะเวลานอน × จำนวนครั้ง inpatient",
    "prev_readmit_lt30": "ครั้งก่อนกลับมาภายใน 30 วัน", "prev_readmit_gt30": "ครั้งก่อนกลับมาหลัง 30 วัน",
    "prev_readmit_no": "ครั้งก่อนไม่ได้กลับมา",
    "n_prior_lt30_in_data": "จำนวนครั้งที่เคยกลับมาภายใน 30 วัน",
    "n_prior_readmit_in_data": "จำนวนครั้งที่เคยกลับมารักษาซ้ำ",
}

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
    27: "Discharged/transferred to a federal health care facility",
    28: "Discharged/transferred/referred to a psychiatric hospital",
}
ADMISSION_SOURCE_MAP = {
    1: "Physician Referral", 2: "Clinic Referral", 3: "HMO Referral",
    4: "Transfer from a hospital", 5: "Transfer from a Skilled Nursing Facility (SNF)",
    6: "Transfer from another health care facility", 7: "Emergency Room",
    8: "Court/Law Enforcement", 9: "Not Available",
    10: "Transfer from critical access hospital", 11: "Normal Delivery",
    13: "Sick Baby", 14: "Extramural Birth", 17: "NULL", 20: "Not Mapped",
    22: "Transfer from hospital inpt (separate claim)", 25: "Transfer from Ambulatory Surgery Center",
}

MAIN_MEDS = ["insulin", "metformin", "glipizide", "glyburide", "glimepiride",
             "pioglitazone", "rosiglitazone", "repaglinide"]
MED_STATUS = ["No", "Steady", "Up", "Down"]
MED_STATUS_TH = {"No": "ไม่ได้ใช้", "Steady": "ใช้ ขนาดเดิม", "Up": "เพิ่มขนาด", "Down": "ลดขนาด"}

GUIDELINE_CITATION = "อ้างอิงจาก AHRQ IDEAL Discharge Planning Framework และ CMS Care Coordination Best Practices"
RECOMMENDATIONS_30D = {
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
    "Low": ["ดูแลตามมาตรฐานปกติ ไม่มีข้อบ่งชี้พิเศษเพิ่มเติมสำหรับช่วง 30 วันแรก"],
}
RECOMMENDATIONS_LONG_TERM = [
    "พิจารณาวางแผนดูแลโรคเรื้อรังต่อเนื่องระยะยาว เช่น นัดติดตามที่คลินิกเบาหวาน",
    "พิจารณาส่งต่อโปรแกรมให้ความรู้และสนับสนุนการดูแลตนเองโรคเบาหวาน (DSMES)",
]
LONG_TERM_CITATION = "อ้างอิงแนวคิด DSMES จาก American Diabetes Association — Standards of Care in Diabetes"

st.set_page_config(page_title="ระบบติดตามความเสี่ยงการกลับเข้ารักษาซ้ำ", page_icon="🏥",
                   layout="wide", initial_sidebar_state="expanded")


# ============================================================
# Resources
# ============================================================
@st.cache_resource
def get_registry():
    return Registry(DB_PATH)


@st.cache_resource(show_spinner="กำลังโหลดโมเดล...")
def get_engine():
    return RiskEngine(MODEL_ROOT)


@st.cache_data
def get_artifacts():
    p = Path(MODEL_ROOT) / "dashboard_artifacts.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


reg = get_registry()
artifacts = get_artifacts()
try:
    engine = get_engine()
    engine_error = None
except Exception as e:   # ยังใช้ระบบบันทึกข้อมูลได้ แม้โหลดโมเดลไม่สำเร็จ
    engine, engine_error = None, e

SPECIALTIES = (artifacts["top_specialties"] if artifacts else
               ["Missing", "InternalMedicine", "Emergency/Trauma", "Family/GeneralPractice", "Cardiology",
                "Surgery-General", "Nephrology", "Orthopedics", "Orthopedics-Reconstructive", "Radiologist"])
SPECIALTIES = [s for s in SPECIALTIES if s != "Missing"] + ["Other", "Missing"]


def seen(col, mapping):
    if artifacts and col in artifacts.get("seen_ids", {}):
        return [i for i in artifacts["seen_ids"][col] if i in mapping]
    return list(mapping)


# ============================================================
# Navigation
# ============================================================
st.session_state.setdefault("page", "list")
st.session_state.setdefault("pid", None)
st.session_state.setdefault("eid", None)          # encounter ที่กำลังดูผลการประเมิน
st.session_state.setdefault("history", [])        # หน้าที่เคยเปิด สำหรับปุ่มย้อนกลับ


def go(page, pid=None, eid=None, push=True, collapse=False):
    """เปลี่ยนหน้า — push=False ใช้เมื่อไม่ต้องการให้ย้อนกลับมาหน้าปัจจุบันได้ (เช่น หลังส่งฟอร์มลงทะเบียน)
    collapse=True ปิด sidebar ให้อัตโนมัติหลังเปลี่ยนหน้า"""
    ss = st.session_state
    current = (ss.page, ss.pid, ss.eid)
    target = (page, pid if pid is not None else ss.pid, eid if eid is not None else ss.eid)
    if page == "list":
        ss.history = []          # หน้ารายชื่อผู้ป่วยคือหน้า home — กลับมาแล้วเริ่มนับประวัติการนำทางใหม่
    elif push and current != target:
        ss.history = (ss.history + [current])[-20:]
    ss.page, ss.pid, ss.eid = target
    if collapse:
        ss["collapse_sidebar"] = True
    st.rerun()


def back():
    ss = st.session_state
    if ss.history:
        ss.page, ss.pid, ss.eid = ss.history.pop()
        if ss.page == "list":
            ss.history = []
    st.rerun()


def collapse_sidebar_if_requested():
    """Streamlit ไม่มีคำสั่งปิด sidebar โดยตรง จึงใช้ JavaScript กดปุ่มปิดแทน"""
    if not st.session_state.pop("collapse_sidebar", False):
        return
    components.html("""
    <script>
    const doc = window.parent.document;
    const sb = doc.querySelector('section[data-testid="stSidebar"]');
    const isOpen = sb && sb.getAttribute('aria-expanded') !== 'false';
    if (isOpen) {
        const btn = doc.querySelector('[data-testid="stSidebarCollapseButton"] button')
                 || doc.querySelector('[data-testid="stSidebarCollapseButton"]')
                 || sb.querySelector('button[kind="header"]')
                 || sb.querySelector('button[kind="headerNoPadding"]');
        if (btn) btn.click();
    }
    </script>""", height=0)


def top_nav():
    """ปุ่มย้อนกลับด้านบน (เมนูหลักอยู่ใน sidebar)"""
    if st.session_state.history and st.session_state.page != "list":
        if st.button("← ย้อนกลับ", key="nav_back"):
            back()


def flash(msg, kind="success"):
    st.session_state["flash"] = (msg, kind)


def show_flash():
    if "flash" in st.session_state:
        msg, kind = st.session_state.pop("flash")
        getattr(st, kind)(msg)


FOLLOWUP_WINDOW = 30          # วันหลังจำหน่ายที่ต้องเฝ้าระวัง (ตามนิยาม readmission 30 วัน)
EARLY_CONTACT_DAYS = 7        # ความเสี่ยงสูงควรติดต่อภายใน 7 วันแรก (AHRQ IDEAL / CMS)


def followup_table(window=FOLLOWUP_WINDOW):
    """ผู้ป่วยที่จำหน่ายภายใน window วันล่าสุด พร้อมสถานะการติดตาม"""
    today = dt.date.today()
    df = reg.discharged_since(today - dt.timedelta(days=window))
    if df.empty:
        return df
    df["days_since"] = [(today - to_date(d)).days for d in df["discharge_date"]]
    df["returned_after"] = [None if pd.isna(n) else (to_date(n) - to_date(d)).days
                            for n, d in zip(df["next_admission"], df["discharge_date"])]
    return df


def followup_due_count():
    df = followup_table()
    if df.empty:
        return 0
    active = df[df["returned_after"].isna()]
    return int(((active["tier_30d"] == "High") & (active["days_since"] <= EARLY_CONTACT_DAYS)).sum())


with st.sidebar:
    st.title("🏥 Readmission Risk")
    st.caption("ระบบติดตามผู้ป่วยเบาหวานและประเมินความเสี่ยงการกลับเข้ารักษาซ้ำ")
    if st.button("📋 รายชื่อผู้ป่วย", key="side_list", use_container_width=True):
        go("list", collapse=True)
    n_due = followup_due_count()
    if st.button(f"🔔 ผู้ป่วยที่ต้องติดตาม" + (f" ({n_due})" if n_due else ""), key="side_followup",
                 use_container_width=True, type="primary" if n_due else "secondary"):
        go("followup", collapse=True)
    if st.button("➕ ลงทะเบียนผู้ป่วยใหม่", key="side_register", use_container_width=True):
        go("register", collapse=True)
    if st.session_state.pid and reg.get_patient(st.session_state.pid):
        p_side = reg.get_patient(st.session_state.pid)
        if st.button(f"👤 {p_side['name'] or p_side['hn']}", key="side_patient", use_container_width=True):
            go("patient", collapse=True)
    st.divider()
    if engine_error:
        st.error("โหลดโมเดลไม่สำเร็จ — บันทึกข้อมูลได้ แต่ยังประเมินความเสี่ยงไม่ได้")
        with st.expander("รายละเอียด"):
            st.code(str(engine_error))
    else:
        st.success("โมเดลพร้อมใช้งาน")
    st.caption("Proof-of-concept สำหรับสนับสนุนการตัดสินใจทางคลินิก ไม่ใช่เครื่องมือวินิจฉัย")


# ============================================================
# Helper: การแสดงผลความเสี่ยง
# ============================================================
def tier_badge(tier, p=None):
    c = TIER_CONFIG[tier]
    txt = f"{c['emoji']} {c['label']}" + (f" · {p*100:.0f}%" if p is not None else "")
    return (f"<span style='background:{c['bg']};color:{c['color']};padding:4px 10px;"
            f"border-radius:12px;font-weight:600'>{txt}</span>")


def risk_card(label, r):
    info, cfg = LABEL_INFO[label], TIER_CONFIG[r["tier"]]
    with st.container(border=True):
        st.markdown(f"**{info['title']}**")
        st.markdown(
            f"<div style='background:{cfg['bg']};border-radius:10px;padding:12px;text-align:center'>"
            f"<div style='font-size:22px;font-weight:700;color:{cfg['color']}'>{cfg['emoji']} {cfg['label']}</div>"
            f"<div style='font-size:30px;font-weight:700;color:{cfg['color']}'>{r['p']*100:.1f}%</div>"
            f"<div style='color:{cfg['color']}'>ความน่าจะเป็นที่ calibrate แล้ว</div></div>",
            unsafe_allow_html=True)
        st.caption(
            f"ค่าเฉลี่ยของผู้ป่วยทั้งหมด {info['base']*100:.1f}% · ผู้ป่วยกลุ่ม {cfg['label']} "
            f"กลับมาจริง {info['tier_rate'][r['tier']]*100:.1f}% (จากข้อมูลทดสอบ)")


def pretty_value(feature, v):
    maps = {"admission_type_id": ADMISSION_TYPE_MAP, "admission_source_id": ADMISSION_SOURCE_MAP,
            "discharge_disposition_id": DISCHARGE_DISPOSITION_MAP}
    if feature in maps:
        return maps[feature].get(int(v), v)
    if feature == "gender":
        return {"Female": "หญิง", "Male": "ชาย"}.get(v, v)
    if feature in ("A1Cresult", "max_glu_serum"):
        return "ไม่ได้ตรวจ" if v == "None" else ("ปกติ" if v == "Norm" else v)
    if feature in ("race", "medical_specialty") and v == "Missing":
        return "ไม่ระบุ"
    if feature in MED_COLS or feature in ("change", "diabetesMed"):
        return {**MED_STATUS_TH, "Ch": "มี", "Yes": "ใช่"}.get(v, v)
    if feature in ("age_ord",):
        return f"{v} ปี (กลางช่วง)"
    if feature.startswith(("prev_readmit", "tested_", "had_prior")):
        return "ใช่" if v == 1 else "ไม่"
    return v


def factor_chart(factors):
    f = pd.DataFrame(factors, columns=["feature", "shap", "value"])
    f["ปัจจัย"] = [f"{FEATURE_LABELS_TH.get(x, x)} = {pretty_value(x, v)}" for x, v in zip(f["feature"], f["value"])]
    f["ทิศทาง"] = np.where(f["shap"] > 0, "เพิ่มความเสี่ยง", "ลดความเสี่ยง")
    f["ขนาด"] = f["shap"].abs()
    return alt.Chart(f).mark_bar().encode(
        x=alt.X("shap:Q", title="ผลต่อความเสี่ยง (SHAP, log-odds)"),
        y=alt.Y("ปัจจัย:N", sort=alt.EncodingSortField(field="ขนาด", order="descending"), title=None,
                axis=alt.Axis(labelLimit=380)),
        color=alt.Color("ทิศทาง:N", scale=alt.Scale(domain=["เพิ่มความเสี่ยง", "ลดความเสี่ยง"],
                                                    range=["#A32D2D", "#0F6E56"]),
                        legend=alt.Legend(title=None, orient="bottom")),
        tooltip=["ปัจจัย", alt.Tooltip("shap:Q", format="+.3f")],
    ).properties(height=40 * len(f) + 40)


def trend_chart(ass):
    t = ass.reset_index(drop=True).copy()
    t["ครั้งที่"] = np.arange(1, len(t) + 1)
    t["รายละเอียด"] = [f"{s} ({c[:16].replace('T', ' ')})" for s, c in zip(t["stage"], t["created_at"])]
    long = t.melt(id_vars=["ครั้งที่", "รายละเอียด"], value_vars=["p_30d", "p_any"], var_name="ชนิด",
                  value_name="ความน่าจะเป็น")
    long["ชนิด"] = long["ชนิด"].map({"p_30d": "ความเสี่ยง 30 วัน", "p_any": "ความเสี่ยงรักษาซ้ำ"})
    return alt.Chart(long).mark_line(point=alt.OverlayMarkDef(size=80)).encode(
        x=alt.X("ครั้งที่:O", title="การประเมินครั้งที่", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("ความน่าจะเป็น:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("ชนิด:N", legend=alt.Legend(title=None, orient="bottom")),
        tooltip=["ครั้งที่", "รายละเอียด", "ชนิด", alt.Tooltip("ความน่าจะเป็น:Q", format=".1%")],
    ).properties(height=280)


SUMMARY_30D = {
    "High": "ผู้ป่วยรายนี้อยู่ในกลุ่มความเสี่ยงสูง — ควรติดตามหลังจำหน่ายเป็นลำดับแรก",
    "Medium": "ผู้ป่วยรายนี้อยู่ในกลุ่มความเสี่ยงปานกลาง — ควรวางแผนติดตามตามความเหมาะสม",
    "Low": "ผู้ป่วยรายนี้อยู่ในกลุ่มความเสี่ยงต่ำ — ดูแลตามมาตรฐานปกติ",
}
# feature ที่เป็นผลคูณ อธิบายเป็นภาษาคนยาก จึงไม่แสดงในรายการเหตุผล (ยังเห็นได้ในกราฟ)
HARD_TO_EXPLAIN = {"age_x_medications", "hospital_stay_x_inpatient"}
UNITS = {"time_in_hospital": "วัน", "number_inpatient": "ครั้ง", "number_emergency": "ครั้ง",
         "number_outpatient": "ครั้ง", "num_medications": "รายการ", "num_lab_procedures": "รายการ",
         "num_procedures": "รายการ", "number_diagnoses": "รายการ", "total_prior_visits": "ครั้ง",
         "n_prior_lt30_in_data": "ครั้ง", "n_prior_readmit_in_data": "ครั้ง", "visit_number": "",
         "num_meds_active": "ชนิด", "medication_change_count": "ชนิด", "comorbidity_count": "กลุ่ม"}


def reason_lines(factors, k=3):
    """เหตุผลหลัก k ข้อเป็นภาษาคน จากปัจจัยที่มีผลมากที่สุด"""
    out = []
    for feat, shap, value in factors:
        if feat in HARD_TO_EXPLAIN:
            continue
        v = pretty_value(feat, value)
        unit = UNITS.get(feat, "")
        up = shap > 0
        color = TIER_CONFIG["High"]["color"] if up else TIER_CONFIG["Low"]["color"]
        out.append(f"<span style='color:{color};font-weight:600'>{'▲ เพิ่ม' if up else '▼ ลด'}ความเสี่ยง</span> "
                   f"— {FEATURE_LABELS_TH.get(feat, feat)}: <b>{v}{' ' + unit if unit else ''}</b>")
        if len(out) == k:
            break
    return out


def render_risk(result, patient, enc):
    r30, rany = result["30d"], result["any"]
    cfg = TIER_CONFIG[r30["tier"]]

    # 1) สรุปประโยคเดียว — ระดับความเสี่ยงเด่นกว่าตัวเลข
    st.markdown(
        f"<div style='background:{cfg['bg']};border-left:8px solid {cfg['color']};border-radius:10px;"
        f"padding:18px 22px;margin-bottom:6px'>"
        f"<div style='font-size:26px;font-weight:700;color:{cfg['color']}'>{cfg['emoji']} {cfg['label']}</div>"
        f"<div style='font-size:19px;color:{cfg['color']};margin-top:4px'>{SUMMARY_30D[r30['tier']]}</div>"
        f"<div style='color:#555;margin-top:8px'>โอกาสกลับมารักษาภายใน 30 วัน ประมาณ <b>{r30['p']*100:.0f}%</b> "
        f"(ผู้ป่วยทั่วไปเฉลี่ย {LABEL_INFO['30d']['base']*100:.0f}%)</div></div>",
        unsafe_allow_html=True)
    st.caption("ℹ️ ใช้ช่วยจัดลำดับความสำคัญในการติดตามผู้ป่วย ไม่ใช่การวินิจฉัย — การตัดสินใจขึ้นกับแพทย์")

    # 2) ความเสี่ยงระยะยาวเป็นข้อมูลเสริม
    st.markdown(f"ความเสี่ยงกลับมารักษาซ้ำในระยะยาว: {tier_badge(rany['tier'], rany['p'])}",
                unsafe_allow_html=True)

    # คำเตือนที่ต้องเห็นเสมอ
    age = age_group(patient["birth_year"], to_date(enc["admission_date"]))
    if age in ("[80-90)", "[90-100)"):
        st.warning("⚠️ ผู้ป่วยอายุ 80 ปีขึ้นไป — ระบบมักประเมินความเสี่ยงต่ำกว่าความจริงในกลุ่มนี้ "
                   "ควรใช้ดุลยพินิจทางคลินิกเป็นหลัก")

    c1, c2 = st.columns(2)
    # 3) เหตุผลหลัก 3 ข้อ
    with c1:
        with st.container(border=True):
            st.markdown("**เหตุผลหลักที่ระบบประเมินแบบนี้**")
            for line in reason_lines(r30["factors"]):
                st.markdown(f"- {line}", unsafe_allow_html=True)
            n_hist = r30["n_history"]
            st.caption(f"ใช้ประวัติการ admit ก่อนหน้าในระบบ {n_hist} ครั้งร่วมด้วย" if n_hist
                       else "ยังไม่มีประวัติการ admit ก่อนหน้าในระบบ")
    with c2:
        with st.container(border=True):
            st.markdown("**แนวทางที่ควรพิจารณา**")
            for rec in RECOMMENDATIONS_30D[r30["tier"]]:
                st.markdown(f"- {rec}")
            if rany["tier"] == "High":
                for rec in RECOMMENDATIONS_LONG_TERM:
                    st.markdown(f"- {rec}")
            st.caption(GUIDELINE_CITATION + (" · " + LONG_TERM_CITATION if rany["tier"] == "High" else ""))

    # 4) รายละเอียดทางเทคนิค — พับไว้
    with st.expander("📊 ดูปัจจัยทั้งหมดแบบกราฟ"):
        tabs = st.tabs(["ความเสี่ยง 30 วัน", "ความเสี่ยงระยะยาว"])
        for tab, label in zip(tabs, ["30d", "any"]):
            with tab:
                st.altair_chart(factor_chart(result[label]["factors"][:6]), use_container_width=True)
        st.caption("แดง = เพิ่มความเสี่ยง, เขียว = ลดความเสี่ยง ไม่ได้หมายถึงสาเหตุทางการแพทย์โดยตรง · "
                   f"คำนวณจากส่วน XGBoost ของโมเดล (น้ำหนัก {1-r30['w_lstm']:.0%} / {1-rany['w_lstm']:.0%}) "
                   "จึงเป็นคำอธิบายโดยประมาณ")
    with st.expander("⚙️ รายละเอียดทางเทคนิค"):
        st.markdown(
            "- **ระดับความเสี่ยงแบ่งตามลำดับ ไม่ใช่เกณฑ์ 50%**: สูง = 20% แรกของผู้ป่วยที่เสี่ยงที่สุด, "
            "ปานกลาง = 30% ถัดไป, ต่ำ = ที่เหลือ เพื่อให้ทีมติดตามได้จริงโดยไม่แจ้งเตือนมากเกินไป\n"
            f"- **ผลจากข้อมูลทดสอบ**: ผู้ป่วยกลุ่มสูง/ปานกลาง/ต่ำ กลับมาภายใน 30 วันจริง "
            f"{LABEL_INFO['30d']['tier_rate']['High']*100:.0f}% / {LABEL_INFO['30d']['tier_rate']['Medium']*100:.0f}% / "
            f"{LABEL_INFO['30d']['tier_rate']['Low']*100:.0f}% และการติดตามเฉพาะกลุ่มสูงครอบคลุมผู้ที่กลับมาได้ราว "
            f"{LABEL_INFO['30d']['capture_high']*100:.0f}%\n"
            "- **สองความเสี่ยงตอบคำถามต่างกัน**: 30 วัน = การดูแลช่วงรอยต่อหลังจำหน่าย, ระยะยาว = ภาระโรคเรื้อรัง\n"
            f"- **ความแม่นยำของโมเดล (ROC-AUC)**: 30 วัน = {LABEL_INFO['30d']['roc_auc']}, "
            f"ระยะยาว = {LABEL_INFO['any']['roc_auc']} อยู่ในระดับใกล้เคียงเครื่องมือประเมินความเสี่ยงที่ใช้ทั่วไป "
            "เหมาะกับการจัดลำดับความสำคัญ ไม่แม่นพอจะใช้แทนการประเมินของแพทย์\n"
            "- **อายุ 80+**: ความแม่นยำต่ำกว่ากลุ่มอื่น (ROC-AUC 0.55–0.66 เทียบกับ ~0.70)")


# ============================================================
# Helper: ฟอร์มข้อมูลการรักษา
# ============================================================
def pick(label, options, default, fmt=None, key=None):
    idx = options.index(default) if default in options else 0
    return st.selectbox(label, options, index=idx, format_func=fmt or (lambda x: x), key=key)


def encounter_form(key, defaults, submit_label, include_disposition=True, extra_top=None, ask_expected_los=True):
    """ฟอร์มข้อมูลทางคลินิก — คืน (submitted, data, extra) โดย extra = ค่าจาก extra_top (เช่น วันที่)"""
    d = defaults or {}
    with st.form(key):
        extra = extra_top() if extra_top else None
        st.markdown("**ข้อมูลการเข้ารับการรักษา**")
        c1, c2, c3 = st.columns(3)
        with c1:
            adm_type = pick("ประเภทการเข้ารับการรักษา", seen("admission_type_id", ADMISSION_TYPE_MAP),
                            d.get("admission_type_id", 1), lambda x: ADMISSION_TYPE_MAP[x], f"{key}_at")
            adm_src = pick("แหล่งที่มาของการเข้ารักษา", seen("admission_source_id", ADMISSION_SOURCE_MAP),
                           d.get("admission_source_id", 7), lambda x: ADMISSION_SOURCE_MAP[x], f"{key}_as")
            spec = pick("แผนกที่รักษา", SPECIALTIES, d.get("medical_specialty", "InternalMedicine"), key=f"{key}_sp")
            if include_disposition:
                disp = pick("ลักษณะการจำหน่าย (คาดการณ์)", seen("discharge_disposition_id", DISCHARGE_DISPOSITION_MAP),
                            d.get("discharge_disposition_id", 1), lambda x: DISCHARGE_DISPOSITION_MAP[x], f"{key}_dd")
            else:
                disp = d.get("discharge_disposition_id", 1)
        with c2:
            n_lab = st.number_input("จำนวนการตรวจ lab", 0, 132, int(d.get("num_lab_procedures", 40)), key=f"{key}_lab")
            n_proc = st.number_input("จำนวนหัตถการ", 0, 6, int(d.get("num_procedures", 0)), key=f"{key}_proc")
            n_med = st.number_input("จำนวนรายการยาทั้งหมด", 1, 81, int(d.get("num_medications", 12)), key=f"{key}_med")
            n_diag = st.number_input("จำนวนการวินิจฉัย", 1, 16, int(d.get("number_diagnoses", 7)), key=f"{key}_ndx")
        with c3:
            n_inp = st.number_input("จำนวนครั้งนอน รพ. (12 เดือนก่อน admit)", 0, 21,
                                    int(d.get("number_inpatient", 0)), key=f"{key}_inp",
                                    help="รวมการนอน รพ. ที่อื่นด้วย ระบบเติมค่าเริ่มต้นจากประวัติในระบบให้")
            n_er = st.number_input("จำนวนครั้ง emergency (12 เดือนก่อน)", 0, 76,
                                   int(d.get("number_emergency", 0)), key=f"{key}_er")
            n_opd = st.number_input("จำนวนครั้ง outpatient (12 เดือนก่อน)", 0, 42,
                                    int(d.get("number_outpatient", 0)), key=f"{key}_opd")
            exp_los = (st.number_input("คาดว่าจะนอนทั้งหมด (วัน)", 1, MAX_STAY_DAYS,
                                       int(d.get("expected_los") or 3), key=f"{key}_elos",
                                       help="โมเดลเรียนรู้จากวันนอนทั้งหมดตอนจำหน่าย ระหว่างนอน รพ. จึงใช้ค่าที่มากกว่า"
                                            "ระหว่างวันนอนจนถึงวันนี้กับค่านี้ พอจำหน่ายแล้วจะใช้วันนอนจริง")
                       if ask_expected_los else d.get("expected_los"))

        st.markdown("**การวินิจฉัยและผลตรวจ**")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            dx1 = pick("กลุ่มโรควินิจฉัยหลัก", DIAG_GROUPS, d.get("diag_1", "Circulatory"), key=f"{key}_dx1")
        dx2 = c2.selectbox("กลุ่มโรควินิจฉัยรอง", DIAG_GROUPS,
                           index=DIAG_GROUPS.index(d.get("diag_2", "Missing")), key=f"{key}_dx2")
        dx3 = c3.selectbox("กลุ่มโรควินิจฉัยที่ 3", DIAG_GROUPS,
                           index=DIAG_GROUPS.index(d.get("diag_3", "Missing")), key=f"{key}_dx3")
        glu_opts, a1c_opts = ["None", "Norm", ">200", ">300"], ["None", "Norm", ">7", ">8"]
        glu = c4.selectbox("ผล max glucose serum", glu_opts,
                           index=glu_opts.index(d.get("max_glu_serum", "None")), key=f"{key}_glu",
                           format_func=lambda x: "ไม่ได้ตรวจ" if x == "None" else x)
        a1c = c5.selectbox("ผล A1C", a1c_opts, index=a1c_opts.index(d.get("A1Cresult", "None")),
                           key=f"{key}_a1c", format_func=lambda x: "ไม่ได้ตรวจ" if x == "None" else x)

        st.markdown("**ยาเบาหวาน**")
        meds = {}
        cols = st.columns(4)
        for i, m in enumerate(MAIN_MEDS):
            with cols[i % 4]:
                meds[m] = pick(m, MED_STATUS, d.get(m, "No"), lambda x: MED_STATUS_TH[x], f"{key}_{m}")
        ch_opts = ["auto", "Ch", "No"]
        ch_th = {"auto": "อัตโนมัติ (มีการเพิ่ม/ลดขนาดยา = มีการปรับ)", "Ch": "มีการปรับ/เปลี่ยนยา", "No": "ไม่มีการปรับยา"}
        change = st.radio("การปรับยาเบาหวานระหว่างนอน รพ.", ch_opts, horizontal=True,
                          index=ch_opts.index(d.get("change") or "auto"), format_func=lambda x: ch_th[x],
                          key=f"{key}_ch", help="รวมการเปลี่ยนชนิดยาด้วย ไม่ใช่แค่ปรับขนาด")
        submitted = st.form_submit_button(submit_label, type="primary", use_container_width=True)

    data = {m: d.get(m, "No") for m in MED_COLS}   # ยาอื่นที่ไม่แสดงในฟอร์ม คงค่าเดิม
    data.update(meds)
    data.update({
        "admission_type_id": adm_type, "admission_source_id": adm_src, "medical_specialty": spec,
        "discharge_disposition_id": disp, "num_lab_procedures": n_lab, "num_procedures": n_proc,
        "num_medications": n_med, "number_diagnoses": n_diag, "number_inpatient": n_inp,
        "number_emergency": n_er, "number_outpatient": n_opd, "diag_1": dx1, "diag_2": dx2,
        "diag_3": dx3, "max_glu_serum": glu, "A1Cresult": a1c,
        "change": None if change == "auto" else change, "diabetesMed": None,
        "expected_los": exp_los,
    })
    return submitted, data, extra


def assess_and_save(patient, encounter_id, stage):
    """ประเมินความเสี่ยงจากข้อมูลล่าสุดในระบบแล้วบันทึกผล"""
    if engine is None:
        flash("บันทึกข้อมูลแล้ว แต่ยังประเมินความเสี่ยงไม่ได้ (โหลดโมเดลไม่สำเร็จ)", "warning")
        return
    encs = reg.get_encounters(patient["patient_id"])
    pos = next(i for i, e in enumerate(encs) if e["encounter_id"] == encounter_id)
    result = engine.predict(patient, encs[:pos + 1])   # ใช้เฉพาะประวัติจนถึงครั้งที่ประเมิน
    reg.add_assessment(encounter_id, stage, result)
    flash(f"บันทึกข้อมูลและประเมินความเสี่ยงแล้ว ({stage})")
    go("result", patient["patient_id"], encounter_id)   # เปิดหน้าผลการประเมิน


def inpatient_in_system_last_year(encs, admission_date):
    start = admission_date - dt.timedelta(days=365)
    return sum(1 for e in encs if e["discharge_date"] and start <= to_date(e["admission_date"]) < admission_date)


# ============================================================
# Page: รายชื่อผู้ป่วย
# ============================================================
def page_list():
    st.title("รายชื่อผู้ป่วย")
    show_flash()
    df = reg.list_patients()
    if df.empty:
        st.info("ยังไม่มีผู้ป่วยในระบบ — เริ่มจากปุ่ม **➕ ลงทะเบียนผู้ป่วยใหม่** ที่แถบด้านซ้าย (กด » มุมซ้ายบนเพื่อเปิด)")
        return
    latest = reg.latest_assessment_by_patient()
    df = df.merge(latest, on="patient_id", how="left")

    c1, c2, c3 = st.columns(3)
    c1.metric("ผู้ป่วยทั้งหมด", len(df))
    c2.metric("กำลังนอนโรงพยาบาล", int((df["n_open"] > 0).sum()))
    c3.metric("ความเสี่ยง 30 วันระดับสูง", int((df["tier_30d"] == "High").sum()))

    q = st.text_input("🔍 ค้นหา HN หรือชื่อ")
    if q:
        df = df[df["hn"].str.contains(q, case=False) | df["name"].fillna("").str.contains(q, case=False)]

    def status(r):
        if r["n_encounters"] == 0:
            return "ยังไม่เคย admit"
        return "🛏️ กำลังนอน รพ." if r["n_open"] > 0 else "จำหน่ายแล้ว"

    def fmt_risk(tier, p):
        return "—" if pd.isna(tier) else f"{TIER_CONFIG[tier]['emoji']} {TIER_CONFIG[tier]['label']} ({p*100:.0f}%)"

    table = pd.DataFrame({
        "HN": df["hn"], "ชื่อ": df["name"].fillna(""),
        "สถานะ": df.apply(status, axis=1),
        "จำนวนครั้งที่ admit": df["n_encounters"],
        "admit ล่าสุด": df["last_admission"].fillna("—"),
        "ความเสี่ยง 30 วัน": [fmt_risk(t, p) for t, p in zip(df["tier_30d"], df["p_30d"])],
        "ความเสี่ยงรักษาซ้ำ": [fmt_risk(t, p) for t, p in zip(df["tier_any"], df["p_any"])],
    })
    st.dataframe(table, hide_index=True, use_container_width=True)

    options = df["patient_id"].tolist()
    labels = {r.patient_id: f"{r.hn} — {r.name or 'ไม่ระบุชื่อ'}" for r in df.itertuples()}
    c1, c2 = st.columns([3, 1])
    pid = c1.selectbox("เลือกผู้ป่วย", options, format_func=lambda x: labels[x], label_visibility="collapsed")
    if c2.button("เปิดข้อมูลผู้ป่วย", type="primary", use_container_width=True):
        go("patient", pid)


# ============================================================
# Page: ลงทะเบียน
# ============================================================
def page_register():
    st.title("ลงทะเบียนผู้ป่วยใหม่")
    with st.form("register"):
        c1, c2 = st.columns(2)
        hn = c1.text_input("HN *")
        name = c2.text_input("ชื่อ-นามสกุล (ไม่บังคับ)")
        this_year = dt.date.today().year
        birth_year = c1.number_input("ปีเกิด (ค.ศ.) *", 1900, this_year, this_year - 65)
        gender = c2.selectbox("เพศ *", ["Female", "Male"], format_func=lambda x: {"Female": "หญิง", "Male": "ชาย"}[x])
        race = c1.selectbox("เชื้อชาติ (ตามหมวดของ dataset)",
                            ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "Missing"],
                            format_func=lambda x: "ไม่ระบุ" if x == "Missing" else x)
        ok = st.form_submit_button("ลงทะเบียน", type="primary")
    st.caption("ข้อมูลเก็บในไฟล์ฐานข้อมูลบนเครื่องนี้เท่านั้น — การใช้งานจริงต้องมีระบบรักษาความปลอดภัยและปฏิบัติตาม PDPA")
    if ok:
        try:
            pid = reg.add_patient(hn, name, birth_year, gender, race)
        except ValueError as e:
            st.error(str(e))
            return
        flash(f"ลงทะเบียน HN {hn} แล้ว — เริ่มบันทึกการ admit ได้เลย")
        go("patient", pid, push=False)   # ย้อนกลับจากหน้าผู้ป่วยจะไปหน้าก่อนลงทะเบียน ไม่ใช่ฟอร์มว่าง


# ============================================================
# Page: ผู้ป่วยรายคน
# ============================================================
RACES = ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "Missing"]


def manage_patient(patient, encs):
    """แก้ไขข้อมูลผู้ป่วย / ลบการ admit ล่าสุด / ลบผู้ป่วย"""
    pid = patient["patient_id"]
    with st.expander("⚙️ แก้ไขหรือลบข้อมูลผู้ป่วย"):
        t_edit, t_enc, t_del = st.tabs(["✏️ แก้ไขข้อมูลผู้ป่วย", "↩️ ลบการ admit ล่าสุด", "🗑️ ลบผู้ป่วย"])
        with t_edit:
            with st.form(f"edit_{pid}"):
                c1, c2 = st.columns(2)
                hn = c1.text_input("HN *", value=patient["hn"])
                name = c2.text_input("ชื่อ-นามสกุล", value=patient["name"] or "")
                by = c1.number_input("ปีเกิด (ค.ศ.) *", 1900, dt.date.today().year, int(patient["birth_year"]))
                with c2:
                    gender = pick("เพศ *", ["Female", "Male"], patient["gender"],
                                  lambda x: {"Female": "หญิง", "Male": "ชาย"}[x], f"edit_g_{pid}")
                with c1:
                    race = pick("เชื้อชาติ", RACES, patient["race"],
                                lambda x: "ไม่ระบุ" if x == "Missing" else x, f"edit_r_{pid}")
                ok = st.form_submit_button("บันทึกการแก้ไข", type="primary")
            st.caption("การแก้ไขมีผลกับการประเมินครั้งต่อไป ผลการประเมินที่บันทึกไว้แล้วจะไม่เปลี่ยน")
            if ok:
                try:
                    reg.update_patient(pid, hn, name, by, gender, race)
                except ValueError as e:
                    st.error(str(e))
                else:
                    flash("บันทึกการแก้ไขข้อมูลผู้ป่วยแล้ว")
                    st.rerun()

        with t_enc:
            if not encs:
                st.info("ยังไม่มีการ admit")
            else:
                last = encs[-1]
                st.markdown(f"การ admit ครั้งล่าสุด: **{last['admission_date']}** "
                            f"({'กำลังนอน รพ.' if last['status'] == 'admitted' else 'จำหน่าย ' + last['discharge_date']})")
                st.caption("ใช้เมื่อบันทึกผิด ลบได้เฉพาะครั้งล่าสุด เพื่อไม่ให้ลำดับประวัติของครั้งอื่นผิดไป "
                           "ผลการประเมินของครั้งนี้จะถูกลบด้วย")
                sure = st.checkbox("ยืนยันการลบการ admit ครั้งนี้", key=f"del_enc_ok_{last['encounter_id']}")
                if st.button("ลบการ admit ล่าสุด", disabled=not sure, key=f"del_enc_{last['encounter_id']}") and sure:
                    reg.delete_last_encounter(pid)
                    flash(f"ลบการ admit วันที่ {last['admission_date']} แล้ว")
                    st.rerun()

        with t_del:
            st.warning(f"จะลบผู้ป่วย HN {patient['hn']} พร้อมการ admit {len(encs)} ครั้งและผลการประเมินทั้งหมด "
                       "ย้อนกลับไม่ได้")
            typed = st.text_input("พิมพ์ HN เพื่อยืนยัน", key=f"del_confirm_{pid}")
            confirmed = typed.strip() == patient["hn"]
            if st.button("ลบผู้ป่วย", type="primary", disabled=not confirmed, key=f"del_pt_{pid}") and confirmed:
                reg.delete_patient(pid)
                st.session_state.history = [h for h in st.session_state.history if h[1] != pid]
                st.session_state.pid, st.session_state.eid = None, None
                flash(f"ลบผู้ป่วย HN {patient['hn']} แล้ว")
                go("list", push=False)

def page_patient():
    pid = st.session_state.pid
    patient = reg.get_patient(pid) if pid else None
    if not patient:
        go("list", push=False)
    encs = reg.get_encounters(pid)
    outcomes = derive_outcomes(encs)
    today = dt.date.today()

    st.title(f"👤 {patient['name'] or 'ไม่ระบุชื่อ'}  ·  HN {patient['hn']}")
    st.caption(f"อายุ {today.year - patient['birth_year']} ปี · "
               f"{'หญิง' if patient['gender'] == 'Female' else 'ชาย'} · {patient['race']} · "
               f"admit ในระบบ {len(encs)} ครั้ง")
    show_flash()
    manage_patient(patient, encs)

    open_enc = next((e for e in encs if e["status"] == "admitted"), None)

    # ---------------- การ admit ปัจจุบัน ----------------
    if open_enc:
        los = length_of_stay(open_enc, today)
        st.subheader(f"🛏️ กำลังนอนโรงพยาบาล — admit {open_enc['admission_date']} (วันที่ {los})")
        if los > MAX_STAY_DAYS:
            st.info(f"นอนเกิน {MAX_STAY_DAYS} วัน ซึ่งเกินช่วงของข้อมูลที่ใช้เทรน — โมเดลจะใช้ค่า {MAX_STAY_DAYS} วัน")
        tab_update, tab_dc = st.tabs(["📝 อัปเดตอาการ / ผลตรวจ / ยา", "🏁 จำหน่ายผู้ป่วย"])
        with tab_update:
            sub, data, _ = encounter_form(f"upd_{open_enc['encounter_id']}", open_enc["data"],
                                          "💾 บันทึกและประเมินความเสี่ยงใหม่")
            if sub:
                reg.update_encounter(open_enc["encounter_id"], data)
                assess_and_save(patient, open_enc["encounter_id"], "ระหว่างนอน รพ.")
                st.rerun()   # กรณีโหลดโมเดลไม่สำเร็จ
        with tab_dc:
            with st.form(f"dc_{open_enc['encounter_id']}"):
                c1, c2 = st.columns(2)
                dc_date = c1.date_input("วันที่จำหน่าย", value=max(today, to_date(open_enc["admission_date"])),
                                        min_value=to_date(open_enc["admission_date"]))
                with c2:
                    disp = pick("ลักษณะการจำหน่าย", seen("discharge_disposition_id", DISCHARGE_DISPOSITION_MAP),
                                open_enc["data"].get("discharge_disposition_id", 1),
                                lambda x: DISCHARGE_DISPOSITION_MAP[x], f"dcd_{open_enc['encounter_id']}")
                st.caption("ระบบจะประเมินความเสี่ยงครั้งสุดท้ายด้วยข้อมูลล่าสุด ใช้วางแผนการติดตามหลังจำหน่าย")
                ok = st.form_submit_button("ยืนยันการจำหน่าย", type="primary")
            if ok:
                data = dict(open_enc["data"], discharge_disposition_id=disp)
                try:
                    reg.discharge(open_enc["encounter_id"], dc_date, data)
                except ValueError as e:
                    st.error(str(e))
                    st.stop()
                assess_and_save(patient, open_enc["encounter_id"], "ตอนจำหน่าย")
                st.rerun()

        latest_summary(pid, open_enc)
    else:
        st.subheader("เริ่มการ admit ใหม่")
        if encs:
            last = encs[-1]
            st.caption(f"จำหน่ายครั้งล่าสุด: {last['discharge_date']}")

        adm_date = st.date_input("วันที่ admit", value=max(today, to_date(encs[-1]["discharge_date"])) if encs else today,
                                 min_value=to_date(encs[-1]["discharge_date"]) if encs else None)
        if encs:
            gap = (adm_date - to_date(encs[-1]["discharge_date"])).days
            st.info(f"จะบันทึกเป็นการกลับมารักษาซ้ำ **{gap} วัน** หลังจำหน่ายครั้งก่อน "
                    f"({'ภายใน 30 วัน' if gap <= 30 else 'หลัง 30 วัน'})")

        defaults = dict(encs[-1]["data"]) if encs else {}
        # ค่าเริ่มต้นของจำนวนครั้งนอน รพ. = อย่างน้อยเท่าจำนวนที่บันทึกในระบบช่วง 12 เดือนก่อนวัน admit
        defaults["number_inpatient"] = max(int(defaults.get("number_inpatient", 0)),
                                           inpatient_in_system_last_year(encs, adm_date))
        defaults["change"] = None
        sub, data, _ = encounter_form(f"new_enc_{len(encs)}", defaults, "เปิดการ admit และประเมินความเสี่ยงแรกรับ")
        if encs:
            st.caption("ฟอร์มเติมค่าเริ่มต้นจากการ admit ครั้งก่อนให้แล้ว กรุณาตรวจสอบและแก้ไขให้ตรงกับครั้งนี้")
        if sub:
            try:
                eid = reg.add_encounter(pid, adm_date, data)
            except ValueError as e:
                st.error(str(e))
                st.stop()
            assess_and_save(patient, eid, "แรกรับ")
            st.rerun()

    # ---------------- ประวัติ ----------------
    if encs:
        st.divider()
        st.subheader("ประวัติการรักษาในระบบ")
        ass = reg.get_assessments(pid)
        final = (ass.sort_values("assessment_id").groupby("encounter_id").tail(1).set_index("encounter_id")
                 if not ass.empty else pd.DataFrame())
        rows = []
        for i, (e, o) in enumerate(zip(encs, outcomes), start=1):
            fa = final.loc[e["encounter_id"]] if e["encounter_id"] in final.index else None
            if e["status"] == "admitted":
                outcome = "กำลังนอน รพ."
            elif o["gap_days"] is None:
                outcome = "ยังไม่กลับมา"
            else:
                outcome = f"กลับมาใน {o['gap_days']} วัน" + (" ⚠️" if o["label"] == "<30" else "")
            rows.append({
                "ครั้งที่": i, "admit": e["admission_date"], "จำหน่าย": e["discharge_date"] or "—",
                "วันนอน": length_of_stay(e, today), "โรคหลัก": e["data"].get("diag_1", ""),
                "ความเสี่ยง 30 วัน (ล่าสุด)": "—" if fa is None else
                f"{TIER_CONFIG[fa['tier_30d']]['emoji']} {fa['p_30d']*100:.0f}%",
                "ความเสี่ยงรักษาซ้ำ (ล่าสุด)": "—" if fa is None else
                f"{TIER_CONFIG[fa['tier_any']]['emoji']} {fa['p_any']*100:.0f}%",
                "ผลลัพธ์จริง": outcome,
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        assessed = [e for e in encs if e["encounter_id"] in final.index]
        if assessed and engine is not None:
            c1, c2 = st.columns([3, 1])
            pos = {e["encounter_id"]: i for i, e in enumerate(encs, start=1)}
            eid_view = c1.selectbox("ดูผลการประเมินฉบับเต็มของ", [e["encounter_id"] for e in assessed][::-1],
                                    format_func=lambda x: f"admit ครั้งที่ {pos[x]} "
                                    f"({next(e['admission_date'] for e in encs if e['encounter_id'] == x)})",
                                    label_visibility="collapsed")
            if c2.button("📊 ดูผลการประเมิน", use_container_width=True):
                go("result", pid, eid_view)

        if not ass.empty:
            st.markdown("**แนวโน้มความเสี่ยงจากการประเมินทุกครั้ง**")
            st.altair_chart(trend_chart(ass), use_container_width=True)


def latest_summary(pid, enc):
    """สรุปผลการประเมินล่าสุดแบบย่อในหน้าผู้ป่วย พร้อมปุ่มไปหน้าผลฉบับเต็ม"""
    ass = reg.get_assessments(pid)
    ass = ass[ass["encounter_id"] == enc["encounter_id"]]
    with st.container(border=True):
        if ass.empty:
            st.markdown("ยังไม่มีผลการประเมินสำหรับการ admit ครั้งนี้")
            return
        a = ass.iloc[-1]
        c1, c2, c3 = st.columns([2, 2, 1.3])
        c1.markdown(f"**30 วัน**<br>{tier_badge(a['tier_30d'], a['p_30d'])}", unsafe_allow_html=True)
        c2.markdown(f"**รักษาซ้ำ**<br>{tier_badge(a['tier_any'], a['p_any'])}", unsafe_allow_html=True)
        if engine is not None and c3.button("📊 ดูผลฉบับเต็ม", type="primary", use_container_width=True,
                                            key=f"view_{enc['encounter_id']}"):
            go("result", pid, enc["encounter_id"])
        st.caption(f"ประเมินล่าสุด: {a['stage']} · {a['created_at'][:16].replace('T', ' ')}")


# ============================================================
# Page: ผู้ป่วยที่ต้องติดตามหลังจำหน่าย
# ============================================================
def suggested_action(tier, days_since):
    if tier == "High":
        return ("📞 ติดต่อและนัด follow-up โดยเร็ว (ภายใน 7 วันหลังจำหน่าย)" if days_since <= EARLY_CONTACT_DAYS
                else "ติดตามต่อเนื่องจนครบ 30 วัน")
    if tier == "Medium":
        return "ติดตามตามความเหมาะสม"
    if tier == "Low":
        return "ดูแลตามมาตรฐานปกติ"
    return "ยังไม่มีผลการประเมิน"


def page_followup():
    st.title("🔔 ผู้ป่วยที่ต้องติดตามหลังจำหน่าย")
    st.caption(f"ผู้ป่วยที่จำหน่ายภายใน {FOLLOWUP_WINDOW} วันล่าสุดและยังไม่กลับมารักษา "
               "เรียงตามความเร่งด่วน โดยใช้ผลการประเมินตอนจำหน่าย")
    show_flash()

    df = followup_table()
    if df.empty:
        st.info(f"ไม่มีผู้ป่วยที่จำหน่ายในช่วง {FOLLOWUP_WINDOW} วันที่ผ่านมา")
        return
    active = df[df["returned_after"].isna()].copy()
    returned = df[df["returned_after"].notna()].copy()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("อยู่ในช่วงเฝ้าระวัง", len(active))
    c2.metric("ความเสี่ยง 30 วันระดับสูง", int((active["tier_30d"] == "High").sum()))
    c3.metric("ควรติดต่อภายใน 7 วัน", int(((active["tier_30d"] == "High") &
                                             (active["days_since"] <= EARLY_CONTACT_DAYS)).sum()))
    c4.metric("กลับมารักษาภายใน 30 วันแล้ว", int((returned["returned_after"] <= FOLLOWUP_WINDOW).sum()))

    tier_th = {t: f"{c['emoji']} {c['label']}" for t, c in TIER_CONFIG.items()}
    show = st.multiselect("แสดงระดับความเสี่ยง", ["High", "Medium", "Low"], default=["High", "Medium"],
                          format_func=lambda t: tier_th[t])
    view = active[active["tier_30d"].isin(show) | (active["tier_30d"].isna() & bool(show))].copy()
    if view.empty:
        st.success("ไม่มีผู้ป่วยในระดับความเสี่ยงที่เลือก")
    else:
        rank = {"High": 0, "Medium": 1, "Low": 2}
        view["_r"] = view["tier_30d"].map(rank).fillna(3)
        view = view.sort_values(["_r", "p_30d", "days_since"], ascending=[True, False, True])
        today = dt.date.today()
        table = pd.DataFrame({
            "HN": view["hn"], "ชื่อ": view["name"].fillna(""),
            "อายุ": [today.year - b for b in view["birth_year"]],
            "จำหน่ายเมื่อ": view["discharge_date"],
            "ผ่านมา (วัน)": view["days_since"],
            "เหลือช่วง 30 วัน": [max(FOLLOWUP_WINDOW - d, 0) for d in view["days_since"]],
            "ความเสี่ยง 30 วัน": ["—" if pd.isna(t) else f"{tier_th[t]} ({p*100:.0f}%)"
                                  for t, p in zip(view["tier_30d"], view["p_30d"])],
            "ความเสี่ยงรักษาซ้ำ": ["—" if pd.isna(t) else f"{TIER_CONFIG[t]['emoji']} {p*100:.0f}%"
                                   for t, p in zip(view["tier_any"], view["p_any"])],
            "จำหน่ายไปที่": [DISCHARGE_DISPOSITION_MAP.get(int(x), "—") if pd.notna(x) else "—"
                             for x in view["disposition_id"]],
            "สิ่งที่ควรทำ": [suggested_action(t, d) for t, d in zip(view["tier_30d"], view["days_since"])],
        })
        st.dataframe(table, hide_index=True, use_container_width=True)

        labels = {r.patient_id: f"{r.hn} — {r.name or 'ไม่ระบุชื่อ'}" for r in view.itertuples()}
        c1, c2 = st.columns([3, 1])
        pid = c1.selectbox("เลือกผู้ป่วย", list(dict.fromkeys(view["patient_id"])), format_func=lambda x: labels[x],
                           label_visibility="collapsed", key="fu_pick")
        if c2.button("เปิดข้อมูลผู้ป่วย", type="primary", use_container_width=True, key="fu_open"):
            go("patient", pid)
    st.caption("แนวทางอ้างอิงจาก AHRQ IDEAL Discharge Planning Framework และ CMS Care Coordination Best Practices · "
               "ผู้ป่วยอายุ 80 ปีขึ้นไปอาจถูกประเมินความเสี่ยงต่ำกว่าความจริง ควรพิจารณาร่วมด้วยเสมอ")

    if not returned.empty:
        with st.expander(f"ผู้ป่วยที่กลับมารักษาแล้วในช่วงนี้ ({len(returned)} ราย) — ใช้ทบทวนว่าระบบประเมินไว้อย่างไร"):
            st.dataframe(pd.DataFrame({
                "HN": returned["hn"], "ชื่อ": returned["name"].fillna(""),
                "จำหน่ายเมื่อ": returned["discharge_date"],
                "กลับมาหลังจำหน่าย (วัน)": returned["returned_after"].astype(int),
                "ความเสี่ยง 30 วันตอนจำหน่าย": ["—" if pd.isna(t) else f"{tier_th[t]} ({p*100:.0f}%)"
                                               for t, p in zip(returned["tier_30d"], returned["p_30d"])],
            }), hide_index=True, use_container_width=True)


# ============================================================
# Page: ผลการประเมิน
# ============================================================
def page_result():
    pid, eid = st.session_state.pid, st.session_state.eid
    patient = reg.get_patient(pid) if pid else None
    encs = reg.get_encounters(pid) if patient else []
    pos = next((i for i, e in enumerate(encs) if e["encounter_id"] == eid), None)
    if patient is None or pos is None:
        go("list", push=False)
    enc = encs[pos]
    today = dt.date.today()

    if st.button("👤 กลับไปหน้าผู้ป่วย", key="to_patient"):
        go("patient", pid)
    st.title("ผลการประเมินความเสี่ยง")
    st.caption(f"{patient['name'] or 'ไม่ระบุชื่อ'} · HN {patient['hn']} · admit ครั้งที่ {pos + 1} "
               f"({enc['admission_date']}"
               + (f" – {enc['discharge_date']}" if enc["discharge_date"] else ", กำลังนอน รพ.") + ")")
    show_flash()

    if engine is None:
        st.error("โหลดโมเดลไม่สำเร็จ — ยังแสดงผลการประเมินไม่ได้")
        return
    if enc["status"] == "admitted":
        los_now, los_model = length_of_stay(enc, today), min(los_for_model(enc, today), MAX_STAY_DAYS)
        st.info(f"**ผลเบื้องต้นระหว่างนอนโรงพยาบาล** — ข้อมูลบางส่วนยังเป็นค่าคาดการณ์ "
                f"(วันนอนที่ใช้คำนวณ {los_model} วัน, ปัจจุบันวันที่ {los_now}; ลักษณะการจำหน่าย) "
                f"ผลที่ใช้วางแผนติดตามหลังจำหน่ายควรใช้ผลตอนจำหน่าย")
    result = engine.predict(patient, encs[:pos + 1], today, n_factors=10)
    render_risk(result, patient, enc)

    ass = reg.get_assessments(pid)
    if len(ass) > 1:
        with st.container(border=True):
            st.subheader("แนวโน้มความเสี่ยงของผู้ป่วยรายนี้")
            st.altair_chart(trend_chart(ass), use_container_width=True)


# ============================================================
# Router
# ============================================================
collapse_sidebar_if_requested()
top_nav()
{"list": page_list, "register": page_register, "patient": page_patient,
 "result": page_result, "followup": page_followup}[st.session_state.page]()
