"""validate_dashboard.py — ตรวจว่า dashboard ให้ผลเท่ากับโมเดลใน notebook 15

วิธีตรวจ: นำผู้ป่วยจริงจาก test set มาป้อนเข้า RiskEngine ของ dashboard ทีละ admit (เหมือนแพทย์บันทึกเอง)
แล้วเทียบความเสี่ยงที่ได้กับ outputs/15_final_*/test_predictions.csv ของ encounter เดียวกัน

dashboard รู้ประวัติจาก "วันที่" ไม่ใช่ label ของ dataset จึงสร้างวันที่จำลองให้สอดคล้องกับ label:
    ครั้งก่อนเป็น <30  → admit ครั้งถัดไปหลังจำหน่าย 15 วัน
    ครั้งก่อนเป็น >30  → 90 วัน
    ครั้งก่อนเป็น NO แต่มี admit ครั้งถัดไป → 400 วัน (dashboard จะนับเป็น >30 จึงต่างจาก dataset แน่นอน)

ผู้ป่วยที่ไม่ตรง สคริปต์จะบอกสาเหตุที่เป็นไปได้จากข้อมูล (ไม่ใช่ความผิดพลาดของระบบ):
    - race / gender ต่างกันระหว่าง encounter (dashboard เก็บเป็นข้อมูลของผู้ป่วย 1 ค่า)
    - ช่วงอายุใน dataset ไม่ต่อเนื่องกับวันที่จำลอง
    - label ครั้งก่อนเป็น NO ทั้งที่มี admit ถัดไป

วิธีรัน (จากโฟลเดอร์ dashboard):
    python validate_dashboard.py                  # สุ่มผู้ป่วย test 300 คน
    python validate_dashboard.py --n-patients 0   # ทุกคนใน test (ช้า)
ผลบันทึกที่ ../outputs/16_slide_figures/dashboard_validation.csv
"""
import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from readmission_registry import RiskEngine, age_group

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
RAW_PATH = ROOT / "data" / "diabetic_data.csv"
SPLIT_PATH = ROOT / "outputs" / "split_assignment.csv"
MODEL_ROOT = ROOT / "models"
OUT_PATH = ROOT / "outputs" / "16_slide_figures" / "dashboard_validation.csv"

PATIENT_COLS = ["encounter_id", "patient_nbr", "race", "gender", "age", "weight", "payer_code",
                "time_in_hospital", "readmitted"]
GAP_DAYS = {"<30": 15, ">30": 90, "NO": 400}
START = dt.date(2000, 1, 1)
TOL = 1e-4


def build_patient(rows):
    """แปลงแถวของ dataset (ผู้ป่วย 1 คน เรียงตาม encounter_id) → patient + encounters แบบที่ dashboard เก็บ"""
    first = rows.iloc[0]
    lo = int(first["age"].strip("[)").split("-")[0])
    patient = {"patient_id": int(first["patient_nbr"]), "race": first["race"], "gender": first["gender"],
               "birth_year": START.year - (lo + 1)}          # อายุ ณ admit แรก = ขอบล่างของช่วง + 1 ปี
    encounters, cur = [], START
    for _, r in rows.iterrows():
        adm = cur
        dis = adm + dt.timedelta(days=int(r["time_in_hospital"]))
        data = {k: (None if pd.isna(v) else v) for k, v in r.items() if k not in PATIENT_COLS}
        encounters.append({"admission_date": adm, "discharge_date": dis, "status": "discharged", "data": data,
                           "_encounter_id": int(r["encounter_id"]), "_age": r["age"]})
        cur = dis + dt.timedelta(days=GAP_DAYS[r["readmitted"]])
    return patient, encounters


def suspected_causes(rows, patient, encounters):
    causes = []
    if rows["race"].fillna("Missing").nunique() > 1:
        causes.append("race ต่างกันระหว่าง encounter")
    if rows["gender"].nunique() > 1:
        causes.append("gender ต่างกันระหว่าง encounter")
    if any(age_group(patient["birth_year"], e["admission_date"]) != e["_age"] for e in encounters):
        causes.append("ช่วงอายุไม่สอดคล้องกับวันที่")
    if (rows["readmitted"].iloc[:-1] == "NO").any():
        causes.append("ครั้งก่อนเป็น NO แต่มี admit ถัดไป")
    return causes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-patients", type=int, default=300, help="จำนวนผู้ป่วยที่สุ่มจาก test (0 = ทั้งหมด)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(RAW_PATH, keep_default_na=False, na_values=["?"], low_memory=False)
    df = df[df["gender"] != "Unknown/Invalid"]
    df = df[~df["discharge_disposition_id"].isin([11, 13, 14, 19, 20, 21])]
    df = df.sort_values(["patient_nbr", "encounter_id"]).reset_index(drop=True)

    split = pd.read_csv(SPLIT_PATH)
    test_ids = set(split.loc[split["split"] == "test", "encounter_id"])
    df = df[df["encounter_id"].isin(test_ids)]
    pts = df["patient_nbr"].unique()
    if args.n_patients:
        pts = np.random.default_rng(args.seed).choice(pts, size=min(args.n_patients, len(pts)), replace=False)

    ref = {lab: pd.read_csv(ROOT / "outputs" / f"15_final_{lab}" / "test_predictions.csv").set_index("encounter_id")
           for lab in ("any", "30d")}
    engine = RiskEngine(str(MODEL_ROOT))
    groups = dict(tuple(df[df["patient_nbr"].isin(pts)].groupby("patient_nbr")))

    out = []
    for n, pid in enumerate(pts, 1):
        rows = groups[pid]
        patient, encs = build_patient(rows)
        causes = suspected_causes(rows, patient, encs)
        for k, e in enumerate(encs):
            res = engine.predict(patient, encs[: k + 1])
            eid = e["_encounter_id"]
            rec = {"patient_nbr": pid, "encounter_id": eid, "visit": k + 1, "n_visits": len(encs)}
            for lab in ("any", "30d"):
                r = ref[lab].loc[eid]
                rec[f"p_{lab}_dashboard"] = res[lab]["p"]
                rec[f"p_{lab}_nb15"] = r["proba_calibrated"]
                rec[f"tier_{lab}_dashboard"] = res[lab]["tier"]
                rec[f"tier_{lab}_nb15"] = r["risk_tier"]
                rec[f"match_{lab}"] = (abs(res[lab]["p"] - r["proba_calibrated"]) < TOL
                                       and res[lab]["tier"] == r["risk_tier"])
            rec["match"] = rec["match_any"] and rec["match_30d"]
            rec["suspected_cause"] = "; ".join(causes)
            out.append(rec)
        if n % 50 == 0:
            print(f"  ตรวจแล้ว {n}/{len(pts)} คน")

    res = pd.DataFrame(out)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT_PATH, index=False)

    by_pt = res.groupby("patient_nbr").agg(match=("match", "all"), cause=("suspected_cause", "first"))
    print(f"\nผู้ป่วยที่ผลตรงกับ notebook 15 ทุก encounter: {by_pt['match'].sum()} / {len(by_pt)} คน")
    print(f"encounter ที่ตรง: {res['match'].sum()} / {len(res)}")
    bad = by_pt[~by_pt["match"]]
    if len(bad):
        print("\nสาเหตุที่เป็นไปได้ของผู้ป่วยที่ไม่ตรง:")
        print(bad["cause"].replace("", "ไม่ทราบสาเหตุ — ควรตรวจเพิ่ม").value_counts().to_string())
    print(f"\nบันทึกรายละเอียดที่ {OUT_PATH}")


if __name__ == "__main__":
    main()
