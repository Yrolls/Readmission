"""export_dashboard_artifacts.py — รันครั้งเดียวหลัง notebook 15 (ทั้ง any และ 30d)

สร้าง models/dashboard_artifacts.json ที่ dashboard ต้องใช้ (รายชื่อแผนก top-10, classes ของ label encoder)
และตรวจว่า feature engineering ของ dashboard ให้ผลตรงกับตอนเทรนทุกคอลัมน์

วิธีรัน (จากโฟลเดอร์ notebooks):  python export_dashboard_artifacts.py
"""
import json
import joblib
import numpy as np
import pandas as pd
from readmission_features import (load_prepared, featurize_raw, encode_for_xgb, nn_features,
                                  build_history_features, build_nn_inputs, attach_split)

RAW_PATH = "../data/diabetic_data.csv"
SPLIT_PATH = "../outputs/split_assignment.csv"
OUT_PATH = "../models/dashboard_artifacts.json"
BUNDLES = {"any": "../models/final_any/final_bundle.pkl", "30d": "../models/final_30d/final_bundle.pkl"}

X, y, groups, meta, R = load_prepared(RAW_PATH, "any")
cat_cols = [c for c in R.columns if pd.api.types.is_string_dtype(R[c]) and c not in ("age",)]
encoder_classes = {c: sorted(R[c].astype(str).unique().tolist()) for c in cat_cols}
top_specialties = [s for s in R["medical_specialty"].unique() if s not in ("Other",)]
artifacts = {
    "top_specialties": sorted(top_specialties),
    "encoder_classes": encoder_classes,
    # รหัส ID ที่พบใน training data — dashboard ให้เลือกได้เฉพาะค่าเหล่านี้ (ค่าที่โมเดลไม่เคยเห็นจะทำนายไม่น่าเชื่อถือ)
    "seen_ids": {c: sorted(int(v) for v in R[c].unique())
                 for c in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]},
}

# ---- ตรวจความตรงกัน: featurize_raw บนข้อมูลดิบ ต้องได้ผลเท่ากับ load_prepared ----
raw = pd.read_csv(RAW_PATH, keep_default_na=False, na_values=["?"], low_memory=False)
raw = raw[raw["gender"] != "Unknown/Invalid"]
raw = raw[~raw["discharge_disposition_id"].isin([11, 13, 14, 19, 20, 21])]
raw = raw.sort_values(["patient_nbr", "encounter_id"]).reset_index(drop=True)
R2 = featurize_raw(raw.drop(columns=["readmitted"]), artifacts["top_specialties"])
R_cmp = R.drop(columns=["readmitted"])
pd.testing.assert_frame_equal(R2[R_cmp.columns].reset_index(drop=True), R_cmp.reset_index(drop=True), check_dtype=False)
print("✓ featurize_raw ตรงกับ load_prepared ทุกคอลัมน์")

hist = build_history_features(meta)
split = attach_split(meta, SPLIT_PATH)
for label, path in BUNDLES.items():
    b = joblib.load(path)
    Xd = encode_for_xgb(R2, encoder_classes, b["xgb_feature_names"], hist)
    ref = pd.concat([X, hist], axis=1)[b["xgb_feature_names"]]
    assert np.allclose(Xd.values.astype(float), ref.values.astype(float)), f"XGB features ไม่ตรง ({label})"
    F_ref, _ = build_nn_inputs(R, hist, np.where(split == "train")[0])
    F_new = nn_features(R2, hist, b["nn_artifacts"])
    assert np.allclose(F_new, F_ref, atol=1e-5), f"LSTM features ไม่ตรง ({label})"
    print(f"✓ feature ของ XGBoost และ LSTM ตรงกับตอนเทรน ({label})")

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(artifacts, f, ensure_ascii=False, indent=1)
print(f"บันทึก {OUT_PATH}")
