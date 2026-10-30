"""readmission_features.py — cleaning + feature engineering ที่ใช้ร่วมกันใน notebook 12–14

ทุกขั้นตอนเหมือน 11_relabel_any_readmission.ipynb ทุกบรรทัด (LabelEncoder ถูก fit บนข้อมูลทั้งชุด
แบบเดียวกับ 11 จึงได้รหัสตรงกัน ใช้กับโมเดลที่บันทึกจาก 11 ได้ทันที)
วางไฟล์นี้ไว้โฟลเดอร์เดียวกับ notebook
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

MED_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
]
AGE_MAP = {"[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35, "[40-50)": 45,
           "[50-60)": 55, "[60-70)": 65, "[70-80)": 75, "[80-90)": 85, "[90-100)": 95}


def map_diagnosis(code):
    if pd.isna(code):
        return "Missing"
    code = str(code)
    if code.startswith("V") or code.startswith("E"):
        return "Other"
    try:
        val = float(code)
    except ValueError:
        return "Other"
    if 390 <= val <= 459 or val == 785:
        return "Circulatory"
    elif 460 <= val <= 519 or val == 786:
        return "Respiratory"
    elif 520 <= val <= 579 or val == 787:
        return "Digestive"
    elif 250 <= val < 251:
        return "Diabetes"
    elif 800 <= val <= 999:
        return "Injury"
    elif 710 <= val <= 739:
        return "Musculoskeletal"
    elif 580 <= val <= 629 or val == 788:
        return "Genitourinary"
    elif 140 <= val <= 239:
        return "Neoplasms"
    else:
        return "Other"


# ---- กลุ่มรหัส ID (อ้างอิง IDS_mapping.csv ของ dataset) ใช้ใน notebook 12 เพื่อการตีความ ----
def group_discharge(i):
    if i == 1: return "Home"
    if i in (6, 8): return "Home health"
    if i == 7: return "Left AMA"
    if i in (2, 3, 4, 5, 15, 22, 23, 24, 27, 28, 29, 30): return "Facility"
    return "Unknown/Other"

def group_admission_type(i):
    if i in (1, 7): return "Emergency"
    if i == 2: return "Urgent"
    if i == 3: return "Elective"
    return "Unknown/Other"

def group_admission_source(i):
    if i == 7: return "Emergency room"
    if i in (1, 2, 3): return "Referral"
    if i in (4, 5, 6, 10, 18, 22, 25, 26): return "Transfer"
    return "Unknown/Other"


def load_prepared(raw_path, label_mode="any"):
    """คืนค่า X (encoded, ไม่มี patient_nbr), y, groups, meta, readable

    meta     = encounter_id, patient_nbr, readmitted_raw (3 ช่วง) — ใช้ merge กับ split_assignment.csv
    readable = DataFrame หลัง feature engineering ก่อน encode (ใช้วิเคราะห์เชิงพรรณนา)
    """
    df = pd.read_csv(raw_path, keep_default_na=False, na_values=["?"], low_memory=False)
    df = df[df["gender"] != "Unknown/Invalid"]
    df = df[~df["discharge_disposition_id"].isin([11, 13, 14, 19, 20, 21])]
    df = df.sort_values(["patient_nbr", "encounter_id"]).reset_index(drop=True)

    meta = df[["encounter_id", "patient_nbr", "readmitted"]].rename(
        columns={"readmitted": "readmitted_raw"}).copy()
    if label_mode == "any":
        df["readmitted"] = (df["readmitted"] != "NO").astype(int)
    elif label_mode == "30d":
        df["readmitted"] = (df["readmitted"] == "<30").astype(int)
    else:
        raise ValueError("label_mode ต้องเป็น 'any' หรือ '30d'")

    med_cols = [c for c in MED_COLS if c in df.columns]
    df["medication_change_count"] = (df[med_cols] == "Up").sum(axis=1) + (df[med_cols] == "Down").sum(axis=1)
    df["num_meds_active"] = (df[med_cols] != "No").sum(axis=1)
    df["tested_glucose"] = (df["max_glu_serum"] != "None").astype(int)
    df["tested_A1C"] = (df["A1Cresult"] != "None").astype(int)

    df["visit_number"] = df.groupby("patient_nbr").cumcount() + 1
    df["change_flag"] = (df["change"] == "Ch").astype(int)
    df["prior_change_count"] = df.groupby("patient_nbr")["change_flag"].cumsum() - df["change_flag"]
    df = df.drop(columns=["change_flag"])

    df["total_prior_visits"] = df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
    df["had_prior_inpatient"] = (df["number_inpatient"] > 0).astype(int)

    diag_groups = pd.DataFrame({c: df[c].apply(map_diagnosis) for c in ["diag_1", "diag_2", "diag_3"]})
    df["comorbidity_count"] = diag_groups.apply(lambda row: row[row != "Missing"].nunique(), axis=1)
    for c in ["diag_1", "diag_2", "diag_3"]:
        df[c] = diag_groups[c]

    df["age_ord"] = df["age"].map(AGE_MAP)
    df["age_x_medications"] = df["age_ord"] * df["num_medications"]
    df["hospital_stay_x_inpatient"] = df["time_in_hospital"] * df["number_inpatient"]

    df.drop(columns=["weight", "payer_code", "encounter_id"], inplace=True)
    df["medical_specialty"] = df["medical_specialty"].fillna("Missing")
    top_specialties = df["medical_specialty"].value_counts().nlargest(10).index
    df["medical_specialty"] = df["medical_specialty"].where(df["medical_specialty"].isin(top_specialties), "Other")
    df["race"] = df["race"].fillna("Missing")

    constant_cols = [c for c in df.columns if df[c].nunique() == 1]
    df.drop(columns=constant_cols, inplace=True)

    readable = df.copy()

    X = df.drop(columns=["readmitted"])
    y = df["readmitted"].copy()
    groups = df["patient_nbr"].values
    categorical_cols = [c for c in X.columns
                        if c != "patient_nbr" and pd.api.types.is_string_dtype(X[c]) and c != "age"]
    X = X.drop(columns=["age"])
    for col in categorical_cols:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    X = X.drop(columns=["patient_nbr"])
    return X, y, groups, meta, readable


def attach_split(meta, split_path):
    """คืน array ชื่อ split ('train'/'val'/'test') เรียงตรงกับแถวของ meta"""
    split = pd.read_csv(split_path)
    merged = meta[["encounter_id"]].merge(split[["encounter_id", "split"]], on="encounter_id", how="left")
    if merged["split"].isna().any():
        raise ValueError("encounter บางตัวไม่มีใน split_assignment.csv — รัน notebook 11 ใหม่ก่อน")
    return merged["split"].values


# ---------------------------------------------------------------------------
# ใช้ใน notebook 12 และ 13: design matrix แบบตีความได้ (หน่วยทางคลินิก + one-hot เทียบกลุ่มอ้างอิง)
# ---------------------------------------------------------------------------
LR_REFS = {
    "gender": "Female", "race": "Caucasian", "admission_type_group": "Emergency",
    "admission_source_group": "Emergency room", "discharge_group": "Home",
    "diag_1": "Circulatory", "A1Cresult": "None", "max_glu_serum": "None",
    "insulin": "No", "metformin": "No", "change": "No", "diabetesMed": "No",
}


def add_id_groups(R):
    """เพิ่มคอลัมน์กลุ่มของ discharge / admission type / admission source ให้ readable df"""
    R = R.copy()
    R["discharge_group"] = R["discharge_disposition_id"].apply(group_discharge)
    R["admission_type_group"] = R["admission_type_id"].apply(group_admission_type)
    R["admission_source_group"] = R["admission_source_id"].apply(group_admission_source)
    return R


def build_lr_design(R, columns=None, min_count=100):
    """R ต้องผ่าน add_id_groups แล้ว | columns = คอลัมน์ของ train (ส่งมาเมื่อสร้างให้ val/test)"""
    D = pd.DataFrame(index=R.index)
    D["age (per 10 y)"] = R["age_ord"] / 10
    D["time_in_hospital (per day)"] = R["time_in_hospital"]
    D["num_lab_procedures (per 10)"] = R["num_lab_procedures"] / 10
    D["num_procedures (per 1)"] = R["num_procedures"]
    D["num_medications (per 5)"] = R["num_medications"] / 5
    D["number_diagnoses (per 1)"] = R["number_diagnoses"]
    for c in ["number_inpatient", "number_emergency", "number_outpatient"]:
        D[f"{c} (per visit, cap 5)"] = R[c].clip(upper=5)
    for c, ref in LR_REFS.items():
        dummies = pd.get_dummies(R[c], prefix=c, prefix_sep=" = ").astype(float)
        dummies = dummies.drop(columns=[f"{c} = {ref}"], errors="ignore")
        if columns is None:
            dummies = dummies.loc[:, dummies.sum() >= min_count]
        D = pd.concat([D, dummies], axis=1)
    if columns is not None:
        D = D.reindex(columns=columns, fill_value=0.0)
    return D.astype(float)


# ---------------------------------------------------------------------------
# ใช้ใน notebook 14–15: history features + input ของ neural network + index ของลำดับ
# ---------------------------------------------------------------------------
def build_history_features(meta):
    """ผลของ encounter ก่อนหน้า + จำนวนครั้งที่เคย readmit (ภายใน dataset) — ไม่ใช้ label ของครั้งปัจจุบัน
    meta ต้องเรียงตาม patient_nbr, encounter_id (ตามที่ load_prepared คืนมา)"""
    raw = meta["readmitted_raw"].values
    pid = meta["patient_nbr"].values
    same_prev = np.r_[False, pid[1:] == pid[:-1]]
    prev_raw = np.where(same_prev, np.concatenate([["none"], raw[:-1]]), "none")
    lt30 = (raw == "<30").astype(int)
    anyr = (raw != "NO").astype(int)
    return pd.DataFrame({
        "prev_readmit_lt30": (prev_raw == "<30").astype(int),
        "prev_readmit_gt30": (prev_raw == ">30").astype(int),
        "prev_readmit_no": (prev_raw == "NO").astype(int),
        "n_prior_lt30_in_data": pd.Series(lt30).groupby(pid).cumsum().values - lt30,
        "n_prior_readmit_in_data": pd.Series(anyr).groupby(pid).cumsum().values - anyr,
    })


def build_nn_inputs(readable, hist, train_rows):
    """one-hot categorical + standardize numeric (ด้วยสถิติของ train) + history (log1p สำหรับจำนวนครั้ง)
    คืน (F float32 array, artifacts สำหรับนำไปใช้กับข้อมูลใหม่ เช่น dashboard)"""
    id_cols = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]
    Rn = readable.drop(columns=["patient_nbr", "readmitted", "age"]).copy()
    Rn[id_cols] = Rn[id_cols].astype(str)
    cat_cols = [c for c in Rn.columns if pd.api.types.is_string_dtype(Rn[c])]
    num_cols = [c for c in Rn.columns if c not in cat_cols]
    onehot = pd.get_dummies(Rn[cat_cols], dtype=np.float32)
    num = Rn[num_cols].astype(np.float32)
    mu, sd = num.iloc[train_rows].mean(), num.iloc[train_rows].std().replace(0, 1)
    num = (num - mu) / sd
    h = hist.astype(np.float32).copy()
    cnt = ["n_prior_lt30_in_data", "n_prior_readmit_in_data"]
    h[cnt] = np.log1p(h[cnt])
    F = np.hstack([num.values, onehot.values, h.values]).astype(np.float32)
    artifacts = {"num_cols": num_cols, "cat_cols": cat_cols, "onehot_columns": list(onehot.columns),
                 "mu": mu, "sd": sd, "hist_columns": list(hist.columns)}
    return F, artifacts


def build_seq_index(visit_number, pid, K):
    """(n × K) index ของแถว — encounter ปัจจุบัน + ก่อนหน้าสูงสุด K−1 ครั้ง, right-padded ด้วย n
    คืน (idx, lengths) พร้อมตรวจว่าไม่มีผู้ป่วยปนกันและ step สุดท้ายคือแถวปัจจุบัน"""
    n = len(visit_number)
    L = np.minimum(np.asarray(visit_number) - 1 + 1, K)
    start = np.arange(n) - (L - 1)
    idx = np.full((n, K), n, dtype=np.int64)
    for j in range(K):
        ok = j < L
        idx[ok, j] = start[ok] + j
    assert (idx[np.arange(n), L - 1] == np.arange(n)).all()
    for j in range(K):
        ok = idx[:, j] < n
        assert (pid[idx[ok, j]] == pid[ok]).all()
    return idx, L


# ---------------------------------------------------------------------------
# ใช้ใน dashboard: feature engineering สำหรับข้อมูล encounter ใหม่ (ไม่ต้องโหลด dataset ทั้งชุด)
# ผ่านการตรวจแล้วว่าให้ผลตรงกับ load_prepared ทุกคอลัมน์ (ดู export_dashboard_artifacts.py)
# ---------------------------------------------------------------------------
DIAG_GROUPS = ["Circulatory", "Respiratory", "Digestive", "Diabetes", "Injury",
               "Musculoskeletal", "Genitourinary", "Neoplasms", "Other", "Missing"]


def _map_diag_any(v):
    """รับได้ทั้งรหัส ICD-9 และชื่อกลุ่มที่ map แล้ว"""
    if isinstance(v, str) and v in DIAG_GROUPS:
        return v
    return map_diagnosis(v)


def featurize_raw(df, top_specialties, constant_cols=("examide", "citoglipton")):
    """feature engineering เหมือน load_prepared ทุกขั้นตอน แต่ใช้กับแถวใดๆ ในรูปแบบคอลัมน์ของ dataset ดิบ
    df ต้องมี patient_nbr และเรียงตาม encounter ภายในผู้ป่วยแล้ว
    top_specialties = รายชื่อ 10 แผนกที่พบบ่อยสุดใน training data (ไม่คำนวณใหม่จาก df)"""
    df = df.copy().reset_index(drop=True)
    med_cols = [c for c in MED_COLS if c in df.columns]
    df["medication_change_count"] = (df[med_cols] == "Up").sum(axis=1) + (df[med_cols] == "Down").sum(axis=1)
    df["num_meds_active"] = (df[med_cols] != "No").sum(axis=1)
    df["tested_glucose"] = (df["max_glu_serum"] != "None").astype(int)
    df["tested_A1C"] = (df["A1Cresult"] != "None").astype(int)
    df["visit_number"] = df.groupby("patient_nbr").cumcount() + 1
    change_flag = (df["change"] == "Ch").astype(int)
    df["prior_change_count"] = change_flag.groupby(df["patient_nbr"]).cumsum() - change_flag
    df["total_prior_visits"] = df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
    df["had_prior_inpatient"] = (df["number_inpatient"] > 0).astype(int)
    diag_groups = pd.DataFrame({c: df[c].apply(_map_diag_any) for c in ["diag_1", "diag_2", "diag_3"]})
    df["comorbidity_count"] = diag_groups.apply(lambda row: row[row != "Missing"].nunique(), axis=1)
    for c in ["diag_1", "diag_2", "diag_3"]:
        df[c] = diag_groups[c]
    df["age_ord"] = df["age"].map(AGE_MAP)
    df["age_x_medications"] = df["age_ord"] * df["num_medications"]
    df["hospital_stay_x_inpatient"] = df["time_in_hospital"] * df["number_inpatient"]
    df = df.drop(columns=[c for c in ["weight", "payer_code", "encounter_id"] if c in df.columns])
    df["medical_specialty"] = df["medical_specialty"].fillna("Missing")
    df["medical_specialty"] = df["medical_specialty"].where(df["medical_specialty"].isin(top_specialties), "Other")
    df["race"] = df["race"].fillna("Missing")
    df = df.drop(columns=[c for c in constant_cols if c in df.columns])
    return df


def encode_for_xgb(readable, encoder_classes, x_columns, hist):
    """แปลง readable → feature ของ XGBoost (label encoding ด้วย classes เดียวกับตอนเทรน) + history"""
    X = readable.drop(columns=[c for c in ["age", "patient_nbr", "readmitted"] if c in readable.columns]).copy()
    for col, classes in encoder_classes.items():
        lookup = {v: i for i, v in enumerate(classes)}
        vals = X[col].astype(str)
        unknown = set(vals) - set(lookup)
        if unknown:
            raise ValueError(f"ค่า {unknown} ในคอลัมน์ {col} ไม่เคยพบตอนเทรน")
        X[col] = vals.map(lookup).astype(int)
    X = pd.concat([X.reset_index(drop=True), hist.reset_index(drop=True)], axis=1)
    return X[x_columns]


def nn_features(readable, hist, nn_artifacts):
    """input ของ LSTM สำหรับแถวใหม่ ด้วย artifacts (mu, sd, คอลัมน์ one-hot) ที่บันทึกจาก notebook 15"""
    id_cols = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]
    Rn = readable.drop(columns=[c for c in ["patient_nbr", "readmitted", "age"] if c in readable.columns]).copy()
    Rn[id_cols] = Rn[id_cols].astype(str)
    a = nn_artifacts
    num = (Rn[a["num_cols"]].astype(np.float32) - a["mu"]) / a["sd"]
    onehot = pd.get_dummies(Rn[a["cat_cols"]], dtype=np.float32).reindex(columns=a["onehot_columns"], fill_value=0.0)
    h = hist[a["hist_columns"]].astype(np.float32).copy()
    cnt = ["n_prior_lt30_in_data", "n_prior_readmit_in_data"]
    h[cnt] = np.log1p(h[cnt])
    return np.hstack([num.values, onehot.values, h.values]).astype(np.float32)
