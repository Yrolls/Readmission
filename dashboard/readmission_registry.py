"""readmission_registry.py — ส่วน backend ของ dashboard (ฐานข้อมูลผู้ป่วย + การคำนวณความเสี่ยง)

แยกจาก UI เพื่อให้ทดสอบได้และแก้ไขง่าย
- Registry   : ฐานข้อมูล SQLite เก็บผู้ป่วย, การ admit (encounter) และผลการประเมินแต่ละครั้ง
- RiskEngine : โหลดโมเดลจาก notebook 15 (label any และ 30d) แล้วคำนวณความเสี่ยงจากประวัติทั้งหมดในระบบ

ประวัติของผู้ป่วย (ใช้โดย LSTM และ history features) สร้างจากข้อมูลในระบบเอง:
ผลลัพธ์ของ admit ครั้งก่อน (<30 / >30 วัน) คำนวณจากวันที่จำหน่ายครั้งก่อนกับวันที่ admit ครั้งถัดมาโดยอัตโนมัติ
"""
import json
import sqlite3
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from readmission_features import (MED_COLS, AGE_MAP, featurize_raw, encode_for_xgb,
                                  nn_features, build_history_features)

AGE_GROUPS = list(AGE_MAP.keys())
MAX_STAY_DAYS = 14          # ข้อมูลเทรนมีระยะนอน 1–14 วัน
READMIT_WINDOW_DAYS = 30


def age_group(birth_year, on_date):
    age = on_date.year - int(birth_year)
    i = min(max(age // 10, 0), 9)
    return AGE_GROUPS[i]


def to_date(s):
    return s if isinstance(s, dt.date) else dt.date.fromisoformat(str(s))


# =============================================================================
# ฐานข้อมูล
# =============================================================================
class Registry:
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS patients (
        patient_id INTEGER PRIMARY KEY AUTOINCREMENT,
        hn TEXT UNIQUE NOT NULL, name TEXT, birth_year INTEGER NOT NULL,
        gender TEXT NOT NULL, race TEXT NOT NULL, created_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS encounters (
        encounter_id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER NOT NULL REFERENCES patients(patient_id),
        admission_date TEXT NOT NULL, discharge_date TEXT,
        status TEXT NOT NULL CHECK (status IN ('admitted', 'discharged')),
        data_json TEXT NOT NULL, updated_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS assessments (
        assessment_id INTEGER PRIMARY KEY AUTOINCREMENT,
        encounter_id INTEGER NOT NULL REFERENCES encounters(encounter_id),
        created_at TEXT NOT NULL, stage TEXT NOT NULL,
        p_any REAL, tier_any TEXT, p_30d REAL, tier_30d TEXT);
    """

    def __init__(self, path="patients.db"):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(self.SCHEMA)
        self.conn.commit()

    @staticmethod
    def _now():
        return dt.datetime.now().isoformat(timespec="seconds")

    # ---------- ผู้ป่วย ----------
    def add_patient(self, hn, name, birth_year, gender, race):
        hn = hn.strip()
        if not hn:
            raise ValueError("ต้องระบุ HN")
        try:
            cur = self.conn.execute(
                "INSERT INTO patients (hn, name, birth_year, gender, race, created_at) VALUES (?,?,?,?,?,?)",
                (hn, name.strip(), int(birth_year), gender, race, self._now()))
        except sqlite3.IntegrityError:
            raise ValueError(f"HN {hn} มีอยู่ในระบบแล้ว")
        self.conn.commit()
        return cur.lastrowid

    def update_patient(self, patient_id, hn, name, birth_year, gender, race):
        hn = hn.strip()
        if not hn:
            raise ValueError("ต้องระบุ HN")
        try:
            self.conn.execute(
                "UPDATE patients SET hn=?, name=?, birth_year=?, gender=?, race=? WHERE patient_id=?",
                (hn, name.strip(), int(birth_year), gender, race, patient_id))
        except sqlite3.IntegrityError:
            raise ValueError(f"HN {hn} มีอยู่ในระบบแล้ว")
        self.conn.commit()

    def delete_patient(self, patient_id):
        """ลบผู้ป่วยพร้อมการ admit และผลการประเมินทั้งหมด"""
        self.conn.execute("""DELETE FROM assessments WHERE encounter_id IN
                             (SELECT encounter_id FROM encounters WHERE patient_id=?)""", (patient_id,))
        self.conn.execute("DELETE FROM encounters WHERE patient_id=?", (patient_id,))
        self.conn.execute("DELETE FROM patients WHERE patient_id=?", (patient_id,))
        self.conn.commit()

    def get_patient(self, patient_id):
        r = self.conn.execute("SELECT * FROM patients WHERE patient_id=?", (patient_id,)).fetchone()
        return dict(r) if r else None

    def list_patients(self):
        return pd.read_sql_query("""
            SELECT p.patient_id, p.hn, p.name, p.birth_year, p.gender,
                   COUNT(e.encounter_id) AS n_encounters,
                   MAX(e.admission_date) AS last_admission,
                   SUM(CASE WHEN e.status='admitted' THEN 1 ELSE 0 END) AS n_open
            FROM patients p LEFT JOIN encounters e ON e.patient_id = p.patient_id
            GROUP BY p.patient_id ORDER BY p.created_at DESC""", self.conn)

    def latest_assessment_by_patient(self):
        return pd.read_sql_query("""
            SELECT e.patient_id, a.p_any, a.tier_any, a.p_30d, a.tier_30d, a.created_at, a.stage
            FROM assessments a JOIN encounters e ON e.encounter_id = a.encounter_id
            WHERE a.assessment_id IN (
                SELECT MAX(a2.assessment_id) FROM assessments a2
                JOIN encounters e2 ON e2.encounter_id = a2.encounter_id GROUP BY e2.patient_id)""", self.conn)

    # ---------- การ admit ----------
    def get_encounters(self, patient_id):
        rows = self.conn.execute(
            "SELECT * FROM encounters WHERE patient_id=? ORDER BY admission_date, encounter_id",
            (patient_id,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["data"] = json.loads(d.pop("data_json"))
            out.append(d)
        return out

    def open_encounter(self, patient_id):
        encs = self.get_encounters(patient_id)
        return next((e for e in encs if e["status"] == "admitted"), None)

    def add_encounter(self, patient_id, admission_date, data):
        admission_date = to_date(admission_date)
        encs = self.get_encounters(patient_id)
        if any(e["status"] == "admitted" for e in encs):
            raise ValueError("ผู้ป่วยยังมีการ admit ที่ยังไม่จำหน่าย — ต้องจำหน่ายก่อนเริ่ม admit ใหม่")
        if encs and admission_date < to_date(encs[-1]["discharge_date"]):
            raise ValueError("วันที่ admit ต้องไม่ก่อนวันที่จำหน่ายครั้งล่าสุด")
        cur = self.conn.execute(
            "INSERT INTO encounters (patient_id, admission_date, status, data_json, updated_at) VALUES (?,?,?,?,?)",
            (patient_id, admission_date.isoformat(), "admitted", json.dumps(data), self._now()))
        self.conn.commit()
        return cur.lastrowid

    def update_encounter(self, encounter_id, data):
        self.conn.execute("UPDATE encounters SET data_json=?, updated_at=? WHERE encounter_id=?",
                          (json.dumps(data), self._now(), encounter_id))
        self.conn.commit()

    def discharge(self, encounter_id, discharge_date, data):
        enc = dict(self.conn.execute("SELECT * FROM encounters WHERE encounter_id=?", (encounter_id,)).fetchone())
        discharge_date = to_date(discharge_date)
        if discharge_date < to_date(enc["admission_date"]):
            raise ValueError("วันที่จำหน่ายต้องไม่ก่อนวันที่ admit")
        self.conn.execute(
            "UPDATE encounters SET discharge_date=?, status='discharged', data_json=?, updated_at=? WHERE encounter_id=?",
            (discharge_date.isoformat(), json.dumps(data), self._now(), encounter_id))
        self.conn.commit()

    def delete_last_encounter(self, patient_id):
        """ลบการ admit ครั้งล่าสุด (ลบได้เฉพาะครั้งล่าสุด เพื่อไม่ให้ลำดับประวัติและผลลัพธ์การกลับมาของครั้งอื่นผิดไป)"""
        encs = self.get_encounters(patient_id)
        if not encs:
            raise ValueError("ไม่มีการ admit ให้ลบ")
        eid = encs[-1]["encounter_id"]
        self.conn.execute("DELETE FROM assessments WHERE encounter_id=?", (eid,))
        self.conn.execute("DELETE FROM encounters WHERE encounter_id=?", (eid,))
        self.conn.commit()
        return encs[-1]

    def discharged_since(self, since_date):
        """การ admit ที่จำหน่ายตั้งแต่ since_date พร้อมผลการประเมินล่าสุดของครั้งนั้น
        และวันที่ admit ครั้งถัดไป (ถ้าผู้ป่วยกลับมาแล้ว)"""
        df = pd.read_sql_query("""
            SELECT p.patient_id, p.hn, p.name, p.birth_year, e.encounter_id,
                   e.admission_date, e.discharge_date, e.data_json,
                   a.p_30d, a.tier_30d, a.p_any, a.tier_any, a.stage,
                   (SELECT MIN(e2.admission_date) FROM encounters e2
                     WHERE e2.patient_id = e.patient_id AND e2.encounter_id != e.encounter_id
                       AND e2.admission_date >= e.discharge_date) AS next_admission
            FROM encounters e
            JOIN patients p ON p.patient_id = e.patient_id
            LEFT JOIN assessments a ON a.assessment_id =
                 (SELECT MAX(a2.assessment_id) FROM assessments a2 WHERE a2.encounter_id = e.encounter_id)
            WHERE e.status = 'discharged' AND e.discharge_date >= ?
            ORDER BY e.discharge_date DESC""", self.conn, params=(to_date(since_date).isoformat(),))
        df["disposition_id"] = [json.loads(d).get("discharge_disposition_id") for d in df.pop("data_json")]
        return df

    # ---------- ผลการประเมิน ----------
    def add_assessment(self, encounter_id, stage, result):
        self.conn.execute(
            "INSERT INTO assessments (encounter_id, created_at, stage, p_any, tier_any, p_30d, tier_30d) VALUES (?,?,?,?,?,?,?)",
            (encounter_id, self._now(), stage, result["any"]["p"], result["any"]["tier"],
             result["30d"]["p"], result["30d"]["tier"]))
        self.conn.commit()

    def get_assessments(self, patient_id):
        return pd.read_sql_query("""
            SELECT a.*, e.admission_date FROM assessments a JOIN encounters e ON e.encounter_id = a.encounter_id
            WHERE e.patient_id=? ORDER BY a.assessment_id""", self.conn, params=(patient_id,))


# =============================================================================
# แปลงข้อมูลในระบบ → feature ของโมเดล
# =============================================================================
def derive_outcomes(encounters):
    """ผลลัพธ์ของแต่ละ admit: '<30' / '>30' ถ้ามีการกลับมาในระบบ, None ถ้ายังไม่ทราบ (ครั้งล่าสุด)"""
    out = []
    for i, e in enumerate(encounters):
        if i + 1 < len(encounters) and e["discharge_date"]:
            gap = (to_date(encounters[i + 1]["admission_date"]) - to_date(e["discharge_date"])).days
            out.append({"gap_days": gap, "label": "<30" if gap <= READMIT_WINDOW_DAYS else ">30"})
        else:
            out.append({"gap_days": None, "label": None})
    return out


def length_of_stay(enc, today=None):
    """จำนวนวันนอนจริง (ถ้ายังไม่จำหน่าย = จำนวนวันจนถึงวันนี้)"""
    end = to_date(enc["discharge_date"]) if enc["discharge_date"] else (today or dt.date.today())
    return max(1, (end - to_date(enc["admission_date"])).days)


def los_for_model(enc, today=None):
    """วันนอนที่ส่งเข้าโมเดล — โมเดลเรียนรู้จากวันนอนทั้งหมดตอนจำหน่าย
    ระหว่างนอน รพ. จึงใช้ค่าที่มากกว่าระหว่างวันนอนจนถึงวันนี้ กับวันนอนที่แพทย์คาดการณ์ไว้"""
    los = length_of_stay(enc, today)
    if not enc["discharge_date"] and enc["data"].get("expected_los"):
        los = max(los, int(enc["data"]["expected_los"]))
    return los


def encounters_to_frames(patient, encounters, top_specialties, today=None):
    """คืน (readable, hist) ของทุก encounter ของผู้ป่วย เรียงตามเวลา — แถวสุดท้ายคือครั้งที่จะประเมิน"""
    rows = []
    for e in encounters:
        d = dict(e["data"])
        d.update({
            "patient_nbr": patient["patient_id"],
            "race": patient["race"], "gender": patient["gender"],
            "age": age_group(patient["birth_year"], to_date(e["admission_date"])),
            "time_in_hospital": min(los_for_model(e, today), MAX_STAY_DAYS),
        })
        d.pop("expected_los", None)
        for m in MED_COLS:
            d.setdefault(m, "No")
        meds = [d[m] for m in MED_COLS]
        # ใช้ค่าที่แพทย์ระบุ ถ้าไม่ระบุจึงอนุมานจากสถานะยา
        # (ใน dataset "Ch" รวมการเปลี่ยนชนิดยาด้วย ไม่ใช่แค่ปรับขนาด จึงควรให้แพทย์ยืนยันเอง)
        if d.get("change") not in ("Ch", "No"):
            d["change"] = "Ch" if any(v in ("Up", "Down") for v in meds) else "No"
        if d.get("diabetesMed") not in ("Yes", "No"):
            d["diabetesMed"] = "Yes" if any(v != "No" for v in meds) else "No"
        rows.append(d)
    readable = featurize_raw(pd.DataFrame(rows), top_specialties)

    labels = [o["label"] for o in derive_outcomes(encounters)]
    meta = pd.DataFrame({"patient_nbr": [patient["patient_id"]] * len(encounters),
                         "readmitted_raw": [l if l else "NO" for l in labels]})
    hist = build_history_features(meta)
    return readable, hist


# =============================================================================
# โมเดล
# =============================================================================
class RiskEngine:
    LABELS = ("any", "30d")

    def __init__(self, model_root="models"):
        import joblib
        import xgboost as xgb
        import torch
        import torch.nn as nn
        from torch.nn.utils.rnn import pack_padded_sequence
        self.xgb, self.torch = xgb, torch
        root = Path(model_root)
        self.artifacts = json.loads((root / "dashboard_artifacts.json").read_text(encoding="utf-8"))

        class SeqNet(nn.Module):   # ต้องเหมือนใน notebook 15 ทุกประการ
            def __init__(self, d_in, h, drop):
                super().__init__()
                self.inp = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Dropout(drop))
                self.lstm = nn.LSTM(h, h, batch_first=True)
                self.head = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Dropout(drop), nn.Linear(h, 1))

            def forward(self, x, lengths):
                packed = pack_padded_sequence(self.inp(x), lengths, batch_first=True, enforce_sorted=False)
                _, (h_n, _) = self.lstm(packed)
                return self.head(h_n[-1]).squeeze(-1)

        self.models = {}
        for label in self.LABELS:
            d = root / f"final_{label}"
            b = joblib.load(d / "final_bundle.pkl")
            booster = xgb.Booster()
            booster.load_model(str(d / "xgb_history.json"))
            cfg = b["lstm_config"]
            nets = []
            for i in range(cfg["N_SEEDS"]):
                net = SeqNet(cfg["D_IN"], cfg["HIDDEN"], cfg["DROPOUT"])
                net.load_state_dict(torch.load(d / f"lstm_seed{i}.pt", map_location="cpu"))
                net.eval()
                nets.append(net)
            self.models[label] = {"bundle": b, "booster": booster, "nets": nets}

    def predict(self, patient, encounters, today=None, n_factors=6):
        """ประเมินความเสี่ยงของ encounter ล่าสุด โดยใช้ประวัติทั้งหมดในระบบ"""
        readable, hist = encounters_to_frames(patient, encounters, self.artifacts["top_specialties"], today)
        result = {}
        for label, m in self.models.items():
            b = m["bundle"]
            X = encode_for_xgb(readable, self.artifacts["encoder_classes"], b["xgb_feature_names"], hist)
            dlast = self.xgb.DMatrix(X.iloc[[-1]])
            it = (0, b["xgb_best_iteration"])
            p_xgb = float(m["booster"].predict(dlast, iteration_range=it)[0])
            contrib = m["booster"].predict(dlast, pred_contribs=True, iteration_range=it)[0, :-1]

            F = nn_features(readable, hist, b["nn_artifacts"])
            K = b["lstm_config"]["K_SEQ"]
            seq = F[-K:]
            x = np.zeros((1, K, F.shape[1]), np.float32)
            x[0, :len(seq)] = seq
            with self.torch.no_grad():
                xt, lt = self.torch.from_numpy(x), self.torch.tensor([len(seq)])
                p_lstm = float(np.mean([self.torch.sigmoid(net(xt, lt)).item() for net in m["nets"]]))

            raw = b["w_lstm"] * p_lstm + (1 - b["w_lstm"]) * p_xgb
            cut = b["tier_cutoffs_raw"]
            tier = "High" if raw >= cut["High"] else "Medium" if raw >= cut["Medium"] else "Low"
            order = np.argsort(-np.abs(contrib))[:n_factors]
            last = X.iloc[-1]
            enc_cls = self.artifacts["encoder_classes"]

            def shown(col):   # ค่าจริงของ feature สำหรับแสดงผล (แปลงรหัส label encoding กลับเป็นข้อความ)
                v = last[col]
                return enc_cls[col][int(v)] if col in enc_cls else (float(v) if not float(v).is_integer() else int(v))
            result[label] = {
                "p": float(b["isotonic"].predict([raw])[0]), "tier": tier, "raw": raw,
                "p_xgb": p_xgb, "p_lstm": p_lstm, "w_lstm": b["w_lstm"],
                "factors": [(X.columns[i], float(contrib[i]), shown(X.columns[i])) for i in order],
                "n_history": len(seq) - 1,
            }
        return result
