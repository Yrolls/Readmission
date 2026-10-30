# Patch: ทำให้ inner CV ตอน tune hyperparameter เป็น group-aware

**ไฟล์ที่ต้องแก้**: `05_improved_pipeline4.ipynb` ขั้นตอนที่ 7 (RandomizedSearchCV)

## ปัญหา
`RandomizedSearchCV(rf_pipeline, rf_param_dist, n_iter=10, cv=3, ...)` — ส่ง `cv=3` เป็น int
sklearn จะใช้ `StratifiedKFold(3)` โดยอัตโนมัติ ซึ่ง**ไม่รู้จัก patient_nbr** ผู้ป่วยคนเดียวกันที่มีหลาย
encounter จึงอาจหลุดไปอยู่ทั้ง train-fold และ validation-fold ระหว่างการ tune hyperparameter — เป็น
leakage ประเภทเดียวกับที่วิจารณ์ Lu & Uddin แต่เกิดที่ inner loop ของ pipeline ตัวเอง

`StackingClassifier(..., cv=5)` ก็มีปัญหาเดียวกัน (ใช้สำหรับสร้าง out-of-fold prediction ของ meta-learner)

## วิธีแก้

```python
from sklearn.model_selection import GroupKFold

# แทนที่ cv=3 (int) ด้วย GroupKFold object
group_cv = GroupKFold(n_splits=3)
groups_train = df.loc[X_train.index, "patient_nbr"]  # ต้องเก็บ patient_nbr ของ X_train ไว้ก่อนหน้านี้

rf_search = RandomizedSearchCV(
    rf_pipeline, rf_param_dist, n_iter=10, cv=group_cv,
    scoring="average_precision", n_jobs=-1, random_state=RANDOM_STATE,
)
# สำคัญ: ต้องส่ง groups= ตอน .fit() ด้วย ไม่งั้น GroupKFold ไม่มีผล
rf_search.fit(X_train, y_train, groups=groups_train)

xgb_search = RandomizedSearchCV(
    xgb_pipeline, xgb_param_dist, n_iter=10, cv=group_cv,
    scoring="average_precision", n_jobs=-1, random_state=RANDOM_STATE,
)
xgb_search.fit(X_train, y_train, groups=groups_train)
```

สำหรับ `StackingClassifier` (ขั้นตอนที่ 8 หรือใน `07_final_calibrated_pipeline.ipynb` ขั้นตอนที่ 6):

```python
stacking_model = StackingClassifier(
    estimators=estimators,
    final_estimator=scaled_meta_learner,
    cv=GroupKFold(n_splits=5),   # แทน cv=5 (int) เดิม
    n_jobs=-1, passthrough=True,
)
# หมายเหตุ: StackingClassifier.fit() ไม่รับ groups= ตรงๆ ผ่าน .fit(X, y)
# ต้องใช้ cross_val_predict(..., groups=...) เองถ้าอยาก group-aware เต็มรูปแบบ
# หรือถ้าเวลาจำกัด อย่างน้อย fix inner CV ของ RandomizedSearchCV (ผลกระทบใหญ่กว่า) ก่อน
```

## ผลกระทบที่คาดว่าจะเกิด
- ตัวเลข PR-AUC ตอน tune (`rf_search.best_score_`) อาจลดลงเล็กน้อย (เพราะ CV ตอนนี้เข้มงวดขึ้นจริง)
- Hyperparameter ที่เลือกได้อาจเปลี่ยนเล็กน้อย แต่ไม่ควรเปลี่ยนแบบสุดขั้ว
- **ค่าที่รายงานใน `07_final_calibrated_pipeline.ipynb` (PR-AUC บน test set จริง) ไม่ควรเปลี่ยนมาก**
  เพราะ test set เดิมก็ถูกกันไว้ด้วย `GroupShuffleSplit` อยู่แล้ว — นี่แค่ทำให้ขั้นตอน tune สมบูรณ์ขึ้น

## ถ้าเวลาไม่พอ
บันทึกเป็น "known limitation" ในรายงานตรงๆ ก็ยังดีกว่าไม่พูดถึงเลย — เขียนแค่ 2-3 ประโยคใน methodology
section ว่า "การ tune hyperparameter ใช้ standard K-fold ไม่ใช่ GroupKFold ซึ่งอาจทำให้ hyperparameter
ที่เลือกได้ optimistic เล็กน้อย แต่ผลการประเมินสุดท้ายบน held-out test set ยังคงป้องกัน patient-level leakage
ไว้ได้ตามปกติ"
