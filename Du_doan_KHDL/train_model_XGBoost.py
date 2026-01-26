import pandas as pd
import numpy as np
import pickle
import re
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, make_scorer
from sklearn.impute import SimpleImputer 
# Đã thay thế RandomForestRegressor
# from sklearn.ensemble import RandomForestRegressor 
import xgboost as xgb # <-- THÊM XGBOOST

# --- 1. Load dữ liệu ---
FILE_PATH = "data_science_job_posts_2025.csv"
try:
    df = pd.read_csv(FILE_PATH)
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file tại đường dẫn '{FILE_PATH}'. Vui lòng kiểm tra lại.")
    exit()

# --- 2. Làm sạch cột lương ---

def clean_salary(s):
    """Hàm làm sạch cột lương (Target Variable)"""
    if pd.isna(s):
        return None
    s = str(s).replace(",", "").lower().replace('€', '') 
    
    numbers = re.findall(r"\d+", s)
    if len(numbers) == 1:
        return float(numbers[0])
    elif len(numbers) >= 2:
        return (float(numbers[0]) + float(numbers[1])) / 2
    return None

df["salary_clean"] = df["salary"].apply(clean_salary)
# Giữ lại các bản ghi có lương đã làm sạch
df = df.dropna(subset=["salary_clean"])
df = df[df["salary_clean"] > 0].copy()

# --- 3. Tách kỹ năng và Trích xuất Features ---
skill_set = ["python", "sql", "r", "spark", "aws", "machine learning", "tensorflow", "pytorch", "excel"]
for skill in skill_set:
    # Tạo các cột skill_... (0 hoặc 1)
    df[f"skill_{skill.replace(' ', '_')}"] = df["skills"].str.contains(skill, case=False, na=False).astype(int)

# --- 4. CHUẨN BỊ DỮ LIỆU TRAIN, VALIDATION VÀ TEST ---
target_job_col = "job_title" 

# CẬP NHẬT FEATURES: Chỉ bao gồm seniority_level, job_title, và các cột skill
features = ["seniority_level", target_job_col] + \
             [f"skill_{s.replace(' ', '_')}" for s in skill_set]

X = df[features]
y = df["salary_clean"]

# PHẦN CHIA DỮ LIỆU: Tỉ lệ 80% | 10% | 10% 
# 1. Chia thành Train (80%) và Temporary (20%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Chia Temporary (20%) thành Validation (10%) và Test (10%)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42) 

print("--- Phân bổ Dữ liệu ---")
print(f"Kích thước tập Train: {len(X_train)} mẫu (80%)")
print(f"Kích thước tập Validation: {len(X_val)} mẫu (10%)")
print(f"Kích thước tập Test: {len(X_test)} mẫu (10%)")
print("-" * 50)

# Áp dụng Log Transformation cho biến mục tiêu
y_train_log = np.log1p(y_train)


# --- 5. Xây dựng Pipeline & Tinh chỉnh Model (GridSearchCV) với K-Fold CV ---

# Định nghĩa các Features
categorical_features = ["seniority_level", target_job_col] 
numeric_features = [f"skill_{s.replace(' ', '_')}" for s in skill_set] 

# Cập nhật Preprocess: Chỉ xử lý Category và các cột skill là Numeric (0/1)
preprocess = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    # Các cột skill (0/1) không cần SimpleImputer vì đã được tạo với na=False
    ('num', 'passthrough', numeric_features) # Giữ nguyên các cột số (skills)
], remainder="drop")

# SỬ DỤNG XGBOOST REGRESSOR
pipeline = Pipeline([
    ("preprocess", preprocess),
    ("regressor", xgb.XGBRegressor(random_state=42, objective='reg:squarederror')) # <-- THAY THẾ MODEL
])

# Lưới tham số cho XGBOOST
param_grid = {
    'regressor__n_estimators': [100, 300, 500], # Số lượng cây
    'regressor__max_depth': [3, 5, 7],        # Độ sâu tối đa của cây
    'regressor__learning_rate': [0.01, 0.1, 0.3], # Tốc độ học
    'regressor__subsample': [0.7, 0.9],       # Tỷ lệ mẫu được lấy để xây dựng mỗi cây
}

scorer = make_scorer(r2_score)
cv_folds = KFold(n_splits=5, shuffle=True, random_state=42)

print("Đang tìm kiếm siêu tham số tối ưu (GridSearchCV) với K-Fold CV trên tập Train...")

# Tìm kiếm Lưới (GridSearchCV):
grid_search = GridSearchCV(
    estimator=pipeline, 
    param_grid=param_grid, 
    scoring=scorer, 
    cv=cv_folds, 
    verbose=2,
    n_jobs=-1
)

# Train model với Log Transformed Target
grid_search.fit(X_train, y_train_log)

best_model = grid_search.best_estimator_

print("\n✅ Tinh chỉnh hoàn thành!")

# --- 6. Lưu Model ---
MODEL_FILE = "model_XGBoost.pkl" # <-- CẬP NHẬT TÊN FILE
with open(MODEL_FILE, "wb") as f:
    pickle.dump(best_model, f)

print(f"\n💾 Đã lưu file '{MODEL_FILE}' mới (chỉ sử dụng Job, Seniority, Skill).")