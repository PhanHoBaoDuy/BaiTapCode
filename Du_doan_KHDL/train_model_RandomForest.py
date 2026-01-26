import pandas as pd
import numpy as np
import joblib 
import re
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# --- 1. Load và Làm sạch Dữ liệu ---
FILE_PATH = "data_science_job_posts_2025.csv"
try:
    df = pd.read_csv(FILE_PATH)
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file tại đường dẫn '{FILE_PATH}'. Vui lòng kiểm tra lại.")
    exit()

def clean_salary(s):
    """Hàm làm sạch cột lương (Target Variable)"""
    if pd.isna(s):
        return None
    s = str(s).replace(",", "").lower().replace('€', '').replace('$', '')
    
    numbers = re.findall(r"[\d\.]+", s)
    numbers = [float(n) for n in numbers if n]

    if len(numbers) == 1:
        return numbers[0]
    elif len(numbers) >= 2:
        return np.mean(numbers)
    return None

df["salary_clean"] = df["salary"].apply(clean_salary)
df = df.dropna(subset=["salary_clean"])
df = df[df["salary_clean"] > 0].copy()

# --- 2. TẠO BIẾN MỤC TIÊU PHÂN LOẠI (CLASSIFICATION TARGET) ---
median_salary = df['salary_clean'].median()
df['is_high_salary'] = (df['salary_clean'] > median_salary).astype(int)

# --- 3. Tách kỹ năng và Trích xuất Features ---
skill_set = ["python", "sql", "r", "spark", "aws", "machine learning", "tensorflow", "pytorch", "excel"]
for skill in skill_set:
    df[f"skill_{skill.replace(' ', '_')}"] = df["skills"].str.contains(skill, case=False, na=False).astype(int)

# --- 4. CHUẨN BỊ DỮ LIỆU TRAIN VÀ TEST ---
target_job_col = "job_title" 

features = ["seniority_level", target_job_col] + \
             [f"skill_{s.replace(' ', '_')}" for s in skill_set]

X = df[features]
y = df["is_high_salary"] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
 
print("\n--- Bắt đầu huấn luyện mô hình Phân loại (Classification) ---")
print(f"Kích thước tập Train: {len(X_train)} mẫu | Kích thước tập Test: {len(X_test)} mẫu")
print("-" * 60)


# --- 5. Xây dựng Pipeline & Tinh chỉnh Model (GridSearchCV) ---

categorical_features = ["seniority_level", target_job_col] 
numeric_features = [f"skill_{s.replace(' ', '_')}" for s in skill_set] 

preprocess = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ('num', 'passthrough', numeric_features)
], remainder="drop")

pipeline = Pipeline([
    ("preprocess", preprocess),
    ("classifier", RandomForestClassifier(random_state=42, class_weight='balanced'))
])

param_grid = {
    'classifier__n_estimators': [100, 200], 
    'classifier__max_depth': [10, 20, None], 
    'classifier__max_features': [0.7, 'sqrt'], 
    'classifier__min_samples_split': [2, 5], 
}

scorer = 'roc_auc' 
cv_folds = KFold(n_splits=5, shuffle=True, random_state=42)

print("Đang tìm kiếm siêu tham số tối ưu (GridSearchCV) với 5-Fold CV...")

# THAY ĐỔI: Đặt verbose=2 để in chi tiết quá trình CV
grid_search = GridSearchCV(
    estimator=pipeline, 
    param_grid=param_grid, 
    scoring=scorer, 
    cv=cv_folds, 
    verbose=2, # Đặt verbose = 2 để hiển thị log chi tiết CV
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("\n✅ Tinh chỉnh hoàn thành!")
print(f"Tham số TỐT NHẤT tìm được: {grid_search.best_params_}")


# --- 6. ĐÁNH GIÁ VÀ IN KẾT QUẢ PHÂN LOẠI ---

y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1] 

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_proba)

# In kết quả theo format mẫu yêu cầu
print("\n--- KẾT QUẢ LOGISTIC REGRESSION (FULL Features) ---")
print("Kết quả RandomForest Classifier (Full Features):")
print(f"Accuracy : {accuracy:.3f}") 
print(f"Precision: {precision:.3f}") 
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")
print(f"ROC AUC  : {roc_auc:.4f}")

# --- 7. Lưu Model ---
MODEL_FILE = "model_RF.pkl"
joblib.dump(best_model, MODEL_FILE) 

print(f"\n💾 Đã lưu model phân loại tốt nhất vào file '{MODEL_FILE}'.")