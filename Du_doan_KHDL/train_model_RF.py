import pandas as pd
import numpy as np
import joblib 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

def train_model():
    print("--- Bắt đầu huấn luyện mô hình Phân loại (Classification) ---")
    try:
        df = pd.read_csv("cleaned_data_science_jobs.csv")
    except FileNotFoundError:
        print("❌ Lỗi: Không tìm thấy file cleaned_data_science_jobs.csv. Hãy chạy process_data.py trước!")
        return

    # Tạo biến mục tiêu (Target): 1 nếu lương > trung bình, 0 nếu ngược lại
    median_salary = df['salary'].median()
    df['Target'] = (df['salary'] > median_salary).astype(int)
    
    X = df.drop(columns=['salary', 'Target'])
    y = df['Target']

    # Chia tập Train/Test 80-20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Kích thước tập Train: {len(X_train)} mẫu | Kích thước tập Test: {len(X_test)} mẫu")
    print(f"Ngưỡng phân loại lương (Median): {int(median_salary)} EURO")
    print("-" * 60)

    # Tiền xử lý dữ liệu
    categorical_features = ['job_title', 'seniority_level']
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ], remainder='passthrough')

    # Xây dựng Pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # Thiết lập tham số cho GridSearchCV
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10, 20, None],
        'classifier__min_samples_split': [2, 5]
    }

    print("Đang tìm kiếm siêu tham số tối ưu (GridSearchCV) với 3-Fold CV...")
    # verbose=2 sẽ in ra log [CV] END giống như bạn yêu cầu
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=3, 
        n_jobs=-1, 
        verbose=2 
    )

    # Chạy huấn luyện
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    print(f"\nTham số TỐT NHẤT tìm được: {grid_search.best_params_}")

    # --- 3. ĐÁNH GIÁ VÀ IN KẾT QUẢ ---
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1] 

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)

    print("\n" + "="*50)
    print("- KẾT QUẢ RANDOM FOREST (FULL Features) -")
    print("="*50)
    print("Kết quả RandomForest Classifier (Làm sạch từ .csv):")
    print(f"Accuracy : {accuracy:.3f}") 
    print(f"Precision: {precision:.3f}") 
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"ROC AUC  : {roc_auc:.4f}")
    print("="*50)

    # Lưu Model
    joblib.dump(best_model, "model_RF.pkl")
    print(f"💾 Đã lưu mô hình tốt nhất vào file: model_RF.pkl")

if __name__ == "__main__":
    train_model()