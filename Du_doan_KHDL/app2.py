from flask import Flask, render_template, request
import joblib 
import pandas as pd
import numpy as np
import re

app = Flask(__name__)

# --- HÀM LÀM SẠCH REVENUE (Cần thiết cho preprocessing model) ---
def clean_revenue(rev):
    """Xử lý chuỗi doanh thu và chuyển về giá trị số thô."""
    if rev is None or str(rev).strip() == '':
        return np.nan
    s = str(rev).upper().replace('€', '').replace('$', '').strip()
    
    multiplier = 1.0
    if 'T' in s:
        multiplier = 1e12
        s = s.replace('T', '')
    elif 'B' in s:
        multiplier = 1e9
        s = s.replace('B', '')
    elif 'M' in s:
        multiplier = 1e6
        s = s.replace('M', '')
        
    try:
        val = float(re.sub(r'[^\d.]', '', s)) * multiplier
        return val
    except ValueError:
        return np.nan
# --- END HÀM LÀM SẠCH REVENUE ---

# Load model (Sử dụng tên model Hồi quy)
MODEL_FILENAME = "model_RF_Regressor.pkl" 
try:
    model = joblib.load(MODEL_FILENAME) 
    print(f"✅ Tải model {MODEL_FILENAME} thành công.")
except Exception as e:
    model = None 
    print(f"❌ Lỗi khi tải model {MODEL_FILENAME}: {e}")
    print("Sử dụng giá trị mặc định để kiểm tra giao diện.")

# Danh sách features (Phải khớp với mô hình đã train)
skill_set = ["python", "sql", "r", "spark", "aws", "machine learning", "tensorflow", "pytorch", "excel"]
ALL_FEATURES = ["job_title", "headquarter_country", "revenue_log"] + \
               [f"skill_{s.replace(' ', '_')}" for s in skill_set]

DEFAULT_REVENUE_MEDIAN = 1e10 # 10 Tỷ EURO/USD là giá trị khuyết giả định

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    
    # 1. Khởi tạo biến lưu trữ lựa chọn mặc định
    selected_skills = []
    selected_seniority = "senior"
    selected_job = "data scientist"
    selected_country = "USA"
    selected_revenue = "10B"
    
    # 2. Khởi tạo chart_data mặc định (Quan trọng để tránh lỗi Jinja2/JS khi GET)
    chart_data = {
        "prediction": 100000.0, # Giá trị float mặc định
        "num_skills_selected": 0,
        "num_skills_total": len(skill_set),
        "num_skills_missing": len(skill_set),
        "seniority": "senior"
    }

    if request.method == "POST":
        # 3. Thu thập và cập nhật dữ liệu từ Form
        seniority = request.form["seniority_level"].lower()
        job_title = request.form["job_title"].lower()
        headquarter_country = request.form["headquarter_country"]
        revenue_raw = request.form["revenue"]
        skills_input = request.form.getlist("skills")
        
        # Cập nhật biến để hiển thị lại trên form
        selected_seniority = seniority
        selected_job = job_title
        selected_country = headquarter_country
        selected_revenue = revenue_raw
        selected_skills = skills_input

        # 4. Xử lý Dữ liệu cho Model
        row = {f"skill_{s.replace(' ', '_')}": (1 if s in skills_input else 0) for s in skill_set}
        
        revenue_clean = clean_revenue(revenue_raw)
        
        if pd.isna(revenue_clean) or revenue_clean <= 0:
            revenue_log = np.log1p(DEFAULT_REVENUE_MEDIAN)
        else:
            revenue_log = np.log1p(revenue_clean)
            
        row["revenue_log"] = revenue_log
        row["job_title"] = job_title
        row["headquarter_country"] = headquarter_country.replace('US', 'USA') 
        
        df_input = pd.DataFrame([row])
        df_input = df_input[ALL_FEATURES] 

        # 5. Dự đoán
        raw_prediction = None
        if model:
            try:
                log_prediction = model.predict(df_input)[0] 
                raw_prediction = np.expm1(log_prediction)
                
                # Giới hạn mức lương tối thiểu (tránh kết quả âm/quá thấp từ expm1)
                raw_prediction = max(10000, raw_prediction) 
                
                prediction = f"{raw_prediction:,.0f}" 
            except Exception as e:
                print(f"❌ Lỗi dự đoán: {e}")
                prediction = "LỖI DỰ ĐOÁN"
                raw_prediction = 0
        else:
            prediction = "75,000"
            raw_prediction = 75000.0
        
        # 6. Cập nhật dữ liệu cho biểu đồ
        chart_data = {
            "prediction": raw_prediction,
            "num_skills_selected": len(skills_input),
            "num_skills_total": len(skill_set),
            "num_skills_missing": len(skill_set) - len(skills_input),
            "seniority": seniority
        }

    # 7. Render Template
    return render_template("index2.html", 
                           prediction=prediction, 
                           chart_data=chart_data,
                           selected_skills=selected_skills,
                           selected_seniority=selected_seniority,
                           selected_job=selected_job,
                           selected_country=selected_country,
                           selected_revenue=selected_revenue)

if __name__ == "__main__":
    app.run(debug=True)