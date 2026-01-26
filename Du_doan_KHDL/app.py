from flask import Flask, render_template, request
import joblib 
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# --- CẤU HÌNH DỰ ĐOÁN HỒI QUY (REGRESSION) ---
MODEL_FILE_REG = "model_RF_regressor.pkl"
SALARY_ERROR_DEFAULT_MONTHLY = 45000 / 12.0 # Giá trị mặc định ~3750 EURO/THÁNG

# Load model (Giữ nguyên)
try:
    if os.path.exists(MODEL_FILE_REG):
        model = joblib.load(open(MODEL_FILE_REG, "rb"))
        print(f"✅ Model Hồi quy loaded successfully: {MODEL_FILE_REG}")
    else:
        model = None
        print(f"❌ Warning: Model file {MODEL_FILE_REG} not found. Using default salary: {round(SALARY_ERROR_DEFAULT_MONTHLY)} EURO/THÁNG")
except Exception as e:
    model = None 
    print(f"❌ Warning: Model loading failed. Error: {e}")

@app.route("/", methods=["GET", "POST"])
def index():
    # ĐÃ SỬA: Khởi tạo giá trị dự đoán là None và thêm biến kiểm soát
    prediction_monthly = None 
    chart_data = {}
    show_results = False 
    
    selected_skills = []
    selected_seniority = "junior"
    selected_job = "Data Scientist"

    if request.method == "POST":
        show_results = True # Đặt là True nếu form được POST (đã dự đoán)
        seniority = request.form["seniority_level"]
        job_title = request.form["job_title"]
        skills_input = request.form.getlist("skills")
        
        selected_seniority = seniority
        selected_job = job_title
        selected_skills = skills_input

        skill_set = ["python", "sql", "r", "spark", "aws", "machine learning", "tensorflow", "pytorch", "excel"]
        
        row = {f"skill_{s.replace(' ', '_')}" : (1 if s in skills_input else 0) for s in skill_set}
        row["seniority_level"] = seniority
        row["job_title"] = job_title 
        
        df = pd.DataFrame([row])
        print(df)
        
        if model:
            try:
                prediction_annual = model.predict(df)[0]
                prediction_monthly = round(prediction_annual / 12.0)
            except Exception as e:
                print(f"🔥 Error predicting with model: {e}")
                prediction_monthly = round(SALARY_ERROR_DEFAULT_MONTHLY) 
        else:
            # Nếu model không tải được, vẫn hiển thị kết quả lỗi sau khi POST
            prediction_monthly = round(SALARY_ERROR_DEFAULT_MONTHLY) 

        chart_data = {
            "prediction": prediction_monthly, 
            "num_skills_selected": len(skills_input),
            "num_skills_total": len(skill_set),
            "num_skills_missing": len(skill_set) - len(skills_input),
            "seniority": seniority
        }

    return render_template("index.html", 
                           prediction=prediction_monthly, 
                           chart_data=chart_data,
                           show_results=show_results, # Truyền biến kiểm soát
                           selected_skills=selected_skills,
                           selected_seniority=selected_seniority,
                           selected_job=selected_job)

if __name__ == "__main__":
    app.run(debug=True)