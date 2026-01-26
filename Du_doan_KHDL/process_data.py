import pandas as pd
import numpy as np
import re

INPUT_FILE = "data_science_job_posts_2025.csv"
OUTPUT_FILE = "cleaned_data_science_jobs.csv"

def clean_salary_to_int(salary_str):
    if pd.isna(salary_str): return None
    # Tìm số trong chuỗi (xử lý cả dấu phẩy)
    numbers = re.findall(r'\d+(?:\.\d+)?', str(salary_str).replace(',', ''))
    numbers = [float(n) for n in numbers]
    if len(numbers) >= 1:
        # Lấy trung bình và làm tròn về số nguyên
        return int(round(np.mean(numbers)))
    return None

def process_data():
    print("--- 1. Bắt đầu làm sạch dữ liệu ---")
    df = pd.read_csv(INPUT_FILE)

    # A. Xử lý Lương thành số nguyên (bỏ .0)
    df['salary'] = df['salary'].apply(clean_salary_to_int)
    
    # B. Tách Skills thành 0/1
    skill_set = ["python", "sql", "r", "spark", "aws", "machine learning", "tensorflow", "pytorch", "excel"]
    for skill in skill_set:
        df[f"skill_{skill.replace(' ', '_')}"] = df["skills"].str.contains(skill, case=False, na=False).astype(int)

    # C. Lọc đúng các cột cần thiết
    skill_cols = [f"skill_{s.replace(' ', '_')}" for s in skill_set]
    final_cols = ['job_title', 'seniority_level', 'salary'] + skill_cols
    
    # Loại bỏ dòng thiếu dữ liệu lương hoặc seniority
    df_clean = df[final_cols].dropna(subset=['salary', 'seniority_level']).copy()

    # Ép kiểu lương về Integer lần cuối để chắc chắn không còn .0
    df_clean['salary'] = df_clean['salary'].astype(int)

    # D. Lưu file
    df_clean.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Đã lưu {len(df_clean)} dòng vào {OUTPUT_FILE}")

if __name__ == "__main__":
    process_data()