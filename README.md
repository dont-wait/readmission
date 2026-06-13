# Readmission Prediction

## 🚀 Quick Start (Cho người dùng Nix/Linux)

Nếu bạn sử dụng Nix, môi trường phát triển đã được cấu hình sẵn:

```bash
# Vào môi trường development
nix develop
# Hoặc nếu dùng direnv
direnv allow

# Chạy server
uvicorn src.api:app --host 127.0.0.1 --port 8001 --reload
```

Server sẽ chạy tại `http://127.0.0.1:8000`.

---

## 🛠 Cài đặt thủ công (Non-Nix)

### Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 📈 Model Pipeline

### 1. Huấn luyện Model
```bash
python -m src.train_xgboost --config configs/xgboost_basic.yaml
```
Kết quả sẽ được lưu tại thư mục `models/` và báo cáo tại `reports/`.

### 2. Tối ưu hóa Threshold
Tìm ngưỡng (threshold) tối ưu để cân bằng giữa Precision và Recall:
```bash
python -m src.find_best_threshold --predictions reports/xgboost_basic_val_predictions.csv
```

### 3. Pipeline Cải thiện (Tuned XGBoost)
So sánh Model gốc với Model đã được tinh chỉnh tham số:
```bash
python -m src.improve_models --config configs/xgboost_basic.yaml --n-iter 25 --cv 3
```
Pipeline này sẽ tự động chọn model tốt nhất và lưu vào `models/improved_best_model.joblib`.

---

## 🖥 FastAPI Server

Server mặc định sử dụng model tại `models/improved_best_model.joblib`.

### Chạy Server
- **Nix**: `start-server`
- **Thủ công**: `uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload`

### Kiểm tra API qua Health Check
```bash
curl http://127.0.0.1:8000/health
```

### Dự đoán (Predict)
Gửi yêu cầu POST tới `/predict` với body JSON:

```bash
curl -X POST http://127.0.0.1:8000/predict \
     -H "Content-Type: application/json" \
     -d '{
       "age": 70,
       "bmi": 28.1,
       "bnp": 456,
       "sodium": 137.5,
       "creatinine": 1.2,
       "systolic_bp": 130,
       "heart_rate": 82,
       "adherence_score": 0.62,
       "distance_to_hospital_km": 24.5
     }'
```

### Tài liệu API (Swagger UI)
Truy cập: `http://127.0.0.1:8000/docs`

---

## 📁 Cấu trúc Output

Sau khi chạy xong pipeline cải thiện, các file quan trọng bao gồm:
- `models/improved_best_model.joblib`: Model tốt nhất kèm threshold đã chọn.
- `reports/improved_report.md`: Báo cáo chi tiết so sánh các model.
- `reports/figures/improved/`: Các biểu đồ trực quan (Confusion Matrix, ROC, PR Curve).

---

## ⚙️ Cấu hình dữ liệu mới

Để sử dụng bộ dữ liệu khác, hãy cập nhật các đường dẫn trong `configs/xgboost_basic.yaml`:

```yaml
data:
  x_train_path: data/X_train_final.csv
  y_train_path: data/y_train_final.csv
  x_val_path: data/X_val.csv
  y_val_path: data/y_val.csv
  target_column: readmitted_30d
```
*Lưu ý: Các file CSV đầu vào phải có cùng tên cột feature như lúc huấn luyện.*
