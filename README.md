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

Sau khi train, pipeline se tao:

- `models/xgboost_basic.joblib`: model da train
- `reports/base/validation_metrics.json`: accuracy, precision, recall, f1, ROC AUC, average precision
- `reports/base/validation_predictions.csv`: xac suat va nhan du doan tren tap validation
- `reports/base/test_metrics.json`: metrics tren tap test neu config co `x_test_path`/`y_test_path`
- `reports/base/test_predictions.csv`: xac suat va nhan du doan tren tap test
- `reports/base/figures/confusion_matrix.png`: ma tran nham lan validation
- `reports/base/figures/test/`: bieu do tren tap test

## Cai thien model

Tim threshold tot nhat tu file prediction da co:

```powershell
python -m src.find_best_threshold --predictions reports/xgboost_basic_val_predictions.csv
```

### 3. Pipeline Cải thiện (Tuned XGBoost)
So sánh Model gốc với Model đã được tinh chỉnh tham số:
```bash
python -m src.improve_models --config configs/xgboost_basic.yaml --n-iter 25 --cv 3
```
Pipeline này sẽ tự động chọn model tốt nhất và lưu vào `models/improved_best_model.joblib`.

---

- `reports/improved/model_comparison.csv`: bang so sanh XGBoost goc va XGBoost tuned
- `reports/improved/key_metrics.csv`: bang cac chi so quan trong kem cach doc nhanh
- `reports/improved/threshold_search.csv`: ket qua quet threshold cho tung model
- `reports/improved/best_val_predictions.csv`: prediction cua model tot nhat tren validation
- `reports/improved/test_metrics.json`: metrics tren tap test cua model tot nhat neu config co `x_test_path`/`y_test_path`
- `reports/improved/best_test_predictions.csv`: prediction tren tap test cua model tot nhat
- `reports/improved/summary.json`: tom tat model, threshold, tham so tuning
- `reports/improved/report.md`: bao cao ngan gon kem bang so sanh va duong dan bieu do
- `reports/improved/figures/`: bang key metrics, confusion matrix, ROC curve, precision-recall curve, feature importance, threshold search
- `models/improved_best_model.joblib`: model tot nhat kem threshold da chon

## Logistic Regression regularized

Chay baseline tuyen tinh de so sanh voi XGBoost va xem huong tac dong cua tung feature:

```powershell
python -m src.train_logistic_regression --config configs/xgboost_basic.yaml --cv 5
```

Pipeline se tao:

- `reports/logistic/report.md`: bao cao ngan gon cua Logistic Regression
- `reports/logistic/model_comparison.csv`: so sanh L2 thuong va L2 `class_weight=balanced`
- `reports/logistic/coefficients.csv`: he so cua tung feature de giai thich model
- `reports/logistic/best_val_predictions.csv`: prediction tren validation cua model Logistic tot nhat
- `reports/logistic/best_test_predictions.csv`: prediction tren test cua model Logistic tot nhat
- `reports/logistic/test_metrics.json`: metrics tren tap test
- `reports/logistic/figures/`: confusion matrix, ROC curve, precision-recall curve, coefficient magnitude
- `models/logistic_regression_best_model.joblib`: Logistic Regression tot nhat kem threshold da chon

## Cau truc bao cao

```text
reports/
  base/
    validation_metrics.json
    validation_predictions.csv
    test_metrics.json
    test_predictions.csv
    figures/
      test/
  improved/
    report.md
    summary.json
    model_comparison.csv
    key_metrics.csv
    threshold_search.csv
    best_val_predictions.csv
    test_metrics.json
    best_test_predictions.csv
    figures/
      test/
  logistic/
    report.md
    summary.json
    model_comparison.csv
    coefficients.csv
    best_val_predictions.csv
    best_test_predictions.csv
    test_metrics.json
    figures/
      test/
```

## Dung voi bo du lieu khac

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
  x_test_path: data/X_test_final.csv
  y_test_path: data/y_test_final.csv
  target_column: readmitted_30d
```

Yeu cau quan trong:

- File `X_train`, `X_val` va `X_test` phai co cung ten cot feature.
- File `y_train`, `y_val` va `y_test` phai co cot target dung voi `target_column`.
- Neu bo du lieu moi co feature khac, can train lai model.
- Neu chi muon du doan tren du lieu moi, file CSV dau vao phai co du cac cot feature giong luc train.

## Du doan tren file moi

Sau khi da train model:

```powershell
python -m src.predict_xgboost --input data/X_val.csv --output reports/new_predictions.csv
```

## Chay FastAPI server

Server mac dinh dung model tot nhat tai `models/improved_best_model.joblib`.

```powershell
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
```

Mo tai lieu API:

```text
http://127.0.0.1:8000/docs
```

Kiem tra server:

```powershell
Invoke-RestMethod -Method Get -Uri http://127.0.0.1:8000/health
```

Goi du doan mot benh nhan:

```powershell
$body = @{
  age = 70
  bmi = 28.1
  bnp = 456
  sodium = 137.5
  creatinine = 1.2
  systolic_bp = 130
  heart_rate = 82
  adherence_score = 0.62
  distance_to_hospital_km = 24.5
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/predict `
  -ContentType "application/json" `
  -Body $body
```

Response mau:

```json
{
  "readmission_probability": 0.33965742588043213,
  "predicted_label": 1,
  "threshold": 0.3,
  "model_path": "models\\improved_best_model.joblib"
}
```

Neu muon dung model khac:

```powershell
$env:READMISSION_MODEL_PATH = "models/xgboost_basic.joblib"
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```
