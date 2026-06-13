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
       "age": 1.4994980475525317,
       "bmi": -0.16957741141794483,
       "bnp": 0.2813581310178311,
       "sodium": 0.3407817314775273,
       "creatinine": 1.3621636626344145,
       "systolic_bp": -0.0403355143463284,
       "heart_rate": 0.43393264135275605,
       "ace_inhibitor": 0.9247012835332911,
       "beta_blocker": 0.9712465382469481,
       "diuretic": -0.9783591080694793,
       "adherence_score": 0.9212000066229001,
       "distance_to_hospital_km": 0.9304731191402703
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

## API Reference

API hien tai nhan dung 12 feature da preprocess/scale giong cac file `Data/X_train_final.csv`, `Data/X_val.csv`, `Data/X_test_final.csv`. Khong gui gia tri raw nhu tuoi 70, BMI 28.1 neu model duoc train tren du lieu da scale.

Chay server:

```powershell
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

### Models

Model id hop le:

| model_id | Model file | Metrics file |
|---|---|---|
| `improved_xgboost` | `models/improved_best_model.joblib` | `reports/improved/test_metrics.json` |
| `logistic` | `models/logistic_regression_best_model.joblib` | `reports/logistic/test_metrics.json` |
| `base_xgboost` | `models/xgboost_basic.joblib` | `reports/base/test_metrics.json` |

Neu khong truyen query `model`, API dung model mac dinh trong `READMISSION_MODEL_ID`, mac dinh la `improved_xgboost`.

### Common query params

| Param | Type | Ap dung | Mo ta |
|---|---|---|---|
| `model` | string | `/health`, `/features`, `/predict`, `/predict/batch` | Mot trong `improved_xgboost`, `logistic`, `base_xgboost`. |
| `threshold` | number, 0-1 | Tat ca endpoint `/predict...` | Neu truyen vao, API dung threshold nay de tao `predicted_label`; neu khong, dung threshold luu trong model bundle. |

### PatientFeatures input

Tat ca field ben duoi la bat buoc, kieu `number`, va la gia tri da preprocess/scale:

```json
{
  "age": 1.4994980475525317,
  "bmi": -0.16957741141794483,
  "bnp": 0.2813581310178311,
  "sodium": 0.3407817314775273,
  "creatinine": 1.3621636626344145,
  "systolic_bp": -0.0403355143463284,
  "heart_rate": 0.43393264135275605,
  "ace_inhibitor": 0.9247012835332911,
  "beta_blocker": 0.9712465382469481,
  "diuretic": -0.9783591080694793,
  "adherence_score": 0.9212000066229001,
  "distance_to_hospital_km": 0.9304731191402703
}
```

API dat `extra="forbid"`, nen field thua hoac field sai ten se bi tra ve loi validation `422`.

### `GET /health`

Kiem tra server va model dang active.

Example:

```powershell
Invoke-RestMethod -Method Get -Uri http://127.0.0.1:8000/health
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/health?model=logistic"
```

Output:

```json
{
  "status": "ok",
  "active_model_id": "improved_xgboost",
  "active_model_path": "models/improved_best_model.joblib",
  "threshold": 0.36,
  "feature_columns": ["age", "bmi", "bnp", "sodium", "creatinine", "systolic_bp", "heart_rate", "ace_inhibitor", "beta_blocker", "diuretic", "adherence_score", "distance_to_hospital_km"],
  "available_models": ["base_xgboost", "improved_xgboost", "logistic"],
  "expects_preprocessed_features": true
}
```

### `GET /models`

Liet ke cac model da dang ky va trang thai file model.

Example:

```powershell
Invoke-RestMethod -Method Get -Uri http://127.0.0.1:8000/models
```

Output:

```json
[
  {
    "model_id": "improved_xgboost",
    "model_path": "models/improved_best_model.joblib",
    "exists": true,
    "selected_threshold": 0.36,
    "feature_columns": ["age", "bmi", "bnp", "sodium", "creatinine", "systolic_bp", "heart_rate", "ace_inhibitor", "beta_blocker", "diuretic", "adherence_score", "distance_to_hospital_km"]
  }
]
```

### `GET /features`

Lay danh sach cot feature model yeu cau va schema example.

Example:

```powershell
Invoke-RestMethod -Method Get -Uri http://127.0.0.1:8000/features
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/features?model=base_xgboost"
```

Output:

```json
{
  "model_id": "improved_xgboost",
  "feature_columns": ["age", "bmi", "bnp", "sodium", "creatinine", "systolic_bp", "heart_rate", "ace_inhibitor", "beta_blocker", "diuretic", "adherence_score", "distance_to_hospital_km"],
  "expects_preprocessed_features": true,
  "example": {
    "age": {
      "type": "number",
      "description": "Preprocessed/scaled age feature."
    }
  }
}
```

### `POST /predict`

Du doan mot benh nhan bang model truyen qua query param.

Example:

```powershell
$body = @{
  age = 1.4994980475525317
  bmi = -0.16957741141794483
  bnp = 0.2813581310178311
  sodium = 0.3407817314775273
  creatinine = 1.3621636626344145
  systolic_bp = -0.0403355143463284
  heart_rate = 0.43393264135275605
  ace_inhibitor = 0.9247012835332911
  beta_blocker = 0.9712465382469481
  diuretic = -0.9783591080694793
  adherence_score = 0.9212000066229001
  distance_to_hospital_km = 0.9304731191402703
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/predict?model=improved_xgboost" `
  -ContentType "application/json" `
  -Body $body
```

Output:

```json
{
  "readmission_probability": 0.73,
  "predicted_label": 1,
  "threshold": 0.36,
  "model_id": "improved_xgboost",
  "model_path": "models/improved_best_model.joblib",
  "model_metrics": {
    "threshold": 0.36,
    "accuracy": 0.575,
    "precision": 0.4908675799086758,
    "recall": 0.8704453441295547,
    "f1": 0.6277372262773723,
    "roc_auc": 0.6994758633345185,
    "average_precision": 0.578637987989787,
    "tn": 130,
    "fp": 223,
    "fn": 32,
    "tp": 215,
    "predicted_positive": 438
  },
  "model_metrics_path": "reports/improved/test_metrics.json"
}
```

Y nghia output:

| Field | Type | Mo ta |
|---|---|---|
| `readmission_probability` | number | Xac suat tai nhap vien 30 ngay, tu 0 den 1. |
| `predicted_label` | integer | `1` neu probability >= threshold, nguoc lai `0`. |
| `threshold` | number | Threshold dung cho request nay. |
| `model_id` | string | Model thuc su duoc dung. |
| `model_path` | string | Duong dan file model. |
| `model_metrics` | object/null | Chi so danh gia model da luu tren tap test. |
| `model_metrics_path` | string/null | Duong dan file metrics. |

Luu y: `readmission_probability` va `predicted_label` la ket qua cua request hien tai. `model_metrics` la chi so danh gia da luu cua model tren tap test, khong phai chi so tinh rieng cho mot benh nhan.

### Model-specific predict endpoints

Neu khong muon truyen query `model`, dung endpoint co san:

| Endpoint | Model |
|---|---|
| `POST /predict/xgboost-improved` | `improved_xgboost` |
| `POST /predict/logistic` | `logistic` |

Hai endpoint nay nhan body `PatientFeatures` va tra ve `PredictionResponse` giong `/predict`.

### Batch predict endpoints

Request body batch co dang:

```json
{
  "items": [
    {
      "age": 1.4994980475525317,
      "bmi": -0.16957741141794483,
      "bnp": 0.2813581310178311,
      "sodium": 0.3407817314775273,
      "creatinine": 1.3621636626344145,
      "systolic_bp": -0.0403355143463284,
      "heart_rate": 0.43393264135275605,
      "ace_inhibitor": 0.9247012835332911,
      "beta_blocker": 0.9712465382469481,
      "diuretic": -0.9783591080694793,
      "adherence_score": 0.9212000066229001,
      "distance_to_hospital_km": 0.9304731191402703
    }
  ]
}
```

Endpoints:

| Endpoint | Model |
|---|---|
| `POST /predict/batch` | Query `model`, hoac model mac dinh |
| `POST /predict/xgboost-improved/batch` | `improved_xgboost` |
| `POST /predict/logistic/batch` | `logistic` |

Example:

```powershell
$batch = @{
  items = @($body | ConvertFrom-Json)
} | ConvertTo-Json -Depth 4

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/predict/logistic/batch?threshold=0.29" `
  -ContentType "application/json" `
  -Body $batch
```

Output:

```json
{
  "predictions": [
    {
      "readmission_probability": 0.71,
      "predicted_label": 1,
      "threshold": 0.29,
      "model_id": "logistic",
      "model_path": "models/logistic_regression_best_model.joblib",
      "model_metrics": {
        "accuracy": 0.575,
        "precision": 0.4911504424778761,
        "recall": 0.8987854251012146,
        "f1": 0.6351931330472103,
        "roc_auc": 0.6917227695519033,
        "average_precision": 0.567186912130848
      },
      "model_metrics_path": "reports/logistic/test_metrics.json"
    }
  ]
}
```

### Common errors

| HTTP status | Khi nao xay ra |
|---|---|
| `422` | Sai body, thieu field, thua field, `threshold` ngoai khoang 0-1, hoac `model` khong hop le. |
| `503` | File model khong ton tai hoac model bundle thieu key bat buoc. Can train model truoc khi chay API. |
| `500` | Loi khi model chay `predict_proba`. |

Neu muon doi model mac dinh khi khong truyen query `model`:

```powershell
$env:READMISSION_MODEL_ID = "logistic"
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```
