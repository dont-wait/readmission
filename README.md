# Readmission Prediction

Train XGBoost de du doan kha nang tai nhap vien trong 30 ngay (`readmitted_30d`).

## Cai dat

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Train model

```powershell
python -m src.train_xgboost --config configs/xgboost_basic.yaml
```

Sau khi train, pipeline se tao:

- `models/xgboost_basic.joblib`: model da train
- `reports/xgboost_basic_metrics.json`: accuracy, precision, recall, f1, ROC AUC, average precision
- `reports/xgboost_basic_val_predictions.csv`: xac suat va nhan du doan tren tap validation
- `reports/figures/confusion_matrix.png`: ma tran nham lan
- `reports/figures/roc_curve.png`: ROC curve
- `reports/figures/precision_recall_curve.png`: precision-recall curve
- `reports/figures/feature_importance.png`: muc do quan trong cua tung feature

## Cai thien model

Tim threshold tot nhat tu file prediction da co:

```powershell
python -m src.find_best_threshold --predictions reports/xgboost_basic_val_predictions.csv
```

Chay pipeline cai thien: so sanh XGBoost goc voi mot phuong phap cai thien duy nhat la XGBoost tuned + `scale_pos_weight`, chi dung cac dac trung that trong file sau loc:

```powershell
python -m src.improve_models --config configs/xgboost_basic.yaml --n-iter 25 --cv 3
```

Pipeline se tao:

- `reports/improved_model_comparison.csv`: bang so sanh ROC AUC, Average Precision, Recall, F1
- `reports/improved_threshold_search.csv`: ket qua quet threshold cho tung model
- `reports/improved_best_val_predictions.csv`: prediction cua model tot nhat tren validation
- `reports/improved_summary.json`: tom tat model, threshold, tham so tuning
- `reports/improved_report.md`: bao cao ngan gon kem bang so sanh va duong dan bieu do
- `reports/figures/improved/`: confusion matrix, ROC curve, precision-recall curve, feature importance, threshold search
- `models/improved_best_model.joblib`: model tot nhat kem threshold da chon

## Dung voi bo du lieu khac

Neu bo du lieu moi da duoc chia thanh `X_train`, `y_train`, `X_val`, `y_val`, hay sua cac duong dan trong `configs/xgboost_basic.yaml`:

```yaml
data:
  x_train_path: data/X_train_final.csv
  y_train_path: data/y_train_final.csv
  x_val_path: data/X_val.csv
  y_val_path: data/y_val.csv
  target_column: readmitted_30d
```

Yeu cau quan trong:

- File `X_train` va `X_val` phai co cung ten cot feature.
- File `y_train` va `y_val` phai co cot target dung voi `target_column`.
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
