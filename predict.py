import numpy as np
import pandas as pd
import mlflow
import joblib


def load_model(run_id: str = None):
    if run_id is None:
        model_path = 'out/model.sav'
    else:
        mlflow_client = mlflow.tracking.MlflowClient()
        model_path = mlflow_client.download_artifacts(run_id, 'model.sav')

    return joblib.load(model_path)


def predict(ds: np.ndarray, model) -> pd.DataFrame:
    x = ds[:, 1:]
    y_log1p = model.predict(x)
    y = np.expm1(y_log1p)
    return pd.DataFrame({
        'row_id': ds[:, 0],
        'meter_reading': y,
    })


if __name__ == '__main__':
    m = load_model('a04bd1e7c0c74a78aae6607882acffc8')
    dataset_test = np.load('dataset_test.npy')
    predict(dataset_test, m).to_csv('submission.csv', index=False)
