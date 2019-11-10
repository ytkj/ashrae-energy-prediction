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


def predict(ds: pd.DataFrame, model) -> pd.DataFrame:
    x = ds.iloc[:, 1:]
    y_log1p = model.predict(x)
    y = np.expm1(y_log1p)
    return pd.DataFrame({
        'row_id': ds['row_id'],
        'meter_reading': y,
    })


if __name__ == '__main__':
    model = load_model()
    dataset_test = pd.read_csv('dataset_test.csv')
    predict(dataset_test, model).to_csv('submission.csv', index=False)
