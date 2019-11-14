from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
import joblib
import mlflow


CURRENT_EXPERIMENT_NAME = 'label encoding + basic feature engineering'


def rmse(y_true, y_pred) -> float:
    return np.sqrt(mean_squared_error(y_true, y_pred))


rmse_score = make_scorer(rmse, greater_is_better=False)


def add_key_prefix(d: Dict, prefix: str = 'best_') -> Dict:
    return {prefix + key: value for key, value in d.items()}


def df_from_cv_results(d: Dict):
    df = pd.DataFrame(d)
    score_columns = ['mean_test_score', 'mean_train_score']
    param_columns = [c for c in df.columns if c.startswith('param_')]
    return pd.concat([
        -df.loc[:, score_columns],
        df.loc[:, param_columns],
    ], axis=1).sort_values(by='mean_test_score')


def sample(a: np.ndarray, frac: float = 0.01) -> np.ndarray:
    return a[np.random.choice(a.shape[0], int(a.shape[0] * frac), replace=False), :]


def grid_search(ds: np.ndarray, n_jobs: int = -1):
    
    y = np.log1p(ds[:, 0])
    x = ds[:, 1:]
        
    param_grid = dict(
        n_estimators=[100],
        max_depth=[None],
        max_features=['auto'],
        min_samples_leaf=[0.0001, 0.0003, 0.0006],
    )
        
    mlflow.set_experiment(CURRENT_EXPERIMENT_NAME)
    with mlflow.start_run():
        
        mlflow.log_params(param_grid)
        
        regressor = GridSearchCV(
            RandomForestRegressor(n_jobs=n_jobs),
            param_grid=param_grid,
            cv=3,
            scoring=rmse_score,
            verbose=2,
            refit=True,
        )

        regressor.fit(x, y)
        
        best_model = regressor.best_estimator_
        best_param = add_key_prefix(regressor.best_params_)
        best_rmse = - regressor.best_score_
        cv_results = df_from_cv_results(regressor.cv_results_)

        joblib.dump(best_model, 'out/model.sav')
        cv_results.to_csv('out/cv_results.csv', index=False)
        
        mlflow.log_params(best_param)
        mlflow.log_metrics(dict(
            rmse=best_rmse,
        ))
        mlflow.log_artifact('./out/model.sav')
        mlflow.log_artifact('./out/cv_results.csv')
        mlflow.end_run()
    return cv_results


def oneshot(ds: np.ndarray, n_jobs: int = -1, **params):
    default_params = dict(
        n_estimators=10,
        max_depth=None,
        max_features='auto',
        min_samples_leaf=1,
    )
    merged_params = {**default_params, **params}
    model = RandomForestRegressor(**merged_params, n_jobs=n_jobs)

    mlflow.set_experiment(CURRENT_EXPERIMENT_NAME)
    with mlflow.start_run():
        mlflow.log_params(merged_params)
        model.fit(ds[:, 1:], np.log1p(ds[:, 0]))
        joblib.dump(model, 'out/model.sav')
        mlflow.log_artifact('./out/model.sav')
        mlflow.log_metrics(dict(
            rmse=-999,
        ))
        mlflow.end_run()


if __name__ == '__main__':

    dataset_train = np.load('e01_label_encoding_and_basic_feature_engineering/dataset_train.npy')
    grid_search(sample(dataset_train, frac=0.2), n_jobs=-1)
