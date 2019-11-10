import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
import joblib
import mlflow


def rmse(y_true, y_pred) -> float:
    return np.sqrt(mean_squared_error(y_true, y_pred))


rmse_score = make_scorer(rmse, greater_is_better=False)


def grid_search(ds: pd.DataFrame):
    y = np.log1p(ds['meter_reading'])
    x = ds.iloc[:, 1:].values

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)

    param_grid = dict(
        n_estimators=[10, 20, 30, 40, 50],
        max_depth=[2, 4, 6, None],
        max_features=['auto', 'sqrt'],
    )

    mlflow.set_experiment('baseline')
    with mlflow.start_run():
        mlflow.log_params(param_grid)

        regressor = GridSearchCV(
            RandomForestRegressor(),
            param_grid=param_grid,
            cv=3,
            scoring=rmse_score,
            verbose=1,
            refit=True,
        )

        regressor.fit(x_train, y_train)

        best_model = regressor.best_estimator_
        best_param = regressor.best_params_
        best_rmse = - regressor.best_score_

        joblib.dump(best_model, 'out/model.sav')

        mlflow.log_params(best_param)
        mlflow.log_metrics(dict(
            rmse=best_rmse,
        ))
        mlflow.log_artifact('./out/model.sav')
        mlflow.end_run()


if __name__ == '__main__':

    dataset_train = pd.read_csv('dataset_train.csv')
    grid_search(dataset_train)
