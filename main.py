from typing import Tuple, List, Dict, Any

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, Imputer, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, KFold
from sklearn.metrics import mean_squared_error, make_scorer
import joblib
import mlflow

pd.options.display.max_columns = None
CURRENT_EXPERIMENT_NAME = 'feature engineering'


def filter_by(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    df_out = df
    for key, value in kwargs.items():
        if type(value) is list:
            df_out = df_out[df_out[key].isin(value)]
        else:
            df_out = df_out[df_out[key] == value]
    return df_out


def missing_rate(df: pd.DataFrame) -> pd.Series:
    return df.isnull().sum() / len(df)


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / (1024 ** 2)    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(
            end_mem, 100 * (start_mem - end_mem) / start_mem)
        )
        
    return df


def rmse(y_true, y_pred) -> float:
    return np.sqrt(mean_squared_error(y_true, y_pred))


rmse_score = make_scorer(rmse, greater_is_better=False)


def add_key_prefix(d: Dict, prefix = 'best_') -> Dict:
    return {prefix + key: value for key, value in d.items()}


def df_from_cv_results(d: Dict):
    df = pd.DataFrame(d)
    score_columns = ['mean_test_score', 'mean_train_score']
    param_columns = [c for c in df.columns if c.startswith('param_')]
    return pd.concat([
        -df.loc[:, score_columns],
        df.loc[:, param_columns],
    ], axis=1).sort_values(by='mean_test_score')


def sample(*args, frac: float = 0.01) -> np.ndarray:
    n_rows = args[0].shape[0]
    random_index = np.random.choice(n_rows, int(n_rows * frac), replace=False)
    gen = (
        a[random_index] for a in args
    )
    if len(args) == 1:
        return next(gen)
    else:
        return gen

    
class BaseTransformer(BaseEstimator, TransformerMixin):
    
    def fit(self, x: pd.DataFrame, y = None):
        return self
    
    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        return x


class ColumnTransformer(BaseTransformer):
    
    def __init__(self, defs: Dict[str, BaseTransformer]):
        self.defs = defs
    
    def fit(self, x: pd.DataFrame, y: np.ndarray = None):
        for col, transformer in self.defs.items():
            transformer.fit(x[col], y)
        return self
        
    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        xp = x.copy()
        for col, transformer in self.defs.items():
            xp[col] = transformer.transform(x[col])
        return xp
    
    def fit_transform(self, x: pd.DataFrame, y: np.ndarray = None) -> pd.DataFrame:
        xp = x.copy()
        for col, transformer in self.defs.items():
            if hasattr(transformer, 'fit_transform'):
                xp[col] = transformer.fit_transform(x[col], y)
            else:
                xp[col] = transformer.fit(x[col], y).transform(x[col])
        return xp


class WrappedLabelEncoder(BaseTransformer):
    
    def __init__(self):
        self.le = LabelEncoder()
    
    def fit(self, x, y = None):
        self.le.fit(x)
        return self

    def transform(self, x):
        return self.le.transform(x)
    
    
class WeatherImputer(BaseTransformer):
    
    def transform(self, w: pd.DataFrame) -> pd.DataFrame:
        
        # add missing datetime
        dt_min, dt_max = w['timestamp'].min(), w['timestamp'].max()
        empty_df = pd.DataFrame({'timestamp': pd.date_range(start=dt_min, end=dt_max, freq='H')})
        w_out = pd.concat([
            ws.merge(
                empty_df, on='timestamp', how='outer'
            ).sort_values(
                by='timestamp'
            ).assign(
                site_id=site_id
            ) for site_id, ws in w.groupby('site_id')
        ], ignore_index=True)

        # large missing rate columns; fill by -999
        w_out['cloud_coverage'] = w_out['cloud_coverage'].fillna(-999).astype(np.int16)

        # small missing rate columns; fill by same value forward and backward
        w_out = pd.concat([
            ws.fillna(method='ffill').fillna(method='bfill') for _, ws in w_out.groupby('site_id')
        ], ignore_index=True)

        # fill nan by mean over all sites
        w_mean = w_out.groupby('timestamp').mean().drop(columns=['site_id']).reset_index()
        w_mean = w_out.loc[:, ['site_id', 'timestamp']].merge(w_mean, on='timestamp', how='left')
        w_out = w_out.where(~w_out.isnull(), w_mean)

        # float -> uint
        w_out['site_id'] = w_out['site_id'].astype(np.uint8)

        return w_out


class WeatherEngineerer(BaseTransformer):
    
    @staticmethod
    def shift_by(wdf: pd.DataFrame, n: int) -> pd.DataFrame:
        method = 'bfill' if n > 0 else 'ffill'
        return pd.concat([
            ws.iloc[:, [2, 4, 8]].shift(n).fillna(method=method) for _, ws in wdf.groupby('site_id')
        ], axis=0)
    
    def weather_weighted_average(self, w: pd.DataFrame, hours: int = 5) -> pd.DataFrame:
        ahours = abs(hours)
        sign = int(hours / ahours)
        w_weighted_average = sum(
            [self.shift_by(w, (i+1)*sign) * (ahours-i) for i in range(ahours)]
        ) / (np.arange(ahours) + 1).sum()

        w_weighted_average.columns = ['{0}_wa{1}'.format(c, hours) for c in w_weighted_average.columns]

        return pd.concat([w, w_weighted_average], axis=1)
    
    @staticmethod
    def dwdt(df: pd.DataFrame, base_col: str) -> pd.DataFrame:
        df_out = df.copy()
        df_out[base_col + '_dt_wa1'] = df[base_col] - df[base_col + '_wa1']
        df_out[base_col + '_dt_wa-1'] = df[base_col] - df[base_col + '_wa-1']
        df_out[base_col + '_dt_wa5'] = df[base_col] - df[base_col + '_wa5']
        df_out[base_col + '_dt_wa-5'] = df[base_col] - df[base_col + '_wa-5']
        return df_out
    
    @staticmethod
    def wet(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
        df_out = df.copy()
        df_out['wet' + suffix] = df['air_temperature' + suffix] - df['dew_temperature' + suffix]
        return df_out
    
    def transform(self, w_in: pd.DataFrame) -> pd.DataFrame:
        w = w_in.pipe(self.weather_weighted_average, hours=1) \
            .pipe(self.weather_weighted_average, hours=-1) \
            .pipe(self.weather_weighted_average) \
            .pipe(self.weather_weighted_average, hours=-5)

        w = w.pipe(self.dwdt, base_col='air_temperature') \
            .pipe(self.dwdt, base_col='dew_temperature') \
            .pipe(self.dwdt, base_col='wind_speed') \
            .pipe(self.wet, suffix='') \
            .pipe(self.wet, suffix='_wa1') \
            .pipe(self.wet, suffix='_wa-1') \
            .pipe(self.wet, suffix='_wa5') \
            .pipe(self.wet, suffix='_wa-5')

        return w



class WindDirectionEncoder(BaseTransformer):
    
    @staticmethod
    def _from_degree(degree: int) -> int:
        val = int((degree / 22.5) + 0.5)
        arr = [i for i in range(0,16)]
        return arr[(val % 16)]
    
    def transform(self, x: pd.Series) -> pd.Series:
        return x.apply(self._from_degree)


class WindSpeedEncoder(BaseTransformer):
    
    def transform(self, x: pd.Series) -> pd.Series:
        return pd.cut(
            x,
            bins=[0, 0.3, 1.6, 3.4, 5.5, 8, 10.8, 13.9, 17.2, 20.8, 24.5, 28.5, 33, 1000],
            right=False, labels=False,
        )

    
weather_pipeline = Pipeline(steps=[
    ('impute_missing_value', WeatherImputer()),
    ('feature_engineering', WeatherEngineerer()),
    ('label_encode', ColumnTransformer({
        'wind_direction': WindDirectionEncoder(),
        'wind_speed': WindSpeedEncoder(),
        'wind_speed_wa1': WindSpeedEncoder(),
        'wind_speed_wa-1': WindSpeedEncoder(),    
        'wind_speed_wa5': WindSpeedEncoder(),
        'wind_speed_wa-5': WindSpeedEncoder(),    
    }))
])


class BuildingMetadataEngineerer(BaseTransformer):
    
    def transform(self, bm_in: pd.DataFrame) -> pd.DataFrame:
        bm = bm_in.copy()
        bm['log_square_feet'] = np.log(bm['square_feet'])
        bm['square_feet_per_floor'] = bm['square_feet'] / bm['floor_count']
        bm['log_square_feet_per_floor'] = bm['log_square_feet'] / bm['floor_count']
        bm['building_age'] = 2019 - bm['year_built']
        bm['square_feet_per_age'] = bm['square_feet'] / bm['building_age']
        bm['log_square_feet_per_age'] = bm['log_square_feet'] / bm['building_age']
        return bm


class BuildingMetadataImputer(BaseTransformer):
    
    def transform(self, bm: pd.DataFrame) -> pd.DataFrame:
        return bm.fillna(-999)


building_metadata_pipeline = Pipeline(steps=[
    ('label_encode', ColumnTransformer({
        'primary_use': WrappedLabelEncoder(),
    })),
    ('feature_engineering', BuildingMetadataEngineerer()),
    ('impute_missing_value', BuildingMetadataImputer()),
])


class BuildingMetaJoiner(BaseTransformer):
    
    def __init__(self, bm: pd.DataFrame = None):
        self.bm = bm
    
    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        if self.bm is None:
            return x
        else:
            return x.merge(
                self.bm,
                on='building_id',
                how='left',
            )

    
class WeatherJoiner(BaseTransformer):
    
    def __init__(self, w: pd.DataFrame = None):
        self.w = w
    
    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        if self.w is None:
            return x
        else:
            return x.merge(
                self.w,
                on=['site_id', 'timestamp'],
                how='left',
            )


class DatetimeFeatureEngineerer(BaseTransformer):
    
    def __init__(self, col: str = 'timestamp'):
        self.col = col
    
    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        xp = x.copy()
        ts = x[self.col]
        xp['month'] = ts.dt.month.astype(np.int8)
        xp['week'] = ts.dt.week.astype(np.int8)
        xp['day_of_week'] = ts.dt.weekday.astype(np.int8)
        xp['time_period'] = pd.cut(
            ts.dt.hour,
            bins=[0, 3, 6, 9, 12, 15, 18, 21, 25],
            right=False, labels=False,
        )
        
        holidays = [
            '2016-01-01', '2016-01-18', '2016-02-15', '2016-05-30', '2016-07-04',
            '2016-09-05', '2016-10-10', '2016-11-11', '2016-11-24', '2016-12-26',
            '2017-01-01', '2017-01-16', '2017-02-20', '2017-05-29', '2017-07-04',
            '2017-09-04', '2017-10-09', '2017-11-10', '2017-11-23', '2017-12-25',
            '2018-01-01', '2018-01-15', '2018-02-19', '2018-05-28', '2018-07-04',
            '2018-09-03', '2018-10-08', '2018-11-12', '2018-11-22', '2018-12-25',
            '2019-01-01'
        ]
        xp['is_holiday'] = (ts.dt.date.astype('str').isin(holidays)).astype(np.int8)
        return xp


class TargetEncoder(BaseTransformer):
    
    def __init__(self, cv: int = 5, smoothing: int = 1):
        self.agg = None
        self.cv = cv
        self.smoothing = 1
    
    def transform(self, x: pd.Series):        
        if self.agg is None:
            raise ValueError('you shold fit() before predict()')
        encoded = pd.merge(x, self.agg, left_on=x.name, right_index=True, how='left')
        encoded = encoded.fillna(encoded.mean())
        xp = encoded['y']
        xp.name = x.name
        return xp
    
    def fit_transform(self, x: pd.Series, y: np.ndarray = None) -> pd.Series:
        df = pd.DataFrame({'x': x, 'y': y})
        self.agg = df.groupby('x').mean()
        fold = KFold(n_splits=self.cv, shuffle=True)
        xp = x.copy()
        for idx_train, idx_test in fold.split(x):
            df_train = df.loc[idx_train, :]
            df_test = df.loc[idx_test, :]
            agg_train = df_train.groupby('x').mean()
            encoded = pd.merge(df_test, agg_train, left_on='x', right_index=True, how='left', suffixes=('', '_mean'))['y_mean']
            encoded = encoded.fillna(encoded.mean())
            xp[encoded.index] = encoded
        return xp


class ColumnDropper(BaseTransformer):
    
    def __init__(self, cols: List[str]):
        self.cols = cols
    
    def transform(self, x: pd.DataFrame, y = None) -> pd.DataFrame:
        return x.drop(columns=self.cols)


class ArrayTransformer(BaseTransformer):
    
    def transform(self, x: pd.DataFrame, y = None) -> np.ndarray:
        return x.values


def pipeline_factory() -> Pipeline:
    return Pipeline(steps=[

        # join
        ('join_building_meta', BuildingMetaJoiner(
            building_metadata_pipeline.fit_transform(
                building_metadata
            )
        )),
        ('join_weather', WeatherJoiner(
            weather_pipeline.fit_transform(
                pd.concat([weather_train, weather_test], axis=0, ignore_index=True)
            )
        )),

        # feature engineering
        ('feature_engineering_from_datetime', DatetimeFeatureEngineerer()),
        ('target_encode', ColumnTransformer({
            'primary_use': TargetEncoder(),
            'meter': TargetEncoder(),
            'cloud_coverage': TargetEncoder(),
            'time_period': TargetEncoder(),
            'wind_direction': TargetEncoder(),
            'wind_speed': TargetEncoder(),
            'wind_speed_wa1': TargetEncoder(),
            'wind_speed_wa-1': TargetEncoder(),
            'wind_speed_wa5': TargetEncoder(),
            'wind_speed_wa-5': TargetEncoder(),
        })),

        # drop columns
        ('drop_columns', ColumnDropper([
            'building_id', 'timestamp', 'site_id', 'precip_depth_1_hr',
        ])),

        # pd.DataFrame -> np.ndarray
        ('df_to_array', ArrayTransformer()),

        # regressor
        ('regressor', RandomForestRegressor()),

    ])


def cv(pipeline: Pipeline, df: pd.DataFrame, n_jobs: int = -1, **params) -> Tuple[float, float]:
    
    x = df.drop(columns='meter_reading')
    y = np.log1p(df['meter_reading'].values)

    default_params = dict(
        n_estimators=10,
        max_depth=None,
        max_features='auto',
        min_samples_leaf=1,
    )
    merged_params = {**default_params, **params}

    pipeline_params = {**merged_params, 'n_jobs': n_jobs}
    pipeline_params = add_key_prefix(pipeline_params, 'regressor__')
    pipeline.set_params(**pipeline_params)
    
    mlflow.set_experiment(CURRENT_EXPERIMENT_NAME)
    with mlflow.start_run():
        
        mlflow.log_params(merged_params)
        scores = cross_validate(
            pipeline, x, y,
            cv=3,
            scoring=rmse_score,
            return_train_score=True,
            verbose=2,
        )
        
        rmse_val = - np.mean(scores['test_score'])
        rmse_train = - np.mean(scores['train_score'])
        mlflow.log_metrics(dict(
            rmse_val=rmse_val,
            rmse_train=rmse_train,
        ))
        return rmse_val, rmse_train


def oneshot(pipeline: Pipeline, df: pd.DataFrame, **params):
    
    x = df.drop(columns='meter_reading')
    y = np.log1p(df['meter_reading'].values)

    default_params = dict(
        n_estimators=10,
        max_depth=None,
        max_features='auto',
        min_samples_leaf=1,
    )
    merged_params = {**default_params, **params}

    pipeline_params = {**merged_params, 'n_jobs': -1, 'verbose': 2}
    pipeline_params = add_key_prefix(pipeline_params, 'regressor__')
    pipeline.set_params(**pipeline_params)

    mlflow.set_experiment(CURRENT_EXPERIMENT_NAME)
    with mlflow.start_run():
        
        mlflow.log_params(merged_params)

        pipeline.fit(x, y)
        joblib.dump(pipeline, 'out/pipeline.sav', compress=1)
        
        score = rmse(y, pipeline.predict(x))
        
        mlflow.log_metrics(dict(rmse_train=score))
        mlflow.log_artifact('out/pipeline.sav')
        
        return pipeline


def grid_search(pipeline: Pipeline, df: pd.DataFrame, n_jobs: int = -1, **param_grid):
            
    x = df.drop(columns='meter_reading')
    y = np.log1p(df['meter_reading'].values)

    default_param_grid = dict(
        n_estimators=[80],
        max_depth=[None],
        max_features=['auto'],
        min_samples_leaf=[0.00003],
    )
    merged_param_grid = {**default_param_grid, **param_grid}
    pipeline_param_grid = add_key_prefix(merged_param_grid, 'regressor__')
    
    pipeline.set_params(regressor__n_jobs=n_jobs)
    
    mlflow.set_experiment(CURRENT_EXPERIMENT_NAME)
    with mlflow.start_run():
        
        mlflow.log_params(merged_param_grid)
        
        regressor = GridSearchCV(
            pipeline,
            param_grid=pipeline_param_grid,
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

    
def load_model(run_id: str = None):
    if run_id is None:
        model_path = 'out/model.joblib'
    else:
        mlflow_client = mlflow.tracking.MlflowClient()
        model_path = mlflow_client.download_artifacts(run_id, 'model.joblib')

    return joblib.load(model_path)


def predict(df: pd.DataFrame, pipeline: Pipeline) -> pd.DataFrame:
    x = df.iloc[:, 1:]
    y_log1p = pipeline.predict(x)
    y = np.expm1(y_log1p)
    return pd.DataFrame({
        'row_id': df.iloc[:, 0],
        'meter_reading': y,
    })[['row_id', 'meter_reading']]


if __name__ == '__main__':
    train = pd.read_csv('data/train.csv', parse_dates=['timestamp']).pipe(reduce_mem_usage)
    building_metadata = pd.read_csv('data/building_metadata.csv')
    weather_train = pd.read_csv('data/weather_train.csv', parse_dates=['timestamp'])

    test = pd.read_csv('data/test.csv', parse_dates=['timestamp']).pipe(reduce_mem_usage)
    weather_test = pd.read_csv('data/weather_test.csv', parse_dates=['timestamp'])

    p = load_model()
    predict(test, p).to_csv('submission.csv', index=False)
