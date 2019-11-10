from typing import Dict

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import mlflow

sns.set_style('whitegrid')

train = pd.read_csv('data/train.csv', parse_dates=['timestamp'])
test = pd.read_csv('data/test.csv')
building_metadata = pd.read_csv('data/building_metadata.csv')
weather_train = pd.read_csv('data/weather_train.csv', parse_dates=['timestamp'])
weather_test = pd.read_csv('data/weather_test.csv', parse_dates=['timestamp'])

meters = {
    0: 'electricity',
    1: 'chilledwater',
    2: 'steam',
    3: 'hotwater',
}


def filter_by(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    df_out = df
    for key, value in kwargs.items():
        if type(value) is list:
            df_out = df_out[df_out[key].isin(value)]
        else:
            df_out = df_out[df_out[key] == value]
    return df_out


def join_building_meta(df: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    return df.merge(
        metadata,
        on="building_id",
        how='left',
    )


def join_weather(df: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    return df.merge(
        weather,
        on=['site_id', 'timestamp'],
        how='left',
    )


def filler_factory(metadata: pd.DataFrame):
    columns = ['year_built', 'floor_count']
    df_mean_pu_si = metadata.groupby(['primary_use', 'site_id'])[columns].mean()
    df_mean_pu = metadata.groupby('primary_use')[columns].mean()
    df_mean_si = metadata.groupby('site_id')[columns].mean()
    df_mean = metadata[columns].mean()

    def filler(site_id: int, primary_use: str, target: str) -> float:
        mean_pu_si = df_mean_pu_si.loc[(primary_use, site_id), target]
        if not np.isnan(mean_pu_si):
            return mean_pu_si
        mean_pu = df_mean_pu.loc[primary_use, target]
        if not np.isnan(mean_pu):
            return mean_pu
        mean_si = df_mean_si.loc[site_id, target]
        if not np.isnan(mean_si):
            return mean_si
        else:
            return df_mean[target]

    return filler


def fix_nan_building_meta(df: pd.DataFrame) -> pd.DataFrame:
    filler = filler_factory(df)

    def fillna(row):
        yb = filler(row['site_id'], row['primary_use'], 'year_built')
        fc = filler(row['site_id'], row['primary_use'], 'floor_count')
        return pd.Series([yb, fc], index=['year_built', 'floor_count'])

    df_out = df.copy()
    df_out.loc[:, ['year_built', 'floor_count']] = df.apply(fillna, axis=1)

    return df_out


def fix_nan_weather(w: pd.DataFrame) -> pd.DataFrame:

    # add missing datetime
    # fill nan forward and backward for each site
    dt_min, dt_max = w['timestamp'].min(), w['timestamp'].max()
    empty_df = pd.DataFrame({'timestamp': pd.date_range(start=dt_min, end=dt_max, freq='H')})
    w_tmp = pd.concat([
        ws.merge(
            empty_df, on='timestamp', how='outer'
        ).sort_values(
            by='timestamp'
        ).fillna(
            method='bfill'
        ).fillna(
            method='ffill'
        ) for site_id, ws in w.groupby('site_id')
    ], ignore_index=True)

    # fill nan by mean over all sites
    w_mean = w_tmp.groupby('timestamp').mean().drop(columns=['site_id']).reset_index()
    w_mean = w_tmp.loc[:, ['site_id', 'timestamp']].merge(w_mean, on='timestamp', how='left')
    return w_tmp.where(~w_tmp.isnull(), w_mean)


def add_features(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()

    # timestamp
    ts = pd.to_datetime(df['timestamp'])
    df['week'] = ts.dt.week
    df['weekend'] = ts.dt.weekday >= 5
    df['time_period_0-6'] = (ts.dt.hour >= 0) & (ts.dt.hour < 6)
    df['time_period_6-12'] = (ts.dt.hour >= 6) & (ts.dt.hour < 12)
    df['time_period_12-18'] = (ts.dt.hour >= 12) & (ts.dt.hour < 18)

    # wind direction
    df['wind_direction_cosine'] = np.cos(np.radians(df['wind_direction']))

    # meter
    df['meter_category'] = df['meter'].map(meters)

    # categorical
    df = pd.concat([
        df,
        pd.get_dummies(df['primary_use'], drop_first=True),
        pd.get_dummies(df['meter_category'], drop_first=True)
    ], axis=1)

    # drop columns
    df = df.drop(columns=[
        'building_id', 'meter', 'timestamp', 'site_id', 'primary_use',
        'meter_category', 'wind_direction'
    ])

    return df


first_train = train \
    .pipe(join_building_meta, metadata=building_metadata.pipe(fix_nan_building_meta)) \
    .pipe(join_weather, weather=weather_train.pipe(fix_nan_weather)) \
    .pipe(add_features)


def foldout(ds: pd.DataFrame, options: Dict = None):
    if options is None:
        options = {}

    y = np.log1p(ds['meter_reading'])
    x = ds.iloc[:, 1:].values

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)

    default_options = dict(
        n_estimators=10,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0,
        max_features='auto',
        max_leaf_nodes=None,
        min_impurity_decrease=0,
    )
    kwargs = {**default_options, **options}

    model = RandomForestRegressor(**kwargs, verbose=1)

    mlflow.set_experiment('baseline')
    with mlflow.start_run():
        mlflow.log_params(kwargs)

        model.fit(x_train, y_train)

        y_pred_train = model.predict(x_train)
        y_pred_val = model.predict(x_val)

        mse_train = mean_squared_error(y_pred_train, y_train)
        mse_val = mean_squared_error(y_pred_val, y_val)

        np.save('out/y_true.npy', y_val)
        np.save('out/y_pred.npy', y_pred_val)
        joblib.dump(model, 'out/model.sav')

        mlflow.log_metrics(dict(
            mse_train=mse_train,
            mse_val=mse_val
        ))
        mlflow.log_artifacts('./out')
        mlflow.end_run()

    return mse_train, mse_val, model


foldout(first_train)
