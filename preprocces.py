import pandas as pd
import numpy as np

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
        ws.merge(empty_df, on='timestamp', how='outer') \
            .sort_values(by='timestamp') \
            .fillna(method='bfill') \
            .fillna(method='ffill') \
        for site_id, ws in w.groupby('site_id')
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

    # categorycal
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


if __name__ == '__main__':

    train = pd.read_csv('data/train.csv', parse_dates=['timestamp'])
    building_metadata = pd.read_csv('data/building_metadata.csv')
    weather_train = pd.read_csv('data/weather_train.csv', parse_dates=['timestamp'])
    train \
        .pipe(join_building_meta, metadata=building_metadata.pipe(fix_nan_building_meta)) \
        .pipe(join_weather, weather=weather_train.pipe(fix_nan_weather)) \
        .pipe(add_features) \
        .to_csv('dataset_train.csv', index=False)

    del train
    del weather_train

    test = pd.read_csv('data/test.csv', parse_dates=['timestamp'])
    weather_test = pd.read_csv('data/weather_test.csv', parse_dates=['timestamp'])
    test \
        .pipe(join_building_meta, metadata=building_metadata.pipe(fix_nan_building_meta)) \
        .pipe(join_weather, weather=weather_test.pipe(fix_nan_weather)) \
        .pipe(add_features) \
        .to_csv('dataset_test.csv', index=False)
