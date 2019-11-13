import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

pd.options.display.max_columns = None

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


def join_building_meta(df: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    return df.merge(
        metadata,
        on="building_id",
        how='left',
    ).drop(columns=['building_id'])


def join_weather(df: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    return df.merge(
        weather,
        on=['site_id', 'timestamp'],
        how='left',
    ).drop(columns=['site_id'])


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

def fix_nan_building_meta_light(df: pd.DataFrame) -> pd.DataFrame:
    df_out = df.copy()
    df_out['year_built'] = df_out['year_built'].fillna(-999).astype(np.int16)
    df_out['floor_count'] = df_out['floor_count'].fillna(-999).astype(np.int16)
    return df_out


def fix_nan_weather(w: pd.DataFrame) -> pd.DataFrame:
    
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


def degToCompass(num):
    val = int((num/22.5)+.5)
    arr = [i for i in range(0,16)]
    return arr[(val % 16)]


def label_encoding(df_in: pd.DataFrame) -> pd.DataFrame:
    
    df = df_in.copy()
    
    # wind direction
    df['wind_direction'] = df['wind_direction'].apply(degToCompass)
    
    # wind speed -> bin
    df['wind_speed'] = pd.cut(
        df['wind_speed'],
        bins=[0, 0.3, 1.6, 3.4, 5.5, 8, 10.8, 13.9, 17.2, 20.8, 24.5, 28.5, 33, 1000],
        right=False, labels=False,
    )
    
    # categorical -> code
    le = LabelEncoder()
    df['primary_use'] = le.fit_transform(df['primary_use']).astype(np.int8)
        
    return df


def feature_engineering(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    
    # timestamp
    ts = pd.to_datetime(df['timestamp'])
    df['month'] = ts.dt.month.astype(np.int8)
    df['week'] = ts.dt.week.astype(np.int8)
    df['day_of_week'] = ts.dt.weekday.astype(np.int8)
    df['time_period'] = pd.cut(
        ts.dt.hour,
        bins=[0, 3, 6, 9, 12, 15, 18, 21, 25],
        right=False, labels=False,
    )
    
    # logarithm
    df['square_feet'] = np.log(df['square_feet'])
    
    df = df.drop(columns=[
        'timestamp',
    ])
    
    return df


def y_first(df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([
        df.loc[:, ['meter_reading']],
        df.drop(columns=['meter_reading']),
    ], axis=1)


if __name__ == '__main__':

    train = pd.read_csv('data/train.csv', parse_dates=['timestamp']).pipe(reduce_mem_usage)
    building_metadata = pd.read_csv('data/building_metadata.csv')
    weather_train = pd.read_csv('data/weather_train.csv', parse_dates=['timestamp'])
    
    dataset_train = train \
        .pipe(join_building_meta, metadata=building_metadata.pipe(fix_nan_building_meta_light)) \
        .pipe(join_weather, weather=weather_train.pipe(fix_nan_weather)) \
        .pipe(label_encoding) \
        .pipe(feature_engineering) \
        .pipe(y_first) \
        .pipe(reduce_mem_usage)
    np.save('dataset_train.npy', dataset_train.values, allow_pickle=False)
    
    del train
    del weather_train
    del dataset_train
    
    test = pd.read_csv('data/test.csv', parse_dates=['timestamp']).pipe(reduce_mem_usage)
    weather_test = pd.read_csv('data/weather_test.csv', parse_dates=['timestamp'])
    dataset_test = test \
        .pipe(join_building_meta, metadata=building_metadata.pipe(fix_nan_building_meta_light)) \
        .pipe(join_weather, weather=weather_test.pipe(fix_nan_weather)) \
        .pipe(label_encoding) \
        .pipe(feature_engineering) \
        .pipe(reduce_mem_usage)
    
    np.save('dataset_test.npy', dataset_test.values, allow_pickle=False)
