import pandas as pd
import numpy as np

def get_data(df, station):
    tmp = df[df['역명'] == station]
    tmp = tmp.reset_index()
    n = len(tmp)
    social_distance_change = [0]*n
    for i in range(1, n, 1):
        if tmp['social_distance'][i] > tmp['social_distance'][i-1]:
            social_distance_change[i] = 1
        elif tmp['social_distance'][i] < tmp['social_distance'][i-1]:
            social_distance_change[i] = -1
    tmp['social_distance_change'] = social_distance_change
    return tmp

def get_tidy_data(df_station):
    tmp_1 = df_station[['사용일자', '승차총승객수', '하차총승객수']]
    tmp_2 = df_station[['사용일자', 'temp_avg', 'is_semester', 'is_offline', 'offline_class', 'precipitation', 'social_distance', 'day_off', 'holiday', 'social_distance_change']]
    tmp_1 = tmp_1.groupby('사용일자').sum().reset_index()
    tmp_2 = tmp_2.groupby('사용일자').median().reset_index()
    station = pd.merge(tmp_1, tmp_2, on='사용일자', how='inner')
    return station

def get_exog_variables(station_train, station_valid, exog_list):
    n = len(exog_list)
    exog_train = []
    exog_valid = []
    for i in range(n):
        exog_series_train = pd.Series(np.array(station_train[exog_list[i]]), index=pd.date_range('2019-01-01', periods=1461, freq='D'))
        exog_series_valid = pd.Series(np.array(station_valid[exog_list[i]]), index=pd.date_range('2023-01-01', periods=365, freq='D'))

        exog_train = exog_train + [exog_series_train]
        exog_valid = exog_valid + [exog_series_valid]

    return [exog_train, exog_valid]
    