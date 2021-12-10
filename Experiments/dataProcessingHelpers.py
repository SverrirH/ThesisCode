import pandas as pd
import numpy as np
import datetime
import pyodbc

def sliding_window(a, window):
    from numpy.lib.stride_tricks import as_strided
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return as_strided(a, shape=shape, strides=strides)

def process_temperature_predictions(df_spa_raw: pd.DataFrame,min_date = datetime.datetime(2015,1,1)):        
        # Fix the datatypes if needed
        df_spa_raw.temperature = df_spa_raw.temperature.replace('-',np.nan).astype(float)
        df_spa_raw.date_time = pd.to_datetime(df_spa_raw.date_time)
        df_spa_raw.value_date = pd.to_datetime(df_spa_raw.value_date)
        
        # Compute offset of prediction (in hours) for pivot table columns
        df_spa_raw.loc[:,'prediction_offset'] = (df_spa_raw.date_time - df_spa_raw.value_date)/ np.timedelta64(1, 'h')
        
        # Pivot table (columns: prediction offest, rows: prediction date, value: Predicted values)
        df_vedurspa_pivot = df_spa_raw.pivot_table(values='temperature',index = 'value_date', columns = 'prediction_offset')
        df_vedurspa_pivot = df_vedurspa_pivot.loc[:,df_vedurspa_pivot.columns[df_vedurspa_pivot.columns > 0]]
        df_vedurspa_pivot = df_vedurspa_pivot.sort_index()

        # Resample rows in case any are missing
        df_vedurspa_pivot = df_vedurspa_pivot.resample('1h').mean()

        # Interpolate along the rows (i.e. If there are values as 12 and 15, interpolate for 13 and 14)
        df_vedurspa_pivot_lin_int = df_vedurspa_pivot.interpolate('linear',limit_direction='backward',axis=1).astype('float32')

        # If an 'old' prediction exists for a given hour but a newer one doesn't exist, use the old one
        # 15 is not arbitrary. When I ran it for higher numbers, it stopped improving after 13
        for i in range(15):
            replacement = df_vedurspa_pivot_lin_int.shift(i).shift(-i,axis=1)
            exists_for_replacement = ~replacement.isna()
            missing_for_original = df_vedurspa_pivot_lin_int.isna()
            
            to_be_replaced = exists_for_replacement & missing_for_original
            df_vedurspa_pivot_lin_int[to_be_replaced] = replacement[to_be_replaced]
        
        # for ts in df_vedurspa_pivot_lin_int.index[1:]:
        #     # Finna hvaða gildi eru til í seinustu röð og núverandi
        #     last = ts - datetime.timedelta(hours=1)
        #     values_last = df_vedurspa_pivot_lin_int.loc[last]
        #     values_curr = df_vedurspa_pivot_lin_int.loc[ts]
        #     missing_last = values_last.isna()
        #     missing_curr = values_curr.isna()

        #     # Finna hvaða gildi er þar af leiðandi hægt að færa fram úr seinustu röð sem nýjasta gildið
        #     to_be_replaced_part = list(missing_curr.values[:-1] & ~(missing_last.values[1:]))
        #     to_be_replaced = np.concatenate((to_be_replaced_part,[False]))
        #     to_replace = np.concatenate(([False],to_be_replaced[:-1]))

        #     # Uppfæra gagnagrunn með seinasta spágildi sem til var fyrir hverja klukkustund
        #     df_vedurspa_pivot_lin_int.loc[ts,to_be_replaced] = df_vedurspa_pivot_lin_int.loc[last,to_replace].values
        
        return df_vedurspa_pivot_lin_int.loc[min_date:]

def connect_to_db():
    sql_conn = pyodbc.connect("""DRIVER={SQL Server};
                                 SERVER=VHG;
                                 DATABASE=Kerfiradur_mirror;
                                 Trusted_Connection=yes""") 
    return sql_conn

def execute_query(conn, sql,**kwargs):
    return pd.read_sql(sql, conn, parse_dates=['ctime','date_time','value_dates'],**kwargs)


def sensor_data_query(ID, sensor_names, earliest_date = datetime.datetime(2015,1,1)):
    date_string = earliest_date.strftime('%Y-%m-%d')
    full_sensor_names = []
    for s_name in sensor_names:
        f_name = 'F-' + ID + '-' + s_name
        full_sensor_names.append(f_name)
    name_list_string_w_brackets = f'([{"], [".join(full_sensor_names)}]) '
    added_quotes = ["'" + i + "'" for i in full_sensor_names]
    name_list_string = f'({", ".join(added_quotes)}) '
    
    query = \
    f'''SELECT * FROM
    (
    SELECT Name,Val,CTime
    FROM  Kerfiradur_mirror.veitur_abb_db1.hour 
    WHERE NAME IN {name_list_string}
        AND CTime >= \'{date_string}\'
        AND CType = 4
        AND Flag = 0
    ) t
    PIVOT(
        AVG(Val)
        FOR Name IN {name_list_string_w_brackets}
    ) AS pivot_table;
    '''
    return query


def process_drainage_sensors(df_sensors):
    is_error_signal = lambda x : x.is_integer() & (x < 10)
    df_sensors = df_sensors.sort_index()
    combined_flow = df_sensors.sum(axis=1)
    combined_flow[combined_flow < 10] = np.nan
    return combined_flow