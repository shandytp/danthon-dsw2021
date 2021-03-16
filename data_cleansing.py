import pandas as pd
import numpy as np
import joblib

NUMERICAL_COLUMNS = ['id',
                    'detection_date_millis',
                    'update_date_millis',
                    's2id_center',
                    'speed',
                    'regular_speed',
                    'delay_seconds',
                    'seconds',
                    'length',
                    'trend',
                    'severity',
                    'jam_level',
                    'drivers_count',
                    'alerts_count',
                    'n_thumbs_up']

CATEGORICAL_COLUMNS = ['street',
                      'city',
                      'is_highway',
                      'line',
                      's2token_center',
                      'type']

def load_data():
    data = joblib.load('output/data_irregularities.pkl')
    
    return data

def separate_dtype(dataset,
                  NUMERICAL_COLUMNS,
                  CATEGORICAL_COLUMNS):
    '''
    Parameters
    ----------
    dataset: DataFrame, dict, Series
    
    NUMERICAL_COLUMNS: list
        Terdiri dari list kolom yang bertipe data numeric
        
    CATEGORICAL_COLUMNS: list
        Terdiri dari list kolom yang bertipe data categorical
    
    Returns
    -------
    numerical: DataFrame
        DataFrame yang memiliki tipe data numerical
    categorical: DataFrame
        DataFrame yang memiliki tipe data categorical
    '''
    
    numerical = dataset[NUMERICAL_COLUMNS].copy()
    categorical = dataset[CATEGORICAL_COLUMNS].copy()
    
    return numerical, categorical

def impute_categorical_transform(categorical_data):
    '''
    Parameters
    ----------
    categorical_data: DataFrame
        DataFrame yang memiliki tipe data categorical
    
    Returns
    -------
    categorical_data: DataFrame
        DataFrame yang sudah terupdate dengan imputasi nilai 'KOSONG'
    '''
    categorical_data = categorical_data.fillna('KOSONG')
    
    return categorical_data

def run():
    data = load_data()
    num_data, cat_data = separate_dtype(data,
                                        NUMERICAL_COLUMNS,
                                        CATEGORICAL_COLUMNS)
    cat_data_imputted = impute_categorical_transform(cat_data)
    
    joblib.dump(num_data,
               'output/num_data.pkl')
    joblib.dump(cat_data_imputted,
              'output/cat_data_imputted.pkl')
    print('---------- DONE CLEANING DATA ----------')
    
if __name__ == '__main__':
    print('---------- START CLEANING DATA ----------')
    run()