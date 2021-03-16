import numpy as np
import pandas as pd
import joblib

PATH = 'data/irregularities.csv'

def read_data(file_dir):
    '''
    Parameters
    ----------
    file_dir : string
        data file location
        
    Returns
    -------
    .pkl file
    '''
    data = pd.read_csv(file_dir)
    joblib.dump(data,
               'output/data_irregularities.pkl')
    print('---------- DONE READ DATA ----------')
    
if __name__ == '__main__':
    print('---------- START READ DATA ----------')
    read_data(PATH)