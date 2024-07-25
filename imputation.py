'''

'''

import pandas as pd
import numpy as np

def median_filling(df, df_features, verbose = True):
    '''
    Fills in all missing values in the numeric columns of df with their median.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe of observations.
    df_features : pandas.DataFrame
        Dataframe of features.
    verbose : bool
        If set to True, will print basic comments.

    Returns
    -------
    df_clean : pandas.DataFrame
        Dataframe of observations with NaN values filled in for the numeric columns.
    '''
    
    df_clean = df.copy()
    
    columns = df_features[((df_features['stat_data_type'] == 'discrete - numeric') | (df_features['stat_data_type'] == 'continuous - numeric')) & (df_features['in_feature_set'] == 1)]['feature'].tolist()
    
    for col in columns:
        if df_clean[col].isnull().values.any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

            if verbose:
                print('Column where NaN is replaced with the median: ', col)
                print('Median value is:', df_clean[col].median(), '\n')
            
    return df_clean