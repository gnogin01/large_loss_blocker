'''

'''

import pandas as pd
import numpy as np

import warnings

import categorical_feature_maps

def clean_date_columns_and_remove_from_feature_set(df, df_features = None, feature_list = None, verbose = True):
    '''
    Converts all date columns to pandas datetime and removes from set of possible features.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe of observations.
    df_features : pandas.DataFrame
        Dataframe of features.
    feature_list : list
        List of features.
    verbose : bool
        If set to True, will print basic comments.

    Returns
    -------
    df_clean : pandas.DataFrame
        Dataframe of observations with date columns converted to pandas datatime.
    df_features_updated : pandas.DataFrame
        Dataframe of features with date columns excluded from model feature set.
    '''
    
    df_clean = df.copy()
    df_features_updated = pd.DataFrame()
    
    if (df_features is None) & (feature_list is None): 
        raise ValueError('Must designate a list of features or pass a features dataframe.')
        
    elif (df_features is not None) & (feature_list is not None):
        df_features_updated = df_features.copy()
    
        columns = feature_list
        for col in columns:
            df_clean[col] = pd.to_datetime(df_clean[col])
            df_features_updated.loc[df_features_updated['feature'] == col, ['in_feature_set', 'reason_dropped']] = [0, 'date column']

            if verbose:
                print('Column being converted to pandas datetime: ', col)
                print('Column removed from feature set. \n')
                
    elif (df_features is None) & (feature_list is not None):
        columns = feature_list
        for col in columns:
            df_clean[col] = pd.to_datetime(df_clean[col])

            if verbose:
                print('Column being converted to pandas datetime: ', col)

    elif (df_features is not None) & (feature_list is None):
        df_features_updated = df_features.copy()
    
        columns = df_features[df_features['stat_data_type'] == 'date']['feature'].tolist()
        for col in columns:
            df_clean[col] = pd.to_datetime(df_clean[col])
            df_features_updated.loc[df_features_updated['feature'] == col, ['in_feature_set', 'reason_dropped']] = [0, 'date column']

            if verbose:
                print('Column being converted to pandas datetime: ', col)
                print('Column removed from feature set. \n')
                
    return df_clean, df_features_updated


def remove_id_columns_from_feature_set(df_features, verbose = True):
    '''
    Remove ID columns from set of possible features.
    
    Parameters
    ----------
    df_features : pandas.DataFrame
        Dataframe of features.
    verbose : bool
        If set to True, will print basic comments.

    Returns
    -------
    df_features_updated : pandas.DataFrame
        Dataframe of features with ID columns excluded from model feature set.
    '''
    
    df_features_updated = df_features.copy()
        
    columns = df_features[df_features['stat_data_type'] == 'ID column']['feature'].tolist()
        
    for col in columns:
        df_features_updated.loc[df_features_updated['feature'] == col, ['in_feature_set', 'reason_dropped']] = [0, 'ID column']
        
        if verbose:
            print('ID column being removed from feature set: ', col, '\n')
        
    return df_features_updated


def convert_percentage_columns_to_decimal(df, columns, verbose = True):
    '''
    Converts all percent columns to decimal.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe of observations.
    verbose : bool
        If set to True, will print basic comments.

    Returns
    -------
    df_clean : pandas.DataFrame
        Dataframe of observations with percent columns converted to decimal.
    '''
    
    df_clean = df.copy()
    
    for col in columns:
        df_clean[col] = df_clean[col].apply(lambda x: x.replace('$', ''))
        df_clean[col] = df_clean[col].apply(lambda x: x.replace('%', '')).astype(int) / 100
        
        # To handle columns that include the dollar amount instead of a percentage.
        df_clean[col] = np.where(df_clean[col] > 1, round(df_clean[col] * 100 / df_clean['coverage_a']).round(5), df_clean[col])
        
        if verbose:
            print('Column being converted from percentage to decimal: ', col)
            print(df_clean[col].value_counts())
            print('\n')
    
    return df_clean


def clean_continuous_numeric_columns(df, df_features = None, feature_list = None, verbose = True):
    '''
    Converts all continuous columns to float64.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe of observations.
    df_features : pandas.DataFrame
        Dataframe of features.
    feature_list : list
        List of features.
    verbose : bool
        If set to True, will print basic comments.

    Returns
    -------
    df_clean : pandas.DataFrame
        Dataframe of observations with continuous columns converted to float64.
    '''
    
    df_clean = df.copy()
    
    if (df_features is None) & (feature_list is None): 
        raise ValueError("Must designate a list of features or pass a features dataframe.")
        
    elif (df_features is not None) & (feature_list is not None):
        warnings.warn('Specified both a features list and a features dataframe. Will default to features in list.')
        columns = feature_list
    
    elif (df_features is None) & (feature_list is not None):
        columns = feature_list
        
    elif (df_features is not None) & (feature_list is None):
        columns = df_features[df_features['stat_data_type'] == 'continuous - numeric']['feature'].tolist()
    
    for col in columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors = 'coerce')
        df_clean[col] = df_clean[col].astype(float)
        
        if verbose:
            print('Column being converted to float64: ', col)
            print(df_clean[col].describe())
            print('\n')
        
    return df_clean


def clean_discrete_numeric_columns(df, df_features = None, feature_list = None, verbose = True):
    '''
    Converts all discrete columns to float64.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe of observations.
    df_features : pandas.DataFrame
        Dataframe of features.
    feature_list : list
        List of features.
    verbose : bool
        If set to True, will print basic comments.

    Returns
    -------
    df_clean : pandas.DataFrame
        Dataframe of observations with discrete columns converted to float64.
    '''
    
    df_clean = df.copy()
    
    if (df_features is None) & (feature_list is None): 
        raise ValueError('Must designate a list of features or pass a features dataframe.')
        
    elif (df_features is not None) & (feature_list is not None):
        warnings.warn('Specified both a features list and a features dataframe. Will default to features in list.')
        columns = feature_list
    
    elif (df_features is None) & (feature_list is not None):
        columns = feature_list
        
    elif (df_features is not None) & (feature_list is None):
        columns = df_features[df_features['stat_data_type'] == 'discrete - numeric']['feature'].tolist()
      
    for col in columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors = 'coerce')
        df_clean[col] = df_clean[col].astype(float)
        
        if verbose:
            print('Column being converted to float64: ', col)
            print(df_clean[col].describe())
            print('\n')
        
    return df_clean


def clean_ordinal_categorical_columns(df, df_features = None, feature_list = None, verbose = True):
    '''
    Converts all ordinal categorical columns to category.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe of observations.
    df_features : pandas.DataFrame
        Dataframe of features.
    feature_list : list
        List of features.
    verbose : bool
        If set to True, will print basic comments.

    Returns
    -------
    df_clean : pandas.DataFrame
        Dataframe of observations with ordinal columns converted to category.
    '''
    
    df_clean = df.copy()
    
    if (df_features is None) & (feature_list is None): 
        raise ValueError('Must designate a list of features or pass a features dataframe.')
        
    elif (df_features is not None) & (feature_list is not None):
        warnings.warn('Specified both a features list and a features dataframe. Will default to features in list.')
        columns = feature_list
    
    elif (df_features is None) & (feature_list is not None):
        columns = feature_list
        
    elif (df_features is not None) & (feature_list is None):
        columns = df_features[df_features['stat_data_type'] == 'ordinal - categorical']['feature'].tolist()
      
    for col in columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors = 'coerce')
        df_clean[col] = df_clean[col].astype('category')
        
        if verbose:
            print('Column being converted to category: ', col)
            print(df_clean[col].value_counts(dropna = False))
            print('\n')
        
    return df_clean


def clean_binary_categorical_columns(df, df_features = None, feature_list = None, verbose = True):
    '''
    Converts all binary categorical columns to category.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe of observations.
    df_features : pandas.DataFrame
        Dataframe of features.
    feature_list : list
        List of features.
    verbose : bool
        If set to True, will print basic comments.

    Returns
    -------
    df_clean : pandas.DataFrame
        Dataframe of observations with binary columns converted to category.
    '''
    
    true_false_dictionary = {
        'false': 0,
        'f': 0,
        '0' : 0,
        'no': 0,
        'never': 0,
        'true': 1,
        't': 1,
        '1': 1,
        'yes': 1,
        'always': 1
    }

    df_clean = df.copy()
    
    if (df_features is None) & (feature_list is None): 
        raise ValueError('Must designate a list of features or pass a features dataframe.')
        
    elif (df_features is not None) & (feature_list is not None):
        warnings.warn('Specified both a features list and a features dataframe. Will default to features in list.')
        columns = feature_list
    
    elif (df_features is None) & (feature_list is not None):
        columns = feature_list
        
    elif (df_features is not None) & (feature_list is None):
        columns = df_features[df_features['stat_data_type'] == 'binary - categorical']['feature'].tolist()
    
    for col in columns:
        with pd.option_context('future.no_silent_downcasting', True):
            df_clean[col] = df_clean[col].astype(str).str.lower().replace(true_false_dictionary)
            df_clean[col] = df_clean[col].astype('category')
        
        if verbose:
            print('Column being converted to 0/1 category: ', col, '\n')
            print(df_clean[col].value_counts())
            print('\n')
    
    return df_clean


def clean_nominal_categorical_columns(df, df_features = None, feature_list = None, verbose = True):
    '''
    Converts all nominal categorical columns to category.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe of observations.
    df_features : pandas.DataFrame
        Dataframe of features.
    feature_list : list
        List of features.
    verbose : bool
        If set to True, will print basic comments.

    Returns
    -------
    df_clean : pandas.DataFrame
        Dataframe of observations with nominal columns converted to category.
    '''
    
    df_clean = df.copy()
    
    if (df_features is None) & (feature_list is None): 
        raise ValueError('Must designate a list of features or pass a features dataframe.')
        
    elif (df_features is not None) & (feature_list is not None):
        warnings.warn('Specified both a features list and a features dataframe. Will default to features in list.')
        columns = feature_list
    
    elif (df_features is None) & (feature_list is not None):
        columns = feature_list
        
    elif (df_features is not None) & (feature_list is None):
        columns = df_features[df_features['stat_data_type'] == 'nominal - categorical']['feature'].tolist()
    
    for col in columns:
        feature_value_dict = categorical_feature_maps.get_category_map(col)
        
        with pd.option_context('future.no_silent_downcasting', True):
            df_clean[col] = df_clean[col].replace(feature_value_dict)
            df_clean[col] = df_clean[col].astype('category')
        
        if verbose:
            print('Column being rempapped and converted to category: ', col)
            print(df_clean[col].describe())
            print('\n')
        
    return df_clean


def remove_duplicate_rows(df, verbose = True):
    '''
    Removes duplicate rows and resets index.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe of observations.
    verbose : bool
        If set to True, will print basic comments.

    Returns
    -------
    df_clean : pandas.DataFrame
        Dataframe of observations with no duplicate rows.
    '''
    df_clean = df.copy().drop_duplicates().reset_index(drop = True)
    
    if verbose:
        print('Number of duplicate rows removed: ', df.shape[0] - df_clean.shape[0], '\n')
    
    return df_clean


def find_columns_need_NaN_filling(df, df_features = None, feature_list = None, verbose = True):
    '''
    '''
    
    if (df_features is None) & (feature_list is None): 
        raise ValueError('Must designate a list of features or pass a features dataframe.')
        
    elif (df_features is not None) & (feature_list is not None):
        warnings.warn('Specified both a features list and a features dataframe. Will default to features in list.')
        relevant_features = feature_list
    
    elif (df_features is None) & (feature_list is not None):
        relevant_features = feature_list
        
    elif (df_features is not None) & (feature_list is None):
        # Numeric and discrete features still in feature set
        relevant_features = df_features[(df_features['in_feature_set'] == 1)]['feature'].tolist()
    
    NaN_columns = df[relevant_features].columns[df[relevant_features].isnull().any()].tolist()
    
    if verbose:
        print('The columns that need NaN filling and are still in the feature set are: \n')
        print(*sorted(NaN_columns), sep = '\n')
        print('\n')
    
    return NaN_columns