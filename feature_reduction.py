'''

'''

import pandas as pd
import numpy as np

import warnings

from scipy.stats.contingency import association

def manually_remove_columns_from_feature_set(df_features, columns, reason = '', verbose = True):
    '''
    Takes all of the "manually removed" columns and excludes them from the model feature set.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe of observations.
    columns : list
        List of curated columns.
    verbose : bool
        If set to True, will print basic comments.

    Returns
    -------
    df_features_updated : pandas.DataFrame
        Dataframe of features with null features excluded from model feature set.
    '''
    df_features_updated = df_features.copy()
    
    for col in columns:
        if df_features_updated[df_features_updated['feature'] == col]['in_feature_set'].item() == 1:
            df_features_updated.loc[df_features_updated['feature'] == col, ['in_feature_set', 'reason_dropped']] = [0, reason]
    
    if verbose:
        print('The columns manually removed from the feature set are: \n')
        print(*sorted(columns), sep = '\n')
        print('\n')
        
    return df_features_updated


def find_columns_with_missing_values(df, df_features, threshold = 0.5, verbose = True):
    '''
    Finds all of the columns with a missing value threshold > x and excludes them from the model feature set.
    
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
    df_features_updated : pandas.DataFrame
        Dataframe of features with too many missing values excluded from model feature set.
    '''
    
    df_features_updated = df_features.copy()
    
    df_missing = df.isnull().sum() / len(df)
    dropped_columns = df_missing[df_missing > threshold].index.tolist()
    
    for col in dropped_columns:
        if df_features_updated[df_features_updated['feature'] == col]['in_feature_set'].item() == 1:
            df_features_updated.loc[df_features_updated['feature'] == col, ['in_feature_set', 'reason_dropped']] = [0, 'more than ' + str(threshold * 100) + '% missing values']
    
    if verbose:
        print('The columns with more than ' + str(threshold * 100) + '% missing values are: \n')
        print(*sorted(dropped_columns), sep = '\n')
        print('\n')
        
    return df_features_updated
    

def find_null_columns(df, df_features = None, feature_list = None, auto_remove = False, verbose = True):
    '''
    Finds all of the null columns and excludes them from the model feature set.
    
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
    df_features_updated : pandas.DataFrame
        Dataframe of features with null features excluded from model feature set.
    '''
    
    df_features_updated = pd.DataFrame()
    
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
    
    df_clean = df.copy().dropna(axis = 1, how = 'all')
    
    dropped_features = list(set(df.columns) - set(df_clean.columns))
    
    if auto_remove:
        # df_clean already calculated
        
        if df_features is not None:
            df_features_updated = df_features.copy()
            for col in dropped_features:
                if df_features_updated[df_features_updated['feature'] == col]['in_feature_set'].item() == 1:
                    df_features_updated.loc[df_features_updated['feature'] == col, ['in_feature_set', 'reason_dropped']] = [0, 'null column']
    else:
        df_clean = df.copy()
    
    if verbose:
        print('The columns with all null values are: \n')
        print(*sorted(dropped_features), sep = '\n')
        print('\n')
        
    return dropped_features, df_clean, df_features_updated


def find_constant_columns(df, df_features = None, feature_list = None, dropna = False, auto_remove = False, verbose = True):
    '''
    Finds all of the constant columns and excludes them from the model feature set.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe of observations.
    df_features : pandas.DataFrame
        Dataframe of features.
    dropna : bool
        If set to True, NA will not be counted as a unique value. 
        If set to False, NA will be counted as a unique value.
    verbose : bool
        If set to True, will print basic comments.

    Returns
    -------
    df_features_updated : pandas.DataFrame
        Dataframe of features with constant features excluded from model feature set.
    '''
    
    df_clean = df.copy()
    df_features_updated = pd.DataFrame()
    
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
        
    dropped_features = df[relevant_features].columns[df[relevant_features].nunique(dropna = dropna) <= 1].tolist()
        
    if auto_remove:
        df_clean = df_clean.drop(columns = dropped_features)
        
        if df_features is not None:
            df_features_updated = df_features.copy()
            for col in dropped_features:
                df_features_updated.loc[df_features_updated['feature'] == col, ['in_feature_set', 'reason_dropped']] = [0, 'constant column']
        
    if verbose:
        if dropna == False:
            print('Counting NA as a separate value, the columns with only one unique value are: \n')
        else:
            print('Not counting NA as a separate value, the columns with only one unique value are: \n')
        
        print(*sorted(dropped_features), sep = '\n')
        print('\n')
    
    return dropped_features, df_clean, df_features_updated


def find_highly_correlated_features(df, df_features = None, feature_list = None, corr_method = 'pearson', corr_threshold = 0.99, auto_remove = False, verbose = True):
    '''
    Finds all of the constant columns and excludes them from the model feature set.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe of observations.
    df_features : pandas.DataFrame
        Dataframe of features.
    feature_list: list
        List of features.
    corr_method : str
        Type of correlation - needs to be supported by panda's corr() function.
    corr_threshold : float
        Threshold for "highly correlated" features.
    auto_remove : bool
        If set to True, will automatically remove correlated features from df and update df_features (if supplied as input).
        If set to False, will return original df and df_features. Rely on df_correlated_features to manually remove features.
    verbose : bool
        If set to True, will print basic comments.

    Returns
    -------
    df_correlated_features : pandas.DataFrame
        Dataframe of highly correlated features. Schema = ['feature1', 'feature2', 'correlation'].
    df_clean : pandas.DataFrame
        Dataframe of raw data with highly correlated columns removed.
    df_features_updated : pandas.DataFrame
        Dataframe of features with highly correlated excluded from model feature set.
    ''' 
    
    df_clean = df.copy()
    df_features_updated = pd.DataFrame()
    
    if (df_features is None) & (feature_list is None): 
        raise ValueError('Must designate a list of features or pass a features dataframe.')
        
    elif (df_features is not None) & (feature_list is not None):
        warnings.warn('Specified both a features list and a features dataframe. Will default to features in list.')
        relevant_features = feature_list
    
    elif (df_features is None) & (feature_list is not None):
        relevant_features = feature_list
        
    elif (df_features is not None) & (feature_list is None):
        # Numeric and discrete features still in feature set
        relevant_features = df_features[(df_features['in_feature_set'] == 1) & 
                                        ((df_features['stat_data_type'] == 'discrete - numeric') | 
                                         (df_features['stat_data_type'] == 'continuous - numeric'))]['feature'].tolist()
    
    # Correlation matrix
    corr_matrix = df_clean[relevant_features].corr(method = corr_method).abs()
    
    # Upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(bool))
    
    # Correlated features
    df_correlated_features = upper.stack().reset_index()
    df_correlated_features.columns = ['feature1', 'feature2', 'correlation']
    df_correlated_features = df_correlated_features[df_correlated_features['correlation'] >= corr_threshold].reset_index(drop = True)
    
    if auto_remove:
        dropped_features = [col for col in upper.columns if any(upper[col] >= corr_threshold)]
        df_clean = df_clean.drop(columns = dropped_features)
        
        if df_features is not None:
            df_features_updated = df_features.copy()
            for col in dropped_features:
                df_features_updated.loc[df_features_updated['feature'] == col, ['in_feature_set', 'reason_dropped']] = [0, 'highly correlated feature, correlation >= ' + str(corr_threshold)]
        
        if verbose:
            print('The features dropped for being highly correlated (correlation >=', str(corr_threshold), ') are: \n')
            print(*sorted(set(dropped_features)), sep = '\n')
            print('\n')
    
    return df_correlated_features, df_clean, df_features_updated


def find_associated_features(df, df_features = None, feature_list = None, threshold = 0.99, auto_remove = 'fewest number of categories', verbose = True):
    '''
    # what are assumptions of cramers v?
    '''
    
    df_clean = df.copy()
    df_features_updated = pd.DataFrame()
    
    if (df_features is None) & (feature_list is None): 
        raise ValueError('Must designate a list of features or pass a features dataframe.')
        
    elif (df_features is not None) & (feature_list is not None):
        warnings.warn('Specified both a features list and a features dataframe. Will default to features in list.')
        relevant_features = feature_list
    
    elif (df_features is None) & (feature_list is not None):
        relevant_features = feature_list
        
    elif (df_features is not None) & (feature_list is None):
        # Categorical variables still in feature set.
        relevant_features = df_features[(df_features['in_feature_set'] == 1) & 
                                        ((df_features['stat_data_type'] == 'nominal - categorical') | 
                                         (df_features['stat_data_type'] == 'ordinal - categorical') |
                                         (df_features['stat_data_type'] == 'binary - categorical'))]['feature'].tolist()

    list_associated_features = []
    for i in relevant_features: 
        for j in relevant_features: 
            if relevant_features.index(j) > relevant_features.index(i): 
                contingency_table = pd.crosstab(df_clean[i], df_clean[j])
                if (contingency_table.shape[0] > 1) & (contingency_table.shape[1] > 1):  # to avoid division by 0
                    v = association(contingency_table, method = "cramer")  # Cramer V for categorical variables

                    if v >= threshold:
                        list_associated_features.append([i, j, v, contingency_table.shape[0], contingency_table.shape[1]])

    df_cramer_v = pd.DataFrame(list_associated_features, columns = ['feature1', 'feature2', 'cramer_v', 'num_categories_feature1', 'num_categories_feature2'])
    
    if auto_remove == 'none':
        if df_features is not None:
            df_features_updated = df_features.copy()
        
    else:
        # Dataframe of flagged features and the number of categories per feature
        df_feature_categories = pd.concat(
            [df_cramer_v[['feature1', 'num_categories_feature1']].rename(columns = {'feature1': 'feature', 'num_categories_feature1': 'num_categories_feature'}), 
             df_cramer_v[['feature2', 'num_categories_feature2']].rename(columns = {'feature2': 'feature', 'num_categories_feature2': 'num_categories_feature'})], 
            ignore_index = True).drop_duplicates()
        df_feature_categories['keep_feature'] = ''
    
        if auto_remove == 'fewest number of categories':
            df_feature_categories = df_feature_categories.sort_values(by = ['num_categories_feature', 'feature'], ascending = [False, True]).reset_index(drop = True)

        if auto_remove == 'largest number of categories':
            df_feature_categories = df_feature_categories.sort_values(by = ['num_categories_feature', 'feature'], ascending = [True, True]).reset_index(drop = True)
        
        # Loop to specify which features are kept/removed
        for i in range(0, df_feature_categories.shape[0]):
            if df_feature_categories['keep_feature'][i] == '':
                df_feature_categories.loc[i, 'keep_feature'] = 'yes'
                remove_features = df_cramer_v[df_cramer_v['feature1'] == df_feature_categories['feature'][i]]['feature2'].tolist() + df_cramer_v[df_cramer_v['feature2'] == df_feature_categories['feature'][i]]['feature1'].tolist()
                df_feature_categories['keep_feature'] = np.where((df_feature_categories['feature'].isin(remove_features)) & (df_feature_categories['keep_feature'] == ''), 'no', df_feature_categories['keep_feature'])

        dropped_features = df_feature_categories[df_feature_categories['keep_feature'] == 'no']['feature'].tolist()
        df_clean = df_clean.drop(columns = dropped_features)
        
        if df_features is not None:
            df_features_updated = df_features.copy()
            for col in dropped_features:
                df_features_updated.loc[df_features_updated['feature'] == col, ['in_feature_set', 'reason_dropped']] = [0, 'highly associated feature, cramer_v >= ' + str(threshold)]
        
        if verbose:
            print('The features dropped for being highly associated (cramer_v >=', str(threshold), ') are: \n')
            print(*sorted(set(dropped_features)), sep = '\n')
            print('\n')
    
    return df_cramer_v, df_clean, df_features_updated