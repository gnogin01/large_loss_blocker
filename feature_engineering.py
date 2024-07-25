'''

'''

import pandas as pd
import numpy as np
from datetime import date

import categorical_feature_maps

def change_coverage_decimal_to_dollars(df, columns, verbose = True):
    '''
    Converts coverage columns from decimal to dollar values.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe of observations.
    columns : list
        List of columns
    verbose : bool
        If set to True, will print basic comments.

    Returns
    -------
    df_clean : pandas.DataFrame
        Dataframe of observations with coverage columns converted from decimal to dollar values.
    '''
    
    df_clean = df.copy()
    
    for col in columns:
        df_clean[col] = df_clean[col] * df_clean['coverage_a']
        
        if verbose:
            print('Column being converted from decimal to dollar amount: ', col, '\n')
    
    return df_clean


def calculate_new_features(df, df_features, columns, verbose = True):
    '''
    Alter existing features or create new features.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe of observations.
    df_features : pandas.DataFrame
        Dataframe of features.
    columns : list
        List of features to create.
    verbose : bool
        If set to True, will print basic comments.

    Returns
    -------
    df_clean : pandas.DataFrame
        Dataframe of observations with coverage columns converted from decimal to dollar values.
    df_features_updated : pandas.DataFrame
        Dataframe of features with new features added to model feature set.
    '''
    
    df_clean = df.copy()
    df_features_updated = df_features.copy()
    
    for col in columns:    
        if col == 'chargeable_losses':
            df_clean['chargeable_losses'] = np.where(df_clean['chargeable_losses'] > 0, 1, 0)
            df_features['stat_data_type'] = np.where(df_features['feature'] == 'chargeable_losses', 'binary - categorical', df_features['stat_data_type'])

        if col == 'water_damage_exclusion':
            df_clean['water_damage_exclusion'] = np.where(df_clean['water_damage'] == 'excluded', 1, 0)
            df_clean['water_damage_exclusion'] = df_clean['water_damage_exclusion'].astype('category')
            
            df_features_updated = pd.concat([pd.DataFrame([['water_damage_exclusion', 1, '', 'binary - categorical']], columns = df_features_updated.columns), df_features_updated], ignore_index = True)

        if col == 'windstorm_roof_shape':
            df_clean['windstorm_roof_shape'] = np.where(df_clean['roof_shape'] == 'Hip', 'Hip', 'Other')
            df_clean['windstorm_roof_shape'] = df_clean['windstorm_roof_shape'].astype('category')
            
            df_features_updated = pd.concat([pd.DataFrame([['windstorm_roof_shape', 1, '', 'nominal - categorical']], columns = df_features_updated.columns), df_features_updated], ignore_index = True)
        
        if col == 'burglar_alarm':
            df_clean['burglar_alarm'] = np.where(df_clean['burglar_alarm'] == 'central', 'central', 'none')
            df_clean['burglar_alarm'] = df_clean['burglar_alarm'].astype('category')

        if col == 'pricing_construction':
            pricing_construction = categorical_feature_maps.get_category_map(col)
            df_clean['pricing_construction'] = df_clean.apply(lambda row: pricing_construction[f'{row["construction_type"]}{row["exterior_wall_finish"]}'] 
                                                            if f'{row["construction_type"]}{row["exterior_wall_finish"]}' in pricing_construction
                                                            else pricing_construction[''], axis = 1)
            df_clean['pricing_construction'] = df_clean['pricing_construction'].fillna('Frame') 
            df_clean['pricing_construction'] = df_clean['pricing_construction'].astype('category')
            
            df_features_updated = pd.concat([pd.DataFrame([['pricing_construction', 1, '', 'nominal - categorical']], columns = df_features_updated.columns), df_features_updated], ignore_index = True)

        if col == 'pricing_roof_type_1':
            roof_type = categorical_feature_maps.get_category_map('roof_type2')
            df_clean['pricing_roof_type_1'] = df_clean['roof_type'].apply(lambda row: roof_type[row][0] if row in roof_type else roof_type[''][0])
            df_clean['pricing_roof_type_1'] = df_clean['pricing_roof_type_1'].astype('category')
            
            df_features_updated = pd.concat([pd.DataFrame([['pricing_roof_type_1', 1, '', 'nominal - categorical']], columns = df_features_updated.columns), df_features_updated], ignore_index = True)

        if col == 'pricing_roof_type_2':
            roof_type = categorical_feature_maps.get_category_map('roof_type2')
            df_clean['pricing_roof_type_2'] = df_clean['roof_type'].apply(lambda row: roof_type[row][1] if row in roof_type else roof_type[''][1])
            df_clean['pricing_roof_type_2'] = df_clean['pricing_roof_type_2'].astype('category')
            
            df_features_updated = pd.concat([pd.DataFrame([['pricing_roof_type_2', 1, '', 'nominal - categorical']], columns = df_features_updated.columns), df_features_updated], ignore_index = True)

        if col == 'pricing_sprinkler':
            pricing_sprinkler = categorical_feature_maps.get_category_map(col)
            df_clean['pricing_sprinkler'] = df_clean.apply(lambda row: pricing_sprinkler[f'{row["fire_alarm_monitoring"]}{row["fire_alarm_sprinkler"]}'] 
                                                         if f'{row["fire_alarm_monitoring"]}{row["fire_alarm_sprinkler"]}' in pricing_sprinkler
                                                         else pricing_sprinkler[''], axis = 1)
            df_clean['pricing_sprinkler'] = df_clean['pricing_sprinkler'].astype('category')
            
            df_features_updated = pd.concat([pd.DataFrame([['pricing_sprinkler', 1, '', 'nominal - categorical']], columns = df_features_updated.columns), df_features_updated], ignore_index = True)

        if col == 'pricing_water_detection':
            pricing_water_detection = categorical_feature_maps.get_category_map(col)
            df_clean['pricing_water_detection'] = df_clean.apply(lambda row: pricing_water_detection[f'{row["water_detection"]}{row["water_alarm"]}'.upper()] 
                                                               if f'{row["water_detection"]}{row["water_alarm"]}'.upper() in pricing_water_detection
                                                               else pricing_water_detection[''], axis = 1)
            df_clean['pricing_water_detection'] = df_clean['pricing_water_detection'].astype('category')
            
            df_features_updated = pd.concat([pd.DataFrame([['pricing_water_detection', 1, '', 'nominal - categorical']], columns = df_features_updated.columns), df_features_updated], ignore_index = True)

        if col == 'policy_year':
            df_clean['policy_year'] = date.today().year - df_clean['endors_term_start_date'].dt.year
            df_clean['policy_year'] = df_clean['policy_year'].astype(float)
            
            df_features_updated = pd.concat([pd.DataFrame([['policy_year', 1, '', 'discrete - numeric']], columns = df_features_updated.columns), df_features_updated], ignore_index = True)

        if col == 'coverage_a_per_100k':
            df_clean['coverage_a_per_100k'] = df_clean['coverage_a'] // 100000
            df_clean['coverage_a_per_100k'] = df_clean['coverage_a_per_100k'].astype(float)
            
            df_features_updated = pd.concat([pd.DataFrame([['coverage_a_per_100k', 1, '', 'discrete - numeric']], columns = df_features_updated.columns), df_features_updated], ignore_index = True)
            
        if col == 'roof_age':
            df_clean['roof_age'] = df_clean['endors_term_start_date'].dt.year - df_clean['roof_construction_year']
            df_clean['roof_age'] = np.where(df_clean['roof_age'] < 0, 0, df_clean['roof_age'])
            df_clean['roof_age'] = df_clean['roof_age'].astype(float)
            
        if col == 'home_value_coverage_a':
            df_clean['home_value_coverage_a'] = np.where(df_clean['coverage_a'] > 0, df_clean['sale_price'] / df_clean['coverage_a'], np.nan)
            df_clean['home_value_coverage_a'] = df_clean['home_value_coverage_a'].astype(float)
            
            df_features_updated = pd.concat([pd.DataFrame([['home_value_coverage_a', 1, '', 'continuous - numeric']], columns = df_features_updated.columns), df_features_updated], ignore_index = True)
                        
        if verbose:
            print('Column being altered/created: ', col)
            print(df_clean[col].value_counts(dropna = False))
            print('\n')
    
    return df_clean, df_features_updated


def calculate_response_variables(df, df_features, columns, verbose = True):
    '''
    Create response variables.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe of observations.
    df_features : pandas.DataFrame
        Dataframe of features.
    columns : list
        List of response variables to create.
    verbose : bool
        If set to True, will print basic comments.

    Returns
    -------
    df_clean : pandas.DataFrame
        Dataframe of observations with response variables included as new columns.
    df_features_updated : pandas.DataFrame
        Dataframe of features with new features added to model feature set.
    '''
    
    df_clean = df.copy()
    df_features_updated = df_features.copy()
    
    for col in columns:   
        if col == 'loss_cost':
            df_clean['loss_cost'] = df_clean['total_inc_loss'] / df_clean['sum_earned_exposure']
            df_clean['loss_cost'] = df_clean['loss_cost'].astype(float)
            df_clean['loss_cost'] = df_clean['loss_cost'].fillna(0)
            
            df_features_updated = pd.concat([pd.DataFrame([['loss_cost', 0, 'response variable', 'continuous - numeric']], columns = df_features_updated.columns), df_features_updated], ignore_index = True)
            
        if col == 'log_loss_cost':
            df_clean['log_loss_cost'] = np.log(df_clean['loss_cost'] + 1)   # adding 1 so that we don't have log(0)
            
            df_features_updated = pd.concat([pd.DataFrame([['log_loss_cost', 0, 'response variable', 'continuous - numeric']], columns = df_features_updated.columns), df_features_updated], ignore_index = True)
            
        if col == 'log_total_inc_loss':
            df_clean['log_total_inc_loss'] = np.log(df_clean['total_inc_loss'] + 1)    # adding 1 so that we don't have log(0)
            
            df_features_updated = pd.concat([pd.DataFrame([['log_total_inc_loss', 0, 'response variable', 'continuous - numeric']], columns = df_features_updated.columns), df_features_updated], ignore_index = True)
            
        if verbose:
            print('Response variable created: ', col)
            print(df_clean[col].describe())
            print('\n')
    
    return df_clean, df_features_updated


def calculate_stratification_variables(df, df_features, columns, verbose = True):
    '''
    Create stratification variables for sampling.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe of observations.
    df_features : pandas.DataFrame
        Dataframe of features.
    columns : list
        List of stratification variables to create.
    verbose : bool
        If set to True, will print basic comments.

    Returns
    -------
    df_clean : pandas.DataFrame
        Dataframe of observations with stratification variables included as new columns.
    df_features_updated : pandas.DataFrame
        Dataframe of features with new features added to model feature set.
    '''
    
    df_clean = df.copy()
    df_features_updated = df_features.copy()
    
    for col in columns:   
        if col == 'loss_bin':
            bins = list(range(-1, 0, 1)) + list(range(0, 10000, 1000)) + list(range(10000, 100000, 5000)) + list(range(100000, 10000000, 100000))
                # [0] bin + (0 to 10,000] by $1000 increments + (10,000 to 100,000] by $5000 increments + (100,000 to 10,000,000] by $100,000 increments
            bin_labels = list(range(0, len(bins)-1))
            
            df_clean['loss_bin'] = pd.cut(df_clean['total_inc_loss'], bins = bins, labels = bin_labels)
            df_clean['loss_bin'] = df_clean['loss_bin'].astype(float)
            
            df_features_updated = pd.concat([pd.DataFrame([['loss_bin', 0, 'stratification variable', 'discrete - numeric']], columns = df_features_updated.columns), df_features_updated], ignore_index = True)
            
        if verbose:
            print('Stratification variable created: ', col)
            print(df_clean[col].describe())
            print('\n')
    
    return df_clean, df_features_updated