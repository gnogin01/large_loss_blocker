'''
Create a dataframe that keeps track of which columns are features,
and which features will be included in the model.
'''

import pandas as pd

'''
Sample of manually created dictionary of datatypes. Data sourced from the following SQL code:
https://github.com/kin/large_loss_blocker/blob/3b2e61f73ce629f899836a370e0dc57eff51b827/data/data_query_with_hazard_hub_subset.sql
'''

feature_data_types = {
    'WUICLASS2020': 'nominal - categorical',
    'additional_coverage_a': 'discrete - numeric',
    'address_county': 'nominal - categorical',
    'affinity_discount': 'unknown',
    'age_of_dwelling': 'discrete - numeric',
    'age_of_dwelling_at_bind': 'discrete - numeric',
    'age_of_insured': 'discrete - numeric',
    'animal_liability': 'ordinal - categorical',
    'animal_tier': 'binary - categorical',
    'ansi_compliance': 'unknown',
    'aop_api_premium': 'continuous - numeric',
    'aop_deductible': 'discrete - numeric',
    'api_premium': 'discrete - numeric',
    'bathrooms': 'discrete - numeric',
    'bceg': 'nominal - categorical',
    'bright_policy_id': 'ID column',
    'builder': 'nominal - categorical',
    'burglar_alarm': 'nominal - categorical',
    'cancellation_date': 'date',
    'census_block_group': 'nominal - categorical',
    'chargeable_lapse': 'binary - categorical',
    'chargeable_losses': 'ordinal - categorical',
    'construction_type': 'nominal - categorical',
    'cova_per_sqft': 'unknown',
    'coverage_a': 'discrete - numeric',
    'coverage_b': 'discrete - numeric',
    'coverage_c': 'discrete - numeric',
    'coverage_d': 'discrete - numeric',
    'coverage_e': 'discrete - numeric',
    'coverage_f': 'discrete - numeric',
    'days_insurance_lapsed': 'unknown',
    'deadbolt': 'binary - categorical',
    'direct_repair': 'binary - categorical',
    'distance_to_actual_coast_feet': 'continuous - numeric',
    'effective_date': 'date',
    'electronic_policy_discount': 'binary - categorical',
    'elevation_amsl': 'discrete - numeric',
    'elevation_certificate': 'binary - categorical',
    'emergency_water_removal_service': 'binary - categorical',
    'endors_term_end_date': 'date',
    'endors_term_start_date': 'date',
    'exterior_wall_finish': 'nominal - categorical',
    'family_type': 'unknown',
    'fbc_wind_speed': 'ordinal - categorical',
    'feet_from_fire_department': 'continuous - numeric',
    'feet_from_fire_hydrant_maximum': 'ordinal - categorical',
    'feet_from_fire_hydrant_minimum': 'ordinal - categorical',
    'fire_alarm_monitoring': 'nominal - categorical',
    'fire_alarm_sprinkler': 'nominal - categorical',
    'fire_extinguisher': 'binary - categorical',
    'flood_and_water_backup': 'binary - categorical',
    'flood_zone': 'nominal - categorical',
    'foundation': 'nominal - categorical',
    'full_policy_number': 'ID column',
    'golf_cart_count': 'discrete - numeric',
    'hurricane_deductible': 'discrete - numeric',
    'hurricane_screened_enclosures': 'ordinal - categorical',
    'hurricane_transition_impact': 'continuous - numeric',
    'identity_theft': 'ordinal - categorical',
    'inland_flood_risk_score': 'discrete - numeric',
    'insurance_score': 'discrete - numeric',
    'latitude': 'continuous - numeric',
    'less_than_thousand_feet_from_water': 'binary - categorical',
    'liability_medical_payments': 'nominal - categorical',
    'limited_fungi': 'ordinal - categorical',
    'line': 'nominal - categorical',
    'longitude': 'continuous - numeric',
    'loss_assessment': 'ordinal - categorical',
    'loss_settlement': 'nominal - categorical',
    'months_occupied': 'discrete - numeric',
    'new_purchase': 'binary - categorical',
    'occupancy': 'nominal - categorical',
    'occupancy_rental_mapping': 'unknown',
    'old_coverage_new_flood_rates': 'discrete - numeric',
    'old_coverage_new_rates': 'discrete - numeric',
    'old_coverage_old_flood_rates': 'discrete - numeric',
    'old_coverage_old_rates': 'discrete - numeric',
    'opening_protection': 'nominal - categorical',
    'ordinance_or_law': 'discrete - numeric',
    'payment_schedule': 'nominal - categorical',
    'personal_injury': 'binary - categorical',
    'policy_term': 'ordinal - categorical',
    'pool': 'binary - categorical',
    'prior_liability_limit': 'nominal - categorical',
    'property_type': 'nominal - categorical',
    'protection_class': 'nominal - categorical',
    'purchase_date': 'date',
    'rating_filing_id': 'ID column',
    'rating_id': 'ID column',
    'replacement_cost_contents': 'binary - categorical',
    'responsible_repair': 'binary - categorical',
    'roof_age': 'discrete - numeric',
    'roof_condition_rating': 'ordinal - categorical',
    'roof_construction_year': 'discrete - numeric',
    'roof_cover': 'nominal - categorical',
    'roof_deck': 'nominal - categorical',
    'roof_deck_attachment': 'nominal - categorical',
    'roof_shape': 'nominal - categorical',
    'roof_solar_panels': 'binary - categorical',
    'roof_surface_payment_schedule': 'binary - categorical',
    'roof_type': 'nominal - categorical',
    'roof_wall_connection': 'nominal - categorical',
    'sale_price': 'discrete - numeric',
    'secondary_water_resistance': 'binary - categorical',
    'secured_community': 'nominal - categorical',
    'self_inspection': 'binary - categorical',
    'sinkhole_exclusion': 'binary - categorical',
    'slope': 'continuous - numeric',
    'special_personal_property': 'binary - categorical',
    'sq_ft': 'discrete - numeric',
    'square_feet': 'discrete - numeric',
    'standardized_address': 'ID column',
    'state': 'nominal - categorical',
    'stories': 'discrete - numeric',
    'stories_above_unit': 'discrete - numeric',
    'storm_surge_risk_score': 'discrete - numeric',
    'sum_earned_exposure': 'continuous - numeric',
    'supplemental_heating': 'binary - categorical',
    'term_end_date': 'date',
    'term_endorsement_most_recent_first': 'ordinal - categorical',
    'term_start_date': 'date',
    'terrain': 'nominal - categorical',
    'times_rented': 'binary - categorical',
    'total_inc_loss': 'continuous - numeric',
    'total_transition_impact': 'continuous - numeric',
    'usage': 'nominal - categorical',
    'vacancy_permission': 'binary - categorical',
    'water_alarm': 'binary - categorical',
    'water_backup': 'ordinal - categorical',
    'water_damage': 'nominal - categorical',
    'water_detection': 'binary - categorical',
    'water_protective_device': 'unknown',
    'welcome_discount': 'unknown',
    'wildfire_grade': 'nominal - categorical',
    'wind_borne_debris_region': 'binary - categorical',
    'wind_hail_exclusion': 'binary - categorical',
    'within_park': 'binary - categorical',
    'written_aop_premium': 'continuous - numeric',
    'written_cat_premium': 'continuous - numeric',
    'written_premium': 'continuous - numeric',
    'year_built': 'discrete - numeric',
    'zipcode': 'nominal - categorical'
}

def create_features_df(df, feature_data_types = feature_data_types, verbose = True):
    '''
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe of observations.
    feature_data_types : dictionary
        Default setting is to use the dictionary provided in this python script.
        Can be used with a different feature: data type dictionary.

    Returns
    -------
    df_features : pandas.DataFrame
        Dataframe of features.
        Columns:
            feature: feature name
            in_feature_set: binary 0/1 value indicating whether feature will be included in the model
            reason_dropped: the reason why a feature will not be included in the model
            stat_data_type: the possible data type categories are 
               'ID column'
               'date column'
               'discrete - numeric'
               'continuous - numeric'
               'binary - categorical'
               'nominal - categorical'
               'ordinal - categorical'
    '''
    
    df_features = pd.DataFrame(data = {'feature': df.columns, 'in_feature_set': 1, 'reason_dropped': ''})
    df_features['stat_data_type'] = df_features['feature'].map(feature_data_types)
    
    if verbose:
        features_missing_data_type = df_features[df_features['stat_data_type'].isna()]['feature'].tolist()
        
        print('The features missing datatype are: \n')
        print(*sorted(features_missing_data_type), sep = '\n')
        print('\n')
    
    return df_features


def get_categorical_features(df_features):
    categorical_features = df_features[(df_features['in_feature_set'] == 1) & 
                                       ((df_features['stat_data_type'] == 'nominal - categorical') | 
                                        (df_features['stat_data_type'] == 'ordinal - categorical') |
                                        (df_features['stat_data_type'] == 'binary - categorical'))]['feature'].tolist()
    return categorical_features


def get_numeric_features(df_features):
    numeric_features = df_features[(df_features['in_feature_set'] == 1) & 
                                   ((df_features['stat_data_type'] == 'discrete - numeric') | 
                                    (df_features['stat_data_type'] == 'continuous - numeric'))]['feature'].tolist()
    
    return numeric_features