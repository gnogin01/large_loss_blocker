''' '''

import numpy as np
from hyperopt import hp
from hyperopt.pyll.base import scope     # scope.int() to change hp.quniform() values from float to int

def create_hyperopt_feature_space(feature_list):
    feature_space = {}
    for col in feature_list:
        feature_space[col] = hp.choice(col, [0, 1])
                
    return feature_space

    
def create_hyperopt_hyperparameter_space(model_type = 'quantile_regressor'):
    # default hyperparameter space
    hyperparameter_space = {
        'colsample_bytree':        hp.uniform('colsample_bytree', 0.5, 1),
        'early_stopping_rounds':   10,
        'enable_categorical':      True, 
        'eta':                     hp.quniform('eta', 0.025, 0.5, 0.025),
        'gamma':                   hp.uniform ('gamma', 0, 9),
        'max_delta_step':          hp.uniform('max_delta_step', 1, 10),
        'max_depth':               scope.int(hp.quniform('max_depth', 3, 15, 1)),
        'min_child_weight':        scope.int(hp.quniform('min_child_weight', 0, 10, 1)),
        'n_estimators':            scope.int(hp.quniform('n_estimators', 100, 1000, 1)),
        'n_jobs':                  -1, 
        'random_state':            50919,
        'reg_alpha':               scope.int(hp.quniform('reg_alpha', 0, 100, 1)),
        'reg_lambda':              hp.uniform('reg_lambda', 0, 1)    
    }
        
    if model_type == 'quantile_regressor':
        hyperparameter_space['eval_metric'] = 'rmse'
        hyperparameter_space['quantile_alpha'] = np.array([0.99])
        hyperparameter_space['objective'] = 'reg:quantileerror'
            
    if model_type == 'binary_classifier':
        hyperparameter_space['eval_metric'] = 'aucpr'
        hyperparameter_space['objective'] = 'binary:logistic'
        hyperparameter_space['scale_pos_weight'] = hp.uniform('scale_pos_weight', 1, 200)
        
    if model_type == 'tweedie_regressor':
        hyperparameter_space['eval_metric'] = 'tweedie-nloglik@1.2'
        # tweedie_variance_power
        hyperparameter_space['objective'] = 'reg:tweedie'
        
            
    return hyperparameter_space