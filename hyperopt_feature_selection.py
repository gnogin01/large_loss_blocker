''' '''

import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, tpe, space_eval
import xgboost as xgb

class hyperopt_xgboost:   
    def __init__(self, model_type, X_train, y_train, X_test, y_test, feature_space, hyperparameter_space):
        self.model_type = model_type
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.feature_space = feature_space
        self.hyperparameter_space = hyperparameter_space
        self.space = {**self.feature_space, **self.hyperparameter_space}
        
        
    def objective(self, params):
        # Dictionary of current hyperparameter values.
        hyperparameters = {k: params[k] for k in self.hyperparameter_space.keys()}

        # List of features are currently in model.
        cols = [i for i, j in params.items() if (i in self.feature_space.keys()) & (j == 1)]
        
        # Specfify model with hyperparameters.
        if self.model_type == 'xgb_regressor':
            model = xgb.XGBRegressor(**hyperparameters)
        if self.model_type == 'xgb_classifier':
            model = xgb.XGBClassifier(**hyperparameters)
        
        # Fit model with subset of features.
        model.fit(self.X_train[cols],
                  self.y_train,
                  eval_set = [(self.X_train[cols], self.y_train), (self.X_test[cols], self.y_test)],
                  verbose = False)
        
        # Check evaluation on test set and record best loss 
        if (hyperparameters['eval_metric'] == 'rmse') | (hyperparameters['eval_metric'] == 'logloss') | (hyperparameters['eval_metric'] == 'tweedie-nloglik@1.2') :
            loss = min(model.evals_result()['validation_1'][hyperparameters['eval_metric']])
        
        if (hyperparameters['eval_metric'] == 'aucpr') | (hyperparameters['eval_metric'] == 'auc'):
            loss = -1 * max(model.evals_result()['validation_1'][hyperparameters['eval_metric']])
   
        return {'loss': loss, 'status': STATUS_OK, 'trained_model': model}
    
    
    def get_best_model(self, trials):
        valid_trial_list = [trial for trial in trials if STATUS_OK == trial['result']['status']]
        losses = [float(trial['result']['loss']) for trial in valid_trial_list]
        best_trial_obj = valid_trial_list[np.argmin(losses)]
        
        return best_trial_obj['result']['trained_model']


    def optimize(self, max_evals = 20):
        # first 20 evals are random
        trials = Trials()
        best = fmin(fn = self.objective,
                    space = self.space,
                    algo = tpe.suggest,
                    max_evals = max_evals,
                    trials = trials)
        model = self.get_best_model(trials)

        return (space_eval(self.space, best), model)
