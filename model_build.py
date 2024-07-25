import numpy as np
from sklearn.model_selection import ParameterGrid
import xgboost as xgb
from sklearn.metrics import brier_score_loss
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier, cv, DMatrix, XGBClassifier
import pandas as pd
import re


def lower_cardinality(df: pd.DataFrame, col_list: list, threshold: int):
    """
    Replace low-frequency values in specified columns of a DataFrame with 'RARE_VALUE'.

    This function iterates through the specified columns in the DataFrame and replaces values
    that occur less frequently than a given threshold with 'RARE_VALUE'. The threshold is specified
    as a percentage of the total number of rows in the DataFrame.

    Parameters:
    ----------
    df : pandas.DataFrame
        The input DataFrame in which you want to reduce cardinality.

    col_list : list
        A list of column names in the DataFrame for which you want to reduce cardinality.

    threshold : float
        The threshold for identifying low-frequency values. It should be a percentage
        (between 0 and 1) of the total number of rows in the DataFrame.

    Returns:
    -------
    pandas.DataFrame
        A modified DataFrame with reduced cardinality in the specified columns.

    Raises:
    -------
    ValueError
        If the threshold is greater than 1, which would be invalid for a percentage.
        If the input is not a pandas DataFrame.
        If col_list is not a list.

    Example:
    --------
    # Reduce cardinality in the 'color' column by replacing values that occur in less
    # than 5% of the rows with 'RARE_VALUE'.
    df = lower_cardinality(df, ['color'], 0.05)
    """
    if threshold > 1:
        raise ValueError("threshold should be a percent")
    if not isinstance(df, pd.DataFrame):
        raise ValueError("only accepts pandas dataframes")
    if not type(col_list) == list:
        raise ValueError("col_list should be a list")
    # threshold should be a precent to be more flexible, so we will convert it first
    threshold_absolute = threshold * df.shape[0]

    for col in col_list:
        df.loc[df[col].value_counts()[df[col]].values < threshold_absolute, col] = (
            "RARE_VALUE"
        )

    return df


######## Create & Stratify Folds
def stratify_folds(df, strat_by, groups=None, nfolds=5, stratseed=1729):
    """
    Stratify data into folds for cross-validation based on grouping and stratification.

    This function stratifies the input DataFrame into multiple folds for cross-validation.
    It ensures that each fold contains a balanced distribution of samples based on the
    specified stratification columns and grouping columns. Stratification is performed to
    achieve a similar distribution of values within each fold.

    Parameters:
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the data to be stratified into folds.

    strat_by : list
        A list of columns in the DataFrame to be used for stratification.

    groups : list or None, optional
        A list of columns in the DataFrame for grouping. If provided, stratification is
        performed within each group. If None, no grouping is applied.

    nfolds : int, optional
        The number of folds to create. Default is 5.

    stratseed : int, optional
        The seed for the random number generator used for stratification. Default is 1729.

    Returns:
    -------
    list of tuples
        A list of tuples where each tuple represents a fold. Each fold contains two lists:
        the first list is the indices of samples in the training set, and the second list is
        the indices of samples in the validation set.

    Example:
    --------
    # Stratify a DataFrame 'data' into 5 folds, considering 'age' as the stratification column,
    # and 'gender' as the grouping column.
    folds = stratify_folds(data, strat_by=['age'], groups=['gender'], nfolds=5)

    for i, (train_indices, val_indices) in enumerate(folds):
        print(f"Fold {i + 1}: Training Set - {len(train_indices)} samples, Validation Set - {len(val_indices)} samples")
    """
    stratcol = list(groups + strat_by)
    n = len(strat_by)
    g = len(groups)
    np.random.seed(stratseed)

    if groups == None:
        group_df = df

    else:
        group_df = df[stratcol].groupby(groups, as_index=False).mean()

    l = len(group_df)

    rand = np.random.normal(loc=0, scale=1, size=1)  # randomness to add to scoring)
    stratscore = np.zeros(l)  # store stratification scores for each grouping

    for i in range(n):
        col = group_df[stratcol[g + i]].to_numpy()
        perc = np.percentile(col, np.arange(1, 99.001, step=1))  # get 99 bin edges
        digi = np.digitize(col, perc)
        stratscore += (digi + rand) / 10 ** (2 * i)
        stratlist = np.argsort(stratscore)
        # stratassign = np.zeros(1).astype(int)

    stratassign = np.zeros(l).astype(int)

    for ifold in range(nfolds):
        stratassign[stratlist[ifold::nfolds]] = ifold
        group_df["fold_assignment"] = pd.DataFrame(stratassign)

    append = list(groups + ["fold_assignment"])

    df1 = df.merge(group_df[append], how="left", left_on=groups, right_on=groups)[
        ["fold_assignment"]
    ]

    folds = []

    for i in range(nfolds):
        outfold = df1[df1["fold_assignment"] == i].index.tolist()
        infold = df1[df1["fold_assignment"] != i].index.tolist()
        folds += [(infold, outfold)]

    return folds


def create_testing_dfs(df2, ids, strat_by, dependent, nfolds=5, seed = 50919):
    """
    Create testing DataFrames and folds for machine learning tasks.

    This function prepares the necessary DataFrames and folds for machine learning tasks.
    It stratifies the input DataFrame, sets up training and validation datasets, and creates
    DMatrix objects for XGBoost modeling.

    Parameters:
    ----------
    df2 : pandas.DataFrame
        The current DataFrame containing all the required columns.

    ids : list
        A list of ID columns that are not used in the modeling process.

    strat_by : list
        A list of columns to stratify by when creating folds for cross-validation.

    dependent : str
        The column name of the dependent variable (target) that will be predicted.

    nfolds : int, optional
        The number of stratification folds to create. Default is 5.

    Returns:
    -------
    tuple
        A tuple containing the following elements:
        - df2: The original DataFrame.
        - training: Training dataset.
        - validation: Validation dataset.
        - X_train: Features of the training dataset.
        - y_train: Target variable of the training dataset.
        - X_val: Features of the validation dataset.
        - y_val: Target variable of the validation dataset.
        - folds: A list of stratified folds for cross-validation, each containing two sets
          (training and validation) without ID columns.
        - dtrain: DMatrix for training data.
        - dval: DMatrix for validation data.
        - fold_strat_sub: Stratified folds used for the training dataset.

    Example:
    --------
    # Prepare testing DataFrames and folds for modeling.
    df2, training, validation, X_train, y_train, X_val, y_val, folds, dtrain, dval, fold_strat_sub = create_testing_dfs(
        df2=dataframe,
        ids=['ID'],
        strat_by=['age'],
        dependent='income',
        nfolds=5
    )
    """

    ##### Stratify
    fold_strat = stratify_folds(
        df=df2, strat_by=strat_by, groups=ids, nfolds=nfolds, stratseed=seed
    )

    # Set validation
    training = df2.iloc[fold_strat[0][0]]
    validation = df2.iloc[fold_strat[0][1]]

    # Set up dmats
    X_train = training[[x for x in df2.columns if x not in dependent + ids]]
    y_train = training[dependent]

    X_val = validation[[x for x in df2.columns if x not in dependent + ids]]
    y_val = validation[dependent]

    # Create folds from training
    fold_strat_sub = stratify_folds(
        training, strat_by=strat_by, groups=ids, nfolds=nfolds, stratseed=seed
    )
    folds = [
        (
            training.iloc[fold_strat_sub[0][0]].drop(columns=ids),
            training.iloc[fold_strat_sub[0][1]].drop(columns=ids),
        ),
        (
            training.iloc[fold_strat_sub[1][0]].drop(columns=ids),
            training.iloc[fold_strat_sub[1][1]].drop(columns=ids),
        ),
        (
            training.iloc[fold_strat_sub[2][0]].drop(columns=ids),
            training.iloc[fold_strat_sub[2][1]].drop(columns=ids),
        ),
    ]

    # Set dmat
    training = training
    validation = validation

    # Drop from train/val
    dtrain = DMatrix(data=X_train, label=y_train, enable_categorical=True)
    dval = DMatrix(data=X_val, label=y_val, enable_categorical=True)

    return (
        df2,
        training,
        validation,
        X_train,
        y_train,
        X_val,
        y_val,
        folds,
        dtrain,
        dval,
        fold_strat_sub,
    )


# LOU TO DO: incorporate custom objective function
def xgboost_train_and_save(
    Xs,
    y,
    parameter_grid,
    cv_folds,
    eval_metric,
    seed=1729,
    boosting_rounds=100,
    early_stop_rounds=3,
):
    """
    Train XGBoost models with hyperparameter tuning and cross-validation, and save evaluation results.

    This function performs hyperparameter tuning and cross-validation using XGBoost. It iterates over
    a grid of hyperparameters, trains XGBoost models, and saves evaluation results in a DataFrame.

    Parameters:
    ----------
    Xs : pandas.DataFrame
        The feature matrix for training the XGBoost models.

    y : pandas.Series
        The target variable to predict.

    parameter_grid : dict
        A dictionary of hyperparameter grids to search during hyperparameter tuning.

    cv_folds : list of tuples
        A list of cross-validation folds, where each tuple contains training and validation data indices.

    eval_metric: string or list
        A string or list to define evaluation metric.
            Example for regression: 'rmse'
            Example for binary classification: 'logloss' or ['logloss', 'auc']

    seed : int, optional
        The random seed for reproducibility. Default is 1729.

    boosting_rounds : int, optional
        The maximum number of boosting rounds for each model. Default is 100.

    early_stop_rounds : int, optional
        The number of early stopping rounds for XGBoost training. Default is 3.

    Returns:
    -------
    pandas.DataFrame
        A DataFrame containing evaluation results for each combination of hyperparameters,
        including metrics such as RMSE.

    Example:
    --------
    # Train XGBoost models with hyperparameter tuning and cross-validation.
    hyperparameter_grid = {'max_depth': [3, 4, 5], 'learning_rate': [0.1, 0.01]}
    cv_folds = [(train_indices1, val_indices1), (train_indices2, val_indices2)]
    eval_metric = 'rmse'
    eval_results = xgboost_train_and_save(X_train, y_train, hyperparameter_grid, cv_folds, seed=42)

    # Analyze the evaluation results to select the best hyperparameters.
    best_params = eval_results.loc[eval_results[eval_metric].idxmin()]
    """
    xgb_param_grid = parameter_grid
    fd = cv_folds
    num_boost_round = boosting_rounds
    early_stopping_rounds = early_stop_rounds
    seed = seed

    dtrain1 = xgb.DMatrix(Xs, label=y, enable_categorical=False)

    for i in range(0, len(ParameterGrid(xgb_param_grid))):
        # print(list(ParameterGrid(xgb_param_grid))[0], type(list(ParameterGrid(xgb_param_grid))[0]))
        xgb_param_grid_temp = list(ParameterGrid(xgb_param_grid))[i]
        xgbCV = xgb.cv(
            params=xgb_param_grid_temp,
            dtrain=dtrain1,
            num_boost_round=num_boost_round,
            nfold=len(fd),
            folds=fd,
            metrics=eval_metric,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=True,
            seed=seed,
        )
    hyperparam_list = []
    hyperparam_list.append(
        pd.concat(
            [
                pd.DataFrame(
                    list(ParameterGrid(xgb_param_grid))[i], index=[0]
                ).reset_index(),
                pd.DataFrame(xgbCV.iloc[xgbCV.shape[0] - 1]).T.reset_index(),
            ],
            axis=1,
        )
    )

    eval_df = pd.concat(hyperparam_list)

    return eval_df


def one_hot_encoding(cat_vars, model_data=None):
    """
    Encode categorical variables using one-hot encoding.

    Parameters:
    cat_vars (list): A list of categorical variables to be one-hot encoded.
    model_data (pandas.DataFrame): Input dataframe containing the data to be encoded. (For training phase)

    Returns:
    dict: A dictionary containing two items:
        - 'encoded_data' (pandas.DataFrame): The dataframe with categorical variables encoded using one-hot encoding.
        - 'new_var_list' (list): A list of new variable names generated after encoding.
        - 'encoding_map' (dict): A dictionary containing mappings of categorical variable values to encoded column names.
    """

    cat_names = []
    new_var_list = []
    encoding_map = {}

    for var in cat_vars:
        cat_list = "var" + "_" + var
        cat_list = pd.get_dummies(model_data[var], prefix=var, drop_first=True)

        # Create encoding map for the current variable
        encoding_map[var] = {}
        for value in model_data[var].unique():
            encoding_map[var][value] = var + "_" + str(value)

        cat_list.index = model_data.index
        cat_names.extend(cat_list.columns)
        new_var_list.extend(cat_list.columns)
        model_data1 = pd.concat([model_data, cat_list], axis=1)
        model_data = model_data1

    model_data = model_data.drop(cat_vars, axis=1)

    return {
        "encoded_data": model_data,
        "new_var_list": new_var_list,
        "encoding_map": encoding_map,
    }


def clean_column_names(df):
    """
    Cleans column names in a DataFrame by replacing square brackets and angle brackets with underscores.

    Parameters:
    - df (pandas.DataFrame): The DataFrame whose column names are to be cleaned.

    Returns:
    - pandas.DataFrame: The DataFrame with cleaned column names.
    """
    regex = re.compile(r"\[|\]|<", re.IGNORECASE)
    df.columns = [
        regex.sub("_", col) if any(x in str(col) for x in set(("[", "]", "<"))) else col
        for col in df.columns.values
    ]
    return df