def model_cross_val_assess(model, X_train, y_train, model_name):
    '''
    Takes *fitted* model in args1 and applies it to the train set (predictors matrix in args2 and respone vector in args3) while also taking model's name as arg4. 
    Returns set of metrics in a dataframe object.

    ### Examplary call of the function:
    model_cross_val_assess(model, X_train, y_train, X_test, y_test, 'my_model')


    ### How to concatenate results of several models for comparison:
    a = model_cross_val_assess(model_1, X_train, y_train, 'my_model_1')
    b = model_cross_val_assess(model_2, X_train, y_train, 'my_model_2')
    c = pd.concat([a, b], axis=1)
    '''
    import pandas as pd
    import numpy as np
    import locale
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
    from sklearn.model_selection import cross_val_score

    def get_num_columns(X):
        if isinstance(X, pd.DataFrame):
            return len(X.columns)
        elif isinstance(X, np.ndarray):
            return X.shape[1]
        else:
            raise TypeError('X must be a pandas DataFrame or a NumPy array')

    scores_dict = {}

    # Use cross-validation to evaluate the model on the training data
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)

    # Calculate the mean and standard deviation of the cross-validation scores
    mean_score = -scores.mean()
    std_score = scores.std()
    coef_variation = std_score / mean_score

    ## Additionally metrics for assessment on train set
    # R2 score
    y_pred = model.predict(X_train)
    r2 = r2_score(y_train, y_pred)

    # Adjusted R2
    p = get_num_columns(X_train)
    n = len(y_train)
    adj_r2 = 1-(1-r2)*(n-1)/(n-p-1)

    # Now return the metrics as the dict
    scores_dict[model_name] = [r2, adj_r2, mean_score, std_score, coef_variation]

    df_train_eval = pd.DataFrame(data=scores_dict, index=['r2', 'adj_r2', 'mean_cv_rmse', 'std_cv_rmse', 'coef_of_var'])
    
    # Set float_format to display numbers without scientific notation and with thousand separators
    locale.setlocale(locale.LC_ALL, '')  # Set the locale to the user's default locale
    pd.options.display.float_format = lambda x: format(locale.atof(f"{x:.1f}"), ',') if abs(x) > 999 else f"{x:.4f}"
    
    
    return df_train_eval
