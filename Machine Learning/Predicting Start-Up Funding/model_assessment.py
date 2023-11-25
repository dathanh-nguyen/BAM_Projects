# Define a function to assess an instance of the fitted model on the test/validation data sets of choosing:
def model_assess(model, X_test, y_test, model_name):
    '''
    Takes *fitted* model in args1 and applies it to the test set (predictors matrix in args2 and respone vector in args3) while also taking model's name as arg4. 
    Returns set of metrics in a dataframe object.

    ### Examplary call of the function:
    model_assess(model, X_test, y_test, 'my_model')

    
    ### How to concatenate results of several models for comparison:
    a = model_assess(model_1, X_test, y_test, 'my_model_1')
    b = model_assess(model_2, X_test, y_test, 'my_model_2')
    c = pd.concat([a, b], axis=1)
    '''
    import pandas as pd
    import numpy as np
    import locale
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

    def get_num_columns(X):
        if isinstance(X, pd.DataFrame):
            return len(X.columns)
        elif isinstance(X, np.ndarray):
            return X.shape[1]
        else:
            raise TypeError('X must be a pandas DataFrame or a NumPy array')

    scores_dict = {}

    # Access predictions of the fitted model on test data
    y_pred = model.predict(X_test)

    ## Metrics
    # R2 score
    r2 = r2_score(y_test, y_pred)

    # Adjusted R2
    p = get_num_columns(X_test)
    n = len(y_test)
    adj_r2 = 1-(1-r2)*(n-1)/(n-p-1)

    # RMSE
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    # MAE
    mae = mean_absolute_error(y_test, y_pred)

    # MAPE
    mape = mean_absolute_percentage_error(y_test, y_pred)

    # Now return the metrics as the dict
    scores_dict[model_name] = [r2, adj_r2, rmse, mae, mape]

    df_eval = pd.DataFrame(data=scores_dict, index=['r2', 'adj_r2', 'rmse', 'mae', 'mape'])
    
    # Set float_format to display numbers without scientific notation and with thousand separators
    locale.setlocale(locale.LC_ALL, '')  # Set the locale to the user's default locale
    pd.options.display.float_format = lambda x: format(locale.atof(f"{x:.1f}"), ',') if abs(x) > 999 else f"{x:.4f}"
    
    
    return df_eval