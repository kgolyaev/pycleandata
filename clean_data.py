import pandas as pd
import numpy as np
from sklearn.ensemble import forest

###############################################################################
def set_random_forest_sample_size(nobs):
    """
    Override default behavior of scikit-learn random forest models to
    sample the same number of rows from the original data with replacement.
    Instead, this forces random forest to fit each tree on a sample of size
    nobs, greatly speeding up the algorithm on large datasets.
    """    
    forest.generate_sample_indices = (lambda rs, n_samples:
        forest.check_random_state(rs).randint(0, n_samples, nobs))
###############################################################################
def reset_random_forest_sample_size():
    """
    Restore the default behavior of scikit-learn random forest models to
    sample the same number of rows from the original data with replacement.
    This should be used if `set_random_forest_sample_size()` had been 
    previously run. 
    """    
    forest.generate_sample_indices = (lambda rs, n_samples:
        forest.check_random_state(rs).randint(0, n_samples, n_samples))
###############################################################################
def create_datetime_features(df, time_colname, include_time=False, 
                             drop=True, inplace=False):
    """Simple feature creation based on time index of a DataFrame
    
    Arguments:
        df {pd.DataFrame} -- DataFrame with time column where the features 
                             will be put into
        time_colname {str} -- name of column in {df} which contains the 
                              time index
    
    Keyword Arguments:
        include_time {bool} -- should time component of datetime index 
                               be included? (default: {False})
        drop {bool} -- should the original time index column be dropped? 
                       (default: {True})
        inplace {bool} -- should the input DataFrame be modified in place? 
                          (default: {False})
    
    Returns:
        Either a transformed DataFrame or `None` if `inplace` is `True`
    """

    if not isinstance(df, pd.DataFrame):
        raise ValueError('Argument `df` must be a pandas DataFrame!')
    if time_colname not in df.columns:
        raise ValueError(('Column {} not found in DataFrame!').format(
            time_colname))
    if not isinstance(include_time, bool):
        raise ValueError('Argument `include_time` must be boolean!')
    if not isinstance(drop, bool):
        raise ValueError('Argument `drop` must be boolean!')
    if not isinstance(inplace, bool):
        raise ValueError('Argument `inplace` must be boolean!')

    dtcol = df[time_colname]
    if not np.subdtype(dtcol.dtype, np.datetime64):
        dtcol = pd.to_datetime(dtcol, infer_datetime_format=True)
    
    features_to_construct = [
        'year', 'quarter', 'month', 'weekofyear', 'dayofyear', 'day', 
        'dayofweek', 'daysinmonth', 'is_leap_year', 'is_year_start', 
        'is_year_end', 'is_quarter_start', 'is_quarter_end', 
        'is_month_start', 'is_month_end']
    if include_time:
        features_to_construct += ['hour', 'minute', 'second']
    if dtcol.dt.tz is not None:
        features_to_construct += ['tz']

    result = df if inplace else df.copy()
    for feature in features_to_construct:
        result[time_colname + "_" + feature] = getattr(dtcol.dt, feature)
    # sort of a trend feature
    result[time_colname + "_elapsed"] = dtcol.astype(np.int64) // (10 ** 9)
    if drop:
        result.drop(time_colname, axis=1, inplace=True)
    return(result if not inplace else None)
###############################################################################
def impute_nans_and_make_dummies(df, colname, fill_values_dict=None, 
                                 inplace=False):
    """Impute missing values in a numeric column of a DataFrame with a constant
    and create an indicator feature for missing rows.
    
    Arguments:
        df {pd.DataFrame} -- DataFrame with the column from which missing 
                             values must be sanitized
        colname {str} -- name of column in {df} which must be sanitized
    
    Keyword Arguments:
        fill_values_dict {dict} -- dictionary of `colname`: `value' pairs that 
                                   had been used on earlier data
                                   (default: {None})
        inplace {bool} -- should the input DataFrame be modified in place? 
                          (default: {False})
    
    Returns:
        A tuple with two elements. First element is either a transformed 
        DataFrame or `None` if `inplace` is `True`. Second element is the 
        updated version of `fill_values_dict`. 
    """

    if not isinstance(df, pd.DataFrame):
        raise ValueError('Argument `df` must be a pandas DataFrame!')
    if colname not in df.columns:
        raise ValueError(('Column {} not found in DataFrame!').format(colname))
    if not pd.api.types.is_numeric_dtype(df[colname]):
        raise ValueError(('`impute_nans_and_make_dummies()` must be called on '
                          + 'a numeric column!'))
    if fill_values_dict is None:
        fill_values_dict = dict()
    elif not isinstance(fill_values_dict, dict):
        raise ValueError(('If specified, argument `fill_values_dict` must be '  
                           + 'a dict!'))
    if not isinstance(inplace, bool):
        raise ValueError('Argument `inplace` must be boolean!')

    result = df if inplace else df.copy()
    if pd.isnull(result[colname]).sum() or (colname in fill_values_dict):
        result[colname + "_nan"] = pd.isnull(result[colname])
        fill_value = fill_values_dict[colname] if colname in fill_values_dict \
                     else result[colname].median()
        result[colname] = result[colname].fillna(fill_value)
        fill_values_dict[colname] = fill_value
    return (result if not inplace else None, fill_values_dict)    
###############################################################################
def handle_nans_numeric(df, fill_values_dict=None, inplace=False):
    """Impute missing values in a numeric column of a DataFrame with a constant
    and create an indicator feature for missing rows.
    
    Arguments:
        df {pd.DataFrame} -- DataFrame with the columns in which missing 
                             values must be sanitized
    Keyword Arguments:
        fill_values_dict {dict} -- dictionary of `colname`: `value' pairs that 
                                   had been used on earlier data
                                   (default: {None})
        inplace {bool} -- should the input DataFrame be modified in place? 
                          (default: {False})
    
    Returns:
        A tuple with two elements. First element is either a transformed 
        DataFrame or `None` if `inplace` is `True`. Second element is the 
        updated version of `fill_values_dict`. 
    """

    if not isinstance(df, pd.DataFrame):
        raise ValueError('Argument `df` must be a pandas DataFrame!')
    if fill_values_dict is None:
        fill_values_dict = dict()
    elif not isinstance(fill_values_dict, dict):
        raise ValueError(('If specified, argument `fill_values_dict` must be '  
                           + 'a dict!'))
    if not isinstance(inplace, bool):
        raise ValueError('Argument `inplace` must be boolean!')

    result = df if inplace else df.copy()
    columns_to_encode = list(df._get_numeric_data().columns)
    for colname in columns_to_encode:
        result, fill_values_dict = impute_nans_and_make_dummies(
            result, colname, fill_values_dict, inplace=False)
    return (result if not inplace else None, fill_values_dict)
###############################################################################
