import pandas as pd

def xgboost_data_loader(train_path, val_path, 
    selected_sensors=['T24', 'T30', 'T50', 'P30', 'Nf', 'Nc', 'epr', 'Ps30', 'NRf', 'NRc', 'htBleed'], 
    target_col='target_rul'
    ):
    """
    Separates the data into X (features) and y (labels) for XGBoost.
    Excludes metadata like unit_id and cycle.
    """
    # Include op_regime if you want it as a feature
    features = ['op_regime'] + selected_sensors
    
    train_df = pd.read_csv(train_path)
    X_train = train_df[features]
    y_train = train_df[target_col]
    
    val_df = pd.read_csv(val_path)
    X_val = val_df[features]
    y_val = val_df[target_col]
    
    return X_train, y_train, X_val, y_val