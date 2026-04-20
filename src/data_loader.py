import pandas as pd
import numpy as np
from utils.DataLoading import load_rul_truth

def xgboost_train_loader(train_path, val_path, 
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

def xgboost_test_loader(test_path, rul_path,
                        selected_sensors=['T24', 'T30', 'T50', 'P30', 'Nf', 'Nc', 'epr', 'Ps30', 'NRf', 'NRc', 'htBleed'],
                        is_snapshot=True,
                        ):
    #load test data
    test_df = pd.read_csv(test_path)
    y_test = load_rul_truth(rul_path)
    features = ['op_regime'] + selected_sensors

    if is_snapshot:
        # Get the last row for each unit_id
        test_last_cycles = test_df.groupby('unit_id').last().reset_index()
        # Separate features (X_test)
        X_test = test_last_cycles[features]

    else:
        # Create unit_id col for y_test
        y_test['unit_id'] = range(1, len(y_test) + 1)
        
        # Get max cycle for each engine in test
        max_cycles = test_df.groupby('unit_id')['cycle'].max().reset_index()
        max_cycles.columns = ['unit_id', 'max_cycle']

        # Merge truth into main test dataframe
        df_merged = test_df.merge(y_test, on='unit_id').merge(max_cycles, on='unit_id')

        # Back-calculate the RUL for every row
        # Formula: Current_True_RUL = Final_True_RUL + (Max_Cycle - Current_Cycle)
        df_merged['true_rul_actual'] = df_merged['true_rul'] + (df_merged['max_cycle'] - df_merged['cycle'])

        y_test = df_merged['true_rul_actual']
        X_test = df_merged[features]
    
    return X_test, y_test

