import pandas as pd
import numpy as np

#round off the 3 regime settings and group in 6 operating conditions conditions
def identify_operating_regimes(df):
    """
    Groups the 3 settings into 6 discrete operating regimes.
    """
    df_regime = df.copy()
    
    # Round settings to collapse minor noise into clusters
    df_regime['alt_bin'] = df_regime['altitude'].round(-3)
    df_regime['mach_bin'] = df_regime['mach_number'].round(1)
    df_regime['tra_bin'] = df_regime['tra'].round()
    
    # Add new column for the combination
    df_regime['op_regime'] = df_regime.groupby(['alt_bin', 'mach_bin', 'tra_bin']).ngroup()
    
    # Drop bin columns
    df_regime = df_regime.drop(columns=['alt_bin', 'mach_bin', 'tra_bin'])
    
    print(f"Identified {df_regime['op_regime'].nunique()} unique operating regimes.")
    return df_regime

#normalize data considering mean and std. dev. per operating regime
def normalize_by_regime(df):
    """
    Standardizes sensors based on their specific operating regime.
    Formula: z = (x - mean_regime) / std_regime
    """
    df_norm = df.copy()

    #identify sensor cols
    sensor_cols = [col for col in df.columns if col not in ['unit_id', 'cycle', 'altitude', 'mach_number', 'tra', 'op_regime']]
    
    # Group by the 'op_regime' and transform each sensor column
    for sensor in sensor_cols:
        # Calculate mean and std for each regime
        df_norm[sensor] = df_norm.groupby('op_regime')[sensor].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )
        
    print(f"Normalization complete for {len(sensor_cols)} sensors.")
    return df_norm