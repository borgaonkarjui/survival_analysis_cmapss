import pandas as pd
import numpy as np
from utils.ExploratoryDataAnalysis import fetch_sensor_cols

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
def normalize_by_regime(df, stats_path='regime_stats.csv'):
    """
    Normalizes a dataframe using pre-calculated statistics from a CSV.
    """
    df_norm = df.copy()
    stats_df = pd.read_csv(stats_path)

    #converting all sensor cols to float
    sensor_cols = stats_df['sensor'].unique().tolist()
    df_norm[sensor_cols] = df_norm[sensor_cols].astype('float64')
    
    # Iterate through each regime and sensor present in the stats file
    for regime in stats_df['op_regime'].unique():
        regime_mask = df_norm['op_regime'] == regime
        
        # Only process if this regime exists in the current dataframe
        if regime_mask.any():
            regime_stats = stats_df[stats_df['op_regime'] == regime]
            
            for _, row in regime_stats.iterrows():
                sensor = row['sensor']
                m = row['mean']
                s = row['std']
                
                # Apply normalization: (x - mean) / std
                if sensor in df_norm.columns:
                    df_norm.loc[regime_mask, sensor] = (df_norm.loc[regime_mask, sensor] - m) / s
                    
    print(f"Normalization complete using stats from {stats_path}")
    return df_norm

#regime wise 
def save_regime_stats(df, output_path='regime_stats.csv'):
    """
    Calculates mean and std for each sensor per regime and saves to CSV.
    The CSV will have columns: op_regime, sensor_name, mean, std
    """
    stats_list = []

    #identify sensor cols
    sensor_cols = fetch_sensor_cols(df)
    
    for regime in sorted(df['op_regime'].unique()):
        regime_data = df[df['op_regime'] == regime]
        
        for sensor in sensor_cols:
            m = regime_data[sensor].mean()
            s = regime_data[sensor].std()
            stats_list.append({
                'op_regime': regime,
                'sensor': sensor,
                'mean': m,
                'std': s if s > 0 else 1.0  # Avoid division by zero
            })
            
    stats_df = pd.DataFrame(stats_list)
    stats_df.to_csv(output_path, index=False)
    print(f"Regime statistics saved to {output_path}")
    return stats_df