import pandas as pd
import numpy as np
import os
from utils.ExploratoryDataAnalysis import fetch_sensor_cols

#round off the 3 regime settings and group in 6 operating conditions conditions
def identify_operating_regimes(df, map_save_path=None):
    """
    Learns regimes from training data and saves the map.
    """
    df_regime = df.copy()
    
    # Round settings to collapse minor noise into clusters
    df_regime['alt_bin'] = df_regime['altitude'].round(-3)
    df_regime['mach_bin'] = df_regime['mach_number'].round(1)
    df_regime['tra_bin'] = df_regime['tra'].round()
    
    # Discover unique combinations and assign IDs
    # drop_duplicates to get the unique 'Rules'
    regime_map = df_regime[['alt_bin', 'mach_bin', 'tra_bin']].drop_duplicates().reset_index(drop=True)
    regime_map['op_regime'] = regime_map.index

    # Save regime map
    if map_save_path:
        os.makedirs(os.path.dirname(map_save_path), exist_ok=True)
        regime_map.to_csv(map_save_path, index=False)
        print(f"Regime map saved to {map_save_path}")

    # Add new column for the combination
    # df_regime['op_regime'] = df_regime.groupby(['alt_bin', 'mach_bin', 'tra_bin']).ngroup()
    df_regime = df_regime.merge(regime_map, on=['alt_bin', 'mach_bin', 'tra_bin'], how='left')
    
    # Drop bin columns
    df_regime = df_regime.drop(columns=['alt_bin', 'mach_bin', 'tra_bin'])
    
    print(f"Identified {df_regime['op_regime'].nunique()} unique operating regimes.")
    return df_regime, regime_map

#regime wise mean and std dev calculation
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
    
    #iterate through each regime and sensor present in the stats file
    for regime in stats_df['op_regime'].unique():
        regime_mask = df_norm['op_regime'] == regime
        
        #if regime exists in the current dataframe
        if regime_mask.any():
            regime_stats = stats_df[stats_df['op_regime'] == regime]
            
            for _, row in regime_stats.iterrows():
                sensor = row['sensor']
                m = row['mean']
                s = row['std']
                
                #normalization: (x - mean) / std
                if sensor in df_norm.columns:
                    df_norm.loc[regime_mask, sensor] = (df_norm.loc[regime_mask, sensor] - m) / s
                    
    print(f"Normalization complete using stats from {stats_path}")
    return df_norm

#RUL claculation for test set
def add_remaining_useful_life(df, cap=125):
    """
    Calculates Piecewise Linear RUL for the training set.
    """
    #find the maximum cycle for each engine
    max_cycle = df.groupby('unit_id')['cycle'].transform('max')
    
    #true rul 0 -> n cycles
    #to use for plots
    df['true_rul'] = max_cycle - df['cycle']
    
    #Piecewise Cap of 125 : target rul - > 0-125 : healthy state clipped to 125 degradation trend preserved for unhealthy state n cycles
    #to use for training
    df['target_rul'] = df['true_rul'].clip(upper=cap)
    
    return df

#apply regime map for test and inference
def apply_regime_map(df, regime_map_path, default_regime=0):
    """
    Uses a saved map to assign regimes to new data (Test or Inference).
    """
    df_regime = df.copy()
    regime_map = pd.read_csv(regime_map_path)
    
    # Create identical bins
    df_regime['alt_bin'] = df_regime['altitude'].round(-3)
    df_regime['mach_bin'] = df_regime['mach_number'].round(1)
    df_regime['tra_bin'] = df_regime['tra'].round()
    
    # Lookup existing regimes
    df_regime = df_regime.merge(regime_map, on=['alt_bin', 'mach_bin', 'tra_bin'], how='left')
    
    # Handle unseen combinations
    # If a combination didn't exist in training, merge results in a NaN
    if df_regime['op_regime'].isnull().any():
        unknown_count = df_regime['op_regime'].isnull().sum()
        print(f"Warning: {unknown_count} rows had unknown regimes. Assigning to default: {default_regime}")
        df_regime['op_regime'] = df_regime['op_regime'].fillna(default_regime).astype(int)
    
    df_regime = df_regime.drop(columns=['alt_bin', 'mach_bin', 'tra_bin'])
    return df_regime