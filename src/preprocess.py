#imports
from utils.DataLoading import load_cmapss_fd004
from utils.DataPreprocessing import identify_operating_regimes, normalize_by_regime, add_remaining_useful_life, save_regime_stats, apply_regime_map
from sklearn.model_selection import train_test_split
import json

def split_data_by_engine(df, test_size=0.2, random_state=42):
    """
    Splits the dataframe into training and validation sets 
    ensuring no engine's data is split across both sets.
    """
    #get unique engine IDs
    unique_engines = df['unit_id'].unique()
    
    #split the IDs into train and validation
    train_ids, val_ids = train_test_split(unique_engines, test_size=test_size, random_state=random_state)
    
    #create dataframes
    train_set = df[df['unit_id'].isin(train_ids)].copy()
    val_set = df[df['unit_id'].isin(val_ids)].copy()
    
    print(f"--- Data Split Summary ---")
    print(f"Total Engines: {len(unique_engines)}")
    print(f"Train Engines: {len(train_ids)} ({len(train_set)} rows)")
    print(f"Val Engines:   {len(val_ids)} ({len(val_set)} rows)")
    
    return train_set, val_set
    
#preprocess train set
def preprocess_train_set(file_path, 
                         stats_path="../data/meta/regime_stats.csv", 
                         map_save_path="../data/meta/regime_map.csv",
                    train_save_path=None, val_save_path=None,
                    sensor_path="../data/meta/selected_sensors.json",
                    test_size=0.2, random_state=42, rul_cap=125):
    """
    Complete data processing pipeline to process raw FD004 text file into a csv ready for training.
    Save metadata from train set.
    """
    #load raw data
    df = load_cmapss_fd004(file_path)

    #load metadata
    with open(sensor_path, 'r') as f:
        selected_sensors = json.load(f)

    #identify and add operating regime column
    df, _ = identify_operating_regimes(df, map_save_path=map_save_path)

    #normalization using regime wise saved stats
    save_regime_stats(df, output_path=stats_path)
    df = normalize_by_regime(df, stats_path)

    #feature selection
    #keeping unit_id and cycle for tracking/plotting, and op_regime for context
    final_cols = ['unit_id', 'cycle', 'op_regime'] + selected_sensors

    #RUL calculation for train set
    df = add_remaining_useful_life(df, cap=rul_cap)

    final_cols.append('target_rul')

    #train and val split
    train_set, val_set = split_data_by_engine(df[final_cols], test_size=test_size, random_state=random_state)
    
    train_set = train_set.reset_index(drop=True)
    val_set = val_set.reset_index(drop=True)

    #save files
    if train_save_path and val_save_path:
        train_set.to_csv(train_save_path, index=False)
        print(f"Train set successfully saved at {train_save_path}!")
        val_set.to_csv(val_save_path, index=False)
        print(f"Validation set successfully saved at {val_save_path}!")

    return train_set, val_set

def preprocess_test_set(file_path=None, df=None, test_save_path=None,
                         stats_path="../data/meta/regime_stats.csv", 
                         map_save_path="../data/meta/regime_map.csv",
                    sensor_path="../data/meta/selected_sensors.json"):
    """
    Complete data processing pipeline to process raw FD004 text file into a csv ready for training.
    Save metadata from train set.
    """
    if not (file_path or df):
        print("Please provide either filepath or dataframe.")
    else:
        if file_path:
            #load raw data
            df = load_cmapss_fd004(file_path)

        #load metadata
        with open(sensor_path, 'r') as f:
            selected_sensors = json.load(f)

        #identify and add operating regime column
        df = apply_regime_map(df, regime_map_path=map_save_path)

        #normalization using regime wise saved stats
        df = normalize_by_regime(df, stats_path)

        #feature selection
        #keeping unit_id and cycle for tracking/plotting, and op_regime for context
        final_cols = ['unit_id', 'cycle', 'op_regime'] + selected_sensors

        if test_save_path:
            df[final_cols].to_csv(test_save_path, index=False)
            print(f"Test set successfully saved at {test_save_path}!")
        return df[final_cols]