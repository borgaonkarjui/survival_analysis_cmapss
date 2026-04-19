#imports
from utils.DataLoading import load_cmapss_fd004
from utils.DataPreprocessing import identify_operating_regimes, normalize_by_regime, add_remaining_useful_life
from sklearn.model_selection import train_test_split

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

def preprocess_data(file_path, stats_path, 
                    train_save_path=None, val_save_path=None, 
                    selected_sensors=['T24', 'T30', 'T50', 'P30', 'Nf', 'Nc', 'epr', 'Ps30', 'NRf', 'NRc', 'htBleed'],
                    test_size=0.2, random_state=42, 
                    is_train=True, rul_cap=125):
    """
    Complete data processing pipeline to process raw FD004 text file into a csv ready for training.
    """
    #load raw data
    df = load_cmapss_fd004(file_path)

    #identify and add operating regime column
    df = identify_operating_regimes(df)

    #normalization using regime wise saved stats
    df = normalize_by_regime(df, stats_path)

    #RUL calculation for train set
    if is_train:
        df = add_remaining_useful_life(df, cap=rul_cap)

    #feature selection
    # We keep unit_id and cycle for tracking/plotting, and op_regime for context
    final_cols = ['unit_id', 'cycle', 'op_regime'] + selected_sensors
    if is_train:
        final_cols.append('target_rul')
    
    #train and val split
    train_set, val_set = split_data_by_engine(df[final_cols], test_size=test_size, random_state=random_state)
    
    train_set = train_set.reset_index(drop=True)
    val_set = val_set.reset_index(drop=True)

    #save files
    if train_save_path and val_save_path:
        train_set.to_csv(train_save_path)
        val_set.to_csv(val_save_path)

    return train_set, val_set