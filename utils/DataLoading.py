import pandas as pd
import numpy as np

def load_cmapss_fd004(file_path):
    # Mapping based on NASA documentation
    index_names = ['unit_id', 'cycle']
    setting_names = ['altitude', 'mach_number', 'tra']
    sensor_names = [
        'T2', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30', 'Nf', 'Nc', 'epr',
        'Ps30', 'phi', 'NRf', 'NRc', 'BPR', 'farB', 'htBleed', 'Nf_dmd',
        'PCNfR_dmd', 'W31', 'W32'
    ]
    
    col_names = index_names + setting_names + sensor_names
    
    # Read space-delimited file
    df = pd.read_csv(file_path, sep=r"\s+", header=None, names=col_names)
    
    return df

def load_rul_truth(file_path):
    return pd.read_csv(file_path, sep=r"\s+", header=None, names=['true_rul'])