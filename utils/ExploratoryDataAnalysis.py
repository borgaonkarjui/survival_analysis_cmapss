import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

#identify sensor columns
def fetch_sensor_cols(df):
    sensor_cols = [col for col in df.columns if col not in ['unit_id', 'cycle', 'altitude', 'mach_number', 'tra', 'op_regime', 'target_rul', 'true_rul']]
    return sensor_cols

#analyze engine cycle stats
def analyze_engines(df, setType="Train"):
    """
    Analyzes the dataset to show total engines and the data available for each.
    """
    # Group by unit_id and get the max cycle for each
    engine_stats = df.groupby('unit_id')['cycle'].max().reset_index()
    engine_stats.columns = ['Engine_ID', 'Total_Cycles']
    
    # Calculate global metrics
    total_engines = engine_stats.shape[0]
    total_datapoints = df.shape[0]
    avg_cycles = engine_stats['Total_Cycles'].mean()
    min_cycles = engine_stats['Total_Cycles'].min()
    max_cycles = engine_stats['Total_Cycles'].max()
    
    print(f"--- FD004 {setType} Set Analysis ---")
    print(f"Total Number of Engines: {total_engines}")
    print(f"Total Datapoints (Rows): {total_datapoints}")
    print(f"Average Cycles per Engine: {avg_cycles:.2f}")
    print(f"Shortest Engine History: {min_cycles} cycles")
    print(f"Longest Engine History: {max_cycles} cycles")
    print("-" * 32)
    
    return engine_stats

#plot engine lifecycle length
def plot_max_cycle_dist(df):
    # Get the last cycle for each engine
    max_cycles = df.groupby('unit_id')['cycle'].max()
    
    plt.figure(figsize=(10, 6))
    sns.histplot(max_cycles, bins=30, kde=True, color='teal')
    
    plt.title('Distribution of Engine Lifespans (Max Cycles) - FD004 Training Set')
    plt.xlabel('Total Cycles until Failure')
    plt.ylabel('Number of Engines')
    plt.grid(axis='y', alpha=0.3)
    
    # Add vertical lines for mean and median
    plt.axvline(max_cycles.mean(), color='red', linestyle='--', label=f'Mean: {max_cycles.mean():.1f}')
    plt.axvline(max_cycles.median(), color='yellow', linestyle='-', label=f'Median: {max_cycles.median():.1f}')
    
    plt.legend()
    plt.show()

#Operating condition/regime analysis
def analyze_regime_clusters(df):
    """
    1. Visualizes raw clusters in 3D.
    2. Identifies 6 regimes via rounding.
    3. Prints datapoint counts and per-engine diversity.
    """
    # Regime identification
    # Round to collapse the simulation noise into 6 distinct IDs
    df['op_regime'] = df.groupby([
        df['altitude'].round(-3), 
        df['mach_number'].round(1), 
        df['tra'].round()
    ]).ngroup()

    # Stastics
    print("--- Datapoints per Operating Regime ---")
    counts = df['op_regime'].value_counts().sort_index()
    print(counts)
    
    # Check how many unique regimes each engine visits
    regimes_per_engine = df.groupby('unit_id')['op_regime'].nunique()
    print(f"\nAverage regimes visited per engine: {regimes_per_engine.mean():.2f}")
    print(f"Min regimes visited by an engine: {regimes_per_engine.min()}")
    print(f"Max regimes visited by an engine: {regimes_per_engine.max()}")

    # Plot clusters
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the RAW values (altitude, mach_number, tra) and color them by rounded regime ID
    scatter = ax.scatter(
        df['altitude'], 
        df['mach_number'], 
        df['tra'], 
        c=df['op_regime'], 
        cmap='viridis', 
        s=5,
        # alpha=0.3
    )
    
    ax.set_xlabel('Altitude (ft)')
    ax.set_ylabel('Mach Number')
    ax.set_zlabel('TRA (%)')
    plt.title('FD004 Settings colored by Regime ID')
    
    # Add colorbar legend
    plt.colorbar(scatter, label='Identified Regime ID', pad=0.1)
    
    plt.tight_layout()
    plt.show()

    return df

#plot regimes for particular engine 
def plot_engine_step_regimes(df, unit_id=1):
    engine_data = df[df['unit_id'] == unit_id]
    
    plt.figure(figsize=(12, 4))
    plt.step(engine_data['cycle'], engine_data['op_regime'], where='post', color='darkblue', linewidth=1.5)
    
    plt.title(f'Flight Regime Transitions: Engine #{unit_id}')
    plt.xlabel('Cycle')
    plt.ylabel('Operating Regime ID')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

#identifying global flat sensors
def identify_global_flat_sensors(df):
    """
    Calculates and prints the standard deviation for all sensors,
    identifying which are globally flat.
    """
    # Identify sensor columns
    sensor_cols = fetch_sensor_cols(df)
    
    # Calculate standard deviation
    std_series = df[sensor_cols].std().sort_values()
    
    print("--- Sensor Variance Profile (Sorted) ---")
    for name, std in std_series.items():
        status = "[FLAT]" if std == 0 else ""
        print(f"{name:10} | StdDev: {std:12.6f} {status}")
        
    flat_sensors = std_series[std_series == 0].index.tolist()
    
    print("-" * 40)
    print(f"Total Flat Sensors Found: {len(flat_sensors)}")
    
    return flat_sensors

#plot single sensor for an engine
def visualize_degradation_start(df, unit_id=1, sensor='T50'):
    """
    Plots a normalized sensor over time to help identify the 'knee' 
    where degradation starts.
    """
    # Filter for one engine
    engine_data = df[df['unit_id'] == unit_id]
    
    plt.figure(figsize=(12, 5))
    plt.plot(engine_data['cycle'], engine_data[sensor], alpha=0.5, label=sensor)
    
    # Add a rolling mean to see the trend through the noise
    plt.plot(engine_data['cycle'], engine_data[sensor].rolling(window=10).mean(), 
             color='red', linewidth=2, label='10-Cycle Rolling Mean')
    
    plt.title(f'Degradation Trend for {sensor} (Engine #{unit_id})')
    plt.xlabel('Cycle')
    plt.ylabel('Normalized Value')
    plt.axhline(0, color='black', linestyle='--', alpha=0.3)
    plt.legend()
    plt.show()

#plot multiple sensors for an multiple engines
def plot_multi_sensor_fleet(df, sensors=['T50', 'Ps30', 'BPR', 'htBleed'], num_engines=10):
    """
    Plots fleet-wide trends for multiple sensors to find a common degradation point.
    """
    fig, axes = plt.subplots(len(sensors), 1, figsize=(15, 4 * len(sensors)))
    unique_units = df['unit_id'].unique()[:num_engines]

    for i, sensor in enumerate(sensors):
        for unit in unique_units:
            engine_data = df[df['unit_id'] == unit].copy()
            
            # Align by end-of-life
            max_c = engine_data['cycle'].max()
            engine_data['cycles_to_fail'] = engine_data['cycle'] - max_c
            
            # Focus on the last 250 cycles
            tail_data = engine_data[engine_data['cycles_to_fail'] > -250]
            
            axes[i].plot(tail_data['cycles_to_fail'], 
                         tail_data[sensor].rolling(window=10).mean(), 
                         alpha=0.5)
        
        axes[i].set_title(f'Fleet Trends: {sensor}')
        axes[i].axvline(-125, color='red', linestyle='--', label='Standard Cap (125)')
        axes[i].set_ylabel('Normalized Value')
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_correlation_with_target(df, redundancy_thresh=0.9, corr_thresh=0.1):
    """
    1. Plots a sensor correlation heatmap with Target RUL.
    2. Identifies sensors directly tracking degradation.
    3. Flags redundant sensor pairs (High Inter-Correlation).
    """
    #define sensor and target cols : ignoring true_rul, engine id and operating consition cols
    sensor_cols = fetch_sensor_cols(df)
    cols_to_corr = sensor_cols + ['target_rul']
    
    plt.figure(figsize=(12, 10))
    correlation_matrix = df[cols_to_corr].corr()
    
    sns.heatmap(correlation_matrix, annot=False, cmap='RdBu_r', center=0)
    plt.title('Feature Correlation Heatmap (Sensors & Target RUL)')
    plt.show()
    
    #sensor correlations with 'target_rul'
    target_corr = correlation_matrix['target_rul']
    print("--- Sensor Correlation with Target RUL ---")
    # print(target_corr)
    # print("Sensors with low correlation")
    for sensor, corr_val in target_corr.items():
        if abs(corr_val) < corr_thresh:
            flag = "--> low correlation"
        else:
            flag = " "
        print(f"{sensor} : {corr_val} {flag}")
        flag = " "
        

    #redundant sensor pairs
    print(f"\n--- Redundant Sensor Pairs (Absolute Correlation > {redundancy_thresh}) ---")
    #absolute values to capture perfect inverse corr
    corr_abs = correlation_matrix.loc[sensor_cols, sensor_cols].abs()
    #scan upped triangle of matrix to avoid redundancy
    upper_tri = corr_abs.where(np.triu(np.ones(corr_abs.shape), k=1).astype(bool)) #k=1 to remove self-corr on the main diagonal
    
    redundant_found = False
    for col in upper_tri.columns:
        for row in upper_tri.index:
            correlation_value = upper_tri.loc[row, col]
            if correlation_value > redundancy_thresh:
                actual_corr = correlation_matrix.loc[row, col]
                print(f"{row:8} <-> {col:8} | Corr Coeff: {actual_corr:.4f}")
                redundant_found = True
                
    if not redundant_found:
        print("No redundant sensor pairs found at this threshold.")