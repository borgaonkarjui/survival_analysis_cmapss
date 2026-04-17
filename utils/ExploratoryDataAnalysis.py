import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

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

#plot operating regime clusters
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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