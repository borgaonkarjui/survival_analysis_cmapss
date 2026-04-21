import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from utils.ExploratoryDataAnalysis import fetch_sensor_cols

#plot training curves
def plot_training_curves(model_lbl, model=None, history=None, save_path=None):
    """
    Plots the RMSE for both training and validation sets across the boosting iterations.
    """
    if not (model or history):
        print("Please provide either model or history object.")
    else:
        if model:
            # Retrieve performance results
            results = model.evals_result()
            train_rmse = results['validation_0']['rmse']
            val_rmse = results['validation_1']['rmse']
        
        if history:
            train_rmse = history['validation_0']['rmse']
            val_rmse = history['validation_1']['rmse']
        
        epochs = len(train_rmse)
        x_axis = range(0, epochs)
        plt.figure(figsize=(10, 6))
        
        # Plot train RMSE
        plt.plot(x_axis, train_rmse, label='Train')
        
        # Plot validation RMSE
        plt.plot(x_axis, val_rmse, label='Validation')
            
        plt.title(f'Regression Error (RMSE) over Time for {model_lbl}')
        plt.xlabel('Number of Trees (Iterations)')
        plt.ylabel('RMSE (Cycles)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        if not save_path == None:
            plt.savefig(save_path)
            print("Training vs validation RMSE plot saved successfully!")
        plt.show()

#plot feature importance
def plot_feature_importance(model, model_lbl, importance_type='gain', save_path=None):
    """
    Plots the importance of sensors based on 'gain' or 'weight'.
    Default strategy 'gain' which is generally better for understanding physical impact.
    """
    
    # Extract importance scores
    importance = model.get_booster().get_score(importance_type=importance_type)
    
    # Convert to DataFrame for plotting
    importance_df = pd.DataFrame({
        'Feature': list(importance.keys()),
        'Importance': list(importance.values())
    }).sort_values(by='Importance', ascending=True)
    
    # Plotting
    plt.figure(figsize=(10, 8))
    plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
    plt.title(f'Sensor Importance for {model_lbl} (Metric: {importance_type.capitalize()})')
    plt.xlabel(f'Relative {importance_type.capitalize()} Score')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    if not save_path == None:
        plt.savefig(save_path)
        print("Feature importance plot saved successfully!")
    plt.show()

#evaluate rmse
def eval_rmse(model, X_test, y_true_test):
    # Predict RUL for the test engines snapshot
    y_pred_test = model.predict(X_test)

    # Clip predictions (RUL cannot be negative)
    y_pred_test = y_pred_test.clip(min=0)

    # Calculate RMSE
    test_rmse = np.sqrt(mean_squared_error(y_true_test, y_pred_test))

    print(f"Final Test RMSE on FD004: {test_rmse:.2f} cycles")

    return y_pred_test, test_rmse

#plot results y_true vs y_pred
def plot_test_results(y_true, y_pred):
    plt.figure(figsize=(10, 6))
    plt.plot(y_true.values, label='Actual RUL', color='blue', marker='o', markersize=3, linestyle='')
    plt.plot(y_pred, label='Predicted RUL', color='red', marker='x', markersize=3, linestyle='')
    plt.title('Actual vs Predicted RUL (Test Set FD004)')
    plt.xlabel('Engine ID')
    plt.ylabel('Remaining Useful Life (Cycles)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

#predict and plot lifecycle for a single engine
def plot_engine_lifecycle(unit_id, test_df, model, selected_sensors):
    """
    Plots the predicted RUL for a single engine over its entire recorded life.
    """
    # 1. Extract all data for this specific engine
    engine_data = test_df[test_df['unit_id'] == unit_id].copy()
    
    # 2. Prepare features
    features = ['op_regime'] + selected_sensors
    X_engine = engine_data[features]
    
    # 3. Predict
    predictions = model.predict(X_engine).clip(min=0)
    
    # 4. Plot
    plt.figure(figsize=(12, 6))
    plt.plot(engine_data['cycle'].values, predictions, label='Predicted RUL', color='red', linewidth=2)
    
    # Theoretical "Perfect" Line (Only for visualization, slope of -1)
    # Note: We don't know the TRUE starting RUL, so we just show the trend
    plt.title(f'RUL Prediction Trend for Engine #{unit_id}')
    plt.xlabel('Current Cycle')
    plt.ylabel('Predicted Remaining Useful Life')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

# Usage:
# plot_engine_lifecycle(5, test_norm, model, selected_sensors)

#plot fleet lifecycle
def plot_fleet_lifecycles(model, X_test, y_true_test, selected_sensors, n_engines=5, random_state=42):
    """
    Plots RUL trajectories for a random sample of engines to see fleet-level stability.
    """
    # unique_units = test_df['unit_id'].unique()
    # sample_units = np.random.choice(unique_units, n_engines, replace=False)
    sample_units = X_test['unit_id'].unique()
    
    plt.figure(figsize=(12, 7))
    # features = ['op_regime'] + selected_sensors
    # features = X_test.cols().tolist()

    for unit_id in sample_units:
        engine_data = X_test[X_test['unit_id'] == unit_id]
        # X_engine = engine_data[features]
        preds = model.predict(engine_data).clip(min=0)
        
        plt.plot(engine_data['cycle'].values, preds, label=f'Engine #{unit_id}', alpha=0.8)

    plt.title(f'RUL Prediction Trajectories for {len(sample_units)} Random Engines')
    plt.xlabel('Cycle count')
    plt.ylabel('Predicted RUL')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()

# Usage:
# plot_fleet_lifecycles(test_norm, model, selected_sensors, n_engines=8)