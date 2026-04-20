import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from utils.ExploratoryDataAnalysis import fetch_sensor_cols

#plot training curves
def plot_training_curves(model):
    """
    Plots the RMSE for both training and validation sets across the boosting iterations.
    """
    # Retrieve performance results
    results = model.evals_result()
    
    epochs = len(results['validation_0']['rmse'])
    x_axis = range(0, epochs)
    
    plt.figure(figsize=(10, 6))
    
    # Plot train RMSE
    plt.plot(x_axis, results['validation_0']['rmse'], label='Train')
    
    # Plot validation RMSE
    plt.plot(x_axis, results['validation_1']['rmse'], label='Validation')
        
    plt.title('XGBoost Regression Error (RMSE) over Time')
    plt.xlabel('Number of Trees (Iterations)')
    plt.ylabel('RMSE (Cycles)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

#plot feature importance
def plot_feature_importance(model, importance_type='gain'):
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
    plt.title(f'Sensor Importance (Metric: {importance_type.capitalize()})')
    plt.xlabel(f'Relative {importance_type.capitalize()} Score')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
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

import numpy as np
from sklearn.metrics import mean_squared_error

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

