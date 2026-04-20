import matplotlib.pyplot as plt
import pandas as pd

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
    
    # Plot validation RMSE
    plt.plot(x_axis, results['validation_0']['rmse'], label='Validation')
    
    # Plot training RMSE (if available in the eval_set)
    if 'validation_1' in results:
        plt.plot(x_axis, results['validation_1']['rmse'], label='Train')
    elif 'train' in results:
        plt.plot(x_axis, results['train']['rmse'], label='Train')
        
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