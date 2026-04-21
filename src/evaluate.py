from utils.modelEvaluation import eval_rmse, plot_training_curves, plot_feature_importance
from data_loader import xgboost_test_loader
import xgboost as xgb
from pathlib import Path
import pandas as pd
import json

def evaluate_xgboost(model_path, model_type, test_path, rul_path, history_path, results_dir="../results"):
    valid_model_types = ['xgboost']
    
    #get model label
    model_path_obj = Path(model_path)
    model_lbl = model_path_obj.stem
    #create folder if not exksts : to save results
    version_dir = Path(f"{results_dir}/{model_lbl}")
    version_dir.mkdir(parents=True, exist_ok=True)

    match model_type:
        #xgboost
        case 'xgboost':
            #loading history
            with open(history_path, 'r') as f:
                history = json.load(f)
            
            #load the model
            model = xgb.XGBRegressor()
            model.load_model(model_path)

            #training plots
            print("---Model Training Plots---")
            training_curves_path = f"{version_dir}/train_val_rmse_plot.png"
            plot_training_curves(model_lbl, history=history, save_path=training_curves_path)
            feature_imp_path = f"{version_dir}/feature_importance_plot.png"
            plot_feature_importance(model, model_lbl, importance_type='gain', save_path=feature_imp_path)

            print(f"---Evaluating model: {model_lbl}---")

            #snapshot test loader
            X_test_snapshot, y_true_test_snapshot = xgboost_test_loader(test_path=test_path, rul_path=rul_path, is_snapshot=True)
            print("---Snapshot evaluation---")
            #snapshot evaluation
            y_pred_test_snap, test_rmse_snap = eval_rmse(model, X_test_snapshot, y_true_test_snapshot)
            
            #global test loader
            X_test_global, y_true_test_global = xgboost_test_loader(test_path=test_path, rul_path=rul_path, is_snapshot=False)
            print("---Global evaluation---")
            #global evaluation
            y_pred_test_global, global_rmse_global = eval_rmse(model, X_test_global, y_true_test_global)

            #write RMSE valuse to results.csv
            results_row = pd.DataFrame({
                'Version' : [model_lbl],
                'Snapshot RMSE' : [test_rmse_snap],
                'Global RMSE' : [global_rmse_global]
            })
            results_path = Path(f"{results_dir}/results.csv")
            # if results_path.is_file():
            #     df = pd.read_csv(results_path)
            #     df = pd.concat([df, results_row], ignore_index=True)
            # else:
            #     df = results_path
            
            results_row.to_csv(results_path, mode='a', header=False, index=False)
            print(f"Results saved to {results_path}!")

        case _:
            return KeyError(f"Unsupported model type! Use one of the following: {valid_model_types}")
    
    return y_true_test_snapshot, y_pred_test_snap, y_true_test_global, y_pred_test_global