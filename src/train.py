from model_defination import get_model
from data_loader import xgboost_train_loader
from sklearn.model_selection import RandomizedSearchCV

def run_training_pipeline(train_path, val_path, target_col='target_rul', model_type="xgboost", params=None, tune=False, param_grid=None, random_state=42, n_iter=None, cv=None):
    """
    Runs model training pipeline.
    """
    valid_model_types = ['xgboost']
    match model_type:
        #xgboost
        case 'xgboost':
            #data loading
            X_train, y_train, X_val, y_val = xgboost_train_loader(
                train_path=train_path, val_path=val_path, target_col=target_col
                )
            #training
            if not tune:
                print("Training XGBoost baseline model...")
                if not params:
                    model, model_params = get_model(model_type="xgboost", params=None)
                else:
                    model, model_params = get_model(model_type="xgboost", params=params)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_train, y_train), (X_val, y_val)],
                    verbose=True
                )
                print("XGBoost baseline model training complete!")
            else:
                #parameter grid
                print("Hyperparameter tuning for XGBoost model...")
                if not param_grid:
                    param_grid = {
                        'n_estimators': [100, 500, 1000],
                        'learning_rate': [0.01, 0.03, 0.05, 0.1],
                        'max_depth': [3, 4, 5, 6],
                        'subsample': [0.7, 0.8, 0.9],
                        'colsample_bytree': [0.7, 0.8, 0.9],
                        'gamma': [0, 0.1, 0.2],
                        'reg_alpha': [0, 0.1, 1],  # L1 regularization
                        'reg_lambda': [1, 5, 10]   # L2 regularization
                    }
                    print("Parameter grid set to default.")

                #model initilization
                if not params:
                    model, model_params = get_model(model_type="xgboost", 
                                                    params={'tree_method':'hist', 'random_state':random_state,
                                        #early stopping
                                        'eval_metric':'rmse',
                                        'early_stopping_rounds':20})
                    print("XGBoost model parameters set to default.")
                else:
                    model, model_params = get_model(model_type="xgboost", params=params)
                
                #random search setup
                random_search = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=param_grid,
                    n_iter=n_iter,
                    scoring='neg_root_mean_squared_error',
                    cv=cv, # 3-fold cross validation
                    verbose=2,
                    random_state=random_state,
                    n_jobs=-1 # use all available CPU cores
                )

                #run search
                random_search.fit(
                    X_train, y_train,
                    eval_set=[(X_train, y_train), (X_val, y_val)],
                    verbose=True
                    )
                
                print("\n--- Best Parameters Found ---")
                print(random_search.best_params_)
                
                # RMSE is returned as negative by sklearn, so reverse the sign
                print(f"Best CV RMSE: {-random_search.best_score_:.2f}")

                model = random_search.best_estimator_
                model_params = random_search.best_params_
                print("Hyperparameter tuning for XGBoost complete!")

        case _:
            return KeyError(f"Unsupported model type! Use one of the following: {valid_model_types}")
        
    return model, model_params