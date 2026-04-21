import xgboost as xgb

def get_model(model_type="xgboost", params=None, random_state=42):
    """
    Returns a model object based on the type and parameters provided.
    """
    valid_model_types = ['xgboost']
    
    match model_type:
        #xgboost
        case 'xgboost':
            if params is None:
                model = xgb.XGBRegressor(
                    n_estimators=100, 
                    learning_rate=0.1, 
                    max_depth=5, 
                    random_state=random_state,
                    #early stopping
                    eval_metric='rmse',
                    early_stopping_rounds=50,
                )
            else:
                model = xgb.XGBRegressor(**params)
        case _:
            return KeyError(f"Unsupported model type! Use one of the following: {valid_model_types}")
    
    return model, model.get_params()