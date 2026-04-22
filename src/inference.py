import xgboost as xgb
import pandas as pd
import json
from preprocess import preprocess_test_set

class RULPredictor:
    def __init__(self, model_type, model_path, stats_path, selected_sensors_path, regime_map_path):
        valid_model_types = ['xgboost']
        match model_type:
            #xgboost
            case 'xgboost':
                #load model
                self.model = xgb.XGBRegressor()
                self.model.load_model(model_path)

            case _:
                return KeyError(f"Unsupported model type! Use one of the following: {valid_model_types}")
        
        #load metadata
        with open(selected_sensors_path, 'r') as f:
            self.selected_sensors = json.load(f)
        
        self.selected_sensors_path =selected_sensors_path
        self.stats_path = stats_path
        self.regime_map_path = regime_map_path

    def predict(self, raw_data_df):
        """
        Takes raw sensor data (DataFrame), 
        preprocesses it, and returns a prediction.
        """
        # Preprocess (is_train=False ensures no RUL calculation)
        processed_df = preprocess_test_set(df=raw_data_df,
                         stats_path=self.stats_path, 
                         map_save_path=self.regime_map_path,
                    sensor_path=self.selected_sensors_path)
        
        # Extract features (match the order used in training)
        features = ['op_regime'] + self.selected_sensors
        X = processed_df[features]
        
        # Predict
        prediction = self.model.predict(X)
        return prediction.tolist()