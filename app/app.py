from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
import pandas as pd
from src.inference import RULPredictor
from utils.DataLoading import load_cmapss_fd004

# Initialize FastAPI
app = FastAPI(title="Engine Health Monitor")
templates = Jinja2Templates(directory="app/templates")

# Initialize the Predictor
predictor = RULPredictor(
    model_type='xgboost',
    model_path='models/xgboost/xgboost_v002.json',
    stats_path='data/meta/regime_stats.csv',
    selected_sensors_path='data/meta/selected_sensors.json',
    regime_map_path='data/meta/regime_map.csv'
)

# Load raw test set for lookup
test_df_raw = load_cmapss_fd004("data/raw/CMAPSSData/test_FD004.txt")
# Get the last row for each unit_id
X_test_raw = test_df_raw.groupby('unit_id').last().reset_index()

unique_engines = sorted(X_test_raw['unit_id'].unique().tolist())

@app.get("/")
def index(request: Request):
    # Populate dropdown with unique engine IDs
    return templates.TemplateResponse(
        request= request, 
        name="index.html",
        context={"engines": unique_engines}
    )

@app.post("/predict")
def predict(request: Request, engine_id: int = Form(...)):
    # Fetch last row for the selected engine (last recorded cycle)
    engine_data = X_test_raw[X_test_raw['unit_id'] == engine_id].tail(1)
    
    # predict
    prediction = predictor.predict(engine_data)
    
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "engines": unique_engines,
            "selected_engine": engine_id,
            "prediction": round(prediction[0], 2)
        }
    )