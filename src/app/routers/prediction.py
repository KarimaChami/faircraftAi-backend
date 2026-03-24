from fastapi import APIRouter, Depends, HTTPException
import logging

from src.app.schemas.prediction import (
    PredictionRequest, PredictionResponse, ExplainResponse,
    SimulationRequest, SimulationResponse, RecommendationResponse
)
from sqlalchemy.orm import Session
from src.app.dependencies.model import get_model
from src.app.db.config import get_db
from src.app.services.prediction_service import PredictionService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1",
    tags=["Prediction"]
)

@router.get("/health")
def health_check():
    return {"status": "ok"}

@router.post("/predict", response_model=PredictionResponse)
def predict_price(req: PredictionRequest, model=Depends(get_model), db: Session = Depends(get_db)):
    try:
        service = PredictionService(model, db)
        return service.predict_price(req)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction.")

@router.post("/explain", response_model=ExplainResponse)
def explain_prediction(req: PredictionRequest, model=Depends(get_model), db: Session = Depends(get_db)):
    try:
        service = PredictionService(model, db)
        return service.explain_prediction(req)
    except Exception as e:
        logger.error(f"Explain failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to explain model prediction.")

@router.post("/simulate", response_model=SimulationResponse)
def simulate_price(req: SimulationRequest, model=Depends(get_model), db: Session = Depends(get_db)):
    try:
        service = PredictionService(model, db)
        
        orig_price_res = service.predict_price(req.original_request, save_to_db=False)
        new_price_res = service.predict_price(req.modified_request, save_to_db=False)
        
        orig_price = orig_price_res.predicted_price
        new_price = new_price_res.predicted_price
        
        diff = round(new_price - orig_price, 2)
        
        return SimulationResponse(
            original_price=orig_price,
            new_price=new_price,
            price_difference=diff
        )
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise HTTPException(status_code=500, detail="Simulation failed.")

@router.post("/recommendations", response_model=RecommendationResponse)
def get_recommendations(req: PredictionRequest, model=Depends(get_model), db: Session = Depends(get_db)):
    try:
        service = PredictionService(model, db)
        
        # We need cost, price and margin to generate recommendations
        prediction = service.predict_price(req, save_to_db=False)
        
        return service.generate_recommendations(
            production_cost=prediction.production_cost,
            predicted_price=prediction.predicted_price,
            margin=prediction.margin
        )
    except Exception as e:
        logger.error(f"Recommendations failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch recommendations.")
