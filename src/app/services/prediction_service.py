import pandas as pd
import numpy as np
import shap
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session

from src.app.schemas.prediction import PredictionRequest, PredictionResponse, ExplainResponse, TopFactor, RecommendationResponse
from src.app.models.prediction import Prediction

class PredictionService:
    def __init__(self, model, db: Optional[Session] = None):
        self.pipeline = model
        self.db = db
    
    def calculate_cost(self, req: PredictionRequest) -> float:
        margin_cost = req.material_cost + (req.labor_hours * req.hourly_rate) + req.overhead_cost
        return round(margin_cost, 2)
    
    def prepare_features(self, req: PredictionRequest) -> pd.DataFrame:
        data = {
            'product_title': [req.product_title],
            'category': [req.category],
            'shop_name': [req.shop_name],
            'title_length': [req.title_length],
            'keyword_count': [req.keyword_count],
            'rating_numeric': [req.rating_numeric],
            'reviews_numeric': [req.reviews_numeric],
            'rating_score': [req.rating_score],
            'popularity_index': [req.popularity_index],
        }
        df = pd.DataFrame(data)
        
        # Feature Engineering match with training
        df['text_composite'] = (df['product_title'].fillna('') + ' ' + df['category'].fillna('')).str.lower()
        df['popularity_weight'] = df['rating_numeric'] * df['reviews_numeric']
        df['keyword_density'] = df['title_length'] / (df['keyword_count'] + 1)
        
        return df
        
    def predict_price(self, req: PredictionRequest, save_to_db: bool = True) -> PredictionResponse:
        production_cost = self.calculate_cost(req)
        
        df = self.prepare_features(req)
        
        # Predict 
        pred = self.pipeline.predict(df)[0]
        
        predicted_price = float(round(float(pred), 2))
        minimum_price = float(production_cost)
        recommended_price = predicted_price
        premium_price = float(round(predicted_price * 1.25, 2))
        margin = float(round(recommended_price - minimum_price, 2))
        
        # Save to DB if session is available and flag is True
        if self.db is not None and save_to_db:
            prediction_record = Prediction(
                product_title=req.product_title,
                category=req.category,
                shop_name=req.shop_name,
                title_length=req.title_length,
                keyword_count=req.keyword_count,
                rating_numeric=req.rating_numeric,
                reviews_numeric=req.reviews_numeric,
                rating_score=req.rating_score,
                popularity_index=req.popularity_index,
                material_cost=req.material_cost,
                labor_hours=req.labor_hours,
                hourly_rate=req.hourly_rate,
                overhead_cost=req.overhead_cost,
                production_cost=production_cost,
                predicted_price=predicted_price,
                minimum_price=minimum_price,
                recommended_price=recommended_price,
                premium_price=premium_price,
                margin=margin
            )
            self.db.add(prediction_record)
            self.db.commit()
            self.db.refresh(prediction_record)
        
        return PredictionResponse(
            production_cost=production_cost,
            predicted_price=predicted_price,
            minimum_price=minimum_price,
            recommended_price=recommended_price,
            premium_price=premium_price,
            margin=margin
        )
        
    def generate_recommendations(self, production_cost: float, predicted_price: float, margin: float) -> RecommendationResponse:
        recommendations = []
        
        # Example logic to match business targets (margins, cost structure)
        if margin < 10.0:
            recommendations.append("The profit margin is low. Consider reducing material costs using alternative suppliers or buying in bulk.")
            recommendations.append("Optimizing labor hours by streamlining production can increase overall margin.")
            
        if predicted_price > (production_cost * 2.0):
            recommendations.append("Your product appears highly valued by the market. Consider offering an even more premium version.")
        
        if margin >= 10.0 and margin < 30.0:
            recommendations.append("Your price logic is healthy. Improve product photos to further increase perceived value and maintain pricing.")
            
        if len(recommendations) < 3:
            recommendations.append("Increase packaging quality to justify higher pricing ranges.")
        
        return RecommendationResponse(recommendations=recommendations)

    def explain_prediction(self, req: PredictionRequest) -> ExplainResponse:
        df = self.prepare_features(req)
        
        preprocessor = self.pipeline.named_steps['preprocessor']
        model = self.pipeline.named_steps['model'].regressor_  # Extract from TransformedTargetRegressor

        # SHAP calculation
        X_test_transformed = preprocessor.transform(df)
        
        if hasattr(X_test_transformed, 'toarray'):
            X_test_transformed = X_test_transformed.toarray()

        # Generate Explainability Model - Use TreeExplainer
        explainer = shap.Explainer(model, X_test_transformed)
        shap_values = explainer(X_test_transformed, check_additivity=False)
        
        # Get feature names from transformer
        try:
            num_cols = list(preprocessor.transformers_[0][1].get_feature_names_out())
        except:
            num_cols = ['title_length', 'keyword_count', 'rating_numeric', 'reviews_numeric', 
                        'rating_score', 'popularity_index', 'popularity_weight', 'keyword_density']
            
        cat_cols = ['shop_name', 'category']
        try:
            text_cols = [f"tfidf_{w}" for w in preprocessor.transformers_[2][1].get_feature_names_out()]
        except:
            text_cols = []
            
        all_features = num_cols + cat_cols + text_cols
        
        # Make sure sizes match, else use fallback indexing
        vals = np.abs(shap_values.values[0])
        
        impacts = []
        for i, val in enumerate(vals):
            feat_name = all_features[i] if i < len(all_features) else f"feature_{i}"
            impacts.append({"feature": feat_name, "impact": float(val)})
            
        # Sorting
        impacts = sorted(impacts, key=lambda x: x["impact"], reverse=True)
        top = [TopFactor(**x) for x in impacts[:5]]
        
        return ExplainResponse(top_factors=top)
