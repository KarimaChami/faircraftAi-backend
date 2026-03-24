from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    product_title: str = Field(..., example="Handmade Silver Ring")
    category: str = Field(..., example="Jewelry")
    shop_name: str = Field(..., example="SilverCraft")
    title_length: int = Field(..., example=20)
    keyword_count: int = Field(..., example=3)
    rating_numeric: float = Field(..., example=4.8)
    reviews_numeric: int = Field(..., example=25)
    rating_score: float = Field(..., example=4.7)
    popularity_index: float = Field(..., example=120)
    material_cost: float = Field(..., example=12.5)
    labor_hours: float = Field(..., example=2.5)
    hourly_rate: float = Field(..., example=8.0)
    overhead_cost: float = Field(..., example=3.0)

class PredictionResponse(BaseModel):
    production_cost: float
    predicted_price: float
    minimum_price: float
    recommended_price: float
    premium_price: float
    margin: float

class TopFactor(BaseModel):
    feature: str
    impact: float

class ExplainResponse(BaseModel):
    top_factors: list[TopFactor]

class SimulationRequest(BaseModel):
    original_request: PredictionRequest
    modified_request: PredictionRequest

class SimulationResponse(BaseModel):
    original_price: float
    new_price: float
    price_difference: float

class RecommendationResponse(BaseModel):
    recommendations: list[str]
