from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from datetime import datetime
from src.app.db.config import Base

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True) # Optional linking if an artisan is logged in
    
    # Inputs
    product_title = Column(String(255), nullable=False)
    category = Column(String(100), nullable=False)
    shop_name = Column(String(100), nullable=False)
    title_length = Column(Integer)
    keyword_count = Column(Integer)
    rating_numeric = Column(Float)
    reviews_numeric = Column(Integer)
    rating_score = Column(Float)
    popularity_index = Column(Float)
    
    material_cost = Column(Float)
    labor_hours = Column(Float)
    hourly_rate = Column(Float)
    overhead_cost = Column(Float)
    
    # Outputs
    production_cost = Column(Float)
    predicted_price = Column(Float)
    minimum_price = Column(Float)
    recommended_price = Column(Float)
    premium_price = Column(Float)
    margin = Column(Float)

    created_at = Column(DateTime, default=datetime.utcnow)
