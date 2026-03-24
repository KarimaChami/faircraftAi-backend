from sqlalchemy import Column, Integer, String, DateTime, Boolean
from datetime import datetime
from src.app.db.config import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)

    role = Column(String(20), default="artisan")  # artisan / admin
    is_active = Column(Boolean, default=True)

    created_at = Column(DateTime, default=datetime.utcnow)
