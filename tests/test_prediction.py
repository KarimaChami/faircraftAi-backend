import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.app.main import app
from src.app.db.config import Base, get_db
from src.app.models.user import User
from src.app.models.prediction import Prediction
from src.app.dependencies.model import get_model
import numpy as np

# --- SETUP TEST DATABASE ---
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_pred.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

# --- MOCK ML MODEL ---
class MockExplainerResult:
    def __init__(self):
        self.values = np.array([[0.1, 0.2, 0.3]])

class MockModel:
    def predict(self, df):
        return np.array([60.0], dtype=np.float32)
    
    @property
    def named_steps(self):
        class MockPreprocessor:
            def transform(self, df): return df
            @property
            def transformers_(self):
                class MockTrans:
                    def get_feature_names_out(self): return ["feat1"]
                    def transform(self, X): return X
                return [("num", MockTrans(), []), ("cat", MockTrans(), []), ("text", MockTrans(), [])]
        
        class MockReg:
            def __init__(self):
                self.regressor_ = self
            def __call__(self, *args, **kwargs):
                return MockExplainerResult()
        
        return {"preprocessor": MockPreprocessor(), "model": MockReg()}

def override_get_model():
    return MockModel()

app.dependency_overrides[get_model] = override_get_model

client = TestClient(app)

@pytest.fixture(scope="module", autouse=True)
def setup_db():
    Base.metadata.create_all(bind=engine)
    # Register a user once for all prediction tests
    client.post(
        "/api/v1/auth/register",
        json={
            "first_name": "Pred",
            "last_name": "Tester",
            "email": "pred_test@example.com",
            "password": "password123",
            "confirm_password": "password123"
        }
    )
    yield
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def auth_token():
    response = client.post(
        "/api/v1/auth/login",
        data={"username": "pred_test@example.com", "password": "password123"}
    )
    return response.json()["access_token"]

# --- TESTS PREDICTION ---
def test_predict_endpoint(auth_token):
    headers = {"Authorization": f"Bearer {auth_token}"}
    payload = {
        "product_title": "Test Ring",
        "category": "Jewelry",
        "shop_name": "MockShop",
        "title_length": 10,
        "keyword_count": 1,
        "rating_numeric": 5.0,
        "reviews_numeric": 5,
        "rating_score": 5.0,
        "popularity_index": 50,
        "material_cost": 20,
        "labor_hours": 2,
        "hourly_rate": 10,
        "overhead_cost": 10
    }
    
    response = client.post("/api/v1/predict", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_price"] == 60.0
    assert data["production_cost"] == 50.0  # 20 + (2*10) + 10
    assert data["margin"] == 10.0


def test_health_check():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
