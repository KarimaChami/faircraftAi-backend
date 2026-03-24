import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.app.main import app
from src.app.db.config import Base, get_db
from src.app.models.user import User
from src.app.models.prediction import Prediction

# --- SETUP TEST DATABASE ---
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_auth.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

# Apply the override to the FastAPI app instance
app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)

@pytest.fixture(scope="module", autouse=True)
def setup_db():
    # Create the tables in the SQLite database
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

# --- TESTS AUTH ---
def test_register_user():
    response = client.post(
        "/api/v1/auth/register",
        json={
            "first_name": "Test",
            "last_name": "User",
            "email": "test_auth@example.com",
            "password": "password123",
            "confirm_password": "password123"
        }
    )
    assert response.status_code == 200
    assert response.json()["email"] == "test_auth@example.com"

def test_login_user():
    # Login with the created user
    response = client.post(
        "/api/v1/auth/login",
        data={"username": "test_auth@example.com", "password": "password123"}
    )
    assert response.status_code == 200
    assert "access_token" in response.json()

def test_profile_access():
    # Login to get token
    login_res = client.post(
        "/api/v1/auth/login",
        data={"username": "test_auth@example.com", "password": "password123"}
    )
    token = login_res.json()["access_token"]
    
    response = client.get(
        "/api/v1/auth/profile",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    assert response.json()["email"] == "test_auth@example.com"
