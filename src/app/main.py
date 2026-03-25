from fastapi import FastAPI
from src.app.routers.authentification import router as auth_router
from src.app.routers.prediction import router as prediction_router
from src.app.db.config import engine, Base
from src.app.models.user import User
from src.app.models.prediction import Prediction
from fastapi.middleware.cors import CORSMiddleware
import sys
print(sys.executable)
print(sys.path)


app = FastAPI(title="FairPrice AI")
origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base.metadata.create_all(bind=engine)
# inclure les routes
app.include_router(auth_router)
# app.include_router(auth_router)
app.include_router(prediction_router)
