from fastapi import FastAPI
from app.api.v1.routes_auth import router as auth_router
from app.db.config import engine, Base
import sys
print(sys.executable)
print(sys.path)
app = FastAPI(title="FairPrice AI")
Base.metadata.create_all(bind=engine)
# inclure les routes
app.include_router(auth_router)

