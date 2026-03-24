
from fastapi.security import OAuth2PasswordRequestForm
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from src.app.schemas.user import UserRegister, UserLogin, UserResponse, Token
from src.app.services.auth_service import register_user, authenticate_user, login_user
from src.app.db.config import get_db
from src.app.dependencies.auth_dependencies import get_current_active_user
from src.app.dependencies.auth_dependencies import require_admin
from src.app.models.user import User
from fastapi import HTTPException


router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=UserResponse)
def register(user: UserRegister, db: Session = Depends(get_db)):
    return register_user(db, user)



@router.post("/login", response_model=Token)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    authenticated_user = authenticate_user(
        db,
        form_data.username,  
        form_data.password
    )

    token = login_user(authenticated_user)

    return {
        "access_token": token,
        "token_type": "bearer"
    }



@router.get("/profile",response_model=UserResponse)
def get_profile(current_user = Depends(get_current_active_user)):
    return current_user

@router.delete("/delete-user/{user_id}")
def delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin)
):
    user = db.query(User).filter(User.id == user_id).first()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    db.delete(user)
    db.commit()
    return {"message": "User deleted"}
