from sqlalchemy.orm import Session
from fastapi import HTTPException
from src.app.models.user import User
from src.app.core.security import hash_password, verify_password, create_access_token

def register_user(db: Session, user_data):
    
    # Vérifier si email existe
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    # Vérifier password match
    if user_data.password != user_data.confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match")

    hashed = hash_password(user_data.password)

    new_user = User(
        first_name=user_data.first_name,
        last_name=user_data.last_name,
        email=user_data.email,
        hashed_password=hashed,
        role="artisan"
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return new_user


def authenticate_user(db: Session, email: str, password: str):
    user = db.query(User).filter(User.email == email).first()

    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if not verify_password(password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if not user.is_active:
        raise HTTPException(status_code=403, detail="Inactive account")

    return user


def login_user(user: User):
    token = create_access_token(
        data={
            "sub": user.email,
            "role": user.role
        }
    )
    return token
