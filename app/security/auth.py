"""
Minimal JWT auth + RBAC for the demo.

In production this is exactly where you'd swap in Keycloak (self-hosted, free,
real enterprise SSO/RBAC) or a paid IdP (Okta/Auth0) -- the FastAPI dependency
`get_current_user` below is the seam: everything downstream only cares about
the returned (username, role) tuple, not how it was obtained.
"""
from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

from app.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")

# Demo user directory. Replace with a real users table / IdP integration.
# Roles map to document classification access -- see security/rbac.py
_DEMO_USERS = {
    "analyst1": {
        "hashed_password": pwd_context.hash("analyst1"),
        "role": "front_office_analyst",
    },
    "compliance1": {
        "hashed_password": pwd_context.hash("compliance1"),
        "role": "compliance_officer",
    },
    "public1": {
        "hashed_password": pwd_context.hash("public1"),
        "role": "public_reader",
    },
}


def authenticate_user(username: str, password: str) -> dict | None:
    user = _DEMO_USERS.get(username)
    if not user or not pwd_context.verify(password, user["hashed_password"]):
        return None
    return {"username": username, "role": user["role"]}


def create_access_token(username: str, role: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expiry_minutes)
    payload = {"sub": username, "role": role, "exp": expire}
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        username: str = payload.get("sub")
        role: str = payload.get("role")
        if username is None or role is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    return {"username": username, "role": role}
