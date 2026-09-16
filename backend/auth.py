"""
auth.py — JWT authentication for Neural-Link.

Changes vs original:
• Uses PyJWT ≥ 2.x API: jwt.encode() returns str (not bytes).
• Expiry stored as int (not float) to comply with RFC 7519.
• Python 3.10+ type hints.
"""

import time
import jwt
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import os

router     = APIRouter()
SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "NEURAL_LINK_SUPER_SECRET_KEY_CHANGE_IN_PROD")
ALGORITHM  = "HS256"
TOKEN_TTL  = 3600   # seconds


class LoginRequest(BaseModel):
    username: str
    password: str


def create_access_token(data: dict) -> str:
    payload = {**data, "exp": int(time.time()) + TOKEN_TTL, "iat": int(time.time())}
    # PyJWT ≥ 2.0 returns a str directly
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


@router.post("/token")
async def login(req: LoginRequest):
    role_map = {
        ("admin",      "admin"): "Admin",
        ("researcher", None):    "Researcher",
        ("student",    None):    "Student",
    }
    role = None
    for (user, pwd), r in role_map.items():
        if req.username == user and pwd is not None and req.password == pwd:
            role = r
            break

    if role is None:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token({"sub": req.username, "role": role})
    return {"access_token": token, "token_type": "bearer", "role": role}


def verify_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=403, detail=f"Could not validate credentials: {exc}")
