"""
base.py — FastAPI app entry
"""
from fastapi import APIRouter
from config import settings

base_router = APIRouter(tags=["base"])

@base_router.get("/")
def welcome():
    app_name = settings.APP_NAME
    app_version = settings.APP_VERSION

    return {
        "app_name": app_name,
        "app_version": app_version
    }

