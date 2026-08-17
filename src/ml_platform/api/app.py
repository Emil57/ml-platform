from fastapi import FastAPI
from ml_platform.api.routes.predictions import router as prediction_router

app = FastAPI(
    title="ML Plaform API",
    description="API for serving machine learning inference",
    version="0.1.0",
)

app.include_router(prediction_router)

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
