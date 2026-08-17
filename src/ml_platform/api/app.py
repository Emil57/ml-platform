from fastapi import FastAPI

app = FastAPI(
    title="ML Plaform API",
    description="API for serving machine learning inference",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
