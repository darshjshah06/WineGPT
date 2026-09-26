from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from api.routes import router

app = FastAPI(
    title="WineGPT",
    version="1.0.0"
)

app.include_router(router)

@app.get("/")
def root():

    return {
        "message": "WineGPT API Running"
    }

@app.get("/lab", include_in_schema=False)
def gradient_lab():

    return FileResponse(Path(__file__).parent / "lab" / "index.html")
