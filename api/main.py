"""CCPFS API - FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.dependencies import store
from api.routers import patients, results, scheduling
from api.schemas import HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load data on startup, cleanup on shutdown."""
    store.load()
    yield


app = FastAPI(
    title="CCPFS API",
    description="Capacity-Constrained Personalized Follow-Up Scheduling",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://frontend:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(results.router, prefix="/api")
app.include_router(patients.router, prefix="/api")
app.include_router(scheduling.router, prefix="/api")


@app.get("/api/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if store.loaded else "loading",
        models_loaded=store.models_loaded,
        patients=store.curves.shape[0] if store.curves is not None else 0,
    )
