from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router as keyword_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for security in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(keyword_router)
