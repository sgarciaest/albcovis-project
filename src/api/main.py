from fastapi import FastAPI
from .routers import cover_data

app = FastAPI(
    title="Album Cover Visual Features API",
    description="API to extract metadata and visual features from album cover art "
                "using MusicBrainz, Discogs, Cover Art Archive and computer vision techniques.",
    version="0.1.0",
    contact={
        "name": "Sergio",
        "url": "https://github.com/sgarciaest/albcovis-project",
        "email": "37sergarest@gmail.com",
    },
)

app.include_router(cover_data.router)

@app.get("/", tags=["root"])
def root():
    return {"message": "Welcome to the Album Cover Visual Features API"}
