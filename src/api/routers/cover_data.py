from fastapi import APIRouter, Query
from albcovis.services.cover_data_aggregator import CoverDataAggregator
from albcovis.models.cover_data import CoverDataModel

router = APIRouter(prefix="/cover", tags=["cover-data"])
aggregator = CoverDataAggregator()

@router.get("/metadata", response_model=CoverDataModel)
def get_metadata(mbid: str = Query(..., description="MusicBrainz Release Group MBID"),
                 discogs_id: str | None = Query(None, description="Optional Discogs master/release ID")):
    """
    Fetch structured metadata (artist, genres, releases, cover art URL).
    """
    return aggregator.get_rg_cover_data(mbid, discogs_id)

@router.get("/features", response_model=CoverDataModel)
def get_metadata_with_features(mbid: str = Query(..., description="MusicBrainz Release Group MBID"),
                               discogs_id: str | None = Query(None, description="Optional Discogs ID")):
    """
    Fetch metadata **and** extract visual features from the cover art (color, textures, faces, text).
    """
    return aggregator.get_rg_cover_data_visual_features(mbid, discogs_id)
