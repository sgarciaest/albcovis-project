from typing import List, Optional, Dict, Any
from pydantic import BaseModel, HttpUrl, Field, ConfigDict
from typing_extensions import Annotated
from pathlib import Path
from albcovis.models.musicbrainz import CoverArtImage

class CoverDataBaseModel(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="ignore",
        coerce_numbers_to_str=True
    )

class Artist(CoverDataBaseModel):
    name: str
    artist_mbid: str
    country: Optional[str] = None
    type: Optional[str] = None
    disambiguation: Optional[str] = None

class ArtistCredit(CoverDataBaseModel):
    full_credit_name: str
    artists: List[Artist] = Field(default_factory=list)

class ExternalLinks(CoverDataBaseModel):
    discogs: Optional[HttpUrl] = None
    allmusic: Optional[HttpUrl] = None
    wikidata: Optional[HttpUrl] = None

class GenreInfo(CoverDataBaseModel):
    genres_discogs: List[str] = Field(default_factory=list)
    styles_discogs: List[str] = Field(default_factory=list)
    genres_mb: List[str] = Field(default_factory=list)
    tags_mb: List[str] = Field(default_factory=list)

class ReleaseSummary(CoverDataBaseModel):
    release_mbid: str
    barcode: Optional[str] = None
    disambiguation: Optional[str] = None
    date: Optional[str] = None
    country: Optional[str] = None
    format: List[str] = Field(default_factory=list)

# class CoverArt(CoverDataBaseModel):
#     has_cover: bool
#     images: List[CoverArtImage] = Field(default_factory=list)

class CoverArtArchiveThumbnails(CoverDataBaseModel):
    px_1200: Optional[HttpUrl] = Field(None, alias="1200")
    px_250: Optional[HttpUrl] = Field(None, alias="250")
    px_500: Optional[HttpUrl] = Field(None, alias="500")
    large: Optional[HttpUrl] = None
    small: Optional[HttpUrl] = None

class ColorSwatch(CoverDataBaseModel):
    hex: Annotated[List[str], Field(min_length=3, max_length=3)]
    rgb: Annotated[List[int], Field(min_length=3, max_length=3)]  # Enforce 3 RGB ints
    lab: Annotated[List[float], Field(min_length=3, max_length=3)]
    weight: Optional[Annotated[float, Field(ge=0.0, le=1.0)]] = None

class ColorExtractionMethods(CoverDataBaseModel):
    dominant_colors: List[ColorSwatch] = Field(default_factory=list)
    prominent_colors: List[ColorSwatch] = Field(default_factory=list)

class TextureDescriptors(CoverDataBaseModel):
    edge_density: Annotated[float, Field(ge=0.0, le=1.0)]
    orientation_entropy: Annotated[float, Field(ge=0.0, le=1.0)]
    pixel_intensity_entropy: Annotated[float, Field(ge=0.0, le=1.0)]
    glcm_contrast: Annotated[float, Field(ge=0.0, le=1.0)]
    glcm_homogeneity: Annotated[float, Field(ge=0.0, le=1.0)]
    glcm_energy: Annotated[float, Field(ge=0.0, le=1.0)]
    glcm_correlation: Annotated[float, Field(ge=0.0, le=1.0)]
    glcm_entropy: Annotated[float, Field(ge=0.0, le=1.0)]
    lbp_entropy: Annotated[float, Field(ge=0.0, le=1.0)]
    lbp_energy: Annotated[float, Field(ge=0.0, le=1.0)]

class VisualComplexity(CoverDataBaseModel):
    texture_descriptors: TextureDescriptors
    visual_complexity: Annotated[float, Field(ge=0.0, le=1.0)]

class Detection(CoverDataBaseModel):
    bbox: List[float] = Field(default_factory=list)
    area: float
    relative_size: float
    confidence: Optional[Annotated[float, Field(ge=0.0, le=1.0)]] = None

class FaceDetectionSummary(CoverDataBaseModel):
    n_faces: int
    mean_area: float
    average_relative_size: float
    largest_face: Optional[Detection] = None
    highest_confidence_face: Optional[Detection] = None
    faces: List[Detection] = Field(default_factory=list)

class TextDetectionSummary(CoverDataBaseModel):
    n_texts: int
    mean_area: float
    average_relative_size: float
    largest_text: Detection
    texts: List[Detection] = Field(default_factory=list)

class VisualFeatures(CoverDataBaseModel):
    colors: ColorExtractionMethods
    visual_complexity_descriptors: VisualComplexity
    face_detection: FaceDetectionSummary
    text_detection: TextDetectionSummary

class CoverArtArchiveImage(CoverDataBaseModel):
    id: str
    types: List[str] = Field(default_factory=list)
    image: HttpUrl
    thumbnails: CoverArtArchiveThumbnails = Field(default_factory=CoverArtArchiveThumbnails)
    approved: Optional[bool] = None
    release: Optional[HttpUrl] = None
    visual_features: Optional[VisualFeatures] = None

class CoverDataModel(CoverDataBaseModel):
    release_group_mbid: str
    title: str
    type: Optional[str] = None
    first_release_date: Optional[str] = None
    artist_credit: ArtistCredit
    external_links: ExternalLinks
    genre_info: GenreInfo
    releases: List[ReleaseSummary] = Field(default_factory=list)
    cover_art: Optional[CoverArtArchiveImage] = None


