# Practical field definition guidance:
#
# 1. Collections (list/dict):
#    - Use Field(default_factory=list/dict) instead of None.
#    - Don't use Optional[]
#    - Why:
#       - keeps JSON shape stable (never null), avoids None checks.
#       - For safety and predictable JSON shape.
#       - Rare for collections to carry different semantic meaning of None conmpared to empty. 
#
# 2. Booleans/ints:
#    - If absence means "unknown": Optional[bool/int] = None.
#    - If absence means false/0: default to False/0 and keep non-optional.
#    - Why: preserves semantic difference between unknown and false/zero.
#
# 3. Strings:
#    - If absence means "unknown": Optional[str] = None.
#    - If present-but-empty is meaningful: default to "".
#    - Why: avoids empty-string noise, makes nulls semantically clear.
# Note: collections default to []/{} for safe iteration, but scalars like str use None to preserve "unknown" vs "empty" semantics.

# 4. Nested models (fields that are other Pydantic models):
#    - If upstream always sends it and your API assumes it exists: keep required (no default, no Optional).
#    - If it can genuinely be absent: use Optional[Model] = None.
#    - Avoid default_factory=Model() unless an "empty" object has clear semantic meaning in your API.
#    - Why: preserves truthfulness of upstream data and avoids fabricating records that hide missing/malformed input.


from typing import List, Optional, Dict
from pydantic import BaseModel, HttpUrl, ConfigDict, Field

def to_dash_case(string: str) -> str:
    return string.replace("_", "-")

class MBBaseModel(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,        # Allow passing snake_case in Python
        alias_generator=to_dash_case, # Map snake_case → dash-case for JSON
        extra="ignore",
        coerce_numbers_to_str=True
    )

class MBTag(MBBaseModel):
    name: str
    count: Optional[int] = None


class MBGenre(MBBaseModel):
    id: Optional[str]  = None
    name: str
    count: Optional[int]  = None
    disambiguation: Optional[str] = None

class MBArea(MBBaseModel):
    id: str
    name: str
    sort_name: Optional[str] = None
    iso_3166_1_codes: List[str] = Field(default_factory=list)
    type: Optional[str] = None
    type_id: Optional[str] = None
    disambiguation: Optional[str] = None

class MBReleaseEvent(MBBaseModel):
    date: Optional[str] = None
    area: Optional[MBArea] = None


class MBMedia(MBBaseModel):
    format: Optional[str] = None
    format_id: Optional[str] = None
    position: Optional[int] = None
    title: Optional[str] = None
    track_count: Optional[int] = None
    id: Optional[str] = None


class MBArtist(MBBaseModel):
    id: str
    name: str
    sort_name: Optional[str] = None
    country: Optional[str] = None
    type: Optional[str] = None
    type_id: Optional[str] = None
    disambiguation: Optional[str] = None
    genres: List[MBGenre] = Field(default_factory=list)
    tags: List[MBTag] = Field(default_factory=list)


class MBArtistCreditEntry(MBBaseModel):
    name: str
    joinphrase: str = "" # Safe for string concatenation, semantically is just a separatro
    artist: MBArtist


class MBRelationURL(MBBaseModel):
    id: Optional[str] = None
    resource: HttpUrl


class MBRelation(MBBaseModel):
    type: str
    type_id: Optional[str] = None
    target_type: Optional[str] = None
    url: MBRelationURL


class MBRelease(MBBaseModel):
    id: str
    title: Optional[str] = None
    status: Optional[str] = None
    date: Optional[str] = None
    country: Optional[str] = None
    barcode: Optional[str] = None
    disambiguation: Optional[str] = None
    media: List[MBMedia] = Field(default_factory=list)
    artist_credit: List[MBArtistCreditEntry] = Field(default_factory=list)
    release_events: List[MBReleaseEvent] = Field(default_factory=list)
    genres: List[MBGenre] = Field(default_factory=list)
    tags: List[MBTag] = Field(default_factory=list)

class MusicBrainzReleaseGroup(MBBaseModel):
    id: str
    title: Optional[str] = None
    primary_type: Optional[str] = None
    first_release_date: Optional[str] = None
    artist_credit: List[MBArtistCreditEntry] = Field(default_factory=list)
    relations: List[MBRelation] = Field(default_factory=list)
    genres: List[MBGenre] = Field(default_factory=list)
    tags: List[MBTag] = Field(default_factory=list)
    releases: List[MBRelease] = Field(default_factory=list)

class CoverArtImage(MBBaseModel):
    types: List[str] = Field(default_factory=list)
    front: Optional[bool] = None
    back: Optional[bool] = None
    comment: Optional[str] = None
    image: HttpUrl
    thumbnails: Dict[str, HttpUrl] = Field(default_factory=dict)
    approved: Optional[bool] = None
    edit: Optional[int] = None
    id: str

class CoverArtResponse(MBBaseModel):
    release: HttpUrl
    images: List[CoverArtImage] = Field(default_factory=list)


