from typing import List, Optional
from pydantic import BaseModel, HttpUrl, Field, ConfigDict


class DGBaseModel(BaseModel):
    # Accept extra keys from Discogs and ignore them
    model_config = ConfigDict(
        extra="ignore",
        populate_by_name=True,
        coerce_numbers_to_str=True
    )


# --- Small primitives / nested bits ---

class DGUser(DGBaseModel):
    resource_url: Optional[HttpUrl] = None
    username: Optional[str] = None


# class DGRating(DGBaseModel):
#     average: Optional[float] = None
#     count: Optional[int] = None


# class DGCommunity(DGBaseModel):
#     contributors: List[DGUser] = Field(default_factory=list)
#     data_quality: Optional[str] = None
#     have: Optional[int] = None
#     rating: Optional[DGRating] = None
#     status: Optional[str] = None
#     submitter: Optional[DGUser] = None
#     want: Optional[int] = None


class DGArtistCredit(DGBaseModel):
    anv: Optional[str] = None
    id: str
    join: Optional[str] = None
    name: str
    resource_url: Optional[HttpUrl] = None
    role: Optional[str] = None
    tracks: Optional[str] = None


class DGCompany(DGBaseModel):
    catno: Optional[str] = None
    entity_type: Optional[str] = None
    entity_type_name: Optional[str] = None
    id: Optional[str] = None
    name: Optional[str] = None
    resource_url: Optional[HttpUrl] = None


class DGFormat(DGBaseModel):
    name: Optional[str] = None
    qty: Optional[str] = None      # Discogs returns this as string
    descriptions: List[str] = Field(default_factory=list)


class DGIdentifier(DGBaseModel):
    type: str
    value: Optional[str] = None


class DGImage(DGBaseModel):
    height: Optional[int] = None
    resource_url: Optional[HttpUrl] = None
    type: Optional[str] = None
    uri: Optional[HttpUrl] = None
    uri150: Optional[HttpUrl] = None
    width: Optional[int] = None


# class DGLabel(DGBaseModel):
#     catno: Optional[str] = None
#     entity_type: Optional[str] = None
#     id: Optional[str] = None
#     name: Optional[str] = None
#     resource_url: Optional[HttpUrl] = None


# class DGTrack(DGBaseModel):
#     duration: Optional[str] = None
#     position: Optional[str] = None
#     title: Optional[str] = None
#     type_: Optional[str] = None
#     extraartists: List[DGArtistCredit] = Field(default_factory=list)


# class DGVideo(DGBaseModel):
#     description: Optional[str] = None
#     duration: Optional[int] = None
#     embed: Optional[bool] = None
#     title: Optional[str] = None
#     uri: Optional[HttpUrl] = None


# --- Release ---

class DiscogsRelease(DGBaseModel):
    id: str
    title: Optional[str] = None
    released: Optional[str] = None
    released_formatted: Optional[str] = None
    year: Optional[int] = None
    country: Optional[str] = None
    master_id: Optional[int] = None
    master_url: Optional[HttpUrl] = None
    resource_url: Optional[HttpUrl] = None
    uri: Optional[HttpUrl] = None
    status: Optional[str] = None
    notes: Optional[str] = None
    artists: List[DGArtistCredit] = Field(default_factory=list)
    extraartists: List[DGArtistCredit] = Field(default_factory=list)
    formats: List[DGFormat] = Field(default_factory=list)
    genres: List[str] = Field(default_factory=list)
    styles: List[str] = Field(default_factory=list)
    identifiers: List[DGIdentifier] = Field(default_factory=list)
    images: List[DGImage] = Field(default_factory=list)

    # data_quality: Optional[str] = None
    # thumb: Optional[HttpUrl] = None
    # community: Optional[DGCommunity] = None
    # companies: List[DGCompany] = Field(default_factory=list)
    # date_added: Optional[str] = None
    # date_changed: Optional[str] = None
    # estimated_weight: Optional[int] = None
    # format_quantity: Optional[int] = None
    # labels: List[DGLabel] = Field(default_factory=list)
    # lowest_price: Optional[float] = None
    # num_for_sale: Optional[int] = None
    # series: List[dict] = Field(default_factory=list)  # rarely used, leave as raw dicts
    # tracklist: List[DGTrack] = Field(default_factory=list)
    # videos: List[DGVideo] = Field(default_factory=list)




# --- Master ---

class DiscogsMaster(DGBaseModel):
    id: str
    title: Optional[str] = None
    year: Optional[int] = None
    main_release: Optional[int] = None
    main_release_url: Optional[HttpUrl] = None
    resource_url: Optional[HttpUrl] = None
    uri: Optional[HttpUrl] = None
    artists: List[DGArtistCredit] = Field(default_factory=list)
    genres: List[str] = Field(default_factory=list)
    styles: List[str] = Field(default_factory=list)
    images: List[DGImage] = Field(default_factory=list)

    # videos: List[DGVideo] = Field(default_factory=list)
    # versions_url: Optional[HttpUrl] = None
    # tracklist: List[DGTrack] = Field(default_factory=list)
    # num_for_sale: Optional[int] = None
    # lowest_price: Optional[float] = None
    # data_quality: Optional[str] = None
