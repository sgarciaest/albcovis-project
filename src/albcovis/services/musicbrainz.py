import uuid
from typing import List
import requests
from albcovis.utils.http import make_session
from albcovis.settings import settings
from albcovis.models.musicbrainz import (
    MusicBrainzReleaseGroup,
    CoverArtResponse
)

from albcovis.utils.rate_limiter import mb_limiter


def ensure_uuid(mbid: str) -> str:
    try:
        uuid_obj = uuid.UUID(mbid)
        return str(uuid_obj)
    except (ValueError, AttributeError):
        raise ValueError(f"Invalid MBID (UUID expected): {mbid!r}")

class MusicBrainzClient:
    BASE = "https://musicbrainz.org/ws/2"
    CAA = "https://coverartarchive.org"

    def __init__(self):
        self.s = make_session(settings.user_agent)
        self.headers = {
            "User-Agent": settings.user_agent,
            "Accept": "application/json"
        }

    def get_release_group(self, mbid: str) -> MusicBrainzReleaseGroup:
        mb_limiter.wait()
        mbid = ensure_uuid(mbid)
        url = f"{self.BASE}/release-group/{mbid}"
        params = {
            "inc": "artist-credits+releases+media+tags+url-rels+genres",
            "fmt": "json"
        }
        r = self.s.get(url, headers=self.headers, params=params, timeout=15)
        r.raise_for_status()
        return MusicBrainzReleaseGroup(**r.json())

    def get_release_group_cover_meta(self, mbid: str) -> CoverArtResponse:
        mb_limiter.wait()
        mbid = ensure_uuid(mbid)
        url = f"{self.CAA}/release-group/{mbid}"
        r = self.s.get(url, headers=self.headers, timeout=15)
        r.raise_for_status()
        return CoverArtResponse(**r.json())

    def search_release_group(self, query: str, inc: str, *, limit: int = 25, offset: int = 0) -> List[MusicBrainzReleaseGroup]:
        limit = max(1, min(limit, 100))
        offset = max(0, offset)
        url = f"{self.BASE}/release-group/"
        params = {
            "query": query,
            "inc": inc,
            "limit": limit,
            "offset": offset,
            "fmt": "json"
        }
        r = self.s.get(url, headers=self.headers, params=params, timeout=15)
        r.raise_for_status()
        results = r.json().get("release-groups", [])
        return [MusicBrainzReleaseGroup(**rg) for rg in results]

        # "inc=artist-credits+releases+media+genres+tags+url-rels"

