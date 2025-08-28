import re
from typing import Any, Dict, Optional, Tuple, Literal
import requests
from albcovis.utils.http import make_session
from albcovis.settings import settings
from albcovis.models.discogs import DiscogsRelease, DiscogsMaster


def ensure_discogs_id(discogs_id: str | int) -> str:
    s = str(discogs_id)
    if not s.isdigit():
        raise ValueError(f"Invalid Discogs ID: {discogs_id!r}")
    return s


class DiscogsClient:
    BASE = "https://api.discogs.com"

    def __init__(self):
        self.s = make_session(settings.user_agent)
        self.headers = {
            "User-Agent": settings.user_agent,
            "Accept": "application/json",
            "Authorization": f"Discogs token={settings.discogs_token}",
        }

    @staticmethod
    def build_url(entity_type: Literal["master", "release", "artist", "label"], discogs_id: str | int) -> str:
        valid = {"master", "release", "artist", "label"}
        if entity_type not in valid:
            raise ValueError(f"Invalid entity type: {entity_type}. Must be one of {valid}.")
        return f"https://www.discogs.com/{entity_type}/{ensure_discogs_id(discogs_id)}"

    @staticmethod
    def parse_url(url: str) -> Optional[Tuple[str, str]]:
        m = re.match(r'https?://(?:www\.)?discogs\.com/(master|release|artist|label)/(\d+)', url, flags=re.I)
        if not m:
            return None
        return m.group(1).lower(), m.group(2)

    # Keep search() returning raw JSON for now (without pydantic model for now)
    # Reason: discogs gets different structure and fields of data for the search endpoint, not as MB which as the same dats schema also for search.
    def search(self, params: Dict[str, Any], *, per_page: int = 50, page: int = 1) -> Dict[str, Any]:
        per_page = max(1, min(per_page, 100))
        page = max(1, page)
        url = f"{self.BASE}/database/search"
        qp = dict(params)
        qp["per_page"] = per_page
        qp["page"] = page
        r = self.s.get(url, headers=self.headers, params=qp, timeout=20)
        r.raise_for_status()
        return r.json()

    def search_id_by_artist_title(
        self, artist: str, title: str
    ) -> Optional[Tuple[Literal["master", "release"], str]]:
        """
        Search Discogs for a given artist/title, preferring master releases.
        Falls back to releases if no masters are found.
        Why use the releases endpoint fallback: if a release only has one release it is not show as a master release directly in discogs database
        Returns a tuple of (entity_type, discogs_id) or None if not found.
        """
        # Prefer master releases
        results = self.search({"artist": artist, "release_title": title, "type": "master"})
        if results.get("pagination", {}).get("items", 0) > 0:
            first = results["results"][0]
            return first["type"], str(first["id"])
        # Fallback to releases
        results = self.search({"artist": artist, "release_title": title})
        if results.get("pagination", {}).get("items", 0) > 0:
            first = results["results"][0]
            return first["type"], str(first["id"])

        return None

    # --- Masterd and releases endpoint returning Pydantic models ---

    def get_master(self, master_id: str | int) -> DiscogsMaster:
        r = self.s.get(f"{self.BASE}/masters/{ensure_discogs_id(master_id)}", headers=self.headers, timeout=20)
        r.raise_for_status()
        return DiscogsMaster(**r.json())

    def get_release(self, release_id: str | int) -> DiscogsRelease:
        r = self.s.get(f"{self.BASE}/releases/{ensure_discogs_id(release_id)}", headers=self.headers, timeout=20)
        r.raise_for_status()
        return DiscogsRelease(**r.json())
