from typing import Optional, Dict, Any, List
import requests
import tempfile
from pathlib import Path
from PIL import Image


from albcovis.settings import settings
from albcovis.services.musicbrainz import MusicBrainzClient, ensure_uuid
from albcovis.services.discogs import DiscogsClient
from albcovis.models.musicbrainz import MusicBrainzReleaseGroup, CoverArtResponse
from albcovis.models.discogs import DiscogsMaster, DiscogsRelease
from albcovis.models.cover_data import CoverDataModel
from albcovis.utils.img import limit_image_size, pil_to_numpy01, rgb_to_gray
from albcovis.services import color_extraction, texture_descriptors, face_detection, text_detection


PREF_ORDER = ("px_500", "large", "px_250", "small", "px_1200")
def pick_caa_url(cover_art) -> tuple[str, str]:
    """
    Returns (url, size_tag) choosing the best available thumbnail or the full image.
    `cover_art` is an instance of CoverArtArchiveImage model from cover_data
    """
    for k in PREF_ORDER:
        url = getattr(cover_art.thumbnails, k, None)
        if url is not None:
            return str(url), k
    return str(cover_art.image), "orig"


class CoverDataAggregator:
    def __init__(self, mb: Optional[MusicBrainzClient] = None, dg: Optional[DiscogsClient] = None):
        self.mb = mb or MusicBrainzClient()
        self.dg = dg or DiscogsClient()

    def get_rg_cover_data(self, mbid: str, discogs_hint_id: Optional[str] = None) -> CoverDataModel:
        mbid = ensure_uuid(mbid)
        mb_rg: MusicBrainzReleaseGroup = self.mb.get_release_group(mbid)

        try:
            caa: Optional[CoverArtResponse] = self.mb.get_release_group_cover_meta(mbid)
        except requests.HTTPError as e:
            # CAA returns 404 if no art; treating it as "no cover"
            if e.response is not None and e.response.status_code == 404:
                caa = None
            else:
                raise

        # --- Artist credit
        full_credit = ""
        artists: List[Dict[str, Any]] = []
        #
        for ac in mb_rg.artist_credit:
            full_credit += ac.name + ac.joinphrase
            artist = ac.artist
            # artists.append({
            #     "name": ac.name,
            #     "artist_mbid": artist.id if artist else "",
            #     "country": artist.country if artist else None,
            #     "type": artist.type if artist else None,
            #     "disambiguation": artist.disambiguation if artist else None,
            # })
            artists.append({
                "name": ac.name,
                "artist_mbid": artist.id,
                "country": artist.country,
                "type": artist.type,
                "disambiguation": artist.disambiguation,
            })

        # --- External links
        ext: Dict[str, str] = {}
        if discogs_hint_id:
            ext["discogs"] = self.dg.build_url("master", discogs_hint_id)

        for rel in mb_rg.relations:
            rtype = rel.type
            # rel.url.resource is a pydantic HttpUrl -> convert to str for downstream functions for regex
            # if rtype in ("discogs", "allmusic", "wikidata") and rel.url and rel.url.resource:
            #     ext[rtype] = str(rel.url.resource)
            if rel.type in ("discogs", "allmusic", "wikidata"):
                ext[rel.type] = str(rel.url.resource)

        # --- Genres (Discogs + MB)
        genre_info: Dict[str, Any] = {
            "genres_discogs": [],
            "styles_discogs": [],
            "genres_mb": [],
            "tags_mb": []
        }

        discogs_tuple: Optional[tuple[str, str]] = None
        if "discogs" in ext:
            discogs_tuple = self.dg.parse_url(ext["discogs"])

        if not discogs_tuple:
            # Try to find one by artist/title
            found = self.dg.search_id_by_artist_title(full_credit, mb_rg.title)
            if found:
                discogs_tuple = found
                ext["discogs"] = self.dg.build_url(found[0], found[1])

        # Fetch Discogs genres/styles
        if discogs_tuple:
            etype, did = discogs_tuple
            if etype == "master":
                djson: DiscogsMaster = self.dg.get_master(did)  # Pydantic DiscogsMaster
            elif etype == "release":
                djson: DiscogsRelease = self.dg.get_release(did)  # Pydantic DiscogsRelease
            else:
                djson = None

            if djson is not None:
                genre_info["genres_discogs"] = djson.genres
                genre_info["styles_discogs"] = djson.styles

        # MB genres/tags
        genre_info["genres_mb"] = [g.name for g in mb_rg.genres]
        genre_info["tags_mb"] = [t.name for t in mb_rg.tags]

        # --- Releases (only official)
        releases_out: List[Dict[str, Any]] = []
        for rel in mb_rg.releases:
            if rel.status != "Official":
                continue
            formats: List[str] = []
            for m in rel.media:
                if m.format and m.format not in formats:
                    formats.append(m.format)
            releases_out.append({
                "release_mbid": rel.id,
                "barcode": rel.barcode,
                "disambiguation": rel.disambiguation,
                "date": rel.date,
                "country": rel.country,
                "format": formats,
            })

        # --- Cover art

        # images = caa.images if (caa and caa.images) else []
        images = caa.images if caa else []
        front_image = next((img for img in images if "Front" in img.types), None)
        if caa and front_image:
            caa_front: Dict[str, Any] = {
                "id": front_image.id,
                "types": front_image.types,
                "image": front_image.image,
                "thumbnails": front_image.thumbnails,
                "approved": front_image.approved,
                "release": caa.release
            }
        else:
            caa_front = None

        return CoverDataModel(
            release_group_mbid=mbid,
            title=mb_rg.title,
            type=mb_rg.primary_type,
            first_release_date=mb_rg.first_release_date,
            artist_credit={"full_credit_name": full_credit, "artists": artists},
            external_links=ext,
            genre_info=genre_info,
            releases=releases_out,
            cover_art=caa_front
        )
    
    from PIL import Image
    from typing import Dict

    from albcovis.utils.img import limit_image_size, pil_to_numpy01, rgb_to_gray
    from albcovis.services import color_extraction, texture_descriptors, face_detection, text_detection

    def extract_visual_features_from_cover(self, path: str) -> Dict:
        # Ensure path format is str and not Path
        path = str(path)

        # Preprocess
        img = Image.open(path)
        img = limit_image_size(img)
        rgb01 = pil_to_numpy01(img)
        gray01 = rgb_to_gray(rgb01)

        c = color_extraction.extract_colors(rgb01)
        vcd = texture_descriptors.extract_visual_complexity_descriptors(gray01)
        fd = face_detection.detect_faces(path)
        td = text_detection.detect_text(path)

        return {
            "colors": c,
            "visual_complexity_descriptors": vcd,
            "face_detection": fd,
            "text_detection": td
        }
    
    def get_rg_cover_data_visual_features(
        self, mbid: str, discogs_hint_id: Optional[str] = None
    ) -> CoverDataModel:
        # Get base metadata
        cover_data = self.get_rg_cover_data(mbid, discogs_hint_id)

        # If no cover art → return the same
        if not cover_data.cover_art or not cover_data.cover_art.image:
            return cover_data

        # Select best url based on priority
        url, size_tag = pick_caa_url(cover_data.cover_art)

        # Download image as a temporary file to process it
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            r = requests.get(url, stream=True, timeout=60)
            r.raise_for_status()
            tmp.write(r.content)
            tmp.flush()
            tmp_path = Path(tmp.name)

        try:
            # Extract visual features
            vf = self.extract_visual_features_from_cover(tmp_path)

            # Add to the data
            cover_data.cover_art.visual_features = vf

        finally:
            # deletetemp file
            tmp_path.unlink(missing_ok=True)

        return cover_data