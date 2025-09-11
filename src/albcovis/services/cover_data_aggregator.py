from typing import Optional, Dict, Any, List
import requests
from albcovis.settings import settings
from albcovis.services.musicbrainz import MusicBrainzClient, ensure_uuid
from albcovis.services.discogs import DiscogsClient
from albcovis.models.cover_data import CoverDataModel
from albcovis.models.musicbrainz import MusicBrainzReleaseGroup, CoverArtResponse
from albcovis.models.discogs import DiscogsMaster, DiscogsRelease


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