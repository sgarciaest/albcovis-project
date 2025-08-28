import re
from typing import Tuple

def split_discogs_title(raw_title: str) -> Tuple[str, str]:
    m = re.match(r"^(.*)\s+-\s+(.*?)\s*$", raw_title or "")
    if not m:
        return "", (raw_title or "").strip()
    return m.group(1).strip(), m.group(2).strip()

def clean_discogs_artist_name(name: str) -> str:
    return re.sub(r'\s*\(\d+\)$', '', (name or "")).strip()
