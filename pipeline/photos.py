"""Find one freely licensed Wikimedia Commons photo per (make, model, year bucket).

Usage: python -m pipeline.photos     # after build_rounds; fills photo fields in rounds.json
Results are cached in photos_cache.json so reruns don't hit the API again.
"""
from __future__ import annotations

import json
import re
import time

import requests

from pipeline import config
from pipeline.common import log

STEP = "photos"


def photo_key(make: str, model: str, year: int) -> str:
    start = year - year % config.PHOTO_YEAR_BUCKET
    return f"{make}|{model}|{start}"


def load_cache() -> dict[str, dict | None]:
    if config.PHOTOS_CACHE_PATH.exists():
        return json.loads(config.PHOTOS_CACHE_PATH.read_text())
    return {}


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "").strip()


def _search(session: requests.Session, query: str) -> list[dict]:
    params = {
        "action": "query", "generator": "search", "gsrnamespace": 6, "gsrsearch": query,
        "gsrlimit": config.PHOTO_SEARCH_LIMIT, "prop": "imageinfo",
        "iiprop": "url|extmetadata", "iiurlwidth": 800, "format": "json",
    }
    for attempt in range(config.PHOTO_MAX_RETRIES):
        resp = session.get(config.WIKIMEDIA_API, params=params, timeout=20)
        if resp.status_code != 429:
            break
        wait = float(resp.headers.get("Retry-After") or config.PHOTO_BACKOFF_S * (attempt + 1))
        log(STEP, f"rate limited; waiting {wait:.0f}s")
        time.sleep(wait)
    resp.raise_for_status()
    pages = resp.json().get("query", {}).get("pages", {})
    return sorted(pages.values(), key=lambda p: p.get("index", 0))


def _pick(pages: list[dict], model: str) -> dict | None:
    model_word = model.lower().split()[0]
    for page in pages:
        title = page.get("title", "").lower()
        if not title.endswith((".jpg", ".jpeg", ".png")) or model_word not in title:
            continue
        info = (page.get("imageinfo") or [{}])[0]
        meta = info.get("extmetadata", {})
        license_name = meta.get("LicenseShortName", {}).get("value", "")
        if not license_name.startswith(config.PHOTO_ALLOWED_LICENSE_PREFIXES):
            continue
        author = _strip_html(meta.get("Artist", {}).get("value", "")) or "Unknown author"
        return {"url": info.get("thumburl") or info.get("url"),
                "credit": f"{author}, {license_name}, via Wikimedia Commons"}
    return None


def fetch_photo(session: requests.Session, key: str) -> dict | None:
    make, model, start = key.split("|")
    for query in (f"{start} {make} {model}", f"{make} {model}"):
        found = _pick(_search(session, query), model)
        time.sleep(config.PHOTO_REQUEST_PAUSE_S)
        if found:
            return found
    return None


def main() -> None:
    data = json.loads(config.ROUNDS_PATH.read_text())
    rounds = data["rounds"]
    cache = load_cache()
    keys = {photo_key(r["car"]["make"], r["car"]["model"], r["car"]["year"]) for r in rounds}
    todo = sorted(k for k in keys if k not in cache)
    log(STEP, f"{len(keys)} photo keys, {len(todo)} not cached")

    session = requests.Session()
    session.headers["User-Agent"] = config.WIKIMEDIA_USER_AGENT
    for i, key in enumerate(todo, 1):
        try:
            cache[key] = fetch_photo(session, key)
        except requests.RequestException as exc:
            log(STEP, f"error for {key}: {exc}")
            continue
        if i % 25 == 0:
            config.PHOTOS_CACHE_PATH.write_text(json.dumps(cache, indent=1))
            log(STEP, f"{i}/{len(todo)} done")
    config.PHOTOS_CACHE_PATH.write_text(json.dumps(cache, indent=1))

    found = 0
    for r in rounds:
        hit = cache.get(photo_key(r["car"]["make"], r["car"]["model"], r["car"]["year"]))
        if hit:
            r["photo"].update(url=hit["url"], credit=hit["credit"])
            found += 1
    config.ROUNDS_PATH.write_text(json.dumps(data, indent=1))
    log(STEP, f"photos found for {found}/{len(rounds)} rounds; rounds.json updated")


if __name__ == "__main__":
    main()
