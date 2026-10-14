"""Find one freely licensed Wikimedia Commons photo per (make, model, year bucket).

Usage: python -m pipeline.photos             # after build_rounds; fills photo fields in rounds.json
       python -m pipeline.photos --refresh   # ignore the cache and search again
Results are cached in photos_cache.json so reruns don't hit the API again.
"""
from __future__ import annotations

import json
import re
import sys
import time

import pandas as pd
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


def _first_year(title: str) -> int | None:
    match = re.search(r"(?<!\d)(19[89]\d|20[0-4]\d)(?!\d)", title)
    return int(match.group(1)) if match else None


def _usable(title: str, model: str, longer_models: list[str]) -> bool:
    """Whole-car photo of this model (not a close-up, not e.g. 'Corolla Cross' for 'Corolla')."""
    t = title.lower().replace("_", " ")
    if not t.endswith((".jpg", ".jpeg")) or model.lower() not in t:
        return False
    if any(word in t for word in config.PHOTO_REJECT_WORDS):
        return False
    return not any(longer.lower() in t for longer in longer_models)


def _pick(pages: list[dict], model: str, start: int, longer_models: list[str]) -> dict | None:
    """Best licensed photo: prefer a year inside the bucket, then any usable one."""
    candidates = []
    for page in pages:
        title = page.get("title", "")
        if not _usable(title, model, longer_models):
            continue
        info = (page.get("imageinfo") or [{}])[0]
        meta = info.get("extmetadata", {})
        license_name = meta.get("LicenseShortName", {}).get("value", "")
        if not license_name.startswith(config.PHOTO_ALLOWED_LICENSE_PREFIXES):
            continue
        author = _strip_html(meta.get("Artist", {}).get("value", "")) or "Unknown author"
        year = _first_year(title.replace("_", " "))
        in_bucket = year is not None and start <= year < start + config.PHOTO_YEAR_BUCKET
        candidates.append((not in_bucket, {"url": info.get("thumburl") or info.get("url"),
                                           "credit": f"{author}, {license_name}, via Wikimedia Commons"}))
    candidates.sort(key=lambda c: c[0])  # stable: in-bucket first, then search rank
    return candidates[0][1] if candidates else None


def fetch_photo(session: requests.Session, key: str, longer_models: list[str]) -> dict | None:
    make, model, start = key.split("|")
    for query in (f"{start} {make} {model}", f"{make} {model}"):
        found = _pick(_search(session, query), model, int(start), longer_models)
        time.sleep(config.PHOTO_REQUEST_PAUSE_S)
        if found:
            return found
    return None


def main(refresh: bool = "--refresh" in sys.argv) -> None:
    data = json.loads(config.ROUNDS_PATH.read_text())
    rounds = data["rounds"]
    cache = {} if refresh else load_cache()
    keys = {photo_key(r["car"]["make"], r["car"]["model"], r["car"]["year"]) for r in rounds}
    todo = sorted(k for k in keys if k not in cache)
    log(STEP, f"{len(keys)} photo keys, {len(todo)} not cached")

    models = {(r["car"]["make"], r["car"]["model"]) for r in rounds}
    if config.CLEAN_PATH.exists():
        clean = pd.read_parquet(config.CLEAN_PATH, columns=["make", "model"]).drop_duplicates()
        models |= set(map(tuple, clean.to_numpy()))
    session = requests.Session()
    session.headers["User-Agent"] = config.WIKIMEDIA_USER_AGENT
    for i, key in enumerate(todo, 1):
        try:
            make, model, _ = key.split("|")
            longer = [m for mk, m in models if mk == make and m != model and model.lower() in m.lower()]
            cache[key] = fetch_photo(session, key, longer)
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
