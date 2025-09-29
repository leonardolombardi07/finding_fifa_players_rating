from __future__ import annotations

import json
import os
import time
import traceback
import certifi
from typing import Any, Dict, Union, Optional

import requests
from requests import Response, Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from shared_types import Player


# --- Config ---
_API_BASE = "https://drop-api.ea.com/rating/ea-sports-fc"
_DEFAULT_LIMIT_PER_REQUEST = 100
_DEFAULT_GENDER = 0  # Male
_DEFAULT_LOCALE = None
_REQUEST_TIMEOUT = 20  # seconds
_MAX_PAGES = None  # set to an int to hard-cap pages for debugging
_SAVE_PATH = "ea_fc_players.json"
_ERROR_LOG_PATH = "ea_fc_players_errors.txt"


def request_players_from_database(
    *,
    limit: int = _DEFAULT_LIMIT_PER_REQUEST,
    gender: int = _DEFAULT_GENDER,
    locale: Union[str, None] = _DEFAULT_LOCALE,
    save_path: str = _SAVE_PATH,
) -> list[Player]:  # type: ignore[name-defined]
    """
    1) Request players in pages of `limit`
    2) Append obtained players to the growing list
    3) Save the combined data as JSON (atomically)
    4) Keep going using `offset` until there are no more items (or MAX_PAGES is reached)

    Returns:
        list[Player]: All fetched players (as dicts compatible with your Player type).

    Notes:
        - Errors are printed and appended to `_ERROR_LOG_PATH` with full traceback.
        - Uses retries/backoff for transient failures.
    """
    all_players: list[Player] = []
    offset = 0
    pages_fetched = 0

    session = _build_session()

    try:
        while True:
            try:
                payload = _fetch_page(
                    session,
                    limit=limit,
                    offset=offset,
                    gender=gender,
                    locale=locale,
                )
            except Exception as exc:
                _log_error(
                    f"[EA FC Ratings] Request failed at offset={offset}, limit={limit}.",
                    exc,
                )
                # stop on hard failure; caller can inspect saved partials
                break

            # The API response is expected to have "items" (a list of players)
            items: list[Player] = payload.get("items") or []
            if not isinstance(items, list):  # type: ignore
                _log_error(
                    f"[EA FC Ratings] Unexpected payload schema at offset={offset}. "
                    f"Expected 'items' as a list, got: {type(items)!r}"
                )
                break

            if not items:
                # no more pages
                break

            all_players.extend(items)
            pages_fetched += 1

            # Save progress after each page (safer for long runs)
            _save_json(
                {"count": len(all_players), "items": all_players},
                save_path,
            )

            # Prepare next page
            offset += len(
                items
            )  # increment by actual number returned (usually == limit)

            # Optional hard cap for debugging
            if _MAX_PAGES is not None and pages_fetched >= _MAX_PAGES:
                break

            # Be nice to the API
            time.sleep(0.15)

    finally:
        session.close()

    # Final save (already saved per page, but ensure final state)
    try:
        _save_json({"count": len(all_players), "items": all_players}, save_path)
    except Exception as exc:
        _log_error(f"[EA FC Ratings] Failed to save final JSON to {save_path}.", exc)

    # Cast to your Player type at the boundary if you like; we avoid importing it here.
    return all_players  # type: ignore[return-value]


def _build_session() -> Session:
    """Create a requests Session with retry/backoff for resilience."""
    session = requests.Session()
    retries = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "Accept": "*/*",
            "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
            "Connection": "keep-alive",
            "Origin": "https://www.ea.com",
            "Referer": "https://www.ea.com/",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/139.0.0.0 Safari/537.36",
        }
    )

    session.verify = certifi.where()  # <— use certifi bundle explicitly
    return session


def _log_error(msg: str, exc: Optional[BaseException] = None) -> None:
    """Print and append a detailed error (with traceback) to a .txt file."""
    print(msg)
    try:
        with open(_ERROR_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(msg.rstrip() + "\n")
            if exc is not None:
                f.write(traceback.format_exc())
                f.write("\n" + "-" * 80 + "\n")

    except Exception:
        # As a last resort, print any failure writing the log.
        print("Failed writing error log:")
        traceback.print_exc()


def _fetch_page(
    session: Session,
    limit: int,
    offset: int,
    gender: int,
    locale: Union[str, None],
) -> Dict[str, Any]:
    """Fetch a single page from the API and return parsed JSON."""
    params = {
        "limit": str(limit),
        "offset": str(offset),
        "gender": str(gender),
    }
    if locale is not None:
        params["locale"] = locale

    resp: Response = session.get(
        _API_BASE,
        params=params,
        timeout=_REQUEST_TIMEOUT,
        verify=False,
    )
    resp.raise_for_status()
    return resp.json()  # Expected to contain an "items" list


def _save_json(obj: Any, path: str) -> None:
    """Write JSON atomically to avoid partial files."""

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            ensure_ascii=False,
            indent=2,
        )

    os.replace(path, path)
