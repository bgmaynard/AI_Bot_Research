"""
WeBull Screener Integration — Asset Type & Market Cap Classification
====================================================================
Uses WeBull's public search and quote endpoints to classify unknown symbols.

WeBull provides:
  - Stock vs ETF detection (template field)
  - Leveraged / inverse ETF flags (isLeveraged, secType)
  - Market cap from quote data (marketValue field, stocks only)
  - ADR detection

WeBull does NOT provide sector/industry (those endpoints are disabled).
We combine WeBull asset-type/cap with Yahoo Finance sector for full classification.

NO production changes. Research-only classification.
"""

import json
import logging
import time
import uuid
from typing import Optional, Dict, Tuple
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

_SEARCH_URL = "https://quotes-gw.webullbroker.com/api/search/pc/tickers"
_QUOTE_URL = "https://quotes-gw.webullbroker.com/api/stock/tickerRealTime/getQuote"

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "*/*",
    "Content-Type": "application/json",
    "platform": "web",
    "hl": "en",
    "os": "web",
    "app": "global",
    "appid": "webull-webapp",
    "ver": "3.39.18",
    "device-type": "Web",
    "did": str(uuid.uuid4()),
}

# Rate limiting
_MIN_REQUEST_INTERVAL = 0.3  # seconds between requests
_last_request_time = 0.0

# Cache: symbol -> (asset_type, cap_bucket, is_leveraged, is_inverse)
_cache: Dict[str, dict] = {}


# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _rate_limit():
    """Enforce minimum interval between WeBull API calls."""
    global _last_request_time
    now = time.time()
    elapsed = now - _last_request_time
    if elapsed < _MIN_REQUEST_INTERVAL:
        time.sleep(_MIN_REQUEST_INTERVAL - elapsed)
    _last_request_time = time.time()


def _fetch_json(url: str, timeout: int = 5) -> Optional[dict]:
    """GET JSON from WeBull API with rate limiting."""
    _rate_limit()
    try:
        req = Request(url, headers=_HEADERS)
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (URLError, HTTPError, json.JSONDecodeError, OSError) as e:
        log.debug("WeBull fetch failed for %s: %s", url, e)
        return None


def _search_ticker(symbol: str) -> Optional[dict]:
    """Search WeBull for a ticker and return the best match.

    Returns dict with keys: tickerId, template, securityType, secType,
    isLeveraged, isAdr, etc. or None.
    """
    url = f"{_SEARCH_URL}?keyword={symbol}&pageIndex=1&pageSize=5&regionId=6"
    data = _fetch_json(url)
    if not data:
        return None

    results = data.get("data", [])
    if not results:
        return None

    # Find exact symbol match (case-insensitive)
    for r in results:
        if r.get("symbol", "").upper() == symbol.upper():
            return r

    # Fall back to first result
    return results[0] if results else None


def _get_quote(ticker_id: str) -> Optional[dict]:
    """Fetch real-time quote for a ticker ID.

    Returns dict with marketValue, template, and price data.
    """
    url = f"{_QUOTE_URL}?tickerId={ticker_id}&includeSecu=1&includeQuote=1&more=1"
    return _fetch_json(url)


def _parse_cap_bucket(market_value_str: Optional[str]) -> str:
    """Parse WeBull marketValue string into cap bucket."""
    if not market_value_str:
        return "unknown"
    try:
        mv = float(market_value_str)
    except (ValueError, TypeError):
        return "unknown"

    if mv < 300_000_000:
        return "micro"
    elif mv < 2_000_000_000:
        return "small"
    elif mv < 10_000_000_000:
        return "mid"
    else:
        return "large"


_INVERSE_KEYWORDS = {"short", "bear", "inverse", "inv ", "ultra short", "ultrashort"}


def _parse_asset_type(search_result: dict) -> Tuple[str, bool]:
    """Determine asset_type from WeBull search result.

    Returns (asset_type, is_inverse).

    WeBull fields:
      - template: "stock" or "etf"
      - isLeveraged: 0 or 1
      - securitySubType: inconsistent (704=inverse sometimes, but SQQQ=701)
      - name: ETF full name — check for "Short", "Bear", "Inverse" keywords
    """
    template = search_result.get("template", "").lower()
    is_leveraged = search_result.get("isLeveraged", 0) == 1
    sub_type = search_result.get("securitySubType")
    name_lower = search_result.get("name", "").lower()

    if template == "etf":
        # Detect inverse: subType 704 OR name contains inverse keywords
        is_inverse = sub_type == 704 or any(kw in name_lower for kw in _INVERSE_KEYWORDS)

        if is_leveraged and is_inverse:
            return "inverse_etf", True
        elif is_inverse:
            return "inverse_etf", True
        elif is_leveraged:
            return "leveraged_etf", False
        else:
            return "sector_etf", False

    if template == "stock":
        return "common_stock", False

    # SPAC or other
    return "common_stock", False


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def classify_symbol(symbol: str) -> Optional[dict]:
    """Classify a symbol using WeBull search + quote endpoints.

    Returns dict with keys:
        asset_type: str (common_stock, leveraged_etf, inverse_etf, sector_etf)
        cap_bucket: str (micro, small, mid, large, etf, unknown)
        is_inverse: bool
        is_leveraged: bool
        is_adr: bool
        exchange: str
        source: "webull"
        confidence: float (0.6-0.9)

    Returns None if WeBull lookup fails entirely.
    """
    symbol = symbol.upper()

    # Check cache
    if symbol in _cache:
        return _cache[symbol]

    # Step 1: Search for ticker
    search = _search_ticker(symbol)
    if not search:
        log.debug("WeBull: no search result for %s", symbol)
        return None

    ticker_id = search.get("tickerId")
    if not ticker_id:
        return None

    # Step 2: Determine asset type from search data
    asset_type, is_inverse = _parse_asset_type(search)
    is_leveraged = search.get("isLeveraged", 0) == 1
    is_adr = search.get("isAdr", 0) == 1
    exchange = search.get("disExchangeCode", "")

    # Step 3: Get market cap from quote (stocks only)
    cap_bucket = "etf" if asset_type in ("leveraged_etf", "inverse_etf", "sector_etf") else "unknown"

    if asset_type == "common_stock" and ticker_id:
        quote = _get_quote(str(ticker_id))
        if quote:
            mv = quote.get("marketValue")
            cap_bucket = _parse_cap_bucket(mv)

    confidence = 0.8
    if cap_bucket == "unknown":
        confidence = 0.6

    result = {
        "asset_type": asset_type,
        "cap_bucket": cap_bucket,
        "is_inverse": is_inverse,
        "is_leveraged": is_leveraged,
        "is_adr": is_adr,
        "exchange": exchange,
        "source": "webull",
        "confidence": confidence,
    }

    _cache[symbol] = result
    return result


def classify_batch(symbols: list) -> Dict[str, Optional[dict]]:
    """Classify multiple symbols. Rate-limited, sequential.

    Args:
        symbols: List of ticker symbols.

    Returns:
        Dict mapping symbol -> classification dict (or None on failure).
    """
    results = {}
    for sym in symbols:
        results[sym] = classify_symbol(sym)
    return results


def get_cache_stats() -> dict:
    """Return cache statistics."""
    if not _cache:
        return {"cached": 0}

    by_type = {}
    by_cap = {}
    for info in _cache.values():
        at = info["asset_type"]
        cb = info["cap_bucket"]
        by_type[at] = by_type.get(at, 0) + 1
        by_cap[cb] = by_cap.get(cb, 0) + 1

    return {
        "cached": len(_cache),
        "by_asset_type": by_type,
        "by_cap_bucket": by_cap,
    }
