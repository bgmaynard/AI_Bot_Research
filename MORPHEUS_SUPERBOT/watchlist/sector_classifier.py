"""
Sector Classifier — Symbol-to-Sector/Type Mapping Engine
=========================================================
Classifies symbols by asset type, sector, and market cap bucket.

Classification sources (priority order):
  1. Static map (known_symbols.json) — confidence=1.0
  2. Heuristic from signal data — confidence=0.5-0.8
  3. Unknown fallback — confidence=0.3

NO production changes. Research-only classification.
"""

import json
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from pathlib import Path


SUPERBOT = Path(__file__).resolve().parent.parent
KNOWN_SYMBOLS_FILE = Path(__file__).resolve().parent / "known_symbols.json"

VALID_ASSET_TYPES = {
    "common_stock", "leveraged_etf", "inverse_etf",
    "sector_etf", "volatility_etf", "spac",
}
VALID_SECTORS = {
    "energy", "technology", "biotech", "crypto_proxy", "semiconductors",
    "materials", "volatility", "broad_market", "financials", "healthcare",
    "real_estate", "defense", "transportation", "retail", "utilities",
    "bonds", "china", "emerging_markets", "europe", "japan",
    "automotive", "entertainment", "industrials", "consumer_staples",
    "small_cap", "unknown",
    # Added 2026-03-08: new sectors from scanner trade data
    "clean_energy", "mining", "quantum_computing", "aerospace",
    "ai_ml", "airlines",
}
VALID_CAP_BUCKETS = {"micro", "small", "mid", "large", "etf", "unknown"}

# Heuristic patterns for ETF suffix detection
_ETF_SUFFIXES = {"L", "S", "X", "U", "D", "Q"}


@dataclass
class SectorClassification:
    symbol: str
    asset_type: str = "common_stock"
    sector: str = "unknown"
    cap_bucket: str = "unknown"
    leverage_factor: float = 1.0
    inverse: bool = False
    confidence: float = 0.3
    source: str = "unknown"

    def to_dict(self) -> dict:
        return asdict(self)


class SectorClassifierEngine:
    """Classify symbols by asset type, sector, and cap bucket."""

    def __init__(self):
        self._classifications: Dict[str, SectorClassification] = {}
        self._known_symbols: Dict[str, dict] = {}
        self._load_known_symbols()

    def _load_known_symbols(self):
        """Load static symbol map from known_symbols.json."""
        if not KNOWN_SYMBOLS_FILE.exists():
            return
        try:
            with open(KNOWN_SYMBOLS_FILE, encoding="utf-8") as f:
                data = json.load(f)
            self._known_symbols = {
                k: v for k, v in data.items() if not k.startswith("_")
            }
        except (json.JSONDecodeError, IOError):
            pass

    def classify(self, symbol: str, signal_data: Optional[dict] = None) -> SectorClassification:
        """Classify a single symbol. Caches result.

        Args:
            symbol: Ticker symbol (e.g. "SOXS", "BATL")
            signal_data: Optional dict with keys like entry_price, spread_pct,
                         gap_pcts, tags, etc. from signal/classifier data.

        Returns:
            SectorClassification with asset_type, sector, cap_bucket, etc.
        """
        # Return cached if already classified
        if symbol in self._classifications:
            return self._classifications[symbol]

        # Source 1: Static map (highest confidence)
        if symbol in self._known_symbols:
            info = self._known_symbols[symbol]
            leverage = info.get("leverage", 1.0)
            cls = SectorClassification(
                symbol=symbol,
                asset_type=info.get("asset_type", "common_stock"),
                sector=info.get("sector", "unknown"),
                cap_bucket=info.get("cap_bucket", "unknown"),
                leverage_factor=abs(leverage),
                inverse=leverage < 0,
                confidence=1.0,
                source="known_symbols",
            )
            self._classifications[symbol] = cls
            return cls

        # Source 2: WeBull API (asset type + market cap) + Yahoo Finance (sector)
        wb_cls = self._classify_webull(symbol)
        if wb_cls:
            # WeBull doesn't provide sector — try Yahoo Finance to fill it in
            if wb_cls.sector == "unknown":
                yahoo_sector = self._lookup_yahoo_sector(symbol)
                if yahoo_sector:
                    wb_cls = SectorClassification(
                        symbol=symbol,
                        asset_type=wb_cls.asset_type,
                        sector=yahoo_sector,
                        cap_bucket=wb_cls.cap_bucket,
                        leverage_factor=wb_cls.leverage_factor,
                        inverse=wb_cls.inverse,
                        confidence=0.75,
                        source="webull+yahoo",
                    )
            self._classifications[symbol] = wb_cls
            return wb_cls

        # Source 3: Heuristic from signal data
        if signal_data:
            cls = self._classify_heuristic(symbol, signal_data)
            self._classifications[symbol] = cls
            return cls

        # Source 4: Unknown fallback
        cls = SectorClassification(
            symbol=symbol,
            asset_type="common_stock",
            sector="unknown",
            cap_bucket="unknown",
            confidence=0.3,
            source="fallback",
        )
        self._classifications[symbol] = cls
        return cls

    def _lookup_yahoo_sector(self, symbol: str) -> Optional[str]:
        """Try Yahoo Finance to get sector for a symbol. Returns internal sector name or None."""
        try:
            from urllib.request import Request, urlopen
            from urllib.error import URLError, HTTPError
            url = f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{symbol}?modules=assetProfile"
            req = Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
            with urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            profile = data.get("quoteSummary", {}).get("result", [{}])[0].get("assetProfile", {})
            yahoo_sector = profile.get("sector", "").lower()
            # Map Yahoo sector names to internal names
            yahoo_map = {
                "technology": "technology", "healthcare": "healthcare",
                "financial services": "financials", "energy": "energy",
                "consumer cyclical": "retail", "consumer defensive": "consumer_staples",
                "industrials": "industrials", "basic materials": "materials",
                "real estate": "real_estate", "utilities": "utilities",
                "communication services": "entertainment",
            }
            internal = yahoo_map.get(yahoo_sector)
            return internal if internal else None
        except Exception:
            return None

    def _classify_webull(self, symbol: str) -> Optional[SectorClassification]:
        """Classify using WeBull search + quote API.

        WeBull provides asset_type and cap_bucket but NOT sector.
        Sector remains 'unknown' and can be enriched by Yahoo fallback
        on the /api/sector/symbol/{SYM} endpoint.
        """
        try:
            from watchlist.webull_classifier import classify_symbol
            wb = classify_symbol(symbol)
        except Exception:
            return None

        if not wb:
            return None

        leverage = 1.0
        inverse = wb.get("is_inverse", False)
        if wb.get("is_leveraged", False):
            leverage = 2.0  # conservative default
        if inverse:
            leverage = -leverage

        return SectorClassification(
            symbol=symbol,
            asset_type=wb["asset_type"],
            sector="unknown",  # WeBull doesn't provide sector
            cap_bucket=wb["cap_bucket"],
            leverage_factor=abs(leverage),
            inverse=inverse,
            confidence=wb.get("confidence", 0.7),
            source="webull",
        )

    def _classify_heuristic(self, symbol: str, data: dict) -> SectorClassification:
        """Heuristic classification from signal/classifier data."""
        entry_price = data.get("entry_price")
        spread_pct = data.get("spread_pct")
        gap_pcts = data.get("gap_pcts", [])
        tags = data.get("tags", [])
        max_gap = max(gap_pcts) if gap_pcts else None

        asset_type = "common_stock"
        sector = "unknown"
        cap_bucket = "unknown"
        confidence = 0.5
        leverage = 1.0
        inverse = False

        # ETF suffix heuristic: 3-4 char symbols ending in common ETF patterns
        sym_upper = symbol.upper()
        if len(sym_upper) <= 4 and sym_upper[-1] in _ETF_SUFFIXES:
            # Could be an ETF — bump confidence slightly but don't override
            confidence = 0.4

        # Price-based cap bucket
        if entry_price is not None:
            if entry_price < 5:
                cap_bucket = "micro"
                confidence = max(confidence, 0.6)
            elif entry_price < 20:
                cap_bucket = "small"
                confidence = max(confidence, 0.5)
            elif entry_price < 100:
                cap_bucket = "mid"
                confidence = max(confidence, 0.5)
            else:
                cap_bucket = "large"
                confidence = max(confidence, 0.5)

        # Tags-based hints
        tag_lower = set(t.lower() for t in tags)

        # Speculative micro-cap detection
        if max_gap and max_gap > 40 and not tag_lower & {"catalyst", "news", "earnings", "fda", "merger"}:
            cap_bucket = "micro"
            confidence = max(confidence, 0.7)

        # Wide spread + low price = micro-cap
        if spread_pct is not None and spread_pct > 0.8 and entry_price is not None and entry_price < 5:
            cap_bucket = "micro"
            confidence = max(confidence, 0.7)

        # Sector hints from tags
        if tag_lower & {"biotech", "fda"}:
            sector = "biotech"
            confidence = max(confidence, 0.8)
        elif tag_lower & {"crypto", "bitcoin", "btc"}:
            sector = "crypto_proxy"
            confidence = max(confidence, 0.8)
        elif tag_lower & {"oil", "gas", "energy"}:
            sector = "energy"
            confidence = max(confidence, 0.7)

        return SectorClassification(
            symbol=symbol,
            asset_type=asset_type,
            sector=sector,
            cap_bucket=cap_bucket,
            leverage_factor=abs(leverage),
            inverse=inverse,
            confidence=confidence,
            source="heuristic",
        )

    def classify_universe(self, symbols: list, signal_data: Optional[Dict[str, dict]] = None):
        """Classify all symbols in the current universe.

        Args:
            symbols: List of ticker symbols.
            signal_data: Optional dict mapping symbol -> signal data dict.
        """
        signal_data = signal_data or {}
        for sym in symbols:
            self.classify(sym, signal_data.get(sym))

    def get_classification(self, symbol: str) -> Optional[SectorClassification]:
        """Retrieve cached classification for a symbol."""
        return self._classifications.get(symbol)

    def get_group(self, asset_type: Optional[str] = None,
                  sector: Optional[str] = None) -> List[SectorClassification]:
        """Filter classifications by asset_type and/or sector."""
        results = []
        for cls in self._classifications.values():
            if asset_type and cls.asset_type != asset_type:
                continue
            if sector and cls.sector != sector:
                continue
            results.append(cls)
        return results

    def to_dict(self) -> dict:
        """Serialize all classifications."""
        by_type = {}
        by_sector = {}
        for cls in self._classifications.values():
            by_type.setdefault(cls.asset_type, []).append(cls.symbol)
            by_sector.setdefault(cls.sector, []).append(cls.symbol)

        return {
            "total": len(self._classifications),
            "classifications": {
                sym: cls.to_dict()
                for sym, cls in sorted(self._classifications.items())
            },
            "by_asset_type": {k: sorted(v) for k, v in sorted(by_type.items())},
            "by_sector": {k: sorted(v) for k, v in sorted(by_sector.items())},
        }
