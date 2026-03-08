# INSTRUCTION: Implement Capped Self-Optimizing Worklist

> **Paste this entire file into the bot's Claude Code session.**
> It is a self-contained specification — every code block is copy-ready.
> Files touched: `worklist/pipeline.py`, `worklist/store.py`, `server/main.py`.
> Files NOT touched: `worklist/scrutiny.py`, `worklist/scoring.py` (existing filters and scoring are correct).

---

## Background

Trade data analysis (2,172 trades, 27 days, IBKR_Algo_BOT_v2) proved that trading too many symbols destroys alpha:

| Slice | PF | Net P&L |
|---|---|---|
| Top 15 symbols | 3.76 | +$1,205 |
| All 192 symbols | 0.59 | -$3,671 |

The bottom 167 symbols destroyed $5,144 in alpha. Only stocks moving 50%+ on the day were profitable (PF 1.39).

**The worklist must be HARD-CAPPED at 15 symbols with automatic attrition** — new qualifiers displace the worst performer. Manual force-adds must go through the same scrutiny pipeline as scanner picks (no more `score=100` bypass).

---

## Change 1 — Hard cap the worklist at 15 active symbols

### 1a. Config

Add `max_active_symbols` to the pipeline config (dataclass or dict, wherever config lives):

```python
max_active_symbols: int = 15
```

If the config is a dataclass, add the field. If it is a plain dict, set the key. The value **15** is non-negotiable — it comes directly from trade data analysis.

### 1b. Displacement logic in `process_scanner_row()`

In `worklist/pipeline.py`, inside `process_scanner_row()`, **after** the new symbol passes scrutiny and its `combined_priority_score` (or equivalent composite score) is computed, but **before** the entry is created and added to the store, insert:

```python
# --- CAP ENFORCEMENT (hard cap = 15) ---
candidates = self._store.get_candidates()       # active + candidate entries for today
if len(candidates) >= self._config.max_active_symbols:
    lowest = self._store.get_lowest_scorer()
    if lowest is None:
        # Shouldn't happen, but guard
        logger.warning(f"[WORKLIST] Cap reached but no lowest scorer found")
        return None

    if combined_score > lowest.combined_priority_score:
        # Displace the lowest scorer
        self._store.update_status(
            lowest.symbol,
            WorklistStatus.EXPIRED,
            reason=f"Displaced by {symbol} "
                   f"(score {combined_score:.1f} > {lowest.combined_priority_score:.1f})"
        )
        self._displacement_log.append({
            "displaced": lowest.symbol,
            "displaced_score": lowest.combined_priority_score,
            "replaced_by": symbol,
            "new_score": combined_score,
            "at": datetime.now(timezone.utc).isoformat(),
        })
        logger.info(
            f"[WORKLIST] DISPLACEMENT: {lowest.symbol} "
            f"(score={lowest.combined_priority_score:.1f}) removed, "
            f"replaced by {symbol} (score={combined_score:.1f})"
        )
    else:
        # New symbol doesn't beat the worst — reject
        self._rejection_log.append({
            "symbol": symbol,
            "score": combined_score,
            "reason": "BELOW_CUTOFF",
            "cutoff_symbol": lowest.symbol,
            "cutoff_score": lowest.combined_priority_score,
            "at": datetime.now(timezone.utc).isoformat(),
        })
        logger.info(
            f"[WORKLIST] BELOW_CUTOFF: {symbol} "
            f"(score={combined_score:.1f}) < worst active "
            f"{lowest.symbol} (score={lowest.combined_priority_score:.1f})"
        )
        await self._emit(
            EventType.WORKLIST_REJECTED,
            symbol=symbol,
            payload={
                "symbol": symbol,
                "rejection_reason": "BELOW_CUTOFF",
                "score": combined_score,
                "cutoff_symbol": lowest.symbol,
                "cutoff_score": lowest.combined_priority_score,
            },
        )
        return None
# --- END CAP ENFORCEMENT ---
# (proceed to create WorklistEntry and add to store)
```

### 1c. Add log lists to pipeline `__init__`

```python
self._displacement_log: list[dict] = []
self._rejection_log: list[dict] = []
```

Expose them in whatever status/diagnostics endpoint already exists so you can verify the cap is working.

### 1d. Apply same logic in `process_news()`

If `process_news()` (or any other method) also adds symbols to the worklist, apply the **identical** cap-enforcement block above. The cleanest approach: extract the cap check into a private method `_enforce_cap(symbol, combined_score) -> bool` and call it from both places.

---

## Change 2 — Add `get_lowest_scorer()` to WorklistStore

In `worklist/store.py`, add this method to the `WorklistStore` class:

```python
def get_lowest_scorer(self) -> Optional[WorklistEntry]:
    """Return the lowest-scoring active/candidate entry for today's session.

    Used by the pipeline's displacement logic to decide who gets evicted
    when the worklist is at capacity.
    """
    with self._lock:
        tradeable = [
            e for e in self._entries.values()
            if e.status in (WorklistStatus.CANDIDATE.value, WorklistStatus.ACTIVE.value)
            and e.session_date == self._session_date
        ]
        if not tradeable:
            return None
        return min(tradeable, key=lambda e: e.combined_priority_score)
```

This method is thread-safe (uses the existing `_lock`) and only considers today's session entries.

---

## Change 3 — Manual adds go through the FULL vetting pipeline

**Replace** the `/api/scanner/force-add/{symbol}` endpoint in `server/main.py`. The old implementation bypasses scrutiny and sets `score=100`. The new implementation fetches real market data and calls `process_scanner_row()` — exactly the same path as a scanner discovery.

```python
@app.post("/api/scanner/force-add/{symbol}")
async def force_add_symbol(symbol: str, reason: str = "manual"):
    """
    Add a symbol through the SAME vetting pipeline as scanner picks.

    Fetches current market data, runs scrutiny and scoring, applies displacement
    if worklist is at capacity. No bypass — manual adds are treated identically
    to scanner discoveries.
    """
    symbol = symbol.upper().strip()

    try:
        # Step 1: Get current market data from quotes/streamer
        price = 0.0
        volume = 0
        rvol = 0.0
        spread_pct = None
        gap_pct = 0.0
        day_change = 0.0

        # Try the streamer first
        if server._streamer_available and server._streamer:
            quote = server._streamer.get_latest_quote(symbol)
            if quote:
                price = quote.get("last", 0.0) or quote.get("price", 0.0)
                volume = quote.get("volume", 0)
                spread_pct = quote.get("spread_pct")
                day_change = quote.get("change_pct", 0.0)
                gap_pct = day_change  # Use day change as gap proxy

        # Fallback: try pipeline's existing data
        if price <= 0 and server._pipeline_available and server._pipeline:
            sym_data = server._pipeline.get_symbol_data(symbol)
            if sym_data:
                price = sym_data.get("price", 0.0)
                volume = sym_data.get("volume", 0)

        if price <= 0:
            return {
                "success": False,
                "symbol": symbol,
                "reason": "NO_MARKET_DATA",
                "details": "Cannot fetch current price/volume for symbol. "
                           "Ensure the symbol is valid and market data is available."
            }

        # Conservative RVOL default for manual adds (no historical avg available)
        if rvol <= 0:
            rvol = 2.0

        # Step 2: Subscribe to data feeds (so we can trade it if accepted)
        server.state.subscribed_symbols.add(symbol)
        if server._streamer_available and server._use_streaming and server._streamer:
            await server._streamer.subscribe_quotes([symbol])

        # Step 3: Process through the SAME worklist pipeline as scanner picks
        if server._worklist_available and server._worklist_pipeline:
            entry = await server._worklist_pipeline.process_scanner_row(
                symbol=symbol,
                scanner_score=60.0,       # Neutral-positive (user spotted it for a reason)
                gap_pct=gap_pct,
                rvol=rvol,
                volume=volume,
                price=price,
                spread_pct=spread_pct,
                news_present=True,        # Manual add implies user sees something noteworthy
                scanner_news_indicator=False,  # Don't fake scanner authority
            )

            if entry:
                # Passed scrutiny — wire into signal pipeline + data feeds
                if server._pipeline_available and server._pipeline:
                    server._pipeline.add_symbol(symbol)
                    asyncio.create_task(server._warmup_pipeline_symbol(symbol))

                if server._databento_feed:
                    server._databento_feed.subscribe([symbol])

                server._save_persistent_watchlist()

                logger.info(
                    f"[MANUAL-ADD] {symbol} ACCEPTED: "
                    f"score={entry.combined_priority_score:.1f}, "
                    f"price=${price:.2f}, vol={volume:,}, rvol={rvol:.1f}x"
                )

                return {
                    "success": True,
                    "symbol": symbol,
                    "score": entry.combined_priority_score,
                    "status": entry.status,
                    "momentum_state": entry.momentum_state,
                    "trigger_reason": entry.trigger_reason,
                    "source": "manual",
                    "pipeline": True,
                }
            else:
                # Failed scrutiny — return reason
                logger.info(f"[MANUAL-ADD] {symbol} REJECTED by scrutiny")
                return {
                    "success": False,
                    "symbol": symbol,
                    "reason": "FAILED_SCRUTINY",
                    "details": "Symbol did not pass worklist scrutiny filters "
                               "(check price, volume, RVOL, spread thresholds). "
                               f"Price=${price:.2f}, Vol={volume:,}, RVOL={rvol:.1f}x"
                }
        else:
            return {
                "success": False,
                "symbol": symbol,
                "reason": "WORKLIST_UNAVAILABLE",
                "details": "Worklist pipeline not initialized"
            }

    except Exception as e:
        logger.error(f"Manual add error for {symbol}: {e}", exc_info=True)
        return {"success": False, "symbol": symbol, "error": str(e)}
```

**Key differences from the old force-add:**
- No `score=100` — real score computed by pipeline
- Scrutiny filters applied (price, volume, spread, RVOL gates)
- Cap enforcement and displacement apply
- Returns rich diagnostics on failure

---

## Change 4 — Update batch endpoint to use same flow

Replace the `/api/scanner/force-add-batch` endpoint to call the new `force_add_symbol` in a loop:

```python
@app.post("/api/scanner/force-add-batch")
async def force_add_batch(symbols: list[str], reason: str = "manual"):
    """Add multiple symbols through the standard vetting pipeline."""
    results = {"accepted": [], "rejected": [], "errors": []}

    for sym in symbols:
        try:
            result = await force_add_symbol(sym, reason=reason)
            if result.get("success"):
                results["accepted"].append({
                    "symbol": sym,
                    "score": result.get("score"),
                })
            else:
                results["rejected"].append({
                    "symbol": sym,
                    "reason": result.get("reason", "unknown"),
                    "details": result.get("details", ""),
                })
        except Exception as e:
            results["errors"].append({"symbol": sym, "error": str(e)})

    logger.info(
        f"[MANUAL-ADD-BATCH] {len(results['accepted'])} accepted, "
        f"{len(results['rejected'])} rejected, {len(results['errors'])} errors"
    )
    return results
```

Note: because the cap is 15, if you batch-add 20 symbols, only the top 15 by score will survive — displacement happens naturally as each symbol is processed sequentially.

---

## Change 5 — Continuous attrition via `rescore_and_trim()`

The scanner already runs continuously via MAX_AI poller. With the displacement logic in `process_scanner_row()`, new high-quality symbols will automatically push out weak ones. No additional scanning code is needed.

However, add a periodic re-score to catch **decaying** symbols whose score has dropped since they were added. In `worklist/pipeline.py`, add:

```python
async def rescore_and_trim(self):
    """
    Periodic re-evaluation of worklist entries.

    Call every 5-10 minutes during market hours.
    Removes symbols whose score has decayed below 80% of the min threshold.
    This opens slots for new, higher-quality discoveries.
    """
    candidates = self._store.get_candidates()
    removed = 0
    threshold = self._config.min_score_for_trading * 0.8

    for entry in candidates:
        if entry.combined_priority_score < threshold:
            self._store.update_status(
                entry.symbol,
                WorklistStatus.EXPIRED,
                reason=f"Score decayed to {entry.combined_priority_score:.1f} "
                       f"(threshold={threshold:.1f})"
            )
            self._displacement_log.append({
                "displaced": entry.symbol,
                "displaced_score": entry.combined_priority_score,
                "replaced_by": None,
                "reason": "DECAY",
                "at": datetime.now(timezone.utc).isoformat(),
            })
            removed += 1
            logger.info(
                f"[WORKLIST] DECAY_REMOVAL: {entry.symbol} "
                f"score={entry.combined_priority_score:.1f} < {threshold:.1f}"
            )

    if removed:
        logger.info(
            f"[WORKLIST] Trimmed {removed} decayed symbols, "
            f"{len(candidates) - removed} remaining"
        )
```

**Wire this into the server's periodic task loop.** Wherever the existing heartbeat / status-check / watchlist-refresh timer runs, add:

```python
# In the periodic task that already runs every N minutes:
if server._worklist_available and server._worklist_pipeline:
    await server._worklist_pipeline.rescore_and_trim()
```

If no periodic task exists yet, create one with `asyncio.create_task` that loops every 5 minutes during market hours (09:30-16:00 ET).

---

## Change 6 — Persistent watchlist respects the cap on startup

When restoring the persistent watchlist on startup in `server/main.py`, **only restore the top N symbols by score** — not all of them. Find the restore logic (likely in `_load_persistent_watchlist()` or server startup) and apply:

```python
def _load_persistent_watchlist(self) -> list[str]:
    """Load persistent watchlist, respecting the active symbol cap."""
    path = self._persistent_watchlist_path
    if not path or not os.path.exists(path):
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to load persistent watchlist: {e}")
        return []

    # data may be a list of dicts with "symbol" and "score" keys,
    # or a list of strings — handle both formats
    entries = []
    for item in data:
        if isinstance(item, dict):
            entries.append(item)
        elif isinstance(item, str):
            entries.append({"symbol": item, "score": 0.0})

    # Sort by score descending, take only top max_active_symbols
    cap = self._config.max_active_symbols if hasattr(self, '_config') else 15
    entries.sort(key=lambda e: e.get("score", 0.0), reverse=True)
    top_entries = entries[:cap]

    restored = [e["symbol"] for e in top_entries if "symbol" in e]
    if len(entries) > len(restored):
        logger.info(
            f"[WORKLIST] Persistent watchlist had {len(entries)} symbols, "
            f"restored top {len(restored)} (cap={cap})"
        )

    return restored
```

Similarly, when **saving** the persistent watchlist, include the score so the restore logic can sort:

```python
def _save_persistent_watchlist(self):
    """Save current worklist entries with scores for ranked restore."""
    if not self._persistent_watchlist_path:
        return

    if self._worklist_available and self._worklist_pipeline:
        candidates = self._worklist_pipeline._store.get_candidates()
        data = [
            {"symbol": e.symbol, "score": e.combined_priority_score}
            for e in sorted(candidates, key=lambda e: -e.combined_priority_score)
        ]
    else:
        data = [{"symbol": s, "score": 0.0} for s in self.state.subscribed_symbols]

    try:
        with open(self._persistent_watchlist_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except OSError as e:
        logger.warning(f"Failed to save persistent watchlist: {e}")
```

---

## Summary of Changes

| File | What Changes |
|---|---|
| `worklist/pipeline.py` | Add cap enforcement block in `process_scanner_row()` (and `process_news()` if applicable). Add `rescore_and_trim()`. Add `_displacement_log` / `_rejection_log`. Set `max_active_symbols=15` in config. |
| `worklist/store.py` | Add `get_lowest_scorer()` method to `WorklistStore`. |
| `server/main.py` | Replace `force_add_symbol` endpoint (full pipeline, no bypass). Replace `force_add_batch` endpoint. Cap-aware `_load_persistent_watchlist()`. Score-preserving `_save_persistent_watchlist()`. Wire `rescore_and_trim()` into periodic task. |
| `worklist/scrutiny.py` | **No changes** — existing filters are correct. |
| `worklist/scoring.py` | **No changes** — existing scoring is correct. |

---

## Testing

### Unit tests

```python
# 1. Add symbol when worklist has room (< 15) — should accept
async def test_add_below_cap():
    pipeline = make_pipeline(max_active_symbols=15)
    for i in range(10):
        await pipeline.process_scanner_row(symbol=f"SYM{i}", scanner_score=70, ...)
    entry = await pipeline.process_scanner_row(symbol="NEW", scanner_score=65, ...)
    assert entry is not None
    assert len(pipeline._store.get_candidates()) == 11


# 2. Add symbol that outscores lowest when at cap — should displace
async def test_displacement_when_at_cap():
    pipeline = make_pipeline(max_active_symbols=15)
    # Fill with scores 50..64
    for i in range(15):
        await pipeline.process_scanner_row(symbol=f"SYM{i}", scanner_score=50 + i, ...)
    assert len(pipeline._store.get_candidates()) == 15

    # Add a strong newcomer (score ~80)
    entry = await pipeline.process_scanner_row(symbol="STRONG", scanner_score=90, ...)
    assert entry is not None
    assert entry.symbol == "STRONG"
    # SYM0 (score ~50) should be displaced
    assert len(pipeline._store.get_candidates()) == 15
    symbols = {e.symbol for e in pipeline._store.get_candidates()}
    assert "SYM0" not in symbols
    assert "STRONG" in symbols


# 3. Add symbol that does NOT outscore lowest when at cap — should reject
async def test_reject_below_cutoff():
    pipeline = make_pipeline(max_active_symbols=15)
    for i in range(15):
        await pipeline.process_scanner_row(symbol=f"SYM{i}", scanner_score=70 + i, ...)
    # Try adding a weak symbol (score ~40)
    entry = await pipeline.process_scanner_row(symbol="WEAK", scanner_score=30, ...)
    assert entry is None
    assert len(pipeline._store.get_candidates()) == 15


# 4. Manual add with bad data — should be rejected by scrutiny
async def test_manual_add_bad_price():
    # Mock streamer to return price=$0.50 (below min price gate)
    response = await client.post("/api/scanner/force-add/PENNY")
    assert response.json()["success"] is False
    assert response.json()["reason"] in ("FAILED_SCRUTINY", "NO_MARKET_DATA")
```

### Integration tests

```python
# 5. Manual add via API — goes through full pipeline, returns real score
async def test_manual_add_full_pipeline():
    # Mock streamer to return realistic data (price=$45, vol=2M, rvol=3.5)
    response = await client.post("/api/scanner/force-add/AAPL")
    data = response.json()
    assert data["success"] is True
    assert data["pipeline"] is True          # Confirms it went through pipeline
    assert 0 < data["score"] < 100           # Real score, not 100
    assert data["source"] == "manual"


# 6. Scanner discovers new symbol that beats worst — auto-displacement logged
async def test_scanner_auto_displacement():
    # Fill worklist to 15, then simulate scanner finding a strong symbol
    pipeline = make_pipeline(max_active_symbols=15)
    fill_worklist(pipeline, count=15, score_range=(50, 65))
    entry = await pipeline.process_scanner_row(symbol="HOT", scanner_score=95, ...)
    assert entry is not None
    assert len(pipeline._displacement_log) >= 1
    last_disp = pipeline._displacement_log[-1]
    assert last_disp["replaced_by"] == "HOT"
```

### Decay test

```python
# 7. Symbol score drops below threshold — removed in next rescore cycle
async def test_decay_removal():
    pipeline = make_pipeline(max_active_symbols=15, min_score_for_trading=50)
    entry = await pipeline.process_scanner_row(symbol="FADING", scanner_score=70, ...)
    assert entry is not None

    # Simulate score decay (manually set the stored entry's score)
    stored = pipeline._store._entries["FADING"]
    stored.combined_priority_score = 35.0    # Below 50 * 0.8 = 40

    await pipeline.rescore_and_trim()

    remaining = {e.symbol for e in pipeline._store.get_candidates()}
    assert "FADING" not in remaining
```

---

## Checklist (for the implementing Claude session)

- [ ] `max_active_symbols = 15` added to pipeline config
- [ ] Cap enforcement block added to `process_scanner_row()`
- [ ] Same cap enforcement applied to `process_news()` (if it adds symbols)
- [ ] `get_lowest_scorer()` added to `WorklistStore`
- [ ] `_displacement_log` and `_rejection_log` initialized in pipeline `__init__`
- [ ] Force-add endpoint replaced (no more `score=100` bypass)
- [ ] Batch endpoint updated to call new force-add
- [ ] `rescore_and_trim()` added and wired into periodic task
- [ ] `_load_persistent_watchlist()` caps restore to top N by score
- [ ] `_save_persistent_watchlist()` includes score in saved data
- [ ] All 7 tests pass
