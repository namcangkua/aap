import os, json, time, threading, requests, numpy as np
from datetime import datetime
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
GROQ_API_KEY        = os.getenv("GROQ_API_KEY")
GROQ_MODEL          = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
TELEGRAM_BOT_TOKEN  = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID    = str(os.getenv("TELEGRAM_CHAT_ID", ""))
POLYMARKET_PK       = os.getenv("POLYMARKET_PRIVATE_KEY")
POLYMARKET_FUNDER   = os.getenv("POLYMARKET_FUNDER_ADDRESS")
MAX_POSITION_USDC   = float(os.getenv("MAX_POSITION_USDC", "3.0"))
MAX_DAILY_LOSS_USDC = float(os.getenv("MAX_DAILY_LOSS_USDC", "3.0"))
AGENT_LOOP_INTERVAL = int(os.getenv("AGENT_LOOP_INTERVAL", "60"))
MIN_MARKET_VOLUME   = float(os.getenv("MIN_MARKET_VOLUME", "5000"))
PROFIT_TARGET_MIN   = float(os.getenv("PROFIT_TARGET_MIN", "0.15"))  # 15% min profit to consider exit
STOP_LOSS           = float(os.getenv("STOP_LOSS", "0.20"))          # 20% stop loss
MISPRICING_MIN      = float(os.getenv("MISPRICING_MIN", "0.08"))     # min 8% mispricing to snipe

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API  = "https://clob.polymarket.com"

# ── State ─────────────────────────────────────────────────────────────────────
_positions: dict  = {}   # token_id → position info
_daily_pnl: float = 0.0
_trade_count: int = 0
_start_time       = datetime.now()
_clob_client      = None
_paused: bool     = False
_tg_offset: int   = 0


# ══════════════════════════════════════════════════════════════════════════════
# TELEGRAM
# ══════════════════════════════════════════════════════════════════════════════
def tg(msg: str, chat_id: str = None):
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": chat_id or TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception as e:
        print(f"[TG] {e}")


def handle_status(chat_id):
    elapsed = (datetime.now() - _start_time).total_seconds()
    tph = (_trade_count / max(elapsed, 1)) * 3600
    pos_lines = "".join(
        f"\n• {p['direction']} {p['size']} @ ${p['entry_price']:.3f} | "
        f"now ${p.get('current_price', p['entry_price']):.3f} | "
        f"pnl={((p.get('current_price', p['entry_price'])/p['entry_price'])-1)*100:+.1f}%"
        for p in _positions.values()
    )
    tg(
        f"📊 <b>Scalper Status</b>\n"
        f"State: {'⏸ PAUSED' if _paused else '🎯 SNIPING'}\n"
        f"Open: {len(_positions)}/{int(os.getenv('MAX_OPEN_POSITIONS','3'))}\n"
        f"Trades: {_trade_count} ({tph:.1f}/hr)\n"
        f"Daily PnL: ${_daily_pnl:.4f}\n"
        f"Uptime: {elapsed/60:.1f}min"
        + (f"\n\n<b>Positions:</b>{pos_lines}" if pos_lines else "\n\nNo open positions"),
        chat_id,
    )


def handle_balance(chat_id):
    try:
        b = get_usdc_balance()
        if b <= 0:
            tg(f"⚠️ <b>Balance Zero</b>\n💰 ${b:.4f} USDC\n\nCheck POLYMARKET_FUNDER address or deposit funds.", chat_id)
        else:
            tg(f"💰 <b>Balance</b>\n${b:.4f} USDC\nMax/trade: ${MAX_POSITION_USDC}\nStop loss: -${MAX_DAILY_LOSS_USDC}", chat_id)
    except Exception as e:
        error_msg = str(e)[:150]
        print(f"[HANDLE_BALANCE] Error: {error_msg}")
        tg(f"❌ <b>Balance Check Failed</b>\n{error_msg}\n\nCheck API keys & Polymarket connection.", chat_id)


def handle_pause(chat_id):
    global _paused
    _paused = True
    tg("⏸ <b>PAUSED</b> — /resume to continue", chat_id)


def handle_resume(chat_id):
    global _paused
    _paused = False
    tg("🎯 <b>RESUMED</b> — Sniping active", chat_id)


def handle_report(chat_id):
    elapsed = (datetime.now() - _start_time).total_seconds()
    tph = (_trade_count / max(elapsed, 1)) * 3600
    tg(
        f"📈 <b>Report</b>\n"
        f"⏱ Uptime: {elapsed/60:.1f}min\n"
        f"🔄 Trades: {_trade_count} ({tph:.1f}/hr)\n"
        f"📂 Open: {len(_positions)}\n"
        f"💵 Daily PnL: ${_daily_pnl:.4f}\n"
        f"🛡 Loss left: ${round(MAX_DAILY_LOSS_USDC+_daily_pnl,2)}\n"
        f"📡 {'⏸ Paused' if _paused else '🎯 Sniping'}",
        chat_id,
    )


def handle_unknown(chat_id):
    tg(
        "🤖 <b>Apex Scalper</b>\n\n"
        "/status — posisi & P&L\n"
        "/balance — saldo USDC\n"
        "/pause — pause\n"
        "/resume — lanjut\n"
        "/report — performa",
        chat_id,
    )


COMMAND_MAP = {
    "/status": handle_status, "/balance": handle_balance,
    "/pause": handle_pause, "/resume": handle_resume,
    "/report": handle_report, "/start": handle_unknown, "/help": handle_unknown,
}


def poll_telegram():
    global _tg_offset
    while True:
        try:
            r = requests.get(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates",
                params={"offset": _tg_offset, "timeout": 10},
                timeout=15,
            )
            for update in r.json().get("result", []):
                _tg_offset = update["update_id"] + 1
                msg     = update.get("message", {})
                text    = msg.get("text", "").strip().lower().split()[0]
                chat_id = str(msg.get("chat", {}).get("id", ""))
                if chat_id != TELEGRAM_CHAT_ID:
                    continue
                COMMAND_MAP.get(text, handle_unknown)(chat_id)
        except Exception as e:
            print(f"[TG POLL] {e}")
        time.sleep(2)


# ══════════════════════════════════════════════════════════════════════════════
# CLOB CLIENT — Magic Link = signature_type=1
# ══════════════════════════════════════════════════════════════════════════════
def get_clob_client():
    global _clob_client
    if _clob_client is None:
        from py_clob_client.client import ClobClient
        from py_clob_client.constants import POLYGON
        _clob_client = ClobClient(
            host=CLOB_API,
            key=POLYMARKET_PK,
            chain_id=POLYGON,
            signature_type=1,    # Magic Link / email login
            funder=POLYMARKET_FUNDER if POLYMARKET_FUNDER else None,
        )
        _clob_client.set_api_creds(_clob_client.create_or_derive_api_creds())
    return _clob_client


def get_usdc_balance() -> float:
    try:
        # Use correct method: get_balance_native() returns balance in wei
        raw = get_clob_client().get_balance_native()
        # Convert from wei (1e6 per USDC on Polygon)
        val = float(raw)
        result = val / 1e6 if val > 100 else val
        print(f"[BALANCE] Retrieved: {result} USDC (raw: {raw})")
        return result
    except AttributeError:
        # Fallback: if get_balance_native not available, try direct CLOB API call
        try:
            client = get_clob_client()
            user = client.get_user()
            usdc_balance = user.get('balances', {}).get('USDC', 0)
            result = float(usdc_balance) / 1e6 if usdc_balance > 100 else float(usdc_balance)
            print(f"[BALANCE] Retrieved via user API: {result} USDC")
            return result
        except Exception as e2:
            print(f"[BALANCE ERROR] Fallback failed: {e2}")
            raise
    except Exception as e:
        print(f"[BALANCE ERROR] Failed to get balance: {e}")
        raise


# ══════════════════════════════════════════════════════════════════════════════
# MISPRICING SCORER
# Detects fresh crypto markets where odds are "wrong" = opportunity to snipe
# ══════════════════════════════════════════════════════════════════════════════
def score_mispricing(market: dict) -> dict:
    """
    Score how mispriced a market is.
    Returns score 0-1 and recommended direction.
    Higher score = bigger opportunity.
    """
    yes_price  = float(market.get("yes_price", 0.5))
    no_price   = float(market.get("no_price", 0.5))
    volume     = float(market.get("volume_24h", 0))
    spread     = float(market.get("spread", 0.05))
    age_hours  = float(market.get("age_hours", 999))
    ob_imbal   = float(market.get("ob_imbalance", 0.5))

    # Fresh markets (< 24h old) = more likely mispriced
    freshness = max(0, 1 - (age_hours / 24))

    # Extreme prices = potential mispricing
    # Market at 0.15 YES = either huge opportunity or obvious NO
    # We look for prices in ranges that suggest early/thin trading
    if yes_price <= 0.25:
        # Cheap YES — could rocket if news comes
        price_score = 0.8
        direction   = "YES"
        entry_price = yes_price
    elif yes_price >= 0.75:
        # Cheap NO — could rocket if narrative shifts
        price_score = 0.8
        direction   = "NO"
        entry_price = no_price
    elif 0.35 <= yes_price <= 0.65:
        # Near 50/50 — less edge for scalping
        price_score = 0.3
        direction   = "YES" if ob_imbal > 0.5 else "NO"
        entry_price = yes_price if direction == "YES" else no_price
    else:
        price_score = 0.5
        direction   = "YES" if yes_price < 0.5 else "NO"
        entry_price = yes_price if direction == "YES" else no_price

    # Low volume = thin market = easier to move
    vol_score = max(0, 1 - (volume / 50000))

    # Tight spread = liquid enough to enter/exit
    spread_score = max(0, 1 - (spread / 0.10))

    # Order book imbalance = momentum signal
    # If buying YES: want high imbalance (buyers > sellers)
    if direction == "YES":
        momentum = ob_imbal
    else:
        momentum = 1 - ob_imbal

    # Combined score
    score = (
        freshness   * 0.30 +
        price_score * 0.30 +
        vol_score   * 0.15 +
        spread_score* 0.10 +
        momentum    * 0.15
    )

    return {
        "score":       round(score, 3),
        "direction":   direction,
        "entry_price": round(entry_price, 4),
        "snipe":       score >= MISPRICING_MIN and spread < 0.08,
    }


# ══════════════════════════════════════════════════════════════════════════════
# TOOLS
# ══════════════════════════════════════════════════════════════════════════════
def tool_scan_fresh_markets(limit: int = None) -> str:
    if limit is None:
        limit = 30
    """Scan for fresh crypto markets sorted by mispricing score."""
    try:
        r = requests.get(
            f"{GAMMA_API}/markets",
            params={
                "active": "true", "closed": "false",
                "limit": limit, "tag": "crypto",
                "order": "startDate", "ascending": "false",
            },
            timeout=15,
        )
        r.raise_for_status()
        data    = r.json()
        markets = data if isinstance(data, list) else data.get("markets", [])

        result = []
        now    = datetime.now(tz=None)

        for m in markets:
            vol = float(m.get("volume24hr") or 0)
            if vol < MIN_MARKET_VOLUME:
                continue

            tokens    = m.get("tokens", [])
            yes_price = no_price = 0.5
            yes_tid   = no_tid = ""
            for t in tokens:
                if t.get("outcome") == "Yes":
                    yes_price = float(t.get("price") or 0.5)
                    yes_tid   = t.get("token_id", "")
                elif t.get("outcome") == "No":
                    no_price = float(t.get("price") or 0.5)
                    no_tid   = t.get("token_id", "")

            # Age in hours
            start_str = m.get("startDate") or m.get("createdAt") or ""
            age_hours = 999.0
            if start_str:
                try:
                    start = datetime.fromisoformat(start_str.replace("Z", "+00:00")).replace(tzinfo=None)
                    age_hours = (now - start).total_seconds() / 3600
                except Exception:
                    pass

            # Order book imbalance
            ob_imbal = 0.5
            spread   = 0.05
            if yes_tid:
                try:
                    book    = requests.get(f"{CLOB_API}/book", params={"token_id": yes_tid}, timeout=8).json()
                    bids    = book.get("bids", [])
                    asks    = book.get("asks", [])
                    bv      = sum(float(b.get("size", 0)) for b in bids)
                    av      = sum(float(a.get("size", 0)) for a in asks)
                    ob_imbal = bv / (bv + av) if (bv + av) > 0 else 0.5
                    if bids and asks:
                        spread = float(asks[0]["price"]) - float(bids[0]["price"])
                except Exception:
                    pass

            mkt = {
                "market_id":    m.get("conditionId", ""),
                "question":     m.get("question", "")[:80],
                "yes_price":    round(yes_price, 4),
                "no_price":     round(no_price, 4),
                "yes_token_id": yes_tid,
                "no_token_id":  no_tid,
                "volume_24h":   vol,
                "age_hours":    round(age_hours, 1),
                "spread":       round(spread, 4),
                "ob_imbalance": round(ob_imbal, 3),
            }
            scored = score_mispricing(mkt)
            mkt.update(scored)
            result.append(mkt)

        # Sort by score descending
        result.sort(key=lambda x: x["score"], reverse=True)
        top = result[:10]

        return json.dumps({
            "status": "ok",
            "total_scanned": len(result),
            "top_opportunities": top,
        })
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


def tool_get_price(token_id: str) -> str:
    """Get current live price of a token."""
    try:
        mid  = requests.get(f"{CLOB_API}/midpoint", params={"token_id": token_id}, timeout=8).json()
        book = requests.get(f"{CLOB_API}/book",     params={"token_id": token_id}, timeout=8).json()
        bids = book.get("bids", [])
        asks = book.get("asks", [])
        best_bid = float(bids[0]["price"]) if bids else 0.0
        best_ask = float(asks[0]["price"]) if asks else 1.0
        return json.dumps({
            "status":    "ok",
            "token_id":  token_id,
            "mid":       float(mid.get("mid", 0.5)),
            "best_bid":  best_bid,
            "best_ask":  best_ask,
            "spread":    round(best_ask - best_bid, 4),
        })
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


def tool_snipe_entry(
    market_id: str,
    token_id: str,
    direction: str,
    price: float,
    usdc_amount: float,
) -> str:
    """
    Snipe entry: BUY token at current ask price.
    usdc_amount = how much USDC to use for this position.
    """
    global _positions, _trade_count
    try:
        if _paused:
            return json.dumps({"status": "skipped", "reason": "agent_paused"})
        max_pos = int(os.getenv("MAX_OPEN_POSITIONS", "3"))
        if len(_positions) >= max_pos:
            return json.dumps({"status": "skipped", "reason": f"max_positions:{max_pos}"})

        balance = get_usdc_balance()
        if balance < 1.0:
            return json.dumps({"status": "skipped", "reason": f"balance_too_low:${balance:.4f}"})

        usdc_to_use = min(float(usdc_amount), MAX_POSITION_USDC, balance * 0.9)
        price       = float(price)
        if not (0 < price < 1):
            return json.dumps({"status": "error", "message": f"invalid_price:{price}"})

        size = round(usdc_to_use / price, 2)
        if size < 1.0:
            return json.dumps({"status": "skipped", "reason": f"size_too_small:{size}"})

        from py_clob_client.clob_types import OrderArgs, OrderType
        from py_clob_client.order_builder.constants import BUY
        client   = get_clob_client()
        signed   = client.create_order(OrderArgs(token_id=token_id, price=price, size=size, side=BUY))
        resp     = client.post_order(signed, OrderType.GTC)
        order_id = resp.get("orderID") or resp.get("id", "unknown")

        _positions[token_id] = {
            "market_id":     market_id,
            "token_id":      token_id,
            "direction":     direction,
            "size":          size,
            "entry_price":   price,
            "current_price": price,
            "usdc_cost":     round(size * price, 4),
            "order_id":      order_id,
            "opened_at":     datetime.now().isoformat(),
        }
        _trade_count += 1

        tg(
            f"🎯 <b>SNIPE ENTRY</b>\n"
            f"Market: {market_id[:12]}\n"
            f"Direction: <b>{direction}</b>\n"
            f"Size: {size} @ ${price:.4f}\n"
            f"Cost: ${size*price:.4f} USDC\n"
            f"Target: +15% → exit consideration"
        )
        return json.dumps({
            "status": "ok", "order_id": order_id,
            "direction": direction, "size": size,
            "entry_price": price, "usdc_cost": round(size*price, 4),
        })
    except Exception as e:
        tg(f"🔴 <b>SNIPE FAILED</b>\n{str(e)[:150]}")
        return json.dumps({"status": "error", "message": str(e)})


def tool_check_and_exit(**kwargs) -> str:
    """
    Check all open positions for exit conditions.
    Agent decides: hold if momentum strong, exit if target hit or reversal detected.
    Returns positions with current P&L and exit recommendations.
    """
    global _positions, _daily_pnl
    if not _positions:
        return json.dumps({"status": "ok", "message": "no_open_positions"})

    updates = []
    for tid, pos in list(_positions.items()):
        try:
            price_data = json.loads(tool_get_price(tid))
            if price_data["status"] != "ok":
                continue

            current = float(price_data["mid"])
            entry   = pos["entry_price"]
            pnl_pct = (current - entry) / entry

            _positions[tid]["current_price"] = current

            # Stop loss
            if pnl_pct <= -STOP_LOSS:
                exit_rec = "EXIT_STOP_LOSS"
            # Strong profit — agent decides
            elif pnl_pct >= 0.50:
                exit_rec = "EXIT_STRONG_PROFIT"
            elif pnl_pct >= PROFIT_TARGET_MIN:
                # Check momentum — if spread tightening = buyers coming in = HOLD
                spread = price_data.get("spread", 0.05)
                exit_rec = "HOLD_MOMENTUM" if spread < 0.03 else "EXIT_TAKE_PROFIT"
            else:
                exit_rec = "HOLD"

            updates.append({
                "token_id":     tid[:12],
                "market_id":    pos["market_id"][:12],
                "direction":    pos["direction"],
                "entry_price":  entry,
                "current_price":current,
                "pnl_pct":      round(pnl_pct * 100, 2),
                "pnl_usdc":     round(pnl_pct * pos["usdc_cost"], 4),
                "recommendation": exit_rec,
                "spread":       price_data.get("spread", 0.05),
            })
        except Exception as e:
            updates.append({"token_id": tid[:12], "error": str(e)})

    return json.dumps({"status": "ok", "positions": updates})


def tool_exit_position(token_id: str, reason: str = "") -> str:
    """
    Exit a position by selling tokens back to the order book.
    """
    global _positions, _daily_pnl
    pos = _positions.get(token_id)
    if not pos:
        return json.dumps({"status": "error", "message": "position_not_found"})
    try:
        # Get current best bid to sell into
        book     = requests.get(f"{CLOB_API}/book", params={"token_id": token_id}, timeout=8).json()
        bids     = book.get("bids", [])
        sell_price = float(bids[0]["price"]) if bids else pos["current_price"]

        from py_clob_client.clob_types import OrderArgs, OrderType
        from py_clob_client.order_builder.constants import SELL
        client = get_clob_client()
        signed = client.create_order(
            OrderArgs(token_id=token_id, price=sell_price, size=pos["size"], side=SELL)
        )
        resp     = client.post_order(signed, OrderType.GTC)
        order_id = resp.get("orderID") or resp.get("id", "unknown")

        pnl      = (sell_price - pos["entry_price"]) * pos["size"]
        pnl_pct  = ((sell_price - pos["entry_price"]) / pos["entry_price"]) * 100
        _daily_pnl += pnl
        del _positions[token_id]

        emoji = "🟢" if pnl > 0 else "🔴"
        tg(
            f"{emoji} <b>EXIT {'✅ PROFIT' if pnl > 0 else '❌ LOSS'}</b>\n"
            f"Direction: {pos['direction']}\n"
            f"Entry: ${pos['entry_price']:.4f} → Exit: ${sell_price:.4f}\n"
            f"PnL: {pnl_pct:+.2f}% (${pnl:+.4f})\n"
            f"Reason: {reason}"
        )
        return json.dumps({
            "status": "ok", "order_id": order_id,
            "sell_price": sell_price, "pnl": round(pnl, 4),
            "pnl_pct": round(pnl_pct, 2),
        })
    except Exception as e:
        tg(f"🔴 <b>EXIT FAILED</b>\n{str(e)[:150]}")
        return json.dumps({"status": "error", "message": str(e)})


def tool_check_positions(**kwargs) -> str:
    elapsed = (datetime.now() - _start_time).total_seconds()
    tph     = (_trade_count / max(elapsed, 1)) * 3600
    return json.dumps({
        "status":                "ok",
        "open_positions":        len(_positions),
        "max_positions":         int(os.getenv("MAX_OPEN_POSITIONS", "3")),
        "total_trades":          _trade_count,
        "trades_per_hour":       round(tph, 1),
        "daily_pnl_usdc":        round(_daily_pnl, 4),
        "daily_loss_limit":      MAX_DAILY_LOSS_USDC,
        "daily_loss_remaining":  round(MAX_DAILY_LOSS_USDC + _daily_pnl, 2),
        "uptime_minutes":        round(elapsed / 60, 1),
        "paused":                _paused,
        "positions": [
            {
                "token_id":     tid[:12],
                "market_id":    p["market_id"][:12],
                "direction":    p["direction"],
                "size":         p["size"],
                "entry_price":  p["entry_price"],
                "current_price":p.get("current_price", p["entry_price"]),
                "usdc_cost":    p["usdc_cost"],
            }
            for tid, p in _positions.items()
        ],
    })


def tool_cancel_all_orders(**kwargs) -> str:
    global _positions
    try:
        get_clob_client().cancel_all()
        count = len(_positions)
        _positions.clear()
        tg(f"🔴 <b>ALL CANCELLED</b>\n{count} positions closed")
        return json.dumps({"status": "ok", "cancelled_count": count})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


# ══════════════════════════════════════════════════════════════════════════════
# TOOL SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════
TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "scan_fresh_markets",
        "description": (
            "Scan for fresh crypto prediction markets on Polymarket, scored by mispricing opportunity. "
            "Returns top markets ranked by snipe score. Call this first every cycle."
        ),
        "parameters": {"type": "object", "properties": {
            "limit": {"type": "integer", "description": "Max markets to scan. Default 30."},
        }, "required": []},
    }},
    {"type": "function", "function": {
        "name": "get_price",
        "description": "Get live price, spread, and order book for a token. Use to check current opportunity before entry or exit.",
        "parameters": {"type": "object", "properties": {
            "token_id": {"type": "string"},
        }, "required": ["token_id"]},
    }},
    {"type": "function", "function": {
        "name": "snipe_entry",
        "description": (
            "Enter a position by buying a token. Use when scan_fresh_markets returns snipe=true "
            "AND score is high AND spread < 0.08. "
            "Use 80-100% of available balance per trade for maximum compounding."
        ),
        "parameters": {"type": "object", "properties": {
            "market_id":   {"type": "string"},
            "token_id":    {"type": "string"},
            "direction":   {"type": "string", "enum": ["YES", "NO"]},
            "price":       {"type": "number", "description": "Entry price (use best_ask from get_price)"},
            "usdc_amount": {"type": "number", "description": "USDC to use. Use most of available balance."},
        }, "required": ["market_id", "token_id", "direction", "price", "usdc_amount"]},
    }},
    {"type": "function", "function": {
        "name": "check_and_exit",
        "description": (
            "Check all open positions for exit conditions. "
            "Returns current P&L and exit recommendation for each position. "
            "Call every cycle after entry."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    }},
    {"type": "function", "function": {
        "name": "exit_position",
        "description": (
            "Sell/exit a position. Call when check_and_exit returns EXIT_* recommendation, "
            "OR when you judge momentum has reversed. "
            "Hold if recommendation is HOLD_MOMENTUM — price still moving in our favor."
        ),
        "parameters": {"type": "object", "properties": {
            "token_id": {"type": "string"},
            "reason":   {"type": "string", "description": "Why exiting: profit_target/stop_loss/reversal/etc"},
        }, "required": ["token_id"]},
    }},
    {"type": "function", "function": {
        "name": "check_positions",
        "description": "Get summary of open positions, P&L, and risk status.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }},
    {"type": "function", "function": {
        "name": "cancel_all_orders",
        "description": "Emergency: cancel all orders. Use when daily loss limit hit.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }},
]

TOOL_MAP = {
    "scan_fresh_markets": tool_scan_fresh_markets,
    "get_price":          tool_get_price,
    "snipe_entry":        tool_snipe_entry,
    "check_and_exit":     tool_check_and_exit,
    "exit_position":      tool_exit_position,
    "check_positions":    tool_check_positions,
    "cancel_all_orders":  tool_cancel_all_orders,
}

# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT — Scalper Brain
# ══════════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """You are an aggressive Polymarket scalper agent. Capital: $3.50 USDC. Goal: multiply capital fast via early entry mispricing.

STRATEGY:
- Find fresh crypto markets where odds are "wrong" (mispriced due to thin early trading)
- Snipe entry EARLY before crowd moves price
- Hold while momentum is strong
- Exit when: profit target hit OR momentum reverses OR stop loss

WORKFLOW every cycle:
1. check_positions → see open positions + P&L
2. check_and_exit → for any open positions, get exit recommendations. EXIT immediately if:
   - recommendation = EXIT_STOP_LOSS
   - recommendation = EXIT_STRONG_PROFIT
   - recommendation = EXIT_TAKE_PROFIT AND you judge momentum fading
   - HOLD if recommendation = HOLD_MOMENTUM (price still moving up)
3. If positions < max: scan_fresh_markets → find best snipe opportunity
4. For top opportunity: get_price → confirm spread < 0.08
5. snipe_entry with 80-90% of balance (aggressive compounding)

ENTRY RULES:
- Only snipe when score >= 0.35 AND snipe=true
- Prefer age_hours < 6 (freshest markets = most mispriced)
- Prefer spread < 0.05 (liquid enough to exit)
- Use 80-90% of balance per trade — we want max compounding

EXIT RULES (YOU DECIDE):
- Stop loss: -20% → EXIT immediately, no hesitation
- +15% to +30%: check spread — if tightening (< 0.03) = HOLD, buyers still coming
- +30% to +50%: hold only if very strong momentum, otherwise take profit
- +50%+: exit, secure the win, find next target
- Momentum reversal (price dropping after peak): EXIT

RISK:
- NEVER trade if daily_loss_remaining < 0.50
- NEVER exceed max_positions
- If unsure: exit and protect capital"""


# ══════════════════════════════════════════════════════════════════════════════
# AGENT LOOP
# ══════════════════════════════════════════════════════════════════════════════
def run_agent_cycle(client, cycle):
    elapsed  = (datetime.now() - _start_time).total_seconds()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"[Cycle #{cycle} | Uptime:{elapsed/60:.1f}min | "
            f"{datetime.now().strftime('%H:%M:%S')} | Paused:{_paused}]\n"
            f"Execute scalper workflow now."
        )},
    ]
    step = 0
    while True:
        step += 1
        if step > 30:
            break
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0.15,
            max_tokens=4096,
        )
        msg = response.choices[0].message
        if not msg.tool_calls:
            return msg.content or "Cycle complete."
        messages.append({
            "role": "assistant", "content": msg.content or "",
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ],
        })
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                # Handle None arguments gracefully
                if tc.function.arguments is None:
                    args = {}
                else:
                    args = json.loads(tc.function.arguments)
            except Exception as e:
                print(f"[TOOL] Failed to parse args for {name}: {e}")
                args = {}
            func   = TOOL_MAP.get(name)
            result = func(**args) if func else json.dumps({"status": "error", "message": f"unknown:{name}"})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
    return "Cycle complete."


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 50)
    print("  🎯 APEX SCALPER — Polymarket")
    print(f"  Model   : {GROQ_MODEL}")
    print(f"  Interval: {AGENT_LOOP_INTERVAL}s")
    print(f"  Max bet : ${MAX_POSITION_USDC}")
    print(f"  Stop at : -${MAX_DAILY_LOSS_USDC}")
    print("=" * 50)

    if not GROQ_API_KEY:
        print("❌ Missing GROQ_API_KEY"); return
    if not POLYMARKET_PK or not POLYMARKET_FUNDER:
        print("❌ Missing POLYMARKET keys"); return

    threading.Thread(target=poll_telegram, daemon=True).start()

    tg(
        f"🎯 <b>Apex Scalper Started</b>\n"
        f"Strategy: Early Entry Mispricing\n"
        f"Model: {GROQ_MODEL}\n"
        f"Max/trade: ${MAX_POSITION_USDC}\n"
        f"Stop loss: -${MAX_DAILY_LOSS_USDC}/day\n"
        f"Profit target: dynamic (agent decides)\n\n"
        f"<b>Commands:</b> /status /balance /pause /resume /report"
    )

    groq_client = Groq(api_key=GROQ_API_KEY)
    cycle = 0

    while True:
        cycle += 1
        try:
            if not _paused:
                run_agent_cycle(groq_client, cycle)
        except KeyboardInterrupt:
            tool_cancel_all_orders()
            tg("⛔ <b>Scalper stopped</b>")
            break
        except Exception as e:
            print(f"[ERROR] cycle#{cycle}: {e}")
            tg(f"⚠️ <b>Cycle #{cycle} error</b>\n{str(e)[:200]}")

        time.sleep(AGENT_LOOP_INTERVAL)


if __name__ == "__main__":
    main()
