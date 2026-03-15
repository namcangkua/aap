import os
import json
import time
import requests
import numpy as np
from datetime import datetime
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY         = os.getenv("GROQ_API_KEY")
GROQ_MODEL           = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
TELEGRAM_BOT_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID     = os.getenv("TELEGRAM_CHAT_ID")
POLYMARKET_PK        = os.getenv("POLYMARKET_PRIVATE_KEY")
POLYMARKET_FUNDER    = os.getenv("POLYMARKET_FUNDER_ADDRESS")
MAX_POSITION_USDC    = float(os.getenv("MAX_POSITION_USDC", "3.0"))
MAX_DAILY_LOSS_USDC  = float(os.getenv("MAX_DAILY_LOSS_USDC", "3.0"))
AGENT_LOOP_INTERVAL  = int(os.getenv("AGENT_LOOP_INTERVAL", "30"))
MIN_EDGE             = float(os.getenv("MIN_EDGE", "0.03"))
MIN_CONFIDENCE       = float(os.getenv("MIN_CONFIDENCE", "0.60"))
KELLY_FRACTION       = float(os.getenv("KELLY_FRACTION", "0.15"))
MIN_MARKET_VOLUME    = float(os.getenv("MIN_MARKET_VOLUME", "5000"))

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API  = "https://clob.polymarket.com"

_positions: dict  = {}
_daily_pnl: float = 0.0
_trade_count: int = 0
_start_time       = datetime.now()
_clob_client      = None


def tg(msg: str):
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception as e:
        print(f"[TG ERROR] {e}")


def get_clob_client():
    global _clob_client
    if _clob_client is None:
        from py_clob_client.client import ClobClient
        from py_clob_client.constants import POLYGON
        _clob_client = ClobClient(
            host=CLOB_API, key=POLYMARKET_PK, chain_id=POLYGON,
            signature_type=0, funder=POLYMARKET_FUNDER,
        )
        _clob_client.set_api_creds(_clob_client.create_or_derive_api_creds())
    return _clob_client


def bayesian_analyze(yes_price, ob_imbalance, trade_direction, spread, volume_24h):
    prior        = 0.5 * 0.5 + 0.5 * yes_price
    ob_score     = float(ob_imbalance)
    trade_score  = (float(trade_direction) + 1) / 2
    spread_score = max(0.1, 1.0 - (float(spread) / 0.10))
    vol_score    = 1 / (1 + np.exp(-(float(volume_24h) - 50000) / 20000))
    likelihood   = ob_score*0.30 + trade_score*0.35 + spread_score*0.15 + vol_score*0.20
    p_d          = likelihood * prior + (1 - likelihood) * (1 - prior)
    posterior    = (likelihood * prior) / p_d if p_d else prior
    confidence   = abs(posterior - 0.5) * 2
    edge_yes     = posterior - yes_price
    edge_no      = (1 - posterior) - (1 - yes_price)
    if abs(edge_yes) >= abs(edge_no):
        edge, direction, trade_prob, ref_price = edge_yes, "YES", posterior, yes_price
    else:
        edge, direction, trade_prob, ref_price = edge_no, "NO", 1 - posterior, 1 - yes_price
    if 0 < ref_price < 1:
        b     = (1 - ref_price) / ref_price
        k     = (trade_prob * b - (1 - trade_prob)) / b
        kelly = max(0.0, min(k * KELLY_FRACTION, 0.20))
    else:
        kelly = 0.0
    return {
        "posterior": round(posterior, 4), "edge": round(edge, 4),
        "confidence": round(confidence, 4), "kelly": round(kelly, 4),
        "direction": direction,
        "should_trade": confidence >= MIN_CONFIDENCE and abs(edge) >= MIN_EDGE and kelly > 0,
    }


def tool_scan_markets(min_volume=5000, limit=20):
    try:
        r = requests.get(f"{GAMMA_API}/markets",
            params={"active":"true","closed":"false","limit":limit,"order":"volume24hr","ascending":"false"},
            timeout=15)
        r.raise_for_status()
        data    = r.json()
        markets = data if isinstance(data, list) else data.get("markets", [])
        result  = []
        for m in markets:
            vol = float(m.get("volume24hr") or 0)
            if vol < min_volume: continue
            tokens = m.get("tokens", [])
            yes_price = no_price = 0.5
            yes_tid = no_tid = ""
            for t in tokens:
                if t.get("outcome") == "Yes":
                    yes_price = float(t.get("price") or 0.5); yes_tid = t.get("token_id","")
                elif t.get("outcome") == "No":
                    no_price = float(t.get("price") or 0.5); no_tid = t.get("token_id","")
            result.append({"market_id":m.get("conditionId",""),"question":m.get("question","")[:80],
                "yes_price":round(yes_price,3),"no_price":round(no_price,3),
                "yes_token_id":yes_tid,"no_token_id":no_tid,"volume_24h":vol})
        return json.dumps({"status":"ok","count":len(result),"markets":result[:15]})
    except Exception as e:
        return json.dumps({"status":"error","message":str(e)})


def tool_analyze_market(market_id, yes_price, volume_24h, yes_token_id=""):
    try:
        ob_imbalance = 0.5; trade_dir = 0.0; spread = 0.05
        if yes_token_id:
            try:
                book   = requests.get(f"{CLOB_API}/book",params={"token_id":yes_token_id},timeout=10).json()
                trades = requests.get(f"{CLOB_API}/trades",params={"token_id":yes_token_id,"limit":20},timeout=10).json()
                mid_r  = requests.get(f"{CLOB_API}/midpoint",params={"token_id":yes_token_id},timeout=10).json()
                bids   = book.get("bids",[]); asks = book.get("asks",[])
                bid_vol = sum(float(b.get("size",0)) for b in bids)
                ask_vol = sum(float(a.get("size",0)) for a in asks)
                total   = bid_vol + ask_vol
                ob_imbalance = bid_vol / total if total > 0 else 0.5
                if bids and asks: spread = float(asks[0]["price"]) - float(bids[0]["price"])
                tl     = trades if isinstance(trades, list) else []
                buys   = sum(1 for t in tl if t.get("side") == "BUY")
                trade_dir = (buys - (len(tl) - buys)) / (len(tl) or 1)
                if mid_r.get("mid"): yes_price = float(mid_r["mid"])
            except Exception: pass
        result = bayesian_analyze(yes_price, ob_imbalance, trade_dir, spread, volume_24h)
        result["market_id"] = market_id; result["status"] = "ok"
        result["recommendation"] = (
            f"{'✅ TRADE' if result['should_trade'] else '❌ SKIP'} "
            f"{result['direction']} @ {yes_price:.3f} | edge={result['edge']:+.3f} conf={result['confidence']:.3f}")
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"status":"error","message":str(e)})


def tool_place_order(market_id, token_id, direction, price, kelly_fraction):
    global _positions, _trade_count
    try:
        if len(_positions) >= int(os.getenv("MAX_OPEN_POSITIONS","3")):
            return json.dumps({"status":"skipped","reason":"max_positions_reached"})
        client  = get_clob_client()
        balance = float(client.get_balance())
        if balance < 1.0:
            return json.dumps({"status":"skipped","reason":f"balance_too_low:${balance:.2f}"})
        usdc_bet = min(kelly_fraction * balance, MAX_POSITION_USDC)
        if not (0 < price < 1):
            return json.dumps({"status":"error","message":f"invalid_price:{price}"})
        size = round(usdc_bet / price, 2)
        if size < 1.0:
            return json.dumps({"status":"skipped","reason":f"size_too_small:{size}"})
        from py_clob_client.clob_types import OrderArgs
        resp     = client.create_and_post_order(OrderArgs(token_id=token_id,price=price,size=size,side="BUY"))
        order_id = resp.get("orderID") or resp.get("id","unknown")
        _positions[order_id] = {"market_id":market_id,"token_id":token_id,"direction":direction,
            "size":size,"entry_price":price,"usdc_cost":round(size*price,2),"opened_at":datetime.now().isoformat()}
        _trade_count += 1
        tg(f"🟢 <b>ORDER PLACED</b>\nMarket: <code>{market_id[:12]}</code>\n"
           f"Direction: <b>{direction}</b>\nSize: {size} @ ${price:.3f}\n"
           f"Cost: ${size*price:.2f} USDC\nOrder: <code>{order_id}</code>")
        return json.dumps({"status":"ok","order_id":order_id,"direction":direction,
            "size":size,"price":price,"usdc_cost":round(size*price,2)})
    except Exception as e:
        tg(f"🔴 <b>ORDER FAILED</b>\n{str(e)}")
        return json.dumps({"status":"error","message":str(e)})


def tool_check_positions():
    global _positions, _daily_pnl, _trade_count, _start_time
    elapsed = (datetime.now() - _start_time).total_seconds()
    tph     = (_trade_count / max(elapsed,1)) * 3600
    return json.dumps({"status":"ok","open_positions":len(_positions),
        "max_positions":int(os.getenv("MAX_OPEN_POSITIONS","3")),
        "total_trades":_trade_count,"trades_per_hour":round(tph,1),
        "daily_pnl_usdc":round(_daily_pnl,4),"daily_loss_limit":MAX_DAILY_LOSS_USDC,
        "daily_loss_remaining":round(MAX_DAILY_LOSS_USDC + _daily_pnl,2),
        "uptime_minutes":round(elapsed/60,1),
        "positions":[{"order_id":oid[:12],"market_id":p["market_id"][:12],
            "direction":p["direction"],"size":p["size"],"entry_price":p["entry_price"],
            "usdc_cost":p["usdc_cost"]} for oid,p in _positions.items()]})


def tool_cancel_order(order_id):
    global _positions
    try:
        get_clob_client().cancel(order_id)
        _positions.pop(order_id, None)
        tg(f"🟡 <b>ORDER CANCELLED</b>\n<code>{order_id}</code>")
        return json.dumps({"status":"ok","cancelled":order_id})
    except Exception as e:
        return json.dumps({"status":"error","message":str(e)})


def tool_cancel_all_orders():
    global _positions
    try:
        get_clob_client().cancel_all()
        count = len(_positions); _positions.clear()
        tg(f"🔴 <b>ALL CANCELLED</b>\n{count} positions closed")
        return json.dumps({"status":"ok","cancelled_count":count})
    except Exception as e:
        return json.dumps({"status":"error","message":str(e)})


TOOL_SCHEMAS = [
    {"type":"function","function":{"name":"scan_markets",
        "description":"Scan Polymarket active markets by volume. Call first every cycle.",
        "parameters":{"type":"object","properties":{
            "min_volume":{"type":"number","description":"Min 24h volume USDC"},
            "limit":{"type":"integer","description":"Max markets to fetch"}},"required":[]}}},
    {"type":"function","function":{"name":"analyze_market",
        "description":"Bayesian analysis. Returns edge, confidence, direction, kelly, should_trade.",
        "parameters":{"type":"object","properties":{
            "market_id":{"type":"string"},"yes_price":{"type":"number"},
            "volume_24h":{"type":"number"},"yes_token_id":{"type":"string"}},"required":["market_id","yes_price","volume_24h"]}}},
    {"type":"function","function":{"name":"place_order",
        "description":"Place BUY order. ONLY when should_trade=true AND edge>=0.03 AND confidence>=0.60.",
        "parameters":{"type":"object","properties":{
            "market_id":{"type":"string"},"token_id":{"type":"string"},
            "direction":{"type":"string","enum":["YES","NO"]},
            "price":{"type":"number"},"kelly_fraction":{"type":"number"}},"required":["market_id","token_id","direction","price","kelly_fraction"]}}},
    {"type":"function","function":{"name":"check_positions",
        "description":"Check open positions, P&L, risk status.",
        "parameters":{"type":"object","properties":{},"required":[]}}},
    {"type":"function","function":{"name":"cancel_order",
        "description":"Cancel specific order.",
        "parameters":{"type":"object","properties":{"order_id":{"type":"string"}},"required":["order_id"]}}},
    {"type":"function","function":{"name":"cancel_all_orders",
        "description":"Emergency: cancel ALL orders when daily loss limit hit.",
        "parameters":{"type":"object","properties":{},"required":[]}}},
]

TOOL_MAP = {"scan_markets":tool_scan_markets,"analyze_market":tool_analyze_market,
    "place_order":tool_place_order,"check_positions":tool_check_positions,
    "cancel_order":tool_cancel_order,"cancel_all_orders":tool_cancel_all_orders}

SYSTEM_PROMPT = """You are an autonomous Polymarket trading agent with $3.50 USDC. Goal: PROFIT via compounding.

WORKFLOW every cycle:
1. check_positions → review state + risk
2. scan_markets → find markets
3. analyze_market top 5-8 markets by volume
4. place_order ONLY when: should_trade=true AND edge>=0.03 AND confidence>=0.60
5. Summarize actions

RULES:
- NEVER trade if daily_loss_remaining < 0.50
- NEVER exceed max_positions
- NEVER trade spread > 0.08 or volume < 5000
- ALWAYS use kelly from analyze_market
- If daily loss limit hit → cancel_all_orders immediately
- Small consistent wins. Protect capital first."""


def run_agent_cycle(client, cycle):
    elapsed  = (datetime.now() - _start_time).total_seconds()
    messages = [
        {"role":"system","content":SYSTEM_PROMPT},
        {"role":"user","content":f"[Cycle #{cycle} | Uptime:{elapsed/60:.1f}min | {datetime.now().strftime('%H:%M:%S')}]\nExecute trading workflow now."},
    ]
    step = 0
    while True:
        step += 1
        if step > 25: break
        response = client.chat.completions.create(
            model=GROQ_MODEL, messages=messages, tools=TOOL_SCHEMAS,
            tool_choice="auto", temperature=0.1, max_tokens=4096)
        msg = response.choices[0].message
        if not msg.tool_calls:
            return msg.content or "Cycle complete."
        messages.append({"role":"assistant","content":msg.content or "",
            "tool_calls":[{"id":tc.id,"type":"function","function":{"name":tc.function.name,"arguments":tc.function.arguments}} for tc in msg.tool_calls]})
        for tc in msg.tool_calls:
            name = tc.function.name
            try: args = json.loads(tc.function.arguments)
            except Exception: args = {}
            print(f"  🔧 [{step}] {name}")
            func   = TOOL_MAP.get(name)
            result = func(**args) if func else json.dumps({"status":"error","message":f"unknown:{name}"})
            messages.append({"role":"tool","tool_call_id":tc.id,"content":result})
    return "Cycle complete."


def main():
    print("="*50)
    print("  🤖 APEX AGENT — Polymarket")
    print(f"  Model   : {GROQ_MODEL}")
    print(f"  Interval: {AGENT_LOOP_INTERVAL}s")
    print(f"  Max bet : ${MAX_POSITION_USDC}")
    print(f"  Stop at : -${MAX_DAILY_LOSS_USDC}")
    print("="*50)
    if not GROQ_API_KEY:
        print("❌ Missing GROQ_API_KEY"); return
    if not POLYMARKET_PK or not POLYMARKET_FUNDER:
        print("❌ Missing POLYMARKET_PRIVATE_KEY or POLYMARKET_FUNDER_ADDRESS"); return
    tg(f"🚀 <b>Apex Agent Started</b>\nModel: {GROQ_MODEL}\nMax/trade: ${MAX_POSITION_USDC}\nStop loss: -${MAX_DAILY_LOSS_USDC}/day")
    groq_client = Groq(api_key=GROQ_API_KEY)
    cycle = 0
    while True:
        cycle += 1
        print(f"\n━━━━━ Cycle #{cycle} | {datetime.now().strftime('%H:%M:%S')} ━━━━━")
        try:
            summary = run_agent_cycle(groq_client, cycle)
            print(f"  📋 {summary[:150]}")
            if cycle % 10 == 0:
                pos = json.loads(tool_check_positions())
                tg(f"📊 <b>Report Cycle #{cycle}</b>\nOpen: {pos['open_positions']}\nTrades: {pos['total_trades']}\nTrades/hr: {pos['trades_per_hour']}\nPnL: ${pos['daily_pnl_usdc']}\nUptime: {pos['uptime_minutes']}min")
        except KeyboardInterrupt:
            print("\n⚠️  Stopping..."); tool_cancel_all_orders(); tg("⛔ <b>Agent stopped</b>"); break
        except Exception as e:
            print(f"  ❌ {e}"); tg(f"⚠️ <b>Cycle #{cycle} error</b>\n{str(e)[:200]}")
        time.sleep(AGENT_LOOP_INTERVAL)

if __name__ == "__main__":
    main()
