import os
import json
import time
import threading
import requests
from datetime import datetime
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ================= CONFIG =================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = str(os.getenv("TELEGRAM_CHAT_ID"))

POLYMARKET_PK = os.getenv("POLYMARKET_PRIVATE_KEY")
POLYMARKET_FUNDER = os.getenv("POLYMARKET_FUNDER_ADDRESS")

AGENT_LOOP_INTERVAL = int(os.getenv("AGENT_LOOP_INTERVAL", "60"))

MAX_POSITION_USDC = float(os.getenv("MAX_POSITION_USDC", "3.0"))
MAX_DAILY_LOSS_USDC = float(os.getenv("MAX_DAILY_LOSS_USDC", "3.0"))

PROFIT_TARGET_MIN = float(os.getenv("PROFIT_TARGET_MIN", "0.15"))
STOP_LOSS = float(os.getenv("STOP_LOSS", "0.20"))

MIN_MARKET_VOLUME = float(os.getenv("MIN_MARKET_VOLUME", "5000"))

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"

# ================= STATE =================

_positions = {}
_daily_pnl = 0.0
_trade_count = 0
_paused = False
_start_time = datetime.now()
_tg_offset = 0
_clob_client = None

# ================= TELEGRAM =================

def tg(msg):
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"},
            timeout=10
        )
    except:
        pass


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

                msg = update.get("message", {})
                text = msg.get("text", "")
                chat = str(msg.get("chat", {}).get("id"))

                if chat != TELEGRAM_CHAT_ID:
                    continue

                if text == "/status":
                    tg(f"Open: {len(_positions)} | Trades: {_trade_count} | PnL: {_daily_pnl}")

                if text == "/pause":
                    global _paused
                    _paused = True
                    tg("⏸ paused")

                if text == "/resume":
                    _paused = False
                    tg("▶ resumed")

        except:
            pass

        time.sleep(2)

# ================= CLOB CLIENT =================

def get_clob_client():
    global _clob_client

    if _clob_client is None:

        from py_clob_client.client import ClobClient
        from py_clob_client.constants import POLYGON

        _clob_client = ClobClient(
            host=CLOB_API,
            key=POLYMARKET_PK,
            chain_id=POLYGON,
            signature_type=1,
            funder=POLYMARKET_FUNDER
        )

        _clob_client.set_api_creds(
            _clob_client.create_or_derive_api_creds()
        )

    return _clob_client


# ================= BALANCE =================

def get_usdc_balance():

    try:

        client = get_clob_client()

        data = client.get_balance_allowance()

        balance = float(data.get("balance", 0))

        return balance / 1e6

    except Exception as e:

        print("balance error", e)

        return 0


# ================= MARKET SCAN =================

def scan_markets():

    try:

        r = requests.get(
            f"{GAMMA_API}/markets",
            params={
                "active": "true",
                "closed": "false",
                "limit": 30,
                "tag": "crypto"
            },
            timeout=10
        )

        markets = r.json()

        results = []

        for m in markets:

            vol = float(m.get("volume24hr") or 0)

            if vol < MIN_MARKET_VOLUME:
                continue

            tokens = m.get("tokens", [])

            yes_price = 0.5
            yes_token = None

            for t in tokens:
                if t.get("outcome") == "Yes":
                    yes_price = float(t.get("price") or 0.5)
                    yes_token = t.get("token_id")

            if not yes_token:
                continue

            results.append({
                "market": m.get("conditionId"),
                "token": yes_token,
                "price": yes_price
            })

        return results

    except:

        return []


# ================= ENTRY =================

def enter_position(market, token, price):

    global _positions
    global _trade_count

    balance = get_usdc_balance()

    if balance < 1:
        return

    size = round((balance * 0.9) / price, 2)

    try:

        from py_clob_client.clob_types import OrderArgs, OrderType
        from py_clob_client.order_builder.constants import BUY

        client = get_clob_client()

        signed = client.create_order(
            OrderArgs(
                token_id=token,
                price=price,
                size=size,
                side=BUY
            )
        )

        resp = client.post_order(signed, OrderType.GTC)

        _positions[token] = {
            "market": market,
            "entry": price,
            "size": size
        }

        _trade_count += 1

        tg(f"ENTRY {size}@{price}")

    except Exception as e:

        tg(f"entry fail {str(e)[:120]}")


# ================= EXIT =================

def exit_position(token):

    global _positions
    global _daily_pnl

    pos = _positions.get(token)

    if not pos:
        return

    try:

        book = requests.get(
            f"{CLOB_API}/book",
            params={"token_id": token},
            timeout=8
        ).json()

        bid = float(book["bids"][0]["price"])

        from py_clob_client.clob_types import OrderArgs, OrderType
        from py_clob_client.order_builder.constants import SELL

        client = get_clob_client()

        signed = client.create_order(
            OrderArgs(
                token_id=token,
                price=bid,
                size=pos["size"],
                side=SELL
            )
        )

        client.post_order(signed, OrderType.GTC)

        pnl = (bid - pos["entry"]) * pos["size"]

        _daily_pnl += pnl

        del _positions[token]

        tg(f"EXIT pnl={pnl}")

    except Exception as e:

        tg(f"exit fail {str(e)[:120]}")


# ================= CHECK POSITIONS =================

def check_positions():

    for token, pos in list(_positions.items()):

        try:

            mid = requests.get(
                f"{CLOB_API}/midpoint",
                params={"token_id": token},
                timeout=8
            ).json()["mid"]

            entry = pos["entry"]

            pnl = (mid - entry) / entry

            if pnl <= -STOP_LOSS:

                exit_position(token)

            elif pnl >= PROFIT_TARGET_MIN:

                exit_position(token)

        except:

            pass


# ================= AGENT =================

SYSTEM_PROMPT = """
You are an aggressive Polymarket scalper.

Goal: grow $3.50 quickly.

Workflow:

1 check positions
2 exit if needed
3 scan markets
4 enter best opportunity
"""


def run_cycle(client):

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "run cycle"}
    ]

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.2
    )

    # we do execution ourselves (safer than tool spam)

    check_positions()

    if len(_positions) < 3:

        markets = scan_markets()

        if markets:

            m = markets[0]

            enter_position(
                m["market"],
                m["token"],
                m["price"]
            )


# ================= MAIN =================

def main():

    print("APEX SCALPER START")

    if not GROQ_API_KEY:
        print("missing groq key")
        return

    threading.Thread(target=poll_telegram, daemon=True).start()

    tg("scalper started")

    client = Groq(api_key=GROQ_API_KEY)

    cycle = 0

    while True:

        cycle += 1

        try:

            if not _paused:

                run_cycle(client)

        except Exception as e:

            print("cycle error", e)

            tg(f"cycle error {str(e)[:120]}")

        time.sleep(AGENT_LOOP_INTERVAL)


if __name__ == "__main__":
    main()
