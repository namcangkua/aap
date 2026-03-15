# Apex Agent — Polymarket

Autonomous Groq-powered prediction market trading agent.

## Railway Variables

| NAME | VALUE |
|------|-------|
| GROQ_API_KEY | gsk_... |
| GROQ_MODEL | llama-3.3-70b-versatile |
| TELEGRAM_BOT_TOKEN | your_token |
| TELEGRAM_CHAT_ID | your_chat_id |
| POLYMARKET_PRIVATE_KEY | 0x... |
| POLYMARKET_FUNDER_ADDRESS | 0x... |
| MAX_POSITION_USDC | 3.0 |
| MAX_DAILY_LOSS_USDC | 3.0 |
| MAX_OPEN_POSITIONS | 3 |
| AGENT_LOOP_INTERVAL | 30 |
| MIN_EDGE | 0.03 |
| MIN_CONFIDENCE | 0.60 |
| KELLY_FRACTION | 0.15 |
| MIN_MARKET_VOLUME | 5000 |
