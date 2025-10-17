README — Railway (Cron Job, English Alerts)

What this package does
----------------------
• Scans Binance USDT perpetual pairs across 1H/2H/4H/1D
• Alerts to Telegram when ≥3 criteria fire on any of the last 4 candles
• Sends a “No signals found” summary if nothing fired
• Designed for Railway's Cron (no need to keep a worker always on)
• Uses data-api.binance.vision (public mirror) to avoid regional blocks

Files
-----
- scan_bot.py      → single run scan (to be called by Railway Schedule)
- requirements.txt → pip dependencies
- Procfile         → optional (for worker runs); Cron will use the command directly
- runtime.txt      → Python 3.11
- README_Railway.txt

How to deploy on Railway (Cron every 1 hour)
--------------------------------------------
1) Create a new Railway project (Python template or empty).
2) Upload these files to your repo connected to Railway.
3) In your Railway service:
   - Go to "Variables" and (optionally) add TELEGRAM_TOKEN / CHAT_ID if you prefer env vars.
     (This build already has your token/chat_id hardcoded as requested.)
4) Go to "Schedules" (Add Schedule):
   - Command:  python scan_bot.py
   - Cron:     0 * * * *          (every 1 hour)
   - Repeat:   Enabled
5) Deploy once and check "Deployments → Logs".
6) You should receive a Telegram message on each run (signals or “No signals found”).
