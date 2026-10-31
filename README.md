# India 10-Min Paper Scalper Simulator

Educational backtest / paper-trading simulator for Indian markets (NSE, IST hours).
Rules: EMA 9/21 cross + RSI + VWAP entry; exit on stop-loss, target, EMA reversal,
momentum fade, 10-minute limit, or 15:15 square-off.

**Not investment advice. Paper trading only.**

## Run
```bash
pip install -r requirements.txt
python simulate.py --demo                                   # offline test with fake data
python simulate.py --symbol ^NSEI --interval 5m --period 30d
python simulate.py --symbol ^NSEBANK --interval 5m --qty 30
python simulate.py --symbol RELIANCE.NS --interval 5m --qty 250
```
Results print in the terminal and save to `trades.csv`.

Notes: Yahoo gives ~7 days of 1m data and ~60 days of 5m data. Index data has no volume,
so VWAP falls back to a running average price.

## Run in the cloud
Push to GitHub, open the repo, click **Code > Codespaces > Create**, then run the commands above.
