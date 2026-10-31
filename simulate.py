"""India 10-Min Paper Scalper - backtest / paper-trading simulator (EDUCATIONAL ONLY).
Same rules as the Pine Script: EMA 9/21 cross + RSI + VWAP, exit on SL/TP/flip/fade/10-min/15:15.
Usage:
  python simulate.py --symbol ^NSEI --interval 5m --period 30d --qty 25
  python simulate.py --demo          # synthetic data, no internet needed
"""
import argparse
import numpy as np
import pandas as pd

IST = "Asia/Kolkata"


def load_data(symbol, interval, period, demo):
    if demo:
        rng = np.random.default_rng(7)
        days = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=15)
        frames = []
        price = 24000.0
        for d in days:
            idx = pd.date_range(d.strftime("%Y-%m-%d") + " 09:15", periods=75, freq="5min", tz=IST)
            r = rng.normal(0, 12, len(idx)).cumsum() + np.sin(np.linspace(0, 6, len(idx))) * 25
            close = price + r
            open_ = np.r_[price, close[:-1]]
            high = np.maximum(open_, close) + rng.uniform(0, 8, len(idx))
            low = np.minimum(open_, close) - rng.uniform(0, 8, len(idx))
            frames.append(pd.DataFrame({"Open": open_, "High": high, "Low": low,
                                        "Close": close, "Volume": rng.integers(1e4, 1e5, len(idx))}, index=idx))
            price = close[-1]
        return pd.concat(frames)
    import yfinance as yf
    df = yf.download(symbol, interval=interval, period=period, progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    df.index = df.index.tz_localize("UTC").tz_convert(IST) if df.index.tz is None else df.index.tz_convert(IST)
    return df


def add_indicators(df, fast, slow, rsi_len, atr_len):
    c = df["Close"]
    df["ema_f"] = c.ewm(span=fast, adjust=False).mean()
    df["ema_s"] = c.ewm(span=slow, adjust=False).mean()
    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / rsi_len, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / rsi_len, adjust=False).mean()
    df["rsi"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - c.shift()).abs(), (df["Low"] - c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / atr_len, adjust=False).mean()
    tp = (df["High"] + df["Low"] + c) / 3
    day = df.index.date
    vol = df["Volume"].replace(0, np.nan)
    if vol.isna().all():  # index data has no volume -> running mean of typical price
        df["vwap"] = tp.groupby(day).expanding().mean().reset_index(level=0, drop=True)
    else:
        df["vwap"] = (tp * vol).groupby(day).cumsum() / vol.groupby(day).cumsum()
    df["cross_up"] = (df["ema_f"] > df["ema_s"]) & (df["ema_f"].shift() <= df["ema_s"].shift())
    df["cross_dn"] = (df["ema_f"] < df["ema_s"]) & (df["ema_f"].shift() >= df["ema_s"].shift())
    return df


def run(df, a, tf_secs):
    horizon_bars = max(1, round(a.horizon * 60 / tf_secs))
    pos, entry, entry_i, sl, tp = 0, 0.0, 0, 0.0, 0.0
    trades = []
    rows = df.reset_index()
    tcol = rows.columns[0]
    for i, r in rows.iterrows():
        mod = r[tcol].hour * 60 + r[tcol].minute
        can_enter = 9 * 60 + 20 <= mod < 15 * 60
        square_off = mod >= 15 * 60 + 15
        closed_now = False
        if pos != 0 and i > entry_i:
            held = i - entry_i
            hit_sl = r.Low <= sl if pos == 1 else r.High >= sl
            hit_tp = r.High >= tp if pos == 1 else r.Low <= tp
            flip = r.cross_dn if pos == 1 else r.cross_up
            fade = (r.Close < r.ema_f and r.rsi < 50) if pos == 1 else (r.Close > r.ema_f and r.rsi > 50)
            why, px = None, r.Close
            if hit_sl: why, px = "Stop-loss", sl
            elif hit_tp: why, px = "Target", tp
            elif square_off: why = "Square-off 15:15"
            elif flip: why = "EMA reversal"
            elif fade: why = "Momentum fade"
            elif held >= horizon_bars: why = "10-min window"
            if why:
                gross = pos * (px - entry) * a.qty
                cost = (entry + px) * a.qty * a.cost_pct / 100
                trades.append({"entry_time": rows.loc[entry_i, tcol], "exit_time": r[tcol],
                               "side": "LONG" if pos == 1 else "SHORT", "entry": round(entry, 2),
                               "exit": round(px, 2), "reason": why, "net_pnl": round(gross - cost, 2)})
                pos, closed_now = 0, True
        long_sig = r.cross_up and 50 < r.rsi < a.rsi_ob and (not a.use_vwap or r.Close > r.vwap)
        short_sig = r.cross_dn and a.rsi_os < r.rsi < 50 and (not a.use_vwap or r.Close < r.vwap)
        if pos == 0 and not closed_now and can_enter and (long_sig or short_sig) and not np.isnan(r.atr):
            pos = 1 if long_sig else -1
            entry, entry_i = r.Close, i
            sl = entry - pos * r.atr * a.sl_mult
            tp = entry + pos * r.atr * a.tp_mult
    return pd.DataFrame(trades)


def summary(t):
    if t.empty:
        print("No trades generated."); return
    net = t["net_pnl"]
    eq = net.cumsum()
    dd = (eq - eq.cummax()).min()
    print(f"Trades: {len(t)} | Win rate: {(net > 0).mean() * 100:.1f}% | Net P&L: Rs {net.sum():,.2f}")
    print(f"Avg win: Rs {net[net > 0].mean() if (net > 0).any() else 0:,.2f} | Avg loss: Rs {net[net <= 0].mean() if (net <= 0).any() else 0:,.2f} | Max drawdown: Rs {dd:,.2f}")
    print("\nExit reasons:\n" + t["reason"].value_counts().to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="^NSEI")
    p.add_argument("--interval", default="5m")
    p.add_argument("--period", default="30d")
    p.add_argument("--qty", type=int, default=25)
    p.add_argument("--horizon", type=int, default=10, help="max holding minutes")
    p.add_argument("--fast", type=int, default=9)
    p.add_argument("--slow", type=int, default=21)
    p.add_argument("--rsi-ob", type=int, default=65)
    p.add_argument("--rsi-os", type=int, default=35)
    p.add_argument("--sl-mult", type=float, default=1.0)
    p.add_argument("--tp-mult", type=float, default=1.5)
    p.add_argument("--cost-pct", type=float, default=0.02)
    p.add_argument("--no-vwap", dest="use_vwap", action="store_false")
    p.add_argument("--demo", action="store_true")
    a = p.parse_args()
    unit = {"m": 60, "h": 3600}[a.interval[-1]]
    tf_secs = int(a.interval[:-1]) * unit
    df = add_indicators(load_data(a.symbol, a.interval, a.period, a.demo), a.fast, a.slow, 14, 14)
    trades = run(df, a, tf_secs)
    summary(trades)
    trades.to_csv("trades.csv", index=False)
    print("\nSaved trades.csv")
