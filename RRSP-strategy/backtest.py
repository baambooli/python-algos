"""
Backtest script for k1-ETFs portfolio.

Reads stooq EOD data, applies a simple momentum-rotation algo on the tickers
in proftolio_lists/k1-ETFs.tls, and back-tests from 2010-01-01 to today
with $10,000 initial capital and $1 commission per buy/sell.

Algo (full-compounding daily rebalance, same-day execution):
  - Universe: tickers from the configured portfolio file.
  - Single shared equity pool — every dollar of profit is redeployed.
  - Signal per ticker:
      Entry: Price > 50d EMA AND 15d EMA > 30d EMA
      Exit : Price < 50d EMA OR  15d EMA < 30d EMA
  - Each trading day (in order):
      1. Sell tickers whose signal has turned OFF.
      2. Rebalance all tickers whose signal is ON to equal weight of the
         total current equity (target = total_equity / num_active_signals).
         Excess shares in a winner are trimmed, under-allocated entries
         are topped up.
    This is "winner concentration + full compounding": if only one
    ticker has a signal, it gets 100% of equity.
  - Same-day execution: signal and fill both happen at today's close
    (simulates a 3:55 PM decision).
  - $1 commission per buy and per sell.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mplcursors

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
DATA_DIR = REPO_ROOT / "stooq_output_us"
PORTFOLIO_DIR = REPO_ROOT / "proftolio_lists"

PORTFOLIO_NAME = "k1-ETFs"

START_DATE = "2010-01-01"
INITIAL_CAPITAL = 10_000.0
COMMISSION = 1.0
EMA_FAST = 15
EMA_MID = 30
EMA_SLOW = 50

OUT_EQUITY_CSV = f"backtest_{PORTFOLIO_NAME}_equity.csv"
OUT_TRADES_CSV = f"backtest_{PORTFOLIO_NAME}_trades.csv"
OUT_ROUND_TRIPS_CSV = f"backtest_{PORTFOLIO_NAME}_round_trips.csv"
OUT_BH_EQUITY_CSV = f"backtest_{PORTFOLIO_NAME}_buy_and_hold_equity.csv"
OUT_EQUITY_PNG = f"plot_{PORTFOLIO_NAME}_equity_curve.png"
OUT_STOCKS_PNG = f"plot_{PORTFOLIO_NAME}_individual_stocks.png"


def load_portfolio(name: str) -> list[str]:
    path = PORTFOLIO_DIR / f"{name}.tls"
    tickers = [
        line.strip().upper()
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    return tickers


def load_ticker(ticker: str) -> pd.Series | None:
    path = DATA_DIR / f"{ticker}.CSV"
    if not path.exists():
        print(f"  ! missing data file: {path.name}", file=sys.stderr)
        return None
    df = pd.read_csv(
        path,
        usecols=["<DATE>", "<CLOSE>"],
        dtype={"<DATE>": str, "<CLOSE>": np.float64},
    )
    df["date"] = pd.to_datetime(df["<DATE>"], format="%Y%m%d")
    s = df.set_index("date")["<CLOSE>"].astype(np.float64)
    s.name = ticker
    return s.sort_index()


def build_price_panel(tickers: list[str]) -> pd.DataFrame:
    series = {}
    for t in tickers:
        s = load_ticker(t)
        if s is not None:
            series[t] = s
    if not series:
        raise RuntimeError("no price data loaded")
    prices = pd.concat(series.values(), axis=1, keys=series.keys())
    prices = prices.sort_index().ffill()
    return prices


def run_backtest(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    warmup = pd.Timedelta(days=EMA_SLOW * 4)
    prices_full = prices.loc[prices.index >= pd.Timestamp(START_DATE) - warmup]

    ema_fast = prices_full.ewm(span=EMA_FAST, adjust=False).mean()
    ema_mid = prices_full.ewm(span=EMA_MID, adjust=False).mean()
    ema_slow = prices_full.ewm(span=EMA_SLOW, adjust=False).mean()

    in_position_mask = (prices_full > ema_slow) & (ema_fast > ema_mid)

    prices = prices_full.loc[prices_full.index >= pd.Timestamp(START_DATE)]
    in_position_mask = in_position_mask.loc[prices.index]

    tickers = list(prices.columns)
    price_arr = prices.to_numpy()
    signal_arr = in_position_mask.to_numpy()
    dates = prices.index

    cash = INITIAL_CAPITAL
    shares = np.zeros(len(tickers), dtype=np.float64)
    equity = np.empty(len(prices), dtype=np.float64)
    trade_log: list[dict] = []

    for i in range(len(prices)):
        dt = dates[i]
        row_px = price_arr[i]
        row_sig = signal_arr[i]

        # --- 1. SELL exited positions ---
        for j, t in enumerate(tickers):
            px = row_px[j]
            if shares[j] > 0 and not bool(row_sig[j]) and not np.isnan(px):
                sh = shares[j]
                cash += sh * px - COMMISSION
                shares[j] = 0.0
                trade_log.append({
                    "date": dt, "action": "SELL", "ticker": t,
                    "price": float(px), "shares": sh,
                    "cash_after": cash,
                })

        # --- 2. Mark-to-market total equity (post-exits) ---
        total_equity = cash
        for j in range(len(tickers)):
            if shares[j] > 0 and not np.isnan(row_px[j]):
                total_equity += shares[j] * row_px[j]

        # --- 3. REBALANCE ONLY when a NEW signal appears (entry event).
        #        Existing winners keep compounding untouched; we just
        #        redeploy available cash equally across the currently
        #        active signals. ---
        active_idx = [j for j in range(len(tickers))
                      if bool(row_sig[j]) and not np.isnan(row_px[j])]
        new_entries = [j for j in active_idx if shares[j] == 0.0]
        num_signals = len(active_idx)
        if new_entries and num_signals > 0:
            # Each active ticker targets total_equity / num_signals.
            # Winners already above target are left alone; only flat
            # tickers are topped up (with available cash).
            target_value = total_equity / num_signals
            for j in new_entries:
                px = row_px[j]
                desired_cost = min(target_value, cash - COMMISSION)
                if desired_cost > px:
                    sh = (desired_cost - COMMISSION) / px
                    if sh > 0:
                        cash -= sh * px + COMMISSION
                        shares[j] = sh
                        trade_log.append({
                            "date": dt, "action": "BUY",
                            "ticker": tickers[j],
                            "price": float(px), "shares": float(sh),
                            "cash_after": cash,
                        })

        # --- 4. End-of-day equity ---
        mark = 0.0
        for j in range(len(tickers)):
            if shares[j] > 0 and not np.isnan(row_px[j]):
                mark += shares[j] * row_px[j]
        equity[i] = cash + mark

    equity_curve = pd.DataFrame({"equity": equity}, index=prices.index)
    trades_df = pd.DataFrame(trade_log)
    return equity_curve, trades_df


def buy_and_hold(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Equal-weighted buy-and-hold across all tickers from START_DATE."""
    px = prices.loc[prices.index >= pd.Timestamp(START_DATE)].copy()
    first_valid = px.apply(lambda s: s.first_valid_index())
    start_row = first_valid.max()
    px = px.loc[start_row:].ffill()

    n = len(px.columns)
    per_ticker_cash = (INITIAL_CAPITAL - n * COMMISSION) / n
    entry_px = px.iloc[0]
    shares = per_ticker_cash / entry_px

    bh_equity = (px * shares).sum(axis=1)
    bh_equity.iloc[0] = INITIAL_CAPITAL - n * COMMISSION + \
        float((entry_px * shares).sum()) - float((entry_px * shares).sum())
    bh_equity.iloc[0] = float((entry_px * shares).sum())

    bh_df = pd.DataFrame({"equity": bh_equity.values}, index=bh_equity.index)
    trades = pd.DataFrame([
        {"date": px.index[0], "action": "BUY", "ticker": t,
         "price": float(entry_px[t]), "shares": float(shares[t]),
         "cash_after": 0.0}
        for t in px.columns
    ])
    return bh_df, trades


def compute_round_trips(trades: pd.DataFrame) -> pd.DataFrame:
    """Pair BUY -> (ADD/TRIM)* -> SELL into one round trip per ticker.

    Cashflow P&L: sum of sell/trim proceeds minus buy/add costs across
    the life of the position, with one commission per event.
    """
    cols = ["ticker", "entry_date", "exit_date", "entry_price",
            "exit_price", "shares", "pnl", "return"]
    if trades.empty:
        return pd.DataFrame(columns=cols)

    rows = []
    open_pos: dict[str, dict] = {}
    for _, tr in trades.iterrows():
        t = tr["ticker"]
        a = tr["action"]
        px = float(tr["price"])
        sh = float(tr["shares"])

        if a == "BUY":
            open_pos[t] = {
                "date": tr["date"], "price": px, "shares": sh,
                "cost": sh * px + COMMISSION, "proceeds": 0.0,
                "initial_shares": sh,
            }
        elif a == "ADD" and t in open_pos:
            p = open_pos[t]
            p["cost"] += sh * px + COMMISSION
            p["shares"] += sh
        elif a == "TRIM" and t in open_pos:
            p = open_pos[t]
            p["proceeds"] += sh * px - COMMISSION
            p["shares"] -= sh
        elif a == "SELL" and t in open_pos:
            p = open_pos.pop(t)
            p["proceeds"] += sh * px - COMMISSION
            pnl = p["proceeds"] - p["cost"]
            ret = pnl / p["cost"] if p["cost"] > 0 else np.nan
            rows.append({
                "ticker": t,
                "entry_date": p["date"], "exit_date": tr["date"],
                "entry_price": p["price"], "exit_price": px,
                "shares": p["initial_shares"], "pnl": pnl, "return": ret,
            })
    return pd.DataFrame(rows, columns=cols)


def compute_metrics(equity: pd.DataFrame, trades: pd.DataFrame,
                    round_trips: pd.DataFrame | None) -> dict:
    start_val = INITIAL_CAPITAL
    end_val = float(equity["equity"].iloc[-1])
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (end_val / start_val) ** (1 / years) - 1 if years > 0 else np.nan

    daily_ret = equity["equity"].pct_change().dropna()
    ann_vol = daily_ret.std() * np.sqrt(252)
    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) \
        if daily_ret.std() > 0 else np.nan
    downside = daily_ret[daily_ret < 0]
    sortino = (daily_ret.mean() / downside.std()) * np.sqrt(252) \
        if len(downside) and downside.std() > 0 else np.nan

    running_max = equity["equity"].cummax()
    drawdown = equity["equity"] / running_max - 1.0
    max_dd = float(drawdown.min())
    calmar = cagr / abs(max_dd) if max_dd < 0 else np.nan

    best_day = float(daily_ret.max()) if len(daily_ret) else np.nan
    worst_day = float(daily_ret.min()) if len(daily_ret) else np.nan
    pct_up_days = float((daily_ret > 0).mean()) if len(daily_ret) else np.nan

    if round_trips is not None and len(round_trips):
        n_rt = len(round_trips)
        wins = round_trips[round_trips["pnl"] > 0]
        losses = round_trips[round_trips["pnl"] <= 0]
        win_rate = len(wins) / n_rt
        loss_rate = len(losses) / n_rt
        avg_win = float(wins["pnl"].mean()) if len(wins) else np.nan
        avg_loss = float(losses["pnl"].mean()) if len(losses) else np.nan
        profit_factor = (wins["pnl"].sum() / abs(losses["pnl"].sum())) \
            if len(losses) and losses["pnl"].sum() != 0 else np.nan
        expectancy = float(round_trips["pnl"].mean())
    else:
        n_rt = 0
        win_rate = loss_rate = avg_win = avg_loss = np.nan
        profit_factor = expectancy = np.nan

    total_commission = len(trades) * COMMISSION

    def _pct(x): return f"{x * 100:.2f}%" if not np.isnan(x) else "n/a"
    def _usd(x): return f"${x:,.2f}" if not np.isnan(x) else "n/a"
    def _num(x): return f"{x:.2f}" if not np.isnan(x) else "n/a"

    return {
        "Period": f"{equity.index[0].date()} -> {equity.index[-1].date()}",
        "Years": f"{years:.2f}",
        "Initial capital": _usd(start_val),
        "Final equity": _usd(end_val),
        "Total return": _pct(end_val / start_val - 1),
        "CAGR": _pct(cagr),
        "Annualized volatility": _pct(ann_vol),
        "Sharpe ratio": _num(sharpe),
        "Sortino ratio": _num(sortino),
        "Calmar ratio": _num(calmar),
        "Max drawdown": _pct(max_dd),
        "Best day": _pct(best_day),
        "Worst day": _pct(worst_day),
        "% up days": _pct(pct_up_days),
        "Round-trip trades": f"{n_rt}" if n_rt else "n/a",
        "Win rate": _pct(win_rate),
        "Loss rate": _pct(loss_rate),
        "Avg win ($)": _usd(avg_win),
        "Avg loss ($)": _usd(avg_loss),
        "Profit factor": _num(profit_factor),
        "Expectancy ($/trade)": _usd(expectancy),
        "Total executions": f"{len(trades)}",
        "Total commissions": _usd(total_commission),
    }


def print_comparison(strategy_metrics: dict, bh_metrics: dict) -> None:
    key_w = max(len(k) for k in strategy_metrics)
    col_w = 22
    total_w = key_w + col_w * 2 + 6
    print("=" * total_w)
    print("BACKTEST METRICS — STRATEGY vs BUY & HOLD".center(total_w))
    print("=" * total_w)
    print(f"{'Metric'.ljust(key_w)} | "
          f"{'Strategy'.rjust(col_w)} | "
          f"{'Buy & Hold'.rjust(col_w)}")
    print("-" * total_w)
    for k in strategy_metrics:
        s = strategy_metrics[k]
        b = bh_metrics.get(k, "n/a")
        print(f"{k.ljust(key_w)} | {s.rjust(col_w)} | {b.rjust(col_w)}")
    print("=" * total_w)


def plot_results(equity: pd.DataFrame, bh_equity: pd.DataFrame,
                 prices: pd.DataFrame, out_dir: Path) -> list[Path]:
    out_paths = []

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})
    last_date = equity.index[-1]
    last_val = float(equity["equity"].iloc[-1])
    bh_last_val = float(bh_equity["equity"].iloc[-1])
    bh_last_date = bh_equity.index[-1]
    eq_line, = ax1.plot(
        equity.index, equity["equity"], color="#1f77b4", lw=1.5,
        label=f"Strategy (last: ${last_val:,.0f})",
    )
    bh_line, = ax1.plot(
        bh_equity.index, bh_equity["equity"], color="#2ca02c", lw=1.5,
        ls="--", label=f"Buy & Hold (last: ${bh_last_val:,.0f})",
    )
    ax1.annotate(
        f"B&H: ${bh_last_val:,.0f}",
        xy=(bh_last_date, bh_last_val),
        xytext=(8, -15), textcoords="offset points",
        color="#2ca02c", fontweight="bold", va="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                  ec="#2ca02c", alpha=0.85),
    )
    ax1.axhline(INITIAL_CAPITAL, color="gray", ls="--", lw=0.8,
                label=f"Initial ${INITIAL_CAPITAL:,.0f}")
    ax1.annotate(
        f"${last_val:,.0f}",
        xy=(last_date, last_val),
        xytext=(8, 0), textcoords="offset points",
        color="#1f77b4", fontweight="bold", va="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                  ec="#1f77b4", alpha=0.85),
    )
    ax1.set_title("Portfolio Equity Curve")
    ax1.set_ylabel("Equity ($)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")
    ax1.set_yscale("log")

    running_max = equity["equity"].cummax()
    dd = (equity["equity"] / running_max - 1.0) * 100
    dd_fill = ax2.fill_between(dd.index, dd.values, 0,
                                color="#d62728", alpha=0.4)
    dd_line, = ax2.plot(dd.index, dd.values, color="#d62728", lw=0.8,
                         alpha=0)  # invisible line for hover hit-testing
    last_dd = float(dd.iloc[-1])
    ax2.annotate(
        f"{last_dd:.2f}%",
        xy=(last_date, last_dd),
        xytext=(8, 0), textcoords="offset points",
        color="#d62728", fontweight="bold", va="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                  ec="#d62728", alpha=0.85),
    )
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)

    cur1 = mplcursors.cursor([eq_line, bh_line, dd_line], hover=True)

    @cur1.connect("add")
    def _fmt1(sel):
        x, y = sel.target
        d = mdates.num2date(x).strftime("%Y-%m-%d")
        if sel.artist is eq_line:
            sel.annotation.set_text(f"Strategy\n{d}\nEquity: ${y:,.2f}")
        elif sel.artist is bh_line:
            sel.annotation.set_text(f"Buy & Hold\n{d}\nEquity: ${y:,.2f}")
        else:
            sel.annotation.set_text(f"{d}\nDrawdown: {y:.2f}%")
        sel.annotation.get_bbox_patch().set(alpha=0.9, fc="#ffffcc")

    fig.tight_layout()
    p1 = out_dir / OUT_EQUITY_PNG
    fig.savefig(p1, dpi=120)
    out_paths.append(p1)

    fig, ax = plt.subplots(figsize=(12, 6))
    sub = prices.loc[prices.index >= pd.Timestamp(START_DATE)]
    normalized = sub.divide(sub.bfill().iloc[0]) * 100.0
    lines = []
    for col in normalized.columns:
        last_n = float(normalized[col].iloc[-1])
        ln, = ax.plot(normalized.index, normalized[col], lw=1.2,
                       label=f"{col} (last: {last_n:,.1f})")
        lines.append(ln)
        ax.annotate(
            f"{col}: {last_n:,.1f}",
            xy=(normalized.index[-1], last_n),
            xytext=(8, 0), textcoords="offset points",
            color=ln.get_color(), fontweight="bold", va="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec=ln.get_color(), alpha=0.85),
        )
    ax.set_title("Individual ETFs — Normalized Price (base = 100)")
    ax.set_ylabel("Price (normalized)")
    ax.set_xlabel("Date")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    cur2 = mplcursors.cursor(lines, hover=True)

    @cur2.connect("add")
    def _fmt2(sel):
        x, y = sel.target
        d = mdates.num2date(x).strftime("%Y-%m-%d")
        label = sel.artist.get_label().split(" (")[0]
        sel.annotation.set_text(f"{label}\n{d}\nValue: {y:,.2f}")
        sel.annotation.get_bbox_patch().set(alpha=0.9, fc="#ffffcc")

    fig.tight_layout()
    p2 = out_dir / OUT_STOCKS_PNG
    fig.savefig(p2, dpi=120)
    out_paths.append(p2)

    return out_paths


def main() -> None:
    tickers = load_portfolio(PORTFOLIO_NAME)
    print(f"Portfolio: {PORTFOLIO_NAME} -> {tickers}")

    prices = build_price_panel(tickers)
    print(f"Loaded price panel: {prices.shape[0]} rows x "
          f"{prices.shape[1]} tickers")

    equity, trades = run_backtest(prices)
    round_trips = compute_round_trips(trades)
    bh_equity, bh_trades = buy_and_hold(prices)

    out_equity = ROOT / OUT_EQUITY_CSV
    out_trades = ROOT / OUT_TRADES_CSV
    out_rt = ROOT / OUT_ROUND_TRIPS_CSV
    out_bh = ROOT / OUT_BH_EQUITY_CSV
    equity.to_csv(out_equity)
    trades.to_csv(out_trades, index=False)
    round_trips.to_csv(out_rt, index=False)
    bh_equity.to_csv(out_bh)

    strat_metrics = compute_metrics(equity, trades, round_trips)
    bh_metrics = compute_metrics(bh_equity, bh_trades, None)
    print_comparison(strat_metrics, bh_metrics)

    plots = plot_results(equity, bh_equity, prices, ROOT)
    print(f"Wrote: {out_equity.name}")
    print(f"Wrote: {out_trades.name}")
    print(f"Wrote: {out_rt.name}")
    print(f"Wrote: {out_bh.name}")
    for p in plots:
        print(f"Wrote: {p.name}")

    plt.show()


if __name__ == "__main__":
    main()
