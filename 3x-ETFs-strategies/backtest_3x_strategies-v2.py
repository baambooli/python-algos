import os
import pandas as pd, numpy as np

# CSV files live in the same folder as this script; resolve relative to it
# so the script works regardless of the current working directory.
# U = os.path.dirname(os.path.abspath(__file__))

U = os.path.dirname("D:/share/stooq_output_us")
def load(t):
    df = pd.read_csv(f"{U}/{t}.CSV")
    df.columns = [c.strip("<>").lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.set_index("date")[["open","high","low","close"]].astype(float)
    return df

data = {t: load(t) for t in ["QQQ","SMH","SOXL","SPXL","SPY","TECL","TLT","TQQQ"]}

def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def atr(df, n=14):
    h,l,c = df["high"], df["low"], df["close"]
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

SLIP = 0.0005
START = "2010-09-01"
END = "2026-06-05"

spy = data["SPY"]["close"]; qqq = data["QQQ"]["close"]; smh = data["SMH"]["close"]
bench_map = {"TQQQ": qqq, "TECL": qqq, "SOXL": smh, "SPXL": spy}

def run_alloc(etf, alloc_fn):
    px = data[etf]["close"]
    tgt = alloc_fn(etf).reindex(px.index).fillna(0.0).clip(0,1)
    tgt = tgt.loc[START:END]; px = px.loc[START:END]
    ret = px.pct_change().fillna(0.0)
    pos = tgt.shift(1).fillna(0.0)
    turnover = tgt.diff().abs().fillna(tgt.iloc[0])
    eq = (1 + pos*ret - turnover*SLIP).cumprod()
    return eq, tgt

def metrics(eq, tgt=None, name=""):
    yrs = (eq.index[-1]-eq.index[0]).days/365.25
    cagr = eq.iloc[-1]**(1/yrs)-1
    dd = (eq/eq.cummax()-1).min()
    dr = eq.pct_change().dropna()
    sharpe = dr.mean()/dr.std()*np.sqrt(252) if dr.std()>0 else np.nan
    avg_exp = tgt.mean() if tgt is not None else 1.0
    ntr = int((tgt.diff().abs()>1e-9).sum()) if tgt is not None else 1
    y2022 = eq.loc["2022"].iloc[-1]/eq.loc["2021"].iloc[-1]-1 if "2021" in eq.index.strftime("%Y") else np.nan
    return dict(strategy=name, final_10k=round(eq.iloc[-1]*10000), CAGR=f"{cagr:.1%}", MaxDD=f"{dd:.1%}",
                Sharpe=round(sharpe,2), yr2022=f"{y2022:.1%}", avg_expo=f"{avg_exp:.0%}", trades=ntr)

# ---------------- Strategy H v1 (original, as written on screenshot) ----------------
def strat_H_v1(etf):
    df = data[etf]; c = df["close"]
    e10,e20,e50,e200 = ema(c,10),ema(c,20),ema(c,50),ema(c,200)
    ratio = atr(df,14)/atr(df,100)
    gate_open = ratio < (ratio.rolling(100).mean() + 2*ratio.rolling(100).std())
    cond = [(e50>e200),(c>e50),(e10>e20)]
    below50 = c<e50; two_below = below50 & below50.shift(1).fillna(False)
    sell1 = (e10<e20); sell2 = two_below
    pos = pd.Series(0.0, index=c.index); cur = 0.0
    s1=sell1.values; s2=sell2.values; g=gate_open.values
    cnt = (cond[0].astype(int)+cond[1].astype(int)+cond[2].astype(int)).values
    for i in range(len(c)):
        if s1[i] and s2[i]: cur = 0.0
        elif s1[i] or s2[i]: cur *= 0.5
        elif g[i]:
            t = 0.33*cnt[i]
            if t > cur: cur = t
        pos.iloc[i] = cur
    return pos

# ---------------- Strategy H v2 (+ master benchmark EMA200 filter) ----------------
def strat_H_v2(etf):
    df = data[etf]; c = df["close"]
    bench = bench_map[etf]; master = (bench > ema(bench,200)).reindex(c.index).fillna(False).values
    e10,e20,e50,e200 = ema(c,10),ema(c,20),ema(c,50),ema(c,200)
    ratio = atr(df,14)/atr(df,100)
    g = (ratio < (ratio.rolling(100).mean()+2*ratio.rolling(100).std())).values
    cnt = ((e50>e200).astype(int)+(c>e50).astype(int)+(e10>e20).astype(int)).values
    below50 = c<e50; two = (below50 & below50.shift(1).fillna(False)).values
    s1=(e10<e20).values; s2=two
    pos=pd.Series(0.0,index=c.index); cur=0.0
    for i in range(len(c)):
        if not master[i]: cur=0.0
        elif s1[i] and s2[i]: cur=0.0          # <- the flaw: forces 100% exit even if Regime still bullish
        elif s1[i] or s2[i]: cur*=0.5
        elif g[i]:
            t=0.33*cnt[i]
            if t>cur: cur=t
        pos.iloc[i]=cur
    return pos

# ---------------- Strategy H v3 (fixed sell logic + trade-noise filter) ----------------
def strat_H_v3(etf, rebalance_threshold=0.05):
    df = data[etf]; c = df["close"]
    bench = bench_map[etf]; master = (bench > ema(bench,200)).reindex(c.index).fillna(False).values
    e10,e20,e50,e200 = ema(c,10),ema(c,20),ema(c,50),ema(c,200)
    ratio = atr(df,14)/atr(df,100)
    g = (ratio < (ratio.rolling(100).mean()+2*ratio.rolling(100).std())).values
    cnt = ((e50>e200).astype(int)+(c>e50).astype(int)+(e10>e20).astype(int)).values
    below50 = c<e50; two = (below50 & below50.shift(1).fillna(False)).values
    s1=(e10<e20).values; s2=two
    raw = pd.Series(0.0,index=c.index); cur=0.0
    for i in range(len(c)):
        if not master[i]:
            cur = 0.0
        else:
            if s1[i]: cur *= 0.5
            if s2[i]: cur *= 0.5
            if not s1[i] and not s2[i] and g[i]:
                t = 0.33*cnt[i]
                if t > cur: cur = t
        raw.iloc[i]=cur
    # trade-noise filter: only actually move your position if the change exceeds threshold
    out = raw.copy(); vals = out.values.copy(); last_traded = vals[0]
    for i in range(1, len(vals)):
        if abs(vals[i]-last_traded) < rebalance_threshold:
            vals[i] = last_traded
        else:
            last_traded = vals[i]
    return pd.Series(vals, index=out.index)

# ---------------- Run comparison ----------------
if __name__ == "__main__":
    rows = []
    for etf in ["TQQQ","SOXL","SPXL","TECL"]:
        for name, fn in [("H v1 (original)",strat_H_v1),
                          ("H v2 (+master filter)",strat_H_v2),
                          ("H v3 (fixed sell logic + trade filter)",strat_H_v3)]:
            eq,tgt = run_alloc(etf, fn)
            rows.append({"ETF":etf, **metrics(eq,tgt,name)})
    res = pd.DataFrame(rows)
    pd.set_option("display.width",220)
    print(res.to_string(index=False))
