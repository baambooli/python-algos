import pandas as pd, numpy as np

# U = "."  # CSV EOD ETFs data are in the current folder
U = "D:/share/stooq_output_us"
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
def hv20(close):
    return np.log(close/close.shift()).rolling(20).std()*np.sqrt(252)*100
def adx(df, n=14):
    h,l,c = df["high"], df["low"], df["close"]
    up = h.diff(); dn = -l.diff()
    plus = np.where((up>dn)&(up>0), up, 0.0); minus = np.where((dn>up)&(dn>0), dn, 0.0)
    trn = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1).ewm(alpha=1/n,adjust=False).mean()
    pdi = 100*pd.Series(plus,index=h.index).ewm(alpha=1/n,adjust=False).mean()/trn
    mdi = 100*pd.Series(minus,index=h.index).ewm(alpha=1/n,adjust=False).mean()/trn
    dx = 100*(pdi-mdi).abs()/(pdi+mdi)
    return dx.ewm(alpha=1/n,adjust=False).mean()

SLIP = 0.0005  # 0.05% per traded notional
START = "2010-09-01"  # after TQQQ/SOXL warmup
END = "2026-06-05"

# shared market series
spy = data["SPY"]["close"]; qqq = data["QQQ"]["close"]; tlt = data["TLT"]["close"]; smh = data["SMH"]["close"]
spy_hv = hv20(spy); spy_hv_e50 = ema(spy_hv.dropna(), 50).reindex(spy.index)
vol_ok = (spy_hv < spy_hv_e50)   # proxy for VIX < EMA50(VIX)

bench_map = {"TQQQ": qqq, "TECL": qqq, "SOXL": smh, "SPXL": spy}
rs_map = {"TQQQ": qqq/tlt, "TECL": qqq/tlt, "SOXL": smh/spy, "SPXL": spy/tlt}

def run_alloc(etf, alloc_fn):
    """alloc_fn returns a Series of target allocation (0..1) computed on close, executed same close."""
    px = data[etf]["close"]
    tgt = alloc_fn(etf).reindex(px.index).fillna(0.0).clip(0,1)
    tgt = tgt.loc[START:END]; px = px.loc[START:END]
    ret = px.pct_change().fillna(0.0)
    # position held during day t's return is target set at close of t-1
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
    tim = tgt[tgt>0].count()/len(tgt) if tgt is not None else 1.0
    avg_exp = tgt.mean() if tgt is not None else 1.0
    ntr = int((tgt.diff().abs()>1e-9).sum()) if tgt is not None else 1
    y2022 = eq.loc["2022"].iloc[-1]/eq.loc["2021"].iloc[-1]-1 if "2021" in eq.index.strftime("%Y") else np.nan
    return dict(strategy=name, final_10k=round(eq.iloc[-1]*10000), CAGR=f"{cagr:.1%}", MaxDD=f"{dd:.1%}",
                Sharpe=round(sharpe,2), yr2022=f"{y2022:.1%}", avg_expo=f"{avg_exp:.0%}", trades=ntr)

# ---------------- Strategies ----------------
def buyhold(etf):
    px = data[etf]["close"]
    return pd.Series(1.0, index=px.index)

def strat_H(etf):
    df = data[etf]; c = df["close"]
    e10,e20,e50,e200 = ema(c,10),ema(c,20),ema(c,50),ema(c,200)
    ratio = atr(df,14)/atr(df,100)
    gate_open = ratio < (ratio.rolling(100).mean() + 2*ratio.rolling(100).std())
    cond = [(e50>e200),(c>e50),(e10>e20)]
    below50 = c<e50; two_below = below50 & below50.shift(1).fillna(False)
    sell1 = (e10<e20); sell2 = two_below
    pos = pd.Series(0.0, index=c.index); cur = 0.0
    n = len(c)
    s1=sell1.values; s2=sell2.values; g=gate_open.values
    cnt = (cond[0].astype(int)+cond[1].astype(int)+cond[2].astype(int)).values
    for i in range(n):
        if s1[i] and s2[i]: cur = 0.0
        elif s1[i] or s2[i]: cur *= 0.5
        elif g[i]:
            t = 0.33*cnt[i]
            if t > cur: cur = t
        pos.iloc[i] = cur
    return pos

def strat_1530(etf):
    c = data[etf]["close"]
    e15,e30,e50 = ema(c,15),ema(c,30),ema(c,50)
    return ((c>e50)&(e15>e30)).astype(float)

def strat_dlite(etf, ema_short=10, ema_mid=20, ema_base=50, hv_soft=45, hv_hard=60):
    df = data[etf]; c = df["close"]
    bench = bench_map[etf]; master = (bench > ema(bench,200)).reindex(c.index).fillna(False)
    e_s,e_m,e_b = ema(c,ema_short),ema(c,ema_mid),ema(c,ema_base)
    p1 = (e_s>e_m)&(c>e_b)
    p3 = vol_ok.reindex(c.index).fillna(False)
    rs = rs_map[etf]; p4 = (rs > ema(rs,100)).reindex(c.index).fillna(False)
    cnt = p1.astype(int)+p3.astype(int)+p4.astype(int)
    tier = cnt.map({0:0.0,1:0.25,2:0.60,3:1.0})
    hv = hv20(c)
    tier = tier.where(hv<=hv_soft, np.minimum(tier,0.25))
    tier = tier.where(hv<=hv_hard, 0.0)
    below = c<e_b; two_below = below & below.shift(1).fillna(False)
    tier = tier.where(~two_below, 0.0)
    tier = tier.where(master, 0.0)
    return tier

def strat_scaledC(etf):
    df = data[etf]; c = df["close"]
    bench = bench_map[etf]; master = (bench > ema(bench,200)).reindex(c.index).fillna(False)
    e10,e20,e50 = ema(c,10),ema(c,20),ema(c,50)
    p1 = (e10>e20)&(c>e50)
    a = atr(df,14)
    p3 = ((a/c) < 0.045) & vol_ok.reindex(c.index).fillna(False)
    ax = adx(df,14); p4 = (ax>20)&(ax>ax.shift(5))
    rs = rs_map[etf]; p5 = (rs > ema(rs,100)).reindex(c.index).fillna(False)
    cnt = p1.astype(int)+p3.astype(int)+p4.astype(int)+p5.astype(int)
    tier = cnt.map({0:0.0,1:0.0,2:0.25,3:0.60,4:1.0})  # 4 available pillars (breadth missing)
    below = c<e50; two_below = below & below.shift(1).fillna(False)
    atr_spike = a > 1.3*a.rolling(20).mean()
    tier = tier.where(~two_below,0.0).where(~atr_spike,0.0).where(master,0.0)
    return tier


def strat_H_master(etf):
    df = data[etf]; c = df["close"]
    bench = bench_map[etf]; master = (bench > ema(bench,200)).reindex(c.index).fillna(False).values
    e10,e20,e50,e200 = ema(c,10),ema(c,20),ema(c,50),ema(c,200)
    ratio = atr(df,14)/atr(df,100)
    g = (ratio < (ratio.rolling(100).mean()+2*ratio.rolling(100).std())).values
    cnt = ((e50>e200).astype(int)+(c>e50).astype(int)+(e10>e20).astype(int)).values
    below50 = c<e50; two = (below50 & below50.shift(1).fillna(False)).values
    s1=(e10<e20).values
    pos=pd.Series(0.0,index=c.index); cur=0.0
    for i in range(len(c)):
        if not master[i]: cur=0.0
        elif s1[i] and two[i]: cur=0.0
        elif s1[i] or two[i]: cur*=0.5
        elif g[i]:
            t=0.33*cnt[i]
            if t>cur: cur=t
        pos.iloc[i]=cur
    return pos

rows = []
for etf in ["TQQQ","SOXL","SPXL","TECL"]:
    for name, fn in [("Buy&Hold",buyhold),("H (as written)",strat_H),
                     ("H + master EMA200",strat_H_master),
                     ("15/30+EMA50 (RRSP doc)",strat_1530),
                     ("D-lite* with HV cap",strat_dlite),
                     ("D-lite* no HV cap",lambda e: strat_dlite(e,hv_soft=1e9,hv_hard=1e9)),
                     ("Scaled C*",strat_scaledC)]:
        eq,tgt = run_alloc(etf, fn)
        rows.append({"ETF":etf, **metrics(eq,tgt,name)})
res = pd.DataFrame(rows)
pd.set_option("display.width",220)
print(res.to_string(index=False))
