
import math
from datetime import date, datetime, time
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Swingtrading Check V1.0", page_icon="📈", layout="centered")

# ----------------------------
# Helpers
# ----------------------------

def clean_yf(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance may return ticker as an extra column level
        if len(df.columns.levels) >= 2:
            try:
                df.columns = df.columns.get_level_values(0)
            except Exception:
                df.columns = ["_".join(map(str, c)).strip() for c in df.columns]
    rename = {c: str(c).title() for c in df.columns}
    df = df.rename(columns=rename)
    needed = ["Open", "High", "Low", "Close", "Volume"]
    for c in needed:
        if c not in df.columns:
            return pd.DataFrame()
    df = df[needed].dropna(subset=["Open", "High", "Low", "Close"]).copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.sort_index()


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_daily(symbol: str, period: str = "2y") -> pd.DataFrame:
    df = yf.download(
        symbol,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    return clean_yf(df)


@st.cache_data(ttl=3600, show_spinner=False)
def load_intraday(symbol: str, period: str = "5d", interval: str = "5m") -> pd.DataFrame:
    df = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
        prepost=False,
    )
    return clean_yf(df)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_ticker_info(symbol: str) -> dict:
    try:
        t = yf.Ticker(symbol)
        info = t.info or {}
        return info
    except Exception:
        return {}


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_earnings(symbol: str):
    """Best-effort earnings date from yfinance. Returns (date|None, source_text)."""
    try:
        t = yf.Ticker(symbol)
        cal = t.calendar
        if isinstance(cal, dict):
            candidates = cal.get("Earnings Date") or cal.get("EarningsDate")
            if candidates:
                if not isinstance(candidates, (list, tuple)):
                    candidates = [candidates]
                for x in candidates:
                    try:
                        d = pd.Timestamp(x).date()
                        if d >= date.today():
                            return d, "Yahoo Finance / yfinance"
                    except Exception:
                        pass
        elif isinstance(cal, pd.DataFrame) and not cal.empty:
            vals = cal.values.flatten().tolist()
            for x in vals:
                try:
                    d = pd.Timestamp(x).date()
                    if d >= date.today():
                        return d, "Yahoo Finance / yfinance"
                except Exception:
                    pass
    except Exception:
        pass
    return None, "nicht verifiziert"


def alpha_overview(symbol: str, api_key: str) -> dict:
    url = "https://www.alphavantage.co/query"
    params = {"function": "OVERVIEW", "symbol": symbol, "apikey": api_key}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    if "Note" in data or "Information" in data:
        raise RuntimeError(data.get("Note") or data.get("Information"))
    return data


def num(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def fmt(x, digits=2, suffix=""):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "?"
    try:
        return f"{float(x):.{digits}f}{suffix}"
    except Exception:
        return "?"


def wilder_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    # Flat/no-loss sequences can legitimately approach 100
    rsi = rsi.fillna(100)
    return rsi


def wilder_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1/period, adjust=False, min_periods=period).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1/period, adjust=False, min_periods=period).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1/period, adjust=False, min_periods=period).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for p in [5, 10, 20, 50, 150, 200]:
        d[f"SMA{p}"] = d["Close"].rolling(p).mean()
    d["EMA20"] = d["Close"].ewm(span=20, adjust=False).mean()
    d["RSI14"] = wilder_rsi(d["Close"], 14)
    d["RSI2"] = wilder_rsi(d["Close"], 2)
    d["ADX14"] = wilder_adx(d, 14)
    d["ADRpct"] = (((d["High"] - d["Low"]) / d["Close"]) * 100).rolling(20).mean()
    d["Vol50"] = d["Volume"].rolling(50).mean()
    return d


def truth_value(v):
    """Normalize Python/NumPy/Pandas booleans; keep missing values unknown."""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    try:
        return bool(v)
    except Exception:
        return None


def status_icon(v):
    t = truth_value(v)
    if t is True:
        return "✅"
    if t is False:
        return "❌"
    return "?"


def business_days_until(d):
    if not d:
        return None
    today = np.datetime64(date.today(), "D")
    target = np.datetime64(d, "D")
    if target < today:
        return -1
    return int(np.busday_count(today, target))


def get_market_symbol(market: str):
    return "^GSPC" if market == "USA" else "^GDAXI"


def get_currency(market: str):
    return "USD" if market == "USA" else "EUR"


def completed_daily_only(df: pd.DataFrame, market: str) -> pd.DataFrame:
    """
    Use only completed daily candles for all daily-plan rules.
    If today's daily bar is already present while the home market is still
    open (plus a small settlement buffer), remove it.
    """
    if df is None or df.empty:
        return df

    d = df.copy()
    if market == "USA":
        now_local = datetime.now(ZoneInfo("America/New_York"))
        # Small buffer so Yahoo/feeds have time to finalize the daily bar.
        completed_after = time(16, 15)
    else:
        now_local = datetime.now(ZoneInfo("Europe/Berlin"))
        completed_after = time(17, 45)

    today_local = now_local.date()

    # If the latest bar is today's session and the official session is not
    # considered finalized yet, exclude it from daily indicators.
    if d.index[-1].date() == today_local and now_local.time() < completed_after:
        d = d.iloc[:-1].copy()

    return d


@st.cache_data(ttl=3600, show_spinner=False)
def usd_per_eur():
    df = load_daily("EURUSD=X", "5d")
    if df.empty:
        return np.nan
    return float(df["Close"].iloc[-1])


def to_eur(amount, currency):
    if amount is None or (isinstance(amount, float) and np.isnan(amount)):
        return np.nan
    if currency == "EUR":
        return float(amount)
    fx = usd_per_eur()  # USD per 1 EUR
    if not np.isfinite(fx) or fx <= 0:
        return np.nan
    return float(amount) / fx


def latest_intraday_vwap(df: pd.DataFrame):
    if df.empty:
        return np.nan
    today = df.index[-1].date()
    x = df[df.index.date == today].copy()
    if x.empty:
        return np.nan
    typical = (x["High"] + x["Low"] + x["Close"]) / 3
    vol = x["Volume"].replace(0, np.nan)
    den = vol.cumsum()
    vwap = (typical * vol).cumsum() / den
    return float(vwap.iloc[-1])


def approx_rvol_at_time(df: pd.DataFrame, lookback_days=10):
    """Approximate cumulative RVOL at same intraday bar using yfinance 5m data."""
    if df.empty:
        return np.nan
    x = df.copy()
    x["d"] = x.index.date
    x["t"] = x.index.strftime("%H:%M")
    dates = sorted(pd.unique(x["d"]))
    if len(dates) < 3:
        return np.nan
    cur_date = dates[-1]
    cur = x[x["d"] == cur_date].copy()
    if cur.empty:
        return np.nan
    last_time = cur.index[-1].strftime("%H:%M")
    cur_cum = float(cur["Volume"].sum())
    prev = dates[max(0, len(dates)-1-lookback_days):-1]
    comps = []
    for d in prev:
        z = x[(x["d"] == d) & (x["t"] <= last_time)]
        if not z.empty:
            comps.append(float(z["Volume"].sum()))
    if not comps or np.mean(comps) <= 0:
        return np.nan
    return cur_cum / float(np.mean(comps))


def momentum_gain_63(close: pd.Series):
    """Max low-to-later-high gain within last 63 sessions."""
    s = close.dropna().tail(63)
    if len(s) < 30:
        return np.nan
    min_so_far = np.inf
    best = -np.inf
    for v in s.values:
        min_so_far = min(min_so_far, v)
        if min_so_far > 0:
            best = max(best, v / min_so_far - 1)
    return best * 100


def table_rows(rows):
    return pd.DataFrame(rows, columns=["Regel", "Wert", "Ergebnis"])


# ----------------------------
# UI
# ----------------------------

st.title("📈 Swingtrading Check V1.0")
st.caption("Ticker rein → Tagesdaten & Setups A/B/C prüfen. Intraday-Werte sind indikativ und vor einer Order in TradingView gegenprüfen.")
st.caption("Hauptdatenquelle: Yahoo Finance über yfinance · Alpha Vantage: optionaler Zusatzcheck")
st.caption("Alle Tagesregeln werden ausschließlich mit der letzten ABGESCHLOSSENEN Tageskerze berechnet.")

with st.sidebar:
    st.header("Einstellungen")
    market = st.selectbox("Heimatmarkt", ["USA", "Deutschland"], index=0)
    depot = st.number_input("Depotwert (€)", min_value=1.0, value=600.0, step=10.0)
    roundtrip_fee = st.number_input("Geschätzte Round-trip-Kosten (€)", min_value=0.0, value=2.0, step=0.1)
    current_open_risk = st.number_input("Aktuell offenes Risiko (€)", min_value=0.0, value=0.0, step=1.0)
    invested_eur = st.number_input("Aktuell investiert (€)", min_value=0.0, value=0.0, step=10.0)
    manual_earnings_text = st.text_input(
        "Earnings manuell (optional, YYYY-MM-DD)",
        value="",
        placeholder="z. B. 2026-10-28"
    )
    st.divider()
    try:
        alpha_key = st.secrets.get("ALPHAVANTAGE_API_KEY", "")
    except Exception:
        alpha_key = ""
    st.write("Alpha Vantage:", "✅ Secret gefunden" if alpha_key else "— kein Secret")
    if "alpha_calls" not in st.session_state:
        st.session_state.alpha_calls = 0
    if "alpha_cache" not in st.session_state:
        st.session_state.alpha_cache = {}
    alpha_counter_placeholder = st.empty()
    alpha_counter_placeholder.caption(
        f"Alpha-Calls in dieser Sitzung: {st.session_state.alpha_calls}"
    )

symbol = st.text_input("Ticker / Yahoo-Symbol", placeholder="z. B. NVDA oder SAP.DE").strip().upper()
analyze = st.button("Aktie prüfen", type="primary", use_container_width=True)

if not analyze:
    st.info("Ticker eingeben und **Aktie prüfen** drücken.")
    st.stop()

if not symbol:
    st.error("Bitte einen Ticker eingeben.")
    st.stop()

manual_earnings = None
if manual_earnings_text.strip():
    try:
        manual_earnings = datetime.strptime(manual_earnings_text.strip(), "%Y-%m-%d").date()
    except ValueError:
        st.error("Earnings-Datum bitte als YYYY-MM-DD eingeben, z. B. 2026-10-28.")
        st.stop()

with st.spinner("Daten werden geladen und Regeln geprüft …"):
    daily_raw = load_daily(symbol, "2y")
    market_symbol = get_market_symbol(market)
    market_raw = load_daily(market_symbol, "2y")
    info = load_ticker_info(symbol)

if daily_raw.empty or len(daily_raw) < 210:
    st.error("Für diesen Ticker konnten nicht genügend Tagesdaten geladen werden. Prüfe den Yahoo-Symbolnamen (z. B. SAP.DE für Xetra).")
    st.stop()

if market_raw.empty or len(market_raw) < 210:
    st.error("Marktdaten für den Vergleichsindex konnten nicht geladen werden.")
    st.stop()

# Tradingplan uses completed daily closes. During the live session, today's
# unfinished candle is excluded from all daily indicators and setup rules.
daily_rules_raw = completed_daily_only(daily_raw, market)
market_rules_raw = completed_daily_only(market_raw, market)

if len(daily_rules_raw) < 210 or len(market_rules_raw) < 210:
    st.error("Nach Ausschluss der laufenden Tageskerze sind nicht genügend abgeschlossene Tagesdaten verfügbar.")
    st.stop()

daily = add_indicators(daily_rules_raw)
market_df = add_indicators(market_rules_raw)

latest = daily.iloc[-1]
prev = daily.iloc[-2]
m = market_df.iloc[-1]

currency = get_currency(market)
price = float(latest["Close"])
asof = daily.index[-1].date()

# Earnings: manual > yfinance best effort
if manual_earnings:
    earnings_date = manual_earnings
    earnings_source = "manuell"
else:
    earnings_date, earnings_source = load_earnings(symbol)
earn_days = business_days_until(earnings_date)

# Long-term metrics
window252 = daily.tail(252)
high52 = float(window252["High"].max())
low52 = float(window252["Low"].min())
below_high = (high52 - price) / high52 * 100 if high52 > 0 else np.nan
above_low = (price - low52) / low52 * 100 if low52 > 0 else np.nan

# RS line alignment
joined = pd.concat(
    [daily["Close"].rename("stock"), market_df["Close"].rename("market")],
    axis=1, join="inner"
).dropna()
rs_now = rs_22 = np.nan
if len(joined) >= 23:
    rs = joined["stock"] / joined["market"]
    rs_now = float(rs.iloc[-1])
    rs_22 = float(rs.iloc[-23])

# Market filters
market_200 = bool(m["Close"] > m["SMA200"])
market_10_20 = bool(m["SMA10"] > m["SMA20"])

# Liquidity
avg_value20 = float((daily["Close"] * daily["Volume"]).tail(20).mean())
min_liq = price >= 5 and avg_value20 >= 5_000_000

# Bid/ask best effort
bid = num(info.get("bid"))
ask = num(info.get("ask"))
spread_abs = ask - bid if np.isfinite(bid) and np.isfinite(ask) and ask >= bid and bid > 0 else np.nan

# Trend template
trend_checks = {
    "Schlusskurs > 50-SMA": price > latest["SMA50"],
    "Schlusskurs > 150-SMA": price > latest["SMA150"],
    "Schlusskurs > 200-SMA": price > latest["SMA200"],
    "50-SMA > 150-SMA": latest["SMA50"] > latest["SMA150"],
    "150-SMA > 200-SMA": latest["SMA150"] > latest["SMA200"],
    "200-SMA heute > vor 22 HT": latest["SMA200"] > daily["SMA200"].iloc[-23],
    "≤ 25 % unter 52W-Hoch": below_high <= 25,
    "≥ 25 % über 52W-Tief": above_low >= 25,
    "RS-Linie höher als vor 22 HT": rs_now > rs_22 if np.isfinite(rs_now) and np.isfinite(rs_22) else None,
    "Keine Earnings in nächsten 5 HT": (earn_days is not None and earn_days > 5),
    "Liquidität (Kurs + Handelswert)": min_liq,
}
trend_all = all(truth_value(v) is True for v in trend_checks.values())

# Setup A
ema_dist_pct = (float(latest["Low"]) - float(latest["EMA20"])) / float(latest["EMA20"]) * 100
adx_rising = bool(latest["ADX14"] > daily["ADX14"].iloc[-4])
rsi_zone = bool((latest["RSI14"] >= 40) and (latest["RSI14"] <= 50))
# Reproducible pullback-volume proxy: from most recent 10-session close high to latest day
last10 = daily.tail(10)
peak_idx = last10["Close"].idxmax()
pullback_slice = daily.loc[peak_idx:].iloc[1:] if peak_idx in daily.index else daily.tail(3)
if len(pullback_slice) == 0:
    pullback_slice = daily.tail(1)
pb_vol_avg = float(pullback_slice["Volume"].mean())
vol_pullback_ok = pb_vol_avg < float(latest["Vol50"])
trigger_ok = bool((latest["Close"] > prev["Close"]) and (latest["Close"] > latest["EMA20"]) and (latest["Volume"] >= 1.5 * latest["Vol50"]))
a_core = {
    "Marktfilter A/B": market_200 and market_10_20,
    "Aktienfilter A/B": trend_all,
    "ADX(14) > 30": latest["ADX14"] > 30,
    "ADX heute > vor 3 HT": adx_rising,
    "Tagestief innerhalb ±1 % der 20-EMA": abs(ema_dist_pct) <= 1,
    "RSI(14) 40–50": rsi_zone,
    "Pullback-Volumen < Vol50": vol_pullback_ok,
    "Trigger-Tag gültig": trigger_ok,
}
a_ready_daily = all(truth_value(v) is True for v in a_core.values())

# Setup B
mom63 = momentum_gain_63(daily["Close"])
sma10_rise = bool(latest["SMA10"] > daily["SMA10"].iloc[-6])
sma20_rise = bool(latest["SMA20"] > daily["SMA20"].iloc[-6])
# Mechanical reference pivot: highest high of prior 20 completed sessions, excluding current day.
pivot20 = float(daily["High"].iloc[-21:-1].max())
b_core = {
    "Marktfilter A/B": market_200 and market_10_20,
    "Aktienfilter A/B": trend_all,
    "Momentum ≥ 30 % in letzten 1–3 Monaten": np.isfinite(mom63) and mom63 >= 30,
    "10-SMA steigt über 5 HT": sma10_rise,
    "20-SMA steigt über 5 HT": sma20_rise,
}
b_candidate = all(truth_value(v) is True for v in b_core.values())
# Consolidation/Pivot remains partly visual by plan.
b_visual_required = True

# Setup C
rsi2_cross = bool((prev["RSI2"] >= 10) and (latest["RSI2"] < 10))
c_core = {
    "Marktindex > 200-SMA": market_200,
    "Aktie/ETF > 200-SMA": price > latest["SMA200"],
    "RSI(2) kreuzt von ≥10 auf <10": rsi2_cross,
    "Liquidität": min_liq,
}
c_ready = all(truth_value(v) is True for v in c_core.values())

# Intraday (best effort, unofficial)
intraday = load_intraday(symbol)
vwap = latest_rvol = np.nan
last_intraday_price = np.nan
if not intraday.empty:
    vwap = latest_intraday_vwap(intraday)
    latest_rvol = approx_rvol_at_time(intraday, 10)
    last_intraday_price = float(intraday["Close"].iloc[-1])

# Alpha optional overview
alpha_data = {}
alpha_status_col1, alpha_status_col2 = st.columns([3,1])
with alpha_status_col1:
    alpha_clicked = st.button("Alpha-Vantage-Zusatzcheck (max. 1 API-Call)", use_container_width=True)
with alpha_status_col2:
    st.metric("Alpha-Calls", st.session_state.alpha_calls)

if alpha_clicked:
    if not alpha_key:
        st.warning("Kein Alpha-Vantage-Key als Streamlit Secret hinterlegt.")
    else:
        cache_key = f"{symbol}:{date.today().isoformat()}"
        if cache_key in st.session_state.alpha_cache:
            alpha_data = st.session_state.alpha_cache[cache_key]
            st.session_state["alpha_last_message"] = "Alpha-Vantage-Daten für diesen Ticker wurden heute in dieser Sitzung bereits geladen – 0 neue API-Calls."
            st.info(st.session_state["alpha_last_message"])
        else:
            try:
                alpha_data = alpha_overview(symbol, alpha_key)
                st.session_state.alpha_cache[cache_key] = alpha_data
                st.session_state.alpha_calls += 1
                alpha_counter_placeholder.caption(
                    f"Alpha-Calls in dieser Sitzung: {st.session_state.alpha_calls}"
                )
                st.session_state["alpha_last_message"] = "Alpha-Vantage-Zusatzcheck erfolgreich – 1 API-Call verbraucht."
                st.success(st.session_state["alpha_last_message"])
            except Exception as e:
                st.warning(f"Alpha-Vantage-Zusatzcheck fehlgeschlagen: {e}")

# ----------------------------
# Results
# ----------------------------

st.subheader("Kurzstatus")
col1, col2, col3 = st.columns(3)
with col1:
    if a_ready_daily:
        st.success("A 🟡\n\nDaily bereit")
    else:
        st.error("A 🔴\n\nNicht erfüllt")
with col2:
    if b_candidate:
        st.warning("B 🟡\n\nKandidat")
    else:
        st.error("B 🔴\n\nNicht erfüllt")
with col3:
    if c_ready:
        st.info("C 🔵\n\nPaper-Signal")
    else:
        st.error("C 🔴\n\nKein Signal")

st.caption("A/B benötigen vor Echtgeld-Entry zusätzlich die im Plan vorgesehenen Intraday-/Chart-Bestätigungen. C bleibt bis 30 eigene Trades Paper-Trading.")

tab1, tab2, tab3, tab4 = st.tabs(["Übersicht", "Setup A", "Setup B", "Setup C"])

with tab1:
    st.subheader("Basisdaten")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Letzter abgeschlossener Schlusskurs", f"{price:.2f} {currency}")
        st.metric("ADR(20)", fmt(latest["ADRpct"], 2, " %"))
        st.metric("RSI(14)", fmt(latest["RSI14"], 1))
        st.metric("RSI(2)", fmt(latest["RSI2"], 1))
    with c2:
        st.metric("ADX(14)", fmt(latest["ADX14"], 1))
        st.metric("52W-Hoch", f"{high52:.2f} {currency}")
        st.metric("52W-Tief", f"{low52:.2f} {currency}")
        st.metric("Ø Handelswert 20T", f"{avg_value20/1_000_000:.1f} Mio. {currency}")

    st.write(f"**Regel-Datenstand (letzte abgeschlossene Tageskerze):** {asof} · **Vergleichsindex:** {market_symbol}")
    if earnings_date:
        st.write(f"**Earnings:** {earnings_date} ({earn_days} Handelstage; Quelle: {earnings_source})")
    else:
        st.warning("Earnings-Termin nicht zuverlässig verifiziert. Vor Echtgeld-Trade in TradingView/Investor Relations prüfen.")

    if np.isfinite(last_intraday_price):
        st.write(f"**Indikativer aktueller Intraday-Preis:** {last_intraday_price:.2f} {currency}")
    if np.isfinite(vwap):
        st.write(f"**Indikativer Intraday-VWAP:** {vwap:.2f} {currency}")
    if np.isfinite(latest_rvol):
        st.write(f"**Indikatives RVOL-at-Time (10 Tage, kumulativ):** {latest_rvol:.2f}")
    st.caption("Intraday-Werte stammen hier aus dem verfügbaren Yahoo/yfinance-Datenfeed und sind nicht identisch garantiert mit TradingView. Vor Order in TradingView bestätigen.")

    st.subheader("Marktfilter")
    market_table = table_rows([
        ("Index > 200-SMA", f"{m['Close']:.2f} vs {m['SMA200']:.2f}", status_icon(market_200)),
        ("10-SMA > 20-SMA (nur A/B)", f"{m['SMA10']:.2f} vs {m['SMA20']:.2f}", status_icon(market_10_20)),
    ])
    st.dataframe(market_table, hide_index=True, use_container_width=True)

    st.subheader("Aktienfilter A/B")
    rows = []
    values = {
        "Schlusskurs > 50-SMA": f"{price:.2f} vs {latest['SMA50']:.2f}",
        "Schlusskurs > 150-SMA": f"{price:.2f} vs {latest['SMA150']:.2f}",
        "Schlusskurs > 200-SMA": f"{price:.2f} vs {latest['SMA200']:.2f}",
        "50-SMA > 150-SMA": f"{latest['SMA50']:.2f} vs {latest['SMA150']:.2f}",
        "150-SMA > 200-SMA": f"{latest['SMA150']:.2f} vs {latest['SMA200']:.2f}",
        "200-SMA heute > vor 22 HT": f"{latest['SMA200']:.2f} vs {daily['SMA200'].iloc[-23]:.2f}",
        "≤ 25 % unter 52W-Hoch": f"{below_high:.1f} %",
        "≥ 25 % über 52W-Tief": f"{above_low:.1f} %",
        "RS-Linie höher als vor 22 HT": f"{fmt(rs_now,4)} vs {fmt(rs_22,4)}",
        "Keine Earnings in nächsten 5 HT": f"{earn_days if earn_days is not None else '?'} HT",
        "Liquidität (Kurs + Handelswert)": f"{price:.2f}; {avg_value20/1_000_000:.1f} Mio.",
    }
    for k, v in trend_checks.items():
        rows.append((k, values[k], status_icon(v)))
    st.dataframe(table_rows(rows), hide_index=True, use_container_width=True)

    st.subheader("Indikatoren")
    ind_df = pd.DataFrame({
        "Indikator": ["SMA5","SMA10","SMA20","SMA50","SMA150","SMA200","EMA20","RSI14","RSI2","ADX14","ADR20","Vol50"],
        "Wert": [
            latest["SMA5"],latest["SMA10"],latest["SMA20"],latest["SMA50"],latest["SMA150"],latest["SMA200"],
            latest["EMA20"],latest["RSI14"],latest["RSI2"],latest["ADX14"],latest["ADRpct"],latest["Vol50"]
        ]
    })
    st.dataframe(ind_df, hide_index=True, use_container_width=True)

    if alpha_data:
        st.subheader("Alpha-Vantage-Zusatzcheck")
        st.write({
            "Sector": alpha_data.get("Sector"),
            "Industry": alpha_data.get("Industry"),
            "52WeekHigh": alpha_data.get("52WeekHigh"),
            "52WeekLow": alpha_data.get("52WeekLow"),
        })

with tab2:
    st.subheader("Setup A – Pullback")
    a_values = {
        "Marktfilter A/B": f"200: {status_icon(market_200)} · 10>20: {status_icon(market_10_20)}",
        "Aktienfilter A/B": "alle erfüllt" if trend_all else "mind. eine Regel fehlt",
        "ADX(14) > 30": fmt(latest["ADX14"],1),
        "ADX heute > vor 3 HT": f"{fmt(latest['ADX14'],1)} vs {fmt(daily['ADX14'].iloc[-4],1)}",
        "Tagestief innerhalb ±1 % der 20-EMA": f"{ema_dist_pct:+.2f} %",
        "RSI(14) 40–50": fmt(latest["RSI14"],1),
        "Pullback-Volumen < Vol50": f"{pb_vol_avg:,.0f} vs {latest['Vol50']:,.0f}",
        "Trigger-Tag gültig": f"Close {latest['Close']:.2f}; Vol {latest['Volume']:,.0f}",
    }
    st.dataframe(table_rows([(k, a_values[k], status_icon(v)) for k,v in a_core.items()]), hide_index=True, use_container_width=True)
    trigger_high = float(latest["High"])
    st.write(f"**Trigger-Hoch:** {trigger_high:.2f} {currency}")
    if a_ready_daily:
        st.warning("Daily-Regeln erfüllt. Echtgeld-Entry erst, wenn der Kurs am Folgetag über dem Trigger-Hoch UND über dem Tages-VWAP liegt.")
    else:
        st.error("Setup A aktuell nicht vollständig erfüllt.")

with tab3:
    st.subheader("Setup B – Momentum-Breakout")
    b_values = {
        "Marktfilter A/B": f"200: {status_icon(market_200)} · 10>20: {status_icon(market_10_20)}",
        "Aktienfilter A/B": "alle erfüllt" if trend_all else "mind. eine Regel fehlt",
        "Momentum ≥ 30 % in letzten 1–3 Monaten": fmt(mom63,1," %"),
        "10-SMA steigt über 5 HT": f"{latest['SMA10']:.2f} vs {daily['SMA10'].iloc[-6]:.2f}",
        "20-SMA steigt über 5 HT": f"{latest['SMA20']:.2f} vs {daily['SMA20'].iloc[-6]:.2f}",
    }
    st.dataframe(table_rows([(k, b_values[k], status_icon(v)) for k,v in b_core.items()]), hide_index=True, use_container_width=True)
    st.write(f"**Mechanischer Referenz-Pivot (20T-Hoch ex heute):** {pivot20:.2f} {currency}")
    st.info("Die 2–8-wöchige Konsolidierung/VCP-Struktur und der endgültige Pivot bleiben nach deinem Plan eine Chartprüfung. Der 20T-Pivot hier ist nur eine reproduzierbare Referenz.")
    if np.isfinite(vwap) and np.isfinite(last_intraday_price):
        st.write(f"Intraday: Preis {last_intraday_price:.2f} · VWAP {vwap:.2f} · RVOL {fmt(latest_rvol,2)}")
    if b_candidate:
        st.warning("Setup B ist ein Daily-Kandidat. Vor Entry: Konsolidierung/Pivot visuell bestätigen, Kurs > Pivot, Kurs > VWAP und RVOL-at-Time ≥ 1,5.")
    else:
        st.error("Setup B aktuell kein regelkonformer Kandidat.")

with tab4:
    st.subheader("Setup C – RSI(2) Mean Reversion")
    c_values = {
        "Marktindex > 200-SMA": f"{m['Close']:.2f} vs {m['SMA200']:.2f}",
        "Aktie/ETF > 200-SMA": f"{price:.2f} vs {latest['SMA200']:.2f}",
        "RSI(2) kreuzt von ≥10 auf <10": f"{prev['RSI2']:.1f} → {latest['RSI2']:.1f}",
        "Liquidität": f"{avg_value20/1_000_000:.1f} Mio. {currency}/Tag",
    }
    st.dataframe(table_rows([(k, c_values[k], status_icon(v)) for k,v in c_core.items()]), hide_index=True, use_container_width=True)
    st.write(f"**5-SMA:** {latest['SMA5']:.2f} {currency}")
    if c_ready:
        st.info("Setup C Signal vorhanden → nach V1.0 PAPER-TRADE; Entry nächste reguläre Eröffnung.")
    else:
        st.error("Setup C aktuell kein Signal.")

# ----------------------------
# Trade planner (only if A daily-ready, B candidate, or C ready)
# ----------------------------

if a_ready_daily or b_candidate or c_ready:
    st.divider()
    st.subheader("Trade-Planer")

    default_entry = float(last_intraday_price) if np.isfinite(last_intraday_price) else price
    entry = st.number_input(f"Geplanter Entry ({currency})", min_value=0.01, value=float(round(default_entry, 2)), step=0.01)

    if a_ready_daily:
        ref_stop = float(latest["Low"])
        stop_label = "Trigger-/Tagestief"
    elif b_candidate:
        ref_stop = float(daily["Low"].tail(10).min())
        stop_label = "10T-Swing-Tief (Referenz)"
    else:
        ref_stop = float(daily["Low"].tail(3).min())
        stop_label = "kurzfristiges Tief (Referenz)"

    stop = st.number_input(f"Technischer Stop ({currency}) – {stop_label}", min_value=0.01, value=float(round(ref_stop, 2)), step=0.01)
    trailing = st.selectbox("Trailing-Regel A/B", ["10-SMA", "20-SMA"])

    if stop >= entry:
        st.error("Stop muss unter dem Entry liegen.")
    else:
        risk_unit_cur = entry - stop
        stop_pct = risk_unit_cur / entry * 100
        adr = float(latest["ADRpct"])
        max_stop_pct = min(adr, 8.0)
        stop_ok = stop_pct <= max_stop_pct

        risk_budget_eur = min(12.0, depot * 0.02)
        risk_unit_eur = to_eur(risk_unit_cur, currency)
        entry_eur = to_eur(entry, currency)
        if not np.isfinite(risk_unit_eur) or not np.isfinite(entry_eur):
            st.error("Währungsumrechnung konnte nicht geladen werden.")
        else:
            available_risk = max(0.0, risk_budget_eur - roundtrip_fee)
            shares_by_risk = math.floor(available_risk / risk_unit_eur) if risk_unit_eur > 0 else 0
            shares_by_value = math.floor(200.0 / entry_eur) if entry_eur > 0 else 0
            shares = max(0, min(shares_by_risk, shares_by_value))
            position_eur = shares * entry_eur
            actual_r = shares * risk_unit_eur + roundtrip_fee if shares > 0 else 0.0
            fee_ratio = roundtrip_fee / actual_r if actual_r > 0 else np.inf
            fee_ok = fee_ratio <= 0.20
            open_risk_limit = min(24.0, depot * 0.04)
            portfolio_ok = current_open_risk + actual_r <= open_risk_limit
            invested_ok = invested_eur + position_eur <= depot * 0.95
            spread_ok = None
            if np.isfinite(spread_abs):
                spread_ok = spread_abs < 0.10 * risk_unit_cur

            c1, c2, c3 = st.columns(3)
            c1.metric("Stückzahl", shares)
            c2.metric("Positionswert", f"{position_eur:.2f} €")
            c3.metric("1R", f"{actual_r:.2f} €")

            rows = [
                ("Stop ≤ min(ADR, 8%)", f"{stop_pct:.2f}% ≤ {max_stop_pct:.2f}%", status_icon(stop_ok)),
                ("Gebührenquote ≤ 20%", f"{fee_ratio*100:.1f}%" if np.isfinite(fee_ratio) else "?", status_icon(fee_ok)),
                ("Offenes Gesamtrisiko", f"{current_open_risk + actual_r:.2f} € ≤ {open_risk_limit:.2f} €", status_icon(portfolio_ok)),
                ("≤ 95% Kapital gebunden", f"{invested_eur + position_eur:.2f} € ≤ {depot*0.95:.2f} €", status_icon(invested_ok)),
                ("Spread < 10% Stopdistanz", f"{fmt(spread_abs,4)} {currency}" if np.isfinite(spread_abs) else "nicht verifiziert", status_icon(spread_ok)),
            ]
            st.dataframe(table_rows(rows), hide_index=True, use_container_width=True)

            if shares <= 0:
                st.error("Mit Risikobudget, Gebühren und 200-€-Positionsgrenze ergibt sich keine handelbare ganze Aktie.")
            elif all(truth_value(x) is True for x in [stop_ok, fee_ok, portfolio_ok, invested_ok]) and truth_value(spread_ok) is not False:
                st.success("Die berechenbaren Risiko-/Kostenregeln passen. Setup-/Intraday-Pflichtregeln trotzdem separat beachten.")
                # R levels in native currency using actual total R translated back approximately
                r_native = (actual_r - roundtrip_fee) / shares if shares > 0 else np.nan
                st.write(f"**+1R:** {entry + r_native:.2f} {currency}  ·  **+2R:** {entry + 2*r_native:.2f}  ·  **+3R:** {entry + 3*r_native:.2f}")
                st.write(f"**Trailing:** {trailing}")
            else:
                st.error("Mindestens eine Risiko-/Kostenregel verhindert den Trade.")

st.divider()
st.caption("Hinweis: Diese App ist ein Regelprüfer, keine Anlageberatung. Datenfeeds können verzögert/abweichend sein. Vor Echtgeld-Orders insbesondere Earnings, Spread, VWAP/RVOL und den tatsächlichen IBKR-Ausführungspreis prüfen.")
