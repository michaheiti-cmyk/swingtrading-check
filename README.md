
# Swingtrading Check V1.0

Eine mobile Streamlit-Web-App für den Tradingplan Version 1.0.

## Was die App automatisch prüft

- Marktfilter: S&P 500 (USA) oder DAX (Deutschland)
- SMA 5/10/20/50/150/200
- EMA 20
- RSI(14), RSI(2), ADX(14)
- ADR(20)
- 52-Wochen-Hoch/-Tief
- RS-Linie gegen den Vergleichsindex
- Liquidität
- Setup A / B / C
- indikatives Intraday-VWAP und RVOL-at-Time
- Risiko, Stückzahl, Gebührenquote und 200-€-Positionsgrenze

## Warum nicht nur Alpha Vantage?

Beim kostenlosen Alpha-Vantage-Zugang liefert `TIME_SERIES_DAILY` standardmäßig nur 100 Tageswerte; `outputsize=full` ist Premium. Das reicht nicht für SMA150/SMA200 und 52-Wochen-Regeln.

Darum nutzt die App für die lange Historie `yfinance` (Yahoo-Finance-Datenfeed) und verwendet Alpha Vantage optional für einen Zusatzcheck. So werden die 25 kostenlosen Alpha-Vantage-Requests pro Tag geschont.

Für eine echte Order bleiben TradingView und der Broker die finale Referenz für Intraday-Werte und Ausführung.

## Dateien

- `app.py` – die App
- `requirements.txt` – Python-Abhängigkeiten
- `.streamlit/secrets.example.toml` – Beispiel für den API-Key
- `.gitignore` – verhindert versehentliches Hochladen echter Secrets

## In 5 Minuten online bringen

### 1. GitHub-Repository erstellen

Erstelle bei GitHub ein neues Repository, z. B.:

`mein-swingtrading-check`

Lade diese Dateien hoch.

Wichtig: Niemals deine echte `secrets.toml` oder deinen API-Key bei GitHub hochladen.

### 2. Streamlit Community Cloud öffnen

Gehe zu:

https://share.streamlit.io

Mit GitHub anmelden und **Create app** / **Deploy app** wählen.

Repository auswählen und als Main file:

`app.py`

angeben.

### 3. Alpha-Vantage-Key als Secret hinterlegen

Bei der Bereitstellung in **Advanced settings / Secrets** eintragen:

```toml
ALPHAVANTAGE_API_KEY = "DEIN_API_KEY"
```

Nicht in `app.py` schreiben.

Die App funktioniert auch ohne Alpha-Key; der Alpha-Zusatzcheck ist dann nur deaktiviert.

### 4. App starten

Nach dem Deployment bekommst du eine URL wie:

`https://deine-app.streamlit.app`

Diese am Handy öffnen und zum Homescreen hinzufügen.

Danach ist die tägliche Nutzung nur:

1. App öffnen
2. Markt auswählen
3. Ticker eingeben, z. B. `NVDA`
4. **Aktie prüfen**

Für Xetra-Symbole verwendet Yahoo Finance oft `.DE`, z. B. `SAP.DE`.

## Alpha-Vantage-Verbrauch

Der Button **Alpha-Vantage-Zusatzcheck** macht maximal einen OVERVIEW-Request je Ticker; das Ergebnis wird 24 Stunden gecacht.

Die eigentlichen Kurs- und Indikatorberechnungen verbrauchen dadurch keine Alpha-Vantage-Requests.

## Wichtige Grenzen

- Das mechanische Setup-B-Pivot ist nur eine Referenz. Die im Tradingplan geforderte Konsolidierungs-/VCP-Struktur bleibt eine Chartprüfung.
- VWAP und RVOL-at-Time aus Yahoo/yfinance können von TradingView abweichen.
- Earnings aus yfinance werden nur best-effort geladen. Wenn unklar, TradingView und Investor Relations prüfen.
- Bid/Ask ist nicht für jeden Ticker zuverlässig verfügbar.
- Vor Echtgeld-Orders immer die Daten im Broker/TradingView prüfen.
