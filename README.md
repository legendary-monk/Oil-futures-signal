# Oil Futures Signal

A Python pipeline that produces a **daily directional oil signal** (`BULLISH`, `BEARISH`, `NEUTRAL`) by combining:

1. Multi-instrument market data (WTI, Brent, USD, XLE, Gold, NatGas)
2. Polymarket prediction-market probabilities
3. Oil-focused RSS news sentiment
4. Technical and macro feature engineering
5. Consensus-aware score fusion (confidence rises only when factors align)
6. Multi-model quant stack spanning no-arbitrage, stochastic, term-structure, risk, and ML math

The output is sent to Telegram and logged for performance tracking in CSV.

---

## Architecture

```text
market_data + polymarket + news_fetcher
                  ↓
              sentiment
                  ↓
               features
                  ↓
            signal_engine
                  ↓
         telegram_bot + validator
```

Core entrypoint: `main.py`.

---

## Setup

### 1) Requirements

- Python 3.10+
- Internet access (Yahoo Finance, RSS feeds, Polymarket API, Telegram API)

Install dependencies:

```bash
pip install -r requirements.txt
```

### 2) Configure secrets

Set environment variables (recommended):

```bash
export TELEGRAM_TOKEN="<bot-token>"
export TELEGRAM_CHAT_ID="<chat-id>"
```

`TELEGRAM_CHAT_ID` supports comma-separated IDs for multiple recipients.

### 3) Optional: adjust tuning

Review `config.py` for:

- model weights (`WEIGHT_POLYMARKET`, `WEIGHT_SENTIMENT`, etc.)
- thresholds (`BULLISH_THRESHOLD`, `BEARISH_THRESHOLD`)
- feed list and network retry settings
- OPEC uncertainty window + calendar path

---

## Run modes

### Full pipeline

```bash
python main.py
# or
python main.py run
```

### Telegram connection test

```bash
python main.py test
```

### Performance report from saved predictions

```bash
python main.py performance
```

---

## Outputs

- Log file: `oil_signal.log`
- Predictions CSV: `oil_predictions.csv`
- Optional Polymarket cache: `polymarket_cache.json`

A single signal includes fields like:

- `signal`, `confidence`, `raw_score`, `normalized_score`
- factor scores (`polymarket_score`, `sentiment_score`, `trend_score`, `macro_signal`)
- price context (`wti_price`, `brent_price`, `brent_wti_spread`)
- feature context (`vol_regime`, `rsi`, `atr_pct`, `opec_uncertainty`)
- model quality context (`factor_participation`, `consensus_strength`)
- quant context (`quant_score`, `quant_diagnostics`)

---

## What was refined for stronger institutional use

The engine now includes a **consensus overlay**:

- factors are dynamically re-normalized when one source is missing
- high agreement across factors boosts conviction
- mixed/disagreeing regimes are automatically damped toward neutral
- low absolute score regimes are filtered to avoid over-trading noise

This keeps the system entirely free (Yahoo/Polymarket/RSS + local Python) while
improving reliability and reducing false directional calls.

It also includes `quant_stack.py`, which computes a no-cost institutional quant layer
covering:
- cost-of-carry + convenience yield + storage-arbitrage bounds + ACT/360
- GBM, OU, Schwartz-family proxies, jump/Levy proxies, and seasonality
- forward/curve interpolation proxies, PCA, cointegration/VECM/state-space/Kalman
- ARIMA/SARIMA/GARCH/EGARCH/regime/Hurst/Bayesian/EVT
- crack/spark/dark spread proxies + copula tail dependence
- LP/Bellman/HJB/Almgren-Chriss/Kelly proxies
- Black-76 Greeks + VaR/CVaR/stress/liquidity-adjusted risk
- microstructure proxies (queue, Kyle lambda, Roll, IV surface proxies)
- ridge/lasso/boosting/random-forest/neural-SDE/Q-learning proxies

---

## OPEC calendar maintenance

OPEC meeting dates are read from:

- `data/opec_calendar.json`

The system loads **current year + next year** values. Update this file at least yearly.

Example format:

```json
{
  "2026": ["2026-02-03", "2026-05-28"],
  "2027": ["2027-02-04"]
}
```

---

## Local dashboard (free)

A Streamlit dashboard is included for quick analytics from `oil_predictions.csv`.

Run:

```bash
streamlit run dashboard.py
```

Dashboard includes:

- rolling 1d/5d directional accuracy
- confidence-band accuracy
- signal distribution
- factor-to-outcome correlations
- OPEC-window filtering

---

## Automation

A GitHub Actions workflow runs daily and uploads artifacts:

- `oil_signal.log`
- `oil_predictions.csv`

Workflow file: `.github/workflows/daily_signal.yml`

Configure repository secrets:

- `TELEGRAM_TOKEN`
- `TELEGRAM_CHAT_ID`

---

## Limitations

- Market data reliability depends on third-party APIs.
- Prediction-market relevance can be sparse at times.
- Outcome labeling relies on available close prices and can lag around market holidays.
- This is a probabilistic signal model, not guaranteed forecasting.

---

## Disclaimer

This project is for research/education. It is **not financial advice**.
