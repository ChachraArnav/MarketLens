# MarketLens

[![CI](https://github.com/ChachraArnav/MarketLens/actions/workflows/ci.yml/badge.svg)](https://github.com/ChachraArnav/MarketLens/actions/workflows/ci.yml)

Historical analytics and downside-risk research for 13 Canadian-listed ETFs, with an interactive Streamlit dashboard.

MarketLens asks one question: **using only what is known at today's close, can a model rank the days most likely to be followed by a sharp five-day drop better than chance?** It answers it with a chronological train / validation / test design, a pre-registered final test on data the models never saw, and results reported against a random-guessing baseline.

![Overview](docs/screenshots/overview.png)

## Results

Every number below comes from `python -m marketlens all` and is checked by CI.

**2026 untouched test.** The model and threshold were chosen on 2025, frozen, then scored once on January 2 to November 3, 2026 (2,743 ETF-days).

| Risk definition | Chosen on 2025 | Average precision | Random guessing | Lift | Recall | Precision |
| --- | --- | --- | --- | --- | --- | --- |
| Fixed: next 5 days fall at least 3% | Random Forest, threshold 0.10 | **0.251** | 0.113 | **2.2×** | 77.1% | 20.7% |
| Volatility-adjusted: fall beyond 1.5σ | Logistic Regression, threshold 0.50 | 0.067 | 0.058 | 1.15× | 40.6% | 5.4% |

**What that means.** For the fixed 3% definition, the Random Forest ranks risky days about twice as well as chance, and it did so both on 2025 (0.150 vs 0.069) and on unseen 2026. At its threshold it flags roughly three quarters of the drops, with about one in five alerts being correct. The volatility-adjusted label is barely predictable from these features. That is a useful negative result: most of what the models pick up is *how volatile an ETF already is*. In the Random Forest, `Volatility_20D` is by far the most important input.

**2025 validation, all models** (average precision; random guessing = event rate):

| Risk definition | Baseline | Logistic Regression | Random Forest | Neural Network |
| --- | --- | --- | --- | --- |
| Fixed 3% | 0.069 | 0.125 | **0.150** | 0.146 |
| Volatility-adjusted | 0.056 | **0.067** | 0.065 | 0.064 |

Accuracy is deliberately not the headline. Because fewer than 1 in 10 days are drops, a model that never predicts a drop scores 93% accuracy and is useless.

![2026 test results](docs/screenshots/test-results.png)

## Method

| | |
| --- | --- |
| **Data** | Daily adjusted prices from Yahoo Finance, January 2021 to November 2026: VFV, XEQT, ZNQ, XIC, XEF, XEC, XFN, XEG, XBB, XGD, XRE, XIT and CGL (all `.TO`). Validated for non-positive prices, duplicates and missing values. |
| **Features** (known at day *t*) | 1-day return, 5-day return, 20-day volatility, price vs 10-day average, 1-day volume change |
| **Labels** (next 5 trading days) | *Fixed*: return ≤ −3%. *Adjusted*: return ≤ −1.5 × 20-day volatility × √5 |
| **Train** | 2021 to 2024, with the last 5 trading days purged because their labels look into 2025 |
| **Validation** | 2025: compare models, choose a model by average precision, then a threshold by F1 |
| **Test** | 2026: scored once with the frozen choices |
| **Models** | Majority-class baseline, logistic regression (balanced), random forest, small neural network (16-8). Scaling lives inside a scikit-learn pipeline, so it's fit on training data only. |

**Leakage checks are tests, not promises.** `tests/test_features.py` shocks every price after a cutoff date and asserts that no feature on or before that date changes. `tests/test_splits.py` asserts that the periods are ordered, don't overlap and that training is purged.

**Limitations.** This covers one test period in one market regime and 13 related ETFs, whose returns are correlated, so the effective sample is smaller than 2,743. Overlapping five-day labels make neighbouring days dependent. The results are historical, exclude costs and are not a trading strategy or investment advice.

## The dashboard

| Page | What it shows |
| --- | --- |
| Overview | Price history, total return, annualized volatility, max drawdown and downside-event frequency per ETF |
| ETF Explorer | Growth of $100 for up to six ETFs over their shared history |
| Portfolio | Buy-and-hold simulator with weights and a benchmark |
| Diversification | Daily-return correlation matrix and pair explorer |
| ML Research | 2025 validation: pick a risk definition, model and threshold; see the confusion matrix and precision-recall curve |
| 2026 Test | The final evaluation above, with precision-recall against random guessing and feature importance |
| Risk Analysis | Downside-event frequency by ETF under both definitions |

![Portfolio simulator](docs/screenshots/portfolio.png)

## Run it

Requires Python 3.11 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows PowerShell
# source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
streamlit run app.py
```

The processed data and results are committed, so the dashboard works immediately.

### Rebuild the research

```bash
python -m marketlens all          # validate prices, build features and labels, evaluate
python -m marketlens download     # optional: fetch fresh prices from Yahoo Finance first
```

Downloading fresh prices changes the 2026 test period, so the numbers above will move. Commit the regenerated `results/` with any README changes.

### Tests and lint

```bash
pip install -r requirements-dev.txt
pytest
ruff check .
```

CI runs both on every push. It also rebuilds the research from the saved prices and fails if the committed results don't reproduce.

## Project structure

```
app.py                     Streamlit dashboard
marketlens/
  config.py                Tickers, dates, features, label definitions, split boundaries
  data.py                  Download, combine and validate prices
  features.py              Features (past only) and labels (next 5 days)
  splits.py                Chronological split with purge gap
  models.py                Model definitions
  evaluate.py              Validation comparison, threshold sweep, single 2026 test
  analytics.py             Returns, volatility, drawdown, buy-and-hold portfolios
  __main__.py              Command line: python -m marketlens ...
tests/                     pytest suite
data/raw/                  One CSV per ETF from Yahoo Finance
data/processed/            Combined prices and the modelling dataset
results/                   Model comparison, threshold sweep, predictions, 2026 test
```

## Deploying (Streamlit Community Cloud, free)

1. Push this repository to GitHub, including `data/` and `results/`.
2. Go to https://share.streamlit.io, sign in with GitHub and click **Create app**.
3. Pick `ChachraArnav/MarketLens`, branch `main`, main file `app.py`.
4. Under **Advanced settings**, choose Python 3.12. Then deploy.

Free apps go to sleep after a few days without visitors and take about 30 seconds to wake.

---

Built by Arnav Chachra as a data science research project. Historical analysis only; not investment advice.
