# CPI x SPX Impact Dashboard

A Streamlit dashboard that helps first-time investing users understand how CPI surprises may affect the S&P 500 and a simple educational stock/bond allocation tilt.

The app combines:

- CPI release inputs: actual CPI, forecast CPI, and portfolio amount
- Week 2 ML models for CPI surprise, CPI surprise direction, and SPX direction
- Historical CPI/SPX comparison charts
- OpenAI-powered reasoning bullets
- OpenAI-assisted stock/bond allocation review
- Allocation rules from local JSON files
- Live market context for SPY/VIX with CSV fallback
- Official BLS CPI trend chart with CSV fallback

## Project Structure

```text
Predict-SP500/
├─ app.py
├─ predict_helper.py
├─ requirements.txt
├─ app.env                  # local API keys, not for GitHub
├─ data/
│  └─ cpi_spx.csv
├─ models/
│  ├─ model1_surprise_regression.json
│  ├─ model2_direction_classifier.json
│  ├─ model3_spx_classifier.joblib
│  ├─ allocation_rules.json
│  └─ allocation_lookup.json
└─ README.md
```

## What The App Does

1. Shows recent CPI context using a 12-month CPI line chart.
2. Lets the user enter or adjust:
   - Actual CPI YoY
   - Forecast CPI YoY
   - Portfolio amount
3. Calculates CPI surprise:

```text
CPI surprise = actual CPI - forecast CPI
```

4. Runs the ML models:
   - Model 1 predicts pre-release CPI surprise magnitude.
   - Model 2 predicts CPI surprise direction: Below, Match, or Above.
   - Model 3 predicts likely SPX direction: Up or Down.
5. Shows model outputs, surprise gauge, historical comparison, reasoning, diagnostics, and allocation guidance.
6. Uses allocation rules plus optional AI review to produce an educational stock/bond split.

## Required Files

The app expects these files to exist:

```text
data/cpi_spx.csv
models/model1_surprise_regression.json
models/model2_direction_classifier.json
models/model3_spx_classifier.joblib
models/allocation_rules.json
models/allocation_lookup.json
```

`allocation_rules.json` is preferred for allocation logic. `allocation_lookup.json` is only used as a fallback if the rules file is missing.

## Setup

Open PowerShell and go to the project folder:

```powershell
cd "C:\Users\hoang\OneDrive\Documents\paul\code in general\Predict-SP500"
```

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## API Key Setup

The OpenAI key is used for:

- Reasoning Layer explanations
- AI-assisted allocation ratio review

Create or edit `app.env`:

```text
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4.1-mini
```

Do not commit `app.env` to GitHub.

If no API key is available, the app still works. It falls back to local rule-based reasoning and allocation.

## Run The App Locally

```powershell
cd "C:\Users\hoang\OneDrive\Documents\paul\code in general\Predict-SP500"
python -m streamlit run app.py
```

Streamlit will print a local URL, usually:

```text
http://localhost:8501
```

Open that URL in your browser.

## Data Sources

The app uses multiple data sources:

- Local CSV: model context, historical CPI/SPX rows, Fed Funds, VIX, SPX returns
- BLS public API: official CPI-U trend when available
- Yahoo Finance chart endpoint: live SPY and VIX context when available

If live endpoints fail, the app falls back to the local CSV so the dashboard does not crash.

## Model Flow

`predict_helper.py` handles loading and prediction.

Model loading preference:

1. `model1_surprise_regression.json`
2. `model2_direction_classifier.json`
3. `model3_spx_classifier.joblib`

The helper also keeps compatibility logic for feature names such as:

- `VIX_Return_Pct_Lag1`
- `SP500_Return_Lag1`
- `FedFunds_Rate`
- `FedFunds_Change_3M`
- `Inflation_Regime_Encoded`

## Allocation Logic

The allocation card and guide use a two-step process.

First, the app selects a baseline allocation from `allocation_rules.json` based on:

- CPI level
- Fed Funds 3-month change
- SPX model direction

Example regimes:

- `goldilocks_bull`
- `reflation_base`
- `stable_growth_easing`
- `stable_growth_tightening`
- `high_inflation_stable`
- `hyper_inflation_tightening`

Second, if the OpenAI API key is available, the app asks the model to review the baseline allocation together with the model outputs and macro context. The API must return JSON with:

```json
{
  "stocks_pct": 30,
  "bonds_pct": 70,
  "reason": "short causal reason"
}
```

If the API response is invalid, the app keeps the baseline allocation.

## Important Notes

This is a research and education prototype. The stock/bond split is a model-based signal for learning and scenario analysis, not personalized financial advice.

The app is designed for first-time investing users, so UI tooltips explain terms like CPI, SPY, VIX, Fed Funds, and SPX reaction.

## Deploying Later

For Streamlit Cloud:

1. Push the project to GitHub.
2. Make sure `requirements.txt`, `app.py`, `predict_helper.py`, `data/`, and `models/` are included.
3. Do not push `app.env`.
4. Add `OPENAI_API_KEY` and optionally `OPENAI_MODEL` in Streamlit Cloud secrets.
5. Set the app entry point to:

```text
app.py
```

## Common Commands

Run locally:

```powershell
python -m streamlit run app.py
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Check Python syntax:

```powershell
python -m py_compile app.py predict_helper.py
```