# DataLens

DataLens is a polished Streamlit app for data profiling, quality auditing, and ML-readiness review.

## Features
- Dataset overview KPIs
- Data quality audit
- Column role detection
- Column profiler with risk labels
- Correlation review and target-aware checks
- Plain-English insights and recommended actions
- Downloadable CSV and text reports

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push this folder to a GitHub repo.
2. Go to https://share.streamlit.io/
3. Click **New app**.
4. Select your repo.
5. Set the main file path to `app.py`.
6. Deploy.

## Notes
- Upload a CSV to begin.
- You can also load a sample dataset from the sidebar.
- Pick a target column to unlock extra ML risk checks.
