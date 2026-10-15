# CANGuard TARA

ML-powered CAN-bus intrusion detection that feeds an ISO/SAE 21434-style **Threat Analysis and Risk Assessment** (Case Study 3 of the Automotive Engineering AI document).

* **ML core:** 27 engineered online features, 6 supervised models + Isolation Forest, group-CV, tuning, zero-day experiment.
* **TARA engine:** deterministic impact x feasibility -> risk, mitigations with a what-if simulator, linked safety hazard, verification tests.
* **UI:** dark Streamlit dashboard (7 pages) + FastAPI service + optional Ollama narrative.

## Quick start

```bash
python -m venv venv && source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python run_pipeline.py --fast                           # generate data + train (about 3-5 min); drop --fast for the full budget
streamlit run app/streamlit_app.py                      # dashboard
uvicorn api.main:app --reload                           # API docs at http://127.0.0.1:8000/docs
pytest -q                                               # tests
```

No model yet? Open the dashboard and click **Generate data & train**.

## Optional LLM narrative (Ollama Cloud)

Paste your key in the sidebar (**LLM settings**) or set it before launching:

```bash
export OLLAMA_API_KEY=your-key            # Windows: set OLLAMA_API_KEY=your-key
export OLLAMA_MODEL=gpt-oss:20b           # any model your key can use
```
Without a key the app uses a built-in template. The LLM never decides risk values.

## Use the real dataset (recommended for final numbers)

Download the HCRL **Car-Hacking** dataset (DoS, Fuzzy, gear, RPM CSV files), put them in one folder, then:

```bash
python run_pipeline.py --real-dir /path/to/folder --max-rows 300000
```
Expected files: `DoS_dataset.csv`, `Fuzzy_dataset.csv`, `gear_dataset.csv`, `RPM_dataset.csv`. The loader maps flag `T` to the attack label and `R` to Normal. It was written against the documented file format and has not been run on the original files here, so check the first run.

## Structure

```
run_pipeline.py      generate/load data -> train -> artifacts/
src/                 simulate, data_io, features, train, detect, tara, llm, config
api/main.py          FastAPI service
app/                 Streamlit app (ui.py theme, pages_*.py)
tests/               pytest
scripts/feature_ablation.py   reproduces the feature-set comparison across seeds
docs/VIVA_QA.md      likely technical questions with answers
```

## Notes on honesty

* The demo data is synthetic; its scores are not real-vehicle results.
* TARA rating tables are illustrative; calibrate with your authorised copy of ISO/SAE 21434.
