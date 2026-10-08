# IIoT Acoustic Dashboard

Local operator dashboard for the gearbox acoustic classifier. The frontend is `dashboard.py`; `acoustic_dashboard_backend.py` loads the saved models and runs inference with the shared steps in `acoustic_preprocessing.py`.

## Run locally

Use the Python environment that has the project's model dependencies installed. Install the dashboard dependencies with `python -m pip install -r requirements-dashboard.txt`, then start it with `python -m streamlit run dashboard.py`. Open the local URL printed by Streamlit, usually `http://localhost:8501`.

The current dashboard reads numeric CSV recordings as 16 kHz mono audio and analyzes the first 3-second window. It uses the first CSV column when a file has multiple columns. Upload a recorded sensor clip, choose CNN or SVM, and review the predicted condition, confidence, waveform and class scores.

The dashboard currently supports Normal, Overhang and Underhang classes from `models/multiclass`. CNN inference needs TensorFlow in that Python environment; the SVM remains available without it. Live microphone capture is not configured; use a recorded CSV until the sensor interface is connected.
