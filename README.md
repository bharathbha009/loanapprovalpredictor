
# Loan Approval Predictor (Ready-to-run)

## What is included
- `loan_dataset.csv` : Synthetic dataset (2000 rows)
- `loan_model.pkl` : Trained RandomForest pipeline (includes preprocessing)
- `app.py` : Flask application (run with `python app.py`)
- `templates/index.html` : Frontend page (card-style result)
- `static/style.css` : Styling for the frontend
- `train_model.py` : Script to retrain the model from `loan_dataset.csv`
- `requirements.txt` : Python dependencies
- `eval_report.json` : Model evaluation on test split

## Run locally
1. (Optional) create and activate a virtualenv:
   ```
   python -m venv venv
   source venv/bin/activate   # on mac/linux
   venv\Scripts\activate    # on Windows
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Start the app:
   ```
   python app.py
   ```
4. Open `http://127.0.0.1:5000` in your browser.

## Notes
- The dataset is synthetic for demo purposes. For production, replace `loan_dataset.csv` with a real dataset and retrain.
- The model uses a RandomForest classifier with a simple one-hot encoder pipeline.
- The result is shown as a colored card: green for Approved, red for Rejected.

