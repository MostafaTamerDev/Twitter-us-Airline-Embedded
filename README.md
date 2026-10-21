# Airline Sentiment AI — Streamlit Deployment

A professional Streamlit interface for the Twitter US Airline Sentiment Bidirectional LSTM model.

## Included
- `app.py` — single-tweet prediction, class probability visualization, and CSV batch predictions
- `Twitter_Airline_BiLSTM.keras` — trained model supplied for this deployment
- `requirements.txt` — Python dependencies
- `runtime.txt` — Python runtime for Streamlit Community Cloud
- `.gitignore` — common local/environment files

## Important: two training artifacts are still required
The `.keras` file alone is not sufficient for text inference. The model was trained with a fitted Keras `Tokenizer`, so deployment also needs these files from the **same notebook run**:

- `twitter_airline_tokenizer.pkl`
- `twitter_airline_label_mapping.pkl`

In the notebook, run the section **“13. Save the Model and Tokenizer”** after training. It should create both files. Copy them into this same project folder beside `app.py`.

Do not fit a new tokenizer on random text or a new dataset: its word-to-index mapping would not match the trained embedding layer.

## Run locally (Windows)
1. Install Python 3.11.
2. Extract this ZIP.
3. Open a terminal in the extracted folder.
4. Create and activate a virtual environment:

   ```powershell
   py -3.11 -m venv .venv
   .venv\Scripts\activate
   ```

5. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

6. Ensure all three artifacts are in the same folder.
7. Start the app:

   ```powershell
   streamlit run app.py
   ```

## Deploy to Streamlit Community Cloud
1. Create a GitHub repository, e.g. `airline-sentiment-streamlit`.
2. Upload `app.py`, `requirements.txt`, `runtime.txt`, `.gitignore`, and the three model artifacts.
3. In Streamlit Community Cloud, choose **Create app**.
4. Select the repository, branch, and `app.py` as the main file.
5. Deploy.

The model file included here is about 31 MB, below GitHub's normal 100 MB per-file limit. Check the repository size and any organization-specific limits before pushing.

## Privacy
Uploaded CSV files are processed by the running app. Do not upload confidential or personally identifying data to a public deployment.
