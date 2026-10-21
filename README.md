# Twitter US Airline Sentiment Analysis

A sentiment analysis project that classifies tweets about US airlines into **Positive**, **Neutral**, or **Negative** sentiment using word embeddings and a Bidirectional LSTM (BiLSTM) neural network.

## 🚀 Live Demo

Try the deployed Streamlit application:

**[Open the Twitter US Airline Sentiment App](https://twitter-us-airline-embedded-bjetrt9n6swogtb8digtly.streamlit.app/)**

## 📌 Project Overview

The goal of this project is to analyze airline-related tweets and predict the sentiment expressed in each tweet.

### Sentiment Classes
- **Positive**
- **Neutral**
- **Negative**

## 🧠 Model & Workflow

1. Prepare and clean the tweet text.
2. Convert text into numerical sequences using the project's text-processing and embedding workflow.
3. Feed the sequences into a Bidirectional LSTM (BiLSTM) model.
4. Predict the sentiment class for each tweet.

## 🛠️ Technologies

- Python
- TensorFlow / Keras
- Streamlit
- Pandas
- NumPy
- Natural Language Processing (NLP)
- Word Embeddings
- Bidirectional LSTM (BiLSTM)

## 📂 Project Links

- **GitHub Repository:** [Twitter-us-Airline-Embedded](https://github.com/MostafaTamerDev/Twitter-us-Airline-Embedded)
- **Kaggle Notebook:** [Twitter US Airline Embedded](https://www.kaggle.com/code/mostafatamerr/twitter-us-airline-embeded)
- **Live Application:** [Streamlit Demo](https://twitter-us-airline-embedded-bjetrt9n6swogtb8digtly.streamlit.app/)

## ▶️ Run Locally

1. Clone the repository:

   ```bash
   git clone https://github.com/MostafaTamerDev/Twitter-us-Airline-Embedded.git
   cd Twitter-us-Airline-Embedded
   ```

2. Create and activate a virtual environment (recommended).

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

## 📁 Main Files

- `app.py` — Streamlit application.
- `Twitter_Airline_BiLSTM.keras` — Trained BiLSTM model (if included in the repository).
- `requirements.txt` — Python dependencies.
- `.python-version` — Specifies the Python version used for deployment.

## 📊 Evaluation

Model evaluation results should be added here if available, such as macro F1-score, precision, recall, and the confusion matrix. No metrics are listed here unless verified from the notebook or evaluation output.

## 🔮 Possible Improvements

- Add a confusion matrix and per-class precision, recall, and F1-score.
- Improve text preprocessing and handling of misspellings or emojis.
- Expand testing with additional tweets and edge cases.

## 👤 Author

**Mostafa Tamer**

- GitHub: [MostafaTamerDev](https://github.com/MostafaTamerDev)
- Kaggle: [mostafatamerr](https://www.kaggle.com/mostafatamerr)
