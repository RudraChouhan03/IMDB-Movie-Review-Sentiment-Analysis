# IMDB Movie Review Sentiment Analysis with RNN

This project demonstrates how to use a Recurrent Neural Network (RNN) for classifying IMDB movie reviews as positive or negative. The workflow covers data preprocessing, model building, training, evaluation, and deployment via a Streamlit web app.

## Project Structure

- `main.py` — Streamlit app for real-time sentiment prediction.
- `prediction.ipynb` — Notebook for testing and experimenting with the model.
- `simple_rnn_imdb.h5` — Saved Keras RNN model.
- `README.md` — Documentation.

## Workflow Overview

### 1. Data Preprocessing

- Load IMDB dataset and word index.
- Lowercase and remove punctuation from reviews.
- Encode words and pad sequences to a fixed length.

### 2. Model Building

- Sequential RNN model with embedding and dense layers.
- Compiled with Adam optimizer and binary cross-entropy loss.

### 3. Training & Evaluation

- Train on IMDB dataset, validate on test set.
- Save the trained model for inference.

### 4. Deployment

- Streamlit app for user-friendly predictions.
- Notebook for manual testing and debugging.

## Requirements

tensorflow
pandas
numpy
scikit-learn
tensorboard
matplotlib
streamlit
scikeras

## Usage

- **Streamlit:**  
  Run `streamlit run main.py` and enter a review for instant sentiment prediction.
- **Notebook:**  
  Open `prediction.ipynb` and run cells to test the model on sample reviews.

## Troubleshooting

- Ensure preprocessing matches training (lowercase, remove punctuation).
- Use English reviews with common vocabulary for best results.

## Creator

- **Rudra Chouhan**
- **+91 7549019916**

## Credits

- Based on tutorials by **Krish Naik Sir**.
