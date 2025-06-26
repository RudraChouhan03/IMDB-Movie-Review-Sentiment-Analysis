# IMDB Movie Review Sentiment Analysis with RNN

This project demonstrates how to use a Recurrent Neural Network (RNN) for classifying IMDB movie reviews as positive or negative. The workflow covers data preprocessing, model building, training, evaluation, and deployment via a Streamlit web app.

---

## Project Structure

- `main.py` — Streamlit app for real-time sentiment prediction.
- `prediction.ipynb` — Notebook for testing and experimenting with the model.
- `simple_rnn_imdb.h5` — Saved Keras RNN model.
- `README.md` — Documentation.

---

## Workflow Overview

### 1. Data Preprocessing

- Load IMDB dataset and word index using Keras:
    ```python
    from tensorflow.keras.datasets import imdb
    word_index = imdb.get_word_index()
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)
    ```
- Lowercase and remove punctuation from reviews:
    ```python
    import string
    def clean_text(text):
        return text.lower().translate(str.maketrans('', '', string.punctuation))
    ```
- Encode words and pad sequences to a fixed length:
    ```python
    from tensorflow.keras.preprocessing import sequence
    maxlen = 500
    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
    ```

---

### 2. Model Building

- Build a Sequential RNN model with embedding and dense layers:
    ```python
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

    model = Sequential([
        Embedding(input_dim=10000, output_dim=32, input_length=500),
        SimpleRNN(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    ```

---

### 3. Training & Evaluation

- Train on IMDB dataset, validate on test set:
    ```python
    history = model.fit(
        X_train, y_train,
        epochs=5,
        batch_size=64,
        validation_data=(X_test, y_test)
    )
    ```
- Save the trained model for inference:
    ```python
    model.save('simple_rnn_imdb.h5')
    ```

---

### 4. Deployment

- **Streamlit app for user-friendly predictions:**  
  The app loads the model and processes user input for real-time sentiment analysis.
    ```python
    import streamlit as st
    from tensorflow.keras.models import load_model

    model = load_model('simple_rnn_imdb.h5')
    user_input = st.text_area('Movie Review')
    # Preprocess and predict as shown in main.py
    ```
- **Notebook for manual testing and debugging:**  
  Use `prediction.ipynb` to run predictions on sample reviews and debug preprocessing.

---

## Requirements

- tensorflow
- pandas
- numpy
- scikit-learn
- tensorboard
- matplotlib
- streamlit
- scikeras

Install with:
```bash
pip install -r requirements.txt
```

---

## Usage

- **Streamlit:**  
  Run `streamlit run main.py` and enter a review for instant sentiment prediction.
- **Notebook:**  
  Open `prediction.ipynb` and run cells to test the model on sample reviews.

---

## Troubleshooting

- Ensure preprocessing matches training (lowercase, remove punctuation).
- Use English reviews with common vocabulary for best results.

---

## Creator

- **Rudra Chouhan**
- **+91 7549019916**

---

## Credits

- Based on tutorials by **Krish Naik Sir**.