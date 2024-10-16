import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

# Load the model once, at the start of the app
@st.cache(allow_output_mutation=True)
def load_lstm_model():
    try:
        return load_model('next_word_lstm.h5')
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Load tokenizer once, at the start of the app
@st.cache(allow_output_mutation=True)
def load_tokenizer():
    try:
        with open('tokenizer.pickle', 'rb') as handle:
            return pickle.load(handle)
    except Exception as e:
        st.error(f"Error loading tokenizer: {str(e)}")
        return None

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    if not model or not tokenizer:
        return "Model or tokenizer is not loaded."

    token_list = tokenizer.texts_to_sequences([text])
    
    if not token_list or not token_list[0]:
        return "No valid tokens found for the input text."

    token_list = token_list[0]

    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]

    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')

    try:
        prediction = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(prediction, axis=1)[0]

        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                return word
    except Exception as e:
        return f"Error during prediction: {str(e)}"

    return "Prediction could not be made."

# Streamlit app
st.title("Next Word Prediction with LSTM")

# Load the model and tokenizer once
model = load_lstm_model()
tokenizer = load_tokenizer()

input_text = st.text_input("Enter the Sequence of Words", "For this releefe much")

if st.button("Predict Next Word"):
    if model:
        max_sequence_len = model.input_shape[1] + 1
        next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
        st.write(f"Next Word Prediction: {next_word}")
    else:
        st.error("Model is not loaded properly.")
