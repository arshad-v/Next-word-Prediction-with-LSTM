# LSTM Next Word Prediction

This project implements a Long Short-Term Memory (LSTM) neural network for next word prediction, trained on Shakespeare's Hamlet. The model predicts the next word in a sequence based on the input text.

## Features

- LSTM-based text prediction model
- Interactive web interface using Streamlit
- Trained on Shakespeare's Hamlet text
- Real-time word predictions

## Project Structure

- `app.py`: Main Streamlit application for the web interface
- `experiemnts.ipynb`: Jupyter notebook containing model training and experimentation
- `hamlet.txt`: Training dataset (Shakespeare's Hamlet)
- `next_word_lstm.h5`: Trained LSTM model
- `tokenizer.pickle`: Saved tokenizer for text processing
- `requirements.txt`: Project dependencies

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Dependencies

- TensorFlow
- Pandas
- NumPy
- Scikit-learn
- TensorBoard
- Matplotlib
- Streamlit
- Scikit-keras
- NLTK
- Pickle

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Enter a sequence of words in the text input field
3. Click "Predict Next Word" to get the model's prediction

## Model Details

The project uses an LSTM (Long Short-Term Memory) neural network architecture for sequence prediction. The model was trained on Shakespeare's Hamlet to learn the patterns and structure of Shakespearean English.

## Technical Details

### Data Preprocessing
- Source text: Shakespeare's Hamlet from NLTK's Gutenberg corpus
- Text is tokenized and converted to lowercase
- Vocabulary size: 4,818 unique words
- Sequences are padded to ensure uniform length

### Model Architecture
- Embedding Layer: 100-dimensional word embeddings
- LSTM Layer: Sequential model with the following layers:
  - Embedding layer (100 dimensions)
  - LSTM layer with dropout
  - Dense layer with softmax activation
- Total Parameters: 1,219,418 (~4.65 MB)

### Training Process
- Training Data: Sequences generated from Hamlet text
- Epochs: 50
- Validation Split: Test set used for validation
- Training Metrics:
  - Loss function: Categorical crossentropy
  - Optimizer: Adam
  - Metrics: Accuracy

### Model Performance
The model shows progressive improvement in training accuracy:
- Initial accuracy: ~3%
- Final accuracy: ~35%
- The model exhibits some overfitting as training progresses, with validation accuracy remaining lower than training accuracy

### Implementation
The project implementation follows these steps:
1. Data Collection: Using NLTK to download and process Hamlet text
2. Text Preprocessing: Tokenization and sequence generation
3. Model Training: Using TensorFlow/Keras for LSTM implementation
4. Model Deployment: Streamlit web interface for real-time predictions

## Files Description

- `app.py`: Contains the Streamlit web application code and prediction logic
- `next_word_lstm.h5`: The trained LSTM model saved in H5 format
- `tokenizer.pickle`: Serialized tokenizer object used for text preprocessing
- `hamlet.txt`: The source text used for training the model

## Contributing

Feel free to fork this repository and submit pull requests to contribute to this project. For major changes, please open an issue first to discuss what you would like to change.
