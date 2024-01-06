"""
Author: Mayank Sharma

Description:
This script utilizes TensorFlow and Keras to build a language model for generating text based on email data.
It includes data preprocessing, model training, and text generation functionalities. The generated model is
saved for reuse, and the script can be extended for various text generation applications.
"""

import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

df = pd.read_csv("emails.csv", nrows=1000)

df = df[['message']]

X_test_padded = None

try:
    model = load_model('email_model.h5')
    with open('email_tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
except (OSError, IOError):

    def preprocess_text(text):
        return text.lower()

    df['processed_message'] = df['message'].apply(preprocess_text)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df['processed_message'])
    total_words = len(tokenizer.word_index) + 1

    input_sequences = []
    for line in df['processed_message']:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    max_sequence_length = max([len(x) for x in input_sequences])
    input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')

    X, y = input_sequences[:, :-1], input_sequences[:, -1]

    y = to_categorical(y, num_classes=total_words)

    model = Sequential()
    model.add(Embedding(total_words, 100, input_length=max_sequence_length-1))
    model.add(LSTM(100))
    model.add(Dense(total_words, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train, epochs=5, verbose=1)

    model.save('email_model.h5')
    with open('email_tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    X_test_padded = pad_sequences(X_test, maxlen=max_sequence_length-1, padding='pre')

def generate_text(seed_text, next_words, model, tokenizer, max_sequence_length):
    generated_text = seed_text
    try:
        for _ in range(next_words):
            token_list = tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list], maxlen=max_sequence_length-1,padding='pre')
            predicted_probs = model.predict(token_list, verbose=0)[0]

            predicted_probs = predicted_probs / np.sum(predicted_probs)

            predicted_index = np.random.choice(len(predicted_probs), p=predicted_probs)

            output_word = ""
            for word, index in tokenizer.word_index.items():
                if index == predicted_index:
                    output_word = word
                    break
            seed_text += " " + output_word
            generated_text += " " + output_word
    except Exception as e:
        print(f"Error during text generation: {e}")
    return generated_text


if X_test_padded is not None:
    num_samples_for_testing = min(1000, len(X_test_padded))
    X_test_padded_subset = X_test_padded[:num_samples_for_testing]
    y_test_one_hot_subset = y_test[:num_samples_for_testing]

    _, accuracy = model.evaluate(X_test_padded_subset, y_test_one_hot_subset, verbose=0)
    print(f"Model Accuracy on Test Data: {accuracy*100:.2f}%")

    seed_text = "I would suggest"
    generated_text = generate_text(seed_text, next_words=10, model=model, tokenizer=tokenizer,max_sequence_length=max_sequence_length)
    print(f"Generated Email: {generated_text}")
