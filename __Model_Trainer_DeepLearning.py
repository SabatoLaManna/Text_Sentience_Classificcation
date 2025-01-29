import os
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

# Paths to the positive and negative review directories
pos_dir = r"aclImdb\train\pos" #Insert the path to the positive reviews
neg_dir = r"aclImdb\train\neg"#Insert the path to the negative reviews

# Function to load data from text files and label based on the folder (pos = 1, neg = 0)
def load_data_from_folder(directory, label):
    texts = []
    labels = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                texts.append(file.read())
                labels.append(label)
    return texts, labels

# Load the positive and negative reviews
positive_texts, positive_labels = load_data_from_folder(pos_dir, 1)
negative_texts, negative_labels = load_data_from_folder(neg_dir, 0)

# Combine positive and negative reviews
texts = positive_texts + negative_texts
labels = positive_labels + negative_labels

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Tokenization and padding
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)

# Pad the sequences to ensure uniform length
X_train_pad = pad_sequences(X_train_sequences, padding='post', maxlen=100)
X_test_pad = pad_sequences(X_test_sequences, padding='post', maxlen=100)

# Build the LSTM model
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=100))
model.add(LSTM(128, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model with epochs
history = model.fit(X_train_pad, np.array(y_train), epochs=1, batch_size=64, validation_data=(X_test_pad, np.array(y_test))) #Change the amount of epochs, more is better

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(X_test_pad, np.array(y_test))
print(f"Test Accuracy: {test_acc}")



# Save the model for future use
model.save('sentiment_model_lstm.h5')

# Save the tokenizer for future use
joblib.dump(tokenizer, 'tokenizer.pkl')

print("Model and tokenizer have been saved.")
