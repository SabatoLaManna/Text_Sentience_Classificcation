import os
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model and tokenizer
model = tf.keras.models.load_model('sentiment_model_lstm.h5') #Insert the path to the model
tokenizer = joblib.load('tokenizer.pkl')

# Function to predict sentiment of a single review and return the score
def predict_sentiment(review):
    # Preprocess the review (tokenize and pad)
    review_sequence = tokenizer.texts_to_sequences([review])
    review_pad = pad_sequences(review_sequence, padding='post', maxlen=100)

    # Predict sentiment score (between 0 and 1), set verbose to 0 to suppress progress bar
    prediction = model.predict(review_pad, verbose=0)
    sentiment_score = prediction[0][0]  # Extract the score
    sentiment = 'positive' if sentiment_score >= 0.5 else 'negative'
    
    return sentiment, sentiment_score

# Main loop to restart after processing
while True:
    test_review = input("Enter a review to test sentiment analysis: ")
    sentiment, score = predict_sentiment(test_review)
    
    # Print the predicted sentiment and score
    print(f"Predicted sentiment: {sentiment}")
    print(f"Sentiment score: {score:.4f}")  # Print score with 4 decimal places

    # Ask if the user wants to restart
    restart = input("Would you like to test another review? (yes/no): ").strip().lower()
    if restart != 'yes':
        print("Exiting the program...")
        break
