import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Paths to the positive and negative review directories
pos_dir = r"aclImdb\train\pos"  # Replace with actual path to positive reviews
neg_dir = r"aclImdb\train\neg"  # Replace with actual path to negative reviews

# Function to load data from text files and label based on the folder (pos = 1, neg = 0)
def load_data_from_folder(directory, label):
    texts = []
    labels = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):  # Ensure it's a file
            with open(file_path, 'r', encoding='utf-8') as file:
                texts.append(file.read())  # Read the text content
                labels.append(label)        # Add the corresponding label
    return texts, labels

# Load the positive and negative reviews
positive_texts, positive_labels = load_data_from_folder(pos_dir, 1)  # Label 1 for positive
negative_texts, negative_labels = load_data_from_folder(neg_dir, 0)  # Label 0 for negative

# Combine positive and negative reviews
texts = positive_texts + negative_texts
labels = positive_labels + negative_labels

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Convert text data into numerical vectors using TF-IDF (Term Frequency-Inverse Document Frequency)
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Evaluate the model's performance on the test data
y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained model and the vectorizer for future use
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("Model and vectorizer have been saved.")
