# Spam Message Detection using Machine Learning

# 1. Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 2. Load dataset
# Make sure spam.csv file is in the same folder
data = pd.read_csv("spam.csv", encoding="latin-1")

# 3. Keep required columns only
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# 4. Convert labels to numeric
# ham = 0, spam = 1
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# 5. Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    data['message'],
    data['label'],
    test_size=0.2,
    random_state=42
)

# 6. Text vectorization (Bag of Words)
vectorizer = CountVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 7. Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 8. Test the model
y_pred = model.predict(X_test_vec)

# 9. Model accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# 10. Classification report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# 11. User input prediction
print("\n--- Spam Message Detection System ---")
while True:
    user_message = input("\nEnter a message (type 'exit' to stop): ")

    if user_message.lower() == 'exit':
        print("Program Ended.")
        break

    message_vector = vectorizer.transform([user_message])
    prediction = model.predict(message_vector)

    if prediction[0] == 1:
        print("📩 Result: SPAM MESSAGE")
    else:
        print("✅ Result: NOT A SPAM MESSAGE")print("Spam Message Detection Project")
