import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# Load the data
data = pd.read_csv('spam.csv')

# Split the data into X (inputs) and y (labels)
X = data['text']
y = data['label']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Convert the text to numerical data
vectorizer = CountVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Train a classifier
classifier = MultinomialNB()
classifier.fit(X_train_vectors, y_train)

# Make predictions on the test set
predictions = classifier.predict(X_test_vectors)
print(predictions)

# Print the accuracy of the classifier
print(metrics.accuracy_score(y_test, predictions))
