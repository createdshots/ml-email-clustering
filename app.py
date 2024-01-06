from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import os

app = Flask(__name__)

# Initialize CountVectorizer and MultinomialNB
vectorizer = CountVectorizer()
classifier = MultinomialNB()

emails = []
labels = []

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        email = request.form['email']
        label = request.form['label']
        emails.append(email)
        labels.append(label)

        # Train the classifier
        X = vectorizer.fit_transform(emails)
        classifier.fit(X, labels)

        # Save the trained model and vectorizer
        with open('model.pkl', 'wb') as f:
            pickle.dump(classifier, f)
        with open('vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)