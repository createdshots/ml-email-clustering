from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import os
from cryptography.fernet import Fernet
app = Flask(__name__)

# Initialize CountVectorizer and MultinomialNB
vectorizer = CountVectorizer()
classifier = MultinomialNB()

emails = []
labels = []

# Generate a key for encryption/decryption
key = Fernet.generate_key()
cipher_suite = Fernet(key)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = ''
    if request.method == 'POST':
        email = request.form['email']
        action = request.form.get('action')  # Use get() instead of []

        if action == 'Train':
            label = request.form['label']
            emails.append(email)
            labels.append(label)

            # Encrypt the email
            encrypted_email = cipher_suite.encrypt(email.encode())

            # Train the classifier
            X = vectorizer.fit_transform(emails)
            classifier.fit(X, labels)

            # Save the trained model and vectorizer
            with open('model.pkl', 'wb') as f:
                pickle.dump(classifier, f)
            with open('vectorizer.pkl', 'wb') as f:
                pickle.dump(vectorizer, f)

        elif action == 'Predict':
            # Load the trained model and vectorizer
            with open('model.pkl', 'rb') as f:
                classifier = pickle.load(f)
            with open('vectorizer.pkl', 'rb') as f:
                vectorizer = pickle.load(f)

            # Decrypt the email
            decrypted_email = cipher_suite.decrypt(email.encode()).decode()

            # Transform the email text to vector
            email_vector = vectorizer.transform([decrypted_email])
            # Make a prediction
            prediction = classifier.predict(email_vector)
            if prediction == ['spam']:
                prediction = 'This email is spam.'
            else:
                prediction = 'This email is not spam.'

    return render_template('index.html', prediction=prediction)



if __name__ == '__main__':
    app.run(debug=True)