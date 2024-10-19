from flask import Flask, render_template, request
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('logistic_sentiment_model.pkl')

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle form submission and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the review from the form
        review = request.form['review']
        
        # Make the prediction
        prediction = model.predict([review])[0]  # Use the trained model to predict
        
        # Map prediction result to a human-readable label
        if prediction == 'pos':
            sentiment = 'Positive'
        else:
            sentiment = 'Negative'
        
        # Render the result page with the prediction
        return render_template('result.html', review=review, sentiment=sentiment)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
