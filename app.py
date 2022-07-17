# import libraries and functions
import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)  # Initialize flask App
model = pickle.load(open('model.pkl', 'rb'))  # load trained model

@app.route('/')  # Homepage
def home():
    return render_template('index.html')

def predict():
    '''
    For rendering results on HTML GUI
    '''

    # retrieve values from form
    init_features = [float(x) for x in request.form.values()]
    final_features = [np.array(init_features)]

    prediction = model.predict(final_features)  # make predictions

    return render_template('index.html',
                           prediction_text='Predicted Class: {}'.format(prediction))  # render the predicted result

if __name__ == "__main__":
    app.run(debug=True)