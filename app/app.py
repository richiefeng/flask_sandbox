from flask import Flask, jsonify, request
from sklearn.externals import joblib

app = Flask(__name__)
lr = joblib.load('model.pkl')


@app.route('/hello')
def hello_world():
    return 'Hello World!'


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    json_ = request.args
    query = [json_['query']]
    prediction = lr.predict(query)[0]
    print(prediction)
    label = 'Positive!' if prediction == 1 else 'Negative!'
    return jsonify({'prediction': label})


if __name__ == '__main__':
    app.run(port=8080, debug=True)
