from flask import Flask, jsonify, request, render_template
import numpy as np
import pickle
app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
@app.route('/')
def home():
	return render_template('index.html')


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
	
	features = [(x) for x in request.form.values()]
	final_features = [np.array(features)]
	prediction = model.predict(final_features)

	output = round(prediction[0], 2)

	return render_template('index.html', prediction_text='Selling price of bike/scooter {} Lacs'.format(output))

if __name__ == '__main__':
	app.run(debug=True)
