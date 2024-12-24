from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model (Ensure 'model_rf.pkl' exists in the same directory)
model = pickle.load(open('model_rf_final.pkl', 'rb'))

@app.route('/')
def index():
    """Serve the HTML form."""
    return render_template('index.html')  # Save the HTML file as 'templates/index.html'

@app.route('/predict', methods=['POST'])
def predict():
    """Handle form submission and return prediction."""
    try:
        # Extract form data
        maternal_age = float(request.form['maternal_age'])
        gestational_age = float(request.form['gestational_age'])
        cervical_dilatation = request.form['cervical_dilatation']
        contraction_frequency = request.form['contraction_frequency']
        fetal_heart_rate = request.form['fetal_heart_rate']
        amniotic = request.form['amniotic']
        pain_location = request.form['pain']
        show_present = 1 if request.form['show_present'] == 'Y' else 0
        bag_of_waters = 1 if request.form['bag_of_waters'] == 'Y' else 0
        parity = 1 if request.form['parity'] == 'Primigravida' else 0
        previous_complications = 1 if request.form['previous_complications'] == 'Y' else 0
        labor_stage = request.form['labor_stage']
        analgesia = 1 if request.form['analgesia'] == 'Y' else 0
        risk = 1 if request.form['risk'] == 'High' else 0
        labor_duration = float(request.form['labor_duration'])

        # Map categorical inputs to numerical values
        cervical_dilatation_map = {
            'Closed': 0, 'Latent Phase': 1, 'Active Phase': 2,
            'Transition Phase': 3, 'Fully Dilated': 4
        }
        contraction_frequency_map = {'Rare': 0, 'Frequent': 1, 'Very Frequent': 2}
        fetal_heart_rate_map = {'Normal': 0, 'Tachycardia': 1, 'Bradycardia': 2}
        amniotic_map = {'Clear': 0, 'Blood-stained': 1, 'Meconium-stained': 2}
        labor_stage_map = {'First': 1, 'Second': 2, 'Third': 3, 'Fourth': 4}
        pain_location_map = {'Groin': 1, 'Abdomen': 2}

        # Convert to numerical values
        cervical_dilatation = cervical_dilatation_map.get(cervical_dilatation, -1)
        contraction_frequency = contraction_frequency_map.get(contraction_frequency, -1)
        fetal_heart_rate = fetal_heart_rate_map.get(fetal_heart_rate, -1)
        amniotic = amniotic_map.get(amniotic, -1)
        labor_stage = labor_stage_map.get(labor_stage, -1)
        pain_location = pain_location_map.get(pain_location, -1)

        # Ensure all mappings are valid
        if -1 in [cervical_dilatation, contraction_frequency, fetal_heart_rate, amniotic, labor_stage, pain_location]:
            raise ValueError("Invalid input detected. Please check your entries.")

        # Prepare feature vector for prediction
        features = np.array([
            maternal_age, gestational_age, cervical_dilatation,
            contraction_frequency, fetal_heart_rate, amniotic, pain_location,
            show_present, bag_of_waters, parity, previous_complications,
            labor_stage, analgesia, risk, labor_duration
        ]).reshape(1, -1)

        # Predict outcome using the trained model
        outcome = model.predict(features)[0]  # Binary outcome: 0 or 1

        # Return prediction as a response
        prediction_message = "Positive labor outcome" if outcome == 1 else "Negative labor outcome"
        return render_template('result - Copy.html', prediction=prediction_message)

    except Exception as e:
        # Log error and return to index page
        print(f"Error: {e}")
        return render_template('index.html', prediction=f"Error: {str(e)}")

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
