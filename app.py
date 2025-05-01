from flask import Flask, render_template, request
import joblib
import numpy as np

# Load the trained model
model = joblib.load("model/salary_model.pkl")

# Initialize the Flask app
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get the form data
        job_profile = int(request.form["job_profile"])
        max_experience = int(request.form["max_experience"])
        min_experience = int(request.form["min_experience"])
        work_type = int(request.form["work_type"])
        company_size = int(request.form["company_size"])
        location = int(request.form["location"])

        # Prepare the input data
        input_data = np.array([job_profile, max_experience, min_experience, work_type, company_size, location])
        processed_input = input_data.reshape(1, -1)

        # Predict the salary
        predicted_salary = model.predict(processed_input)
        predicted_salary = predicted_salary * -1  # Ensure positive salary

        return render_template("index.html", predicted_salary=predicted_salary[0])

    return render_template("index.html", predicted_salary=None)

if __name__ == "__main__":
    app.run(debug=True)
