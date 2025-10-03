from flask import Flask, render_template
from signals import eeg, ecg, radar, doppler

app = Flask(__name__)

# 🛑 CRITICAL FIX: Set a secret key for session management
app.secret_key = 'a_secure_and_random_string_of_your_choice_1234567890' 

@app.route("/")
def home():
    return render_template("home.html")
#
# EEG routes
app.register_blueprint(eeg.bp, url_prefix="/eeg")

# ECG routes
app.register_blueprint(ecg.ECG_BP, url_prefix="/ecg")

# RADAR routes
app.register_blueprint(radar.bp, url_prefix="/radar")

# DOPPLER routes
app.register_blueprint(doppler.bp, url_prefix="/doppler")

if __name__ == "__main__":
    app.run(debug=True)