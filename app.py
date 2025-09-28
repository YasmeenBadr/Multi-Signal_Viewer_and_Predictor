from flask import Flask, render_template
from signals import eeg, ecg, radar, doppler

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

# EEG routes
app.register_blueprint(eeg.bp, url_prefix="/eeg")

# ECG routes
app.register_blueprint(ecg.bp, url_prefix="/ecg")

# RADAR routes
app.register_blueprint(radar.bp, url_prefix="/radar")

# DOPPLER routes
app.register_blueprint(doppler.bp, url_prefix="/doppler")

if __name__ == "__main__":
    app.run(debug=True)
