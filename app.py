from flask import Flask, render_template
# Added 'voice' to the imports to reflect the new module/blueprint
from signals import eeg, ecg, radar, doppler , sar, voice 

app = Flask(__name__)


app.secret_key = 'a_secure_and_random_string_of_your_choice_1234567890' 

@app.route("/")
def home():
    # Assuming 'home.html' is the dashboard where the links are
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


# SAR routes
app.register_blueprint(sar.bp, url_prefix="/sar")

# VOICE routes - New registration for the Voice Processing Suite link
app.register_blueprint(voice.bp, url_prefix="/voice")

if __name__ == "__main__":
    app.run(debug=True)
