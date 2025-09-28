# signals/ecg.py
from flask import Blueprint, render_template

# Define blueprint
bp = Blueprint("ecg", __name__, template_folder="../templates")

@bp.route("/")
def index():
    # This will look for templates/ecg.html
    return render_template("ecg.html")
