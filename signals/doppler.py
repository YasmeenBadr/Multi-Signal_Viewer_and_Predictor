# signals/doppler.py
from flask import Blueprint, render_template

bp = Blueprint("doppler", __name__, template_folder="../templates")

@bp.route("/")
def index():
    return render_template("doppler.html")
