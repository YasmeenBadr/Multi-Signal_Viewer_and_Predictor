# signals/radar.py
from flask import Blueprint, render_template

bp = Blueprint("radar", __name__, template_folder="../templates")

@bp.route("/")
def index():
    return render_template("radar.html")
