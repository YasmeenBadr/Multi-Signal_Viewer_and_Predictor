from flask import Blueprint, render_template

# Define the blueprint for the Voice Processing Suite
bp = Blueprint('voice', __name__, template_folder='templates')

@bp.route("/")
def voice_dashboard():
    """Renders the voice processing template, accessible via /voice."""
    return render_template("voice.html")
