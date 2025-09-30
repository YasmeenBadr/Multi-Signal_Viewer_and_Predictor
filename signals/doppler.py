from flask import Blueprint, render_template, request, send_file
import numpy as np
from scipy.io.wavfile import write
from scipy.signal import butter, filtfilt
import io

bp = Blueprint("doppler", __name__, template_folder="../templates")

c = 343.0

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

@bp.route("/")
def index():
    return render_template("doppler.html")

@bp.route("/generate", methods=["POST"])
def generate():
    
    v = float(request.form.get("velocity", 0.0))      # m/s
    f0 = float(request.form.get("frequency", 150.0))  # Hz
    d0 = float(request.form.get("distance", 5.0))     # m
    duration = float(request.form.get("duration", 6.0)) # s

    
    fs = 44100
    N = int(duration * fs)
    t = np.linspace(0, duration, N, endpoint=False)

    
    x = v * (t - duration/2.0)
    r = np.sqrt(x**2 + d0**2)

    
    dr_dt = (x * v) / (r + 1e-9)
    v_radial = -dr_dt

    
    denom = c - v_radial
    denom = np.where(np.abs(denom) < 1e-6, np.sign(denom) * 1e-6, denom)
    f_inst = f0 * (c / denom)

    
    phase = 2.0 * np.pi * np.cumsum(f_inst) / fs

    
    sig = 0.6 * np.sin(phase) + 0.28 * np.sin(2*phase) + 0.12 * np.sin(3*phase)
    sig += 0.25 * np.sin(0.5 * phase)

    
    r_ref = 5.0
    amp = 1.0 / (1.0 + (r / r_ref)**2)
    amp *= (1.0 + 0.15 * np.sin(2.0 * np.pi * 3.5 * (1 + v/10.0) * t))
    sig *= amp

    
    noise_level = 0.008   
    sig += noise_level * np.random.normal(size=sig.shape)

    
    lowcut = 50.0     
    highcut = 4000.0  
    b, a = butter_bandpass(lowcut, highcut, fs, order=4)
    sig = filtfilt(b, a, sig)

    
    kernel = np.ones(3) / 3.0
    sig = np.convolve(sig, kernel, mode='same')

    
    sig = sig / (np.max(np.abs(sig)) + 1e-9) * 0.95

    
    audio = (sig * 32767).astype(np.int16)
    buf = io.BytesIO()
    write(buf, fs, audio)
    buf.seek(0)
    return send_file(buf, mimetype="audio/wav",
                     as_attachment=True,
                     download_name="car_doppler_filtered.wav")
