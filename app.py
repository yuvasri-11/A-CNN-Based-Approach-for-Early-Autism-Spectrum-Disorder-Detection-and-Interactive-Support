import os
import random
import csv
import io
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from apscheduler.schedulers.background import BackgroundScheduler
from model.predict import ensure_model_ready, predict_video, predict_image

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
# TODO: User must configure these for actual emails
SENDER_EMAIL = "sherlinyuvasri@gmail.com"
SENDER_PASSWORD = "sztuekoitijzrabe"

def send_email(to_email, subject, body):
    if SENDER_EMAIL == "your_email@gmail.com":
        print("[EMAIL MOCK] Please set SENDER_EMAIL and SENDER_PASSWORD in app.py to send real emails.")
        return False
    try:
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        print(f"Failed to send email to {to_email}: {e}")
        return False

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
DOCS_DIR = BASE_DIR / "uploads" / "documents"
MODEL_PATH = BASE_DIR / "model" / "model.h5"

IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp", "gif"}
VIDEO_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "webm"}
ALLOWED_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
ALLOWED_DOC_EXTENSIONS = {"pdf", "png", "jpg", "jpeg", "doc", "docx"}

app = Flask(__name__)
app.config["SECRET_KEY"] = "very-secret-dev-key"
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
app.config["DOCS_FOLDER"] = str(DOCS_DIR)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200MB
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access interactive modules.'
login_manager.login_message_category = 'error'
login_manager.init_app(app)

# Setup directories
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    parent_email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    child_name = db.Column(db.String(150), nullable=False)

class Document(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)

class MedicationSchedule(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    tablet_name = db.Column(db.String(150), nullable=False)
    time_string = db.Column(db.String(10), nullable=False) # e.g. "09:30"

class Reward(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    activity_name = db.Column(db.String(150), nullable=False)
    stars = db.Column(db.Integer, default=1)
    earned_at = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Initialize DB
with app.app_context():
    db.create_all()

# --- HELPERS ---
def allowed_file(filename: str, extensions: set) -> bool:
    if "." not in filename:
        return True
    return filename.rsplit(".", 1)[1].lower() in extensions

# --- AUTH ROUTES ---
@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == "POST":
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = User.query.filter_by(parent_email=email).first()
        if not user or not check_password_hash(user.password_hash, password):
            flash("Invalid email or password.", "error")
            return redirect(url_for('login'))
        
        login_user(user)
        send_email(email, "Login Successful", f"Hello! You have successfully logged into the ASD Platform.")
        flash("Logged in successfully!", "success")
        return redirect(url_for('dashboard'))

    return render_template("login.html", is_register=False)

@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == "POST":
        email = request.form.get('email')
        password = request.form.get('password')
        child_name = request.form.get('child_name')

        user = User.query.filter_by(parent_email=email).first()
        if user:
            flash("Email already registered. Please login.", "info")
            return redirect(url_for('login'))
        
        hashed_pw = generate_password_hash(password)
        new_user = User(parent_email=email, password_hash=hashed_pw, child_name=child_name)
        db.session.add(new_user)
        db.session.commit()

        login_user(new_user)
        
        # Send Welcome email
        send_email(email, "Welcome to the ASD Platform", "Hello! Your account has been securely created.")
        
        flash("Registration successful. You are now logged in.", "success")
        return redirect(url_for('dashboard'))

    return render_template("login.html", is_register=True)

@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for('login'))

# --- DASHBOARD & DOCUMENTS ---
@app.route("/dashboard", methods=["GET"])
@login_required
def dashboard():
    docs = Document.query.filter_by(user_id=current_user.id).all()
    meds = MedicationSchedule.query.filter_by(user_id=current_user.id).all()
    rewards = Reward.query.filter_by(user_id=current_user.id).order_by(Reward.earned_at.desc()).all()
    meds_data = [{"id": m.id, "tablet_name": m.tablet_name, "time": m.time_string} for m in meds]
    return render_template("dashboard.html", user=current_user, documents=docs, medications=meds_data, rewards=rewards)

@app.route("/profile", methods=["GET"])
@login_required
def profile():
    rewards = Reward.query.filter_by(user_id=current_user.id).order_by(Reward.earned_at.desc()).all()
    return render_template("profile.html", user=current_user, rewards=rewards)

@app.route("/upload_document", methods=["POST"])
@login_required
def upload_document():
    if "doc" not in request.files:
        flash("No document file found.", "error")
        return redirect(url_for('dashboard'))

    file = request.files["doc"]
    if not file or file.filename == "":
        flash("No file selected.", "error")
        return redirect(url_for('dashboard'))

    if not allowed_file(file.filename, ALLOWED_DOC_EXTENSIONS):
        flash("Unsupported document type.", "error")
        return redirect(url_for('dashboard'))

    filename = secure_filename(file.filename)
    # Prepend user id to namespace and combat dupes
    save_name = f"user_{current_user.id}_{int(datetime.utcnow().timestamp())}_{filename}"
    save_path = DOCS_DIR / save_name
    file.save(str(save_path))

    new_doc = Document(user_id=current_user.id, filename=save_name)
    db.session.add(new_doc)
    db.session.commit()

    flash("Document uploaded successfully.", "success")
    return redirect(url_for('dashboard'))

@app.route("/upload_schedule", methods=["POST"])
@login_required
def upload_schedule():
    tablet_names = request.form.getlist('tablet_name[]')
    hours = request.form.getlist('hour[]')
    mins = request.form.getlist('minute[]')
    ampms = request.form.getlist('ampm[]')
    
    if not tablet_names or not hours or not mins or not ampms or len(tablet_names) != len(hours):
        flash("Invalid data submitted.", "error")
        return redirect(url_for('dashboard'))
        
    try:
        parsed_count = 0
        for pill, h, m, ampm in zip(tablet_names, hours, mins, ampms):
            pill = pill.strip()
            if pill and h and m and ampm:
                hr_int = int(h)
                if ampm == "PM" and hr_int < 12:
                    hr_int += 12
                elif ampm == "AM" and hr_int == 12:
                    hr_int = 0
                
                t_str = f"{hr_int:02d}:{m}"
                
                db.session.add(MedicationSchedule(user_id=current_user.id, tablet_name=pill, time_string=t_str))
                parsed_count += 1
                
        db.session.commit()
        flash(f"Successfully saved {parsed_count} medication timings.", "success")
    except Exception as e:
        flash(f"Failed to save schedule: {str(e)}", "error")
        
    return redirect(url_for('dashboard'))

@app.route("/delete_medication/<int:med_id>", methods=["POST"])
@login_required
def delete_medication(med_id):
    med = MedicationSchedule.query.get(med_id)
    if med and med.user_id == current_user.id:
        db.session.delete(med)
        db.session.commit()
        flash("Medication removed.", "success")
    return redirect(url_for('dashboard'))

@app.route("/api/trigger_email_notification", methods=["POST"])
@login_required
def trigger_email_notification():
    data = request.get_json() or {}
    title = data.get("title", "No Title")
    body = data.get("body", "No message body")
    
    # Send actual email
    print(f"\n[EMAIL TRIGGER] Attempting to send '{title}' to {current_user.parent_email}...\n")
    success = send_email(current_user.parent_email, title, body)
    if not success:
        print("[MOCK FALLBACK] To config real email, update SENDER_EMAIL in app.py")
    
    return jsonify({"ok": True, "message": "Email sent"})

@app.route("/api/reward", methods=["POST"])
@login_required
def add_reward():
    data = request.get_json() or {}
    activity = data.get("activity", "Unknown Activity")
    new_reward = Reward(user_id=current_user.id, activity_name=activity, stars=1)
    db.session.add(new_reward)
    db.session.commit()
    return jsonify({"ok": True, "message": "Reward saved!"})

# --- ORIGINAL ROUTES ---
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/interactive", methods=["GET"])
@login_required
def interactive():
    return render_template("interactive.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "video" not in request.files:
        return jsonify({"ok": False, "error": "No file field named 'video' found."}), 400

    file = request.files["video"]
    if not file or file.filename == "":
        return jsonify({"ok": False, "error": "No file selected."}), 400

    if not allowed_file(file.filename, ALLOWED_EXTENSIONS):
        return jsonify(
            {
                "ok": False,
                "error": "Unsupported file type. Please upload a supported video or photo format.",
            }
        ), 400

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    filename = secure_filename(file.filename)
    save_path = UPLOAD_DIR / filename
    file.save(str(save_path))

    try:
        ensure_model_ready(model_path=MODEL_PATH)
        
        is_image = False
        if file.content_type and file.content_type.startswith("image/"):
            is_image = True
        elif "." in filename and filename.rsplit(".", 1)[1].lower() in IMAGE_EXTENSIONS:
            is_image = True
            
        if is_image:
            result = predict_image(image_path=save_path, model_path=MODEL_PATH)
        else:
            result = predict_video(video_path=save_path, model_path=MODEL_PATH, max_frames=16)
    except Exception as e:
        return jsonify({"ok": False, "error": f"Prediction failed: {e}"}), 500
    finally:
        try:
            if save_path.exists():
                save_path.unlink()
        except Exception:
            pass

    return jsonify({"ok": True, **result})

# --- Scheduler ---
def check_and_send_notifications():
    print("[SCHEDULER] Checking medication schedules...")
    with app.app_context():
        now_local = datetime.now()
        current_time_str = now_local.strftime("%H:%M")
        
        schedules = MedicationSchedule.query.all()
        for schedule in schedules:
            if schedule.time_string == current_time_str:
                user = User.query.get(schedule.user_id)
                if user:
                    subject = f"Medication Reminder: {schedule.tablet_name}"
                    body = f"Hello,\n\nIt is time for your child ({user.child_name}) to take their medication: {schedule.tablet_name}.\n\nStay healthy!"
                    print(f"[SCHEDULER] Triggering email to {user.parent_email} for {schedule.tablet_name} at {current_time_str}")
                    send_email(user.parent_email, subject, body)

scheduler = BackgroundScheduler()
scheduler.add_job(func=check_and_send_notifications, trigger="interval", minutes=1)

if not app.debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
    scheduler.start()

if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    app.run(host="0.0.0.0", port=5000, debug=True)
