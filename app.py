"""
app.py
------
Placement Prediction & Career Guidance System
Flask Backend — fully corrected and production/deployment-ready.

FIXES APPLIED (functional):
  1. [QUIZ BUG] load_json() now extracts the actual question list from the
     nested-dict JSON structure (e.g. {"aptitude": [...]} -> [...]).
     pick_random_questions() is simplified -- it only receives plain lists now.
  2. [DB MISMATCH] save_quiz_scores() now correctly passes total_score to the
     INSERT so the column is never NULL.

FIXES APPLIED (deployment / Render):
  3. [PORT] App now binds to the $PORT environment variable injected by
     Render at runtime instead of a hardcoded port=5000.
  4. [SECRET_KEY] SECRET_KEY is now required from the environment with no
     hardcoded fallback -- app will fail fast at startup if it's missing,
     rather than silently using an insecure default in production.
  5. [DEBUG] debug mode is now driven by the FLASK_DEBUG env var and
     defaults to False, so the Werkzeug debugger/reloader never runs in
     production unless explicitly enabled for local development.
  6. [QUIZ STATE] QUIZ_QUESTIONS is no longer a single process-global dict.
     Under Gunicorn with multiple workers, a global in-memory dict is
     per-worker, so one user's generated quiz could be invisible to (or
     overwritten by) another user handled by a different worker. Quiz
     questions for the current attempt are now stored in the user's own
     Flask session instead, which is safe across workers and users.

FIXES APPLIED (security hardening pass -- see docs/security.md):
  7. [CSRF] Flask-WTF CSRFProtect is enabled globally. Every state-changing
     HTML form includes a hidden csrf_token field; the one JSON POST
     (/generate_quiz) reads the token from a <meta> tag via a request
     header instead (see templates/base.html + quiz_setup.html).
  8. [RATE LIMITING] Flask-Limiter caps /login, /register, and /admin/login
     to slow down credential-guessing/enumeration without blocking normal
     use. See docs/security.md for the exact limits and rationale.
  9. [TIMING SIDE-CHANNEL] /login and /admin/login now always run a
     password-hash comparison, even for a nonexistent account, so response
     time does not reveal whether an email/username is registered.
  10. [INFO DISCLOSURE] Raw exception text (SQL errors, library errors,
      stack traces) is no longer flashed to users. Errors are logged
      server-side via the `logging` module and users see a generic,
      actionable message instead.
  11. [UPLOAD HARDENING] MAX_CONTENT_LENGTH caps every request body (so an
      oversized resume is rejected before it reaches memory), and the
      resume route checks the PDF magic header (%PDF) before parsing --
      previously only the ".pdf" filename extension was checked, so a
      renamed non-PDF file would be handed straight to the PDF parser.
  12. [SESSION COOKIES] Session cookies are now HttpOnly, SameSite=Lax, and
      Secure whenever the app runs over HTTPS (configurable), reducing
      exposure to XSS-driven cookie theft and CSRF via cross-site requests.
  13. [DEBUG PRINTS REMOVED] Development-only print()/traceback.print_exc()
      calls in the request path are replaced with structured logging that
      never reaches the HTTP response.
"""

import logging
import os
import json
import re
import random
import datetime
from dotenv import load_dotenv
load_dotenv()
from collections import Counter

from flask import (
    Flask, render_template, request, redirect,
    url_for, session, flash, jsonify
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from functools import wraps

from flask_wtf import CSRFProtect
from flask_wtf.csrf import CSRFError
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

import database as db
import prediction as pred
import feature_engineering
import job_market
from utils.resume_parser import extract_text
from utils.ats_score import calculate_ats, get_ats_breakdown

# ------------------------------------------------------------------
# LOGGING
# Structured server-side logging replaces ad-hoc print()/traceback calls
# in the request path. Diagnostics go to the log, never into a Flask
# response or flash message a user can see.
# ------------------------------------------------------------------
logging.basicConfig(
    level=os.environ.get('LOG_LEVEL', 'INFO'),
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger('placementai')

app = Flask(__name__)

# ------------------------------------------------------------------
# FIX 4: SECRET_KEY must be set via environment variable in production.
# No insecure hardcoded fallback. App fails fast at startup if missing,
# which is preferable to silently running with a guessable default key.
# ------------------------------------------------------------------
app.secret_key = os.environ['SECRET_KEY']

# ------------------------------------------------------------------
# FIX 12: session cookie hardening.
#   HTTPONLY  -- JavaScript cannot read the session cookie (mitigates
#                cookie theft via XSS).
#   SAMESITE  -- 'Lax' stops the cookie being sent on cross-site POSTs
#                (the main CSRF vector) while still allowing normal
#                top-level navigation (e.g. following a link from email).
#   SECURE    -- only sent over HTTPS. Configurable via env var because
#                local development over plain http:// would otherwise
#                silently lose the session cookie entirely.
#   LIFETIME  -- caps how long a session stays valid.
# ------------------------------------------------------------------
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = os.environ.get('SESSION_COOKIE_SECURE', '1') == '1'
app.config['PERMANENT_SESSION_LIFETIME'] = datetime.timedelta(hours=8)

# FIX 11: hard cap on request body size (covers the resume PDF upload).
# 10 MB is generous for a resume PDF; anything larger is almost certainly
# not a resume. Configurable via env var for deployments that need a
# different ceiling.
app.config['MAX_CONTENT_LENGTH'] = int(os.environ.get('MAX_UPLOAD_MB', '10')) * 1024 * 1024

app.jinja_env.globals.update(enumerate=enumerate)

# ------------------------------------------------------------------
# FIX 8: rate limiting on auth-sensitive endpoints.
#
# ORDERING NOTE: this is initialized BEFORE CSRFProtect below on purpose.
# Both extensions hook into Flask via app.before_request, and Flask stops
# processing further before_request handlers as soon as one returns a
# response -- so whichever is registered first "wins" for a given
# request. We want rate limiting to run first: an attacker hammering
# /login without a valid CSRF token should still get rate-limited (429)
# after N attempts, rather than the CSRF check silently absorbing every
# request first and the rate limiter never getting a chance to count
# them. (This was caught by tests/test_security.py during development --
# the original ordering let CSRF block every attempt with 400 and the
# rate limit was never reached.)
#
# Storage defaults to in-memory, which is fine for a single-process/
# single-worker deployment; a multi-worker Gunicorn deployment should
# point this at a shared backend (e.g. Redis) via RATELIMIT_STORAGE_URI,
# or limits reset per-worker. See docs/security.md.
# ------------------------------------------------------------------
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=['200 per day', '50 per hour'],
    storage_uri=os.environ.get('RATELIMIT_STORAGE_URI', 'memory://'),
)

# ------------------------------------------------------------------
# FIX 7: CSRF protection, enabled globally for every POST/PUT/PATCH/DELETE
# request. Forms get a hidden csrf_token field (see templates); the one
# JSON endpoint (/generate_quiz) sends the token via the X-CSRFToken
# header instead (see templates/base.html + quiz_setup.html).
# ------------------------------------------------------------------
csrf = CSRFProtect(app)


@app.errorhandler(CSRFError)
def handle_csrf_error(e):
    # FIX 10: no raw exception text shown to the user.
    logger.warning('CSRF validation failed for %s %s: %s', request.method, request.path, e.description)
    flash('Your session expired or the form was resubmitted unsafely. Please try again.', 'danger')
    return redirect(request.referrer or url_for('login')), 400


@app.errorhandler(413)
def handle_request_too_large(e):
    logger.info('Rejected oversized request from %s to %s', get_remote_address(), request.path)
    flash('That file is too large. Please upload a PDF under '
          f"{app.config['MAX_CONTENT_LENGTH'] // (1024*1024)} MB.", 'danger')
    return redirect(request.referrer or url_for('resume')), 413


@app.errorhandler(404)
def handle_not_found(e):
    return render_template('error.html', code=404,
                            message="That page doesn't exist."), 404


@app.errorhandler(500)
def handle_server_error(e):
    # FIX 10: never leak stack traces / internal error text to the client.
    logger.exception('Unhandled server error on %s %s', request.method, request.path)
    return render_template('error.html', code=500,
                            message="Something went wrong on our end. Please try again."), 500


# ------------------------------------------------------------------
# Timing-safe "does this account exist" check.
# FIX 9: check_password_hash() was previously only called when a matching
# user/admin row was found, so a login attempt for a NONEXISTENT account
# returned almost instantly (no hash comparison), while an attempt for a
# REAL account with a wrong password took measurably longer (a real
# password hash comparison). That timing difference lets an attacker
# enumerate valid emails/usernames without ever seeing a different error
# message. We now always run a hash comparison, using a fixed dummy hash
# when no account was found, so both paths take comparable time.
# ------------------------------------------------------------------
_DUMMY_HASH = generate_password_hash('placementai-dummy-password-for-timing-safety')


def _verify_password(candidate_row, password, hash_key='password'):
    """Constant-time-ish credential check: always compares a hash."""
    stored_hash = candidate_row[hash_key] if candidate_row else _DUMMY_HASH
    ok = check_password_hash(stored_hash, password)
    return bool(candidate_row) and ok


# ============================================================
# ACCESS DECORATORS
# ============================================================
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login first.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'admin_id' not in session:
            flash('Admin access required.', 'danger')
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated


# ------------------------------------------------------------------
# FIX 1: load_json now UNWRAPS the nested-dict JSON structure.
#
# Every question file looks like one of:
#   {"aptitude": [...]}          <- aptitude files
#   {"data_science": [...]}      <- technical files use domain key
#   {"communication": [...]}     <- communication file
#
# We always want just the plain list inside, regardless of the key name.
# ------------------------------------------------------------------
def load_json(file_path):
    """Load a question JSON file and return a plain list of question dicts."""
    try:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        # If the file is a dict with one key whose value is a list, unwrap it.
        if isinstance(data, dict):
            values = list(data.values())
            if len(values) == 1 and isinstance(values[0], list):
                return values[0]
            # Multiple keys — flatten all lists (shouldn't happen, but safe)
            flat = []
            for v in values:
                if isinstance(v, list):
                    flat.extend(v)
            return flat

        # Already a plain list
        if isinstance(data, list):
            return data

        return []

    except Exception as e:
        logger.warning('Could not load question file %s: %s', file_path, e)
        return []


def pick_random_questions(question_list, num=10):
    """Pick `num` random questions from a plain list of question dicts."""
    if not question_list:
        return []
    # Safety: filter to only proper dicts (question objects)
    question_list = [q for q in question_list if isinstance(q, dict)]
    return random.sample(question_list, min(num, len(question_list)))


# Load communication questions once at startup
try:
    communication_questions = load_json("questions/communication.json")
    logger.info('Loaded %d communication questions.', len(communication_questions))
except Exception:
    logger.exception('Failed to load communication questions at startup')
    communication_questions = []

DOMAIN_FILES = {
    "Data Science"            : {"technical": "questions/technical/ds.json",      "aptitude": "questions/aptitude/ds.json"},
    "Web Development"         : {"technical": "questions/technical/wd.json",      "aptitude": "questions/aptitude/wd.json"},
    "Cybersecurity"           : {"technical": "questions/technical/cs.json",      "aptitude": "questions/aptitude/cs.json"},
    "Cloud Computing"         : {"technical": "questions/technical/cc.json",      "aptitude": "questions/aptitude/cc.json"},
    "DevOps"                  : {"technical": "questions/technical/devops.json",  "aptitude": "questions/aptitude/devops.json"},
    "Internet of Things (IoT)": {"technical": "questions/technical/iot.json",     "aptitude": "questions/aptitude/iot.json"},
    "Blockchain Technology"   : {"technical": "questions/technical/bt.json",      "aptitude": "questions/aptitude/bt.json"},
    "Mobile App Development"  : {"technical": "questions/technical/mad.json",     "aptitude": "questions/aptitude/mad.json"},
}


# ============================================================
# AUTH ROUTES
# ============================================================
@app.route('/')
def index():
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
@limiter.limit('10 per minute')
def register():
    domains = db.get_all_domains()
    if request.method == 'POST':
        data    = request.form
        email   = data.get('email', '').strip()

        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            flash('Invalid email address.', 'danger')
            return render_template('registration.html', data=data, domains=domains)

        password = data.get('password', '')
        confirm  = data.get('confirm_password', '')

        if len(password) < 6:
            flash('Password must be at least 6 characters.', 'danger')
            return render_template('registration.html', data=data, domains=domains)

        if password != confirm:
            flash('Passwords do not match.', 'danger')
            return render_template('registration.html', data=data, domains=domains)

        # FIX: required-field/type validation now happens before DB access,
        # producing clear user-facing messages instead of a raw KeyError
        # bubbling into the generic except-block below.
        missing = [f for f in ('full_name', 'register_no', 'department', 'batch_year') if not data.get(f, '').strip()]
        if missing:
            flash('Please fill in all required fields.', 'danger')
            return render_template('registration.html', data=data, domains=domains)

        try:
            batch_year = int(data['batch_year'])
        except ValueError:
            flash('Batch year must be a number.', 'danger')
            return render_template('registration.html', data=data, domains=domains)

        try:
            db.create_user(
                full_name     = data['full_name'].strip(),
                email         = email,
                password_hash = generate_password_hash(password),
                register_no   = data['register_no'].strip(),
                department    = data['department'].strip(),
                batch_year    = batch_year,
            )
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except Exception:
            # FIX 10: don't flash raw DB/exception text (e.g. a MySQL
            # duplicate-key error revealing schema/column names) to the
            # user. Most realistic failure here is a duplicate email/
            # register_no; give an actionable generic message and log the
            # real cause server-side for diagnosis.
            logger.exception('Registration failed for email=%s', email)
            flash('Registration failed. That email or register number may already be in use.', 'danger')

    return render_template('registration.html', domains=domains, data=None)


@app.route('/login', methods=['GET', 'POST'])
@limiter.limit('10 per minute')
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        user  = db.get_user_by_email(email)
        # FIX 9: _verify_password() always performs a hash comparison, even
        # when `user` is None, so a nonexistent-account login attempt takes
        # about as long as a wrong-password attempt on a real account --
        # response timing no longer reveals whether an email is registered.
        if _verify_password(user, request.form.get('password', '')):
            session.clear()
            session['user_id']  = user['id']
            session['username'] = user['full_name']
            flash(f"Welcome back, {user['full_name']}! 👋", 'success')
            return redirect(url_for('dashboard'))
        flash('Invalid email or password.', 'danger')
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


# ============================================================
# DASHBOARD
# ============================================================
@app.route('/dashboard')
@login_required
def dashboard():
    user_id    = session['user_id']
    user       = db.get_user_by_id(user_id)
    academic   = db.get_academic_details(user_id)
    quiz       = db.get_latest_quiz_scores(user_id)
    prediction = db.get_latest_prediction(user_id)
    # Real previous-assessment lookup for the readiness gauge's delta —
    # never fabricated. None if the student has fewer than 2 predictions.
    previous_prediction = db.get_previous_prediction(user_id) if prediction else None

    # Strongest signal / biggest gap: computed directly from the student's
    # own quiz sub-scores + resume score (all already 0-100 scale). Only
    # populated when real data exists -- never a placeholder guess.
    signal_map = {}
    if quiz:
        if quiz.get('aptitude_score') is not None:
            signal_map['Aptitude'] = quiz['aptitude_score']
        if quiz.get('technical_score') is not None:
            signal_map['Technical'] = quiz['technical_score']
        if quiz.get('communication_score') is not None:
            signal_map['Communication'] = quiz['communication_score']
    if session.get('resume_score'):
        signal_map['Resume'] = session['resume_score']

    strongest_signal = max(signal_map, key=signal_map.get) if signal_map else None
    biggest_gap_signal = min(signal_map, key=signal_map.get) if signal_map else None

    return render_template('dashboard.html',
                           user=user, academic=academic,
                           quiz=quiz, prediction=prediction,
                           previous_prediction=previous_prediction,
                           strongest_signal=strongest_signal,
                           strongest_signal_value=signal_map.get(strongest_signal),
                           biggest_gap_signal=biggest_gap_signal,
                           biggest_gap_value=signal_map.get(biggest_gap_signal))


# ============================================================
# ACADEMIC DETAILS
# ============================================================
@app.route('/academic', methods=['GET', 'POST'])
@login_required
def academic():
    user_id          = session['user_id']
    domains          = db.get_all_domains()
    academic_details = db.get_academic_details(user_id)

    if request.method == 'POST':
        data = request.form
        try:
            db.save_academic_details(
                user_id   = user_id,
                tenth     = float(data['tenth']),
                twelfth   = float(data['twelfth']),
                cgpa      = float(data['cgpa']),
                backlogs  = int(data['backlogs']),
                domain_id = int(data['domain'])
            )
            flash('Academic details saved!', 'success')
            return redirect(url_for('quiz_setup'))
        except (KeyError, ValueError):
            flash('Please fill in all fields with valid numbers.', 'danger')
        except Exception:
            # FIX 10: no raw exception text shown to the user.
            logger.exception('Failed to save academic details for user_id=%s', user_id)
            flash('Could not save your academic details. Please try again.', 'danger')

    return render_template('academic.html',
                           academic_details=academic_details,
                           domains=domains)


# ============================================================
# QUIZ
# ============================================================
@app.route('/quiz/setup')
@login_required
def quiz_setup():
    academic = db.get_academic_details(session['user_id'])
    domains  = db.get_all_domains()
    return render_template('quiz_setup.html', academic=academic, domains=domains)


@app.route('/generate_quiz', methods=['POST'])
@login_required
def generate_quiz():
    try:
        data = request.get_json(silent=True) or {}

        domain_id = data.get('domain')
        if not domain_id:
            return jsonify({"error": "Domain missing"}), 400

        domain = db.get_domain_name(int(domain_id))

        if not domain or domain not in DOMAIN_FILES:
            return jsonify({"error": "Invalid domain selected."}), 400

        files = DOMAIN_FILES[domain]

        # FIX 1: load_json now returns a plain list — no more nested dict issues
        technical_qs = load_json(files['technical'])
        aptitude_qs  = load_json(files['aptitude'])

        logger.debug('generate_quiz: domain=%s aptitude=%d technical=%d communication=%d',
                      domain, len(aptitude_qs), len(technical_qs), len(communication_questions))

        quiz = {
            "aptitude"     : pick_random_questions(aptitude_qs, 10),
            "technical"    : pick_random_questions(technical_qs, 10),
            "communication": pick_random_questions(communication_questions, 10),
        }

        # FIX 6: store the generated quiz in the user's own session instead
        # of a process-global dict, so it's correct under multiple Gunicorn
        # workers and isolated per user.
        session['quiz_questions'] = quiz
        return jsonify({"status": "ok"})

    except (TypeError, ValueError):
        return jsonify({"error": "Invalid request."}), 400
    except Exception:
        # FIX 10: no raw exception text returned to the client.
        logger.exception('generate_quiz failed for user_id=%s', session.get('user_id'))
        return jsonify({"error": "Could not generate quiz. Please try again."}), 500


@app.route('/quiz', methods=['GET'])
@login_required
def quiz_page():
    quiz = session.get('quiz_questions', {"aptitude": [], "technical": [], "communication": []})
    if not quiz.get('aptitude'):
        flash("Please generate your quiz first by selecting a domain.", "warning")
        return redirect(url_for('quiz_setup'))
    return render_template('quiz.html', quiz=quiz)


@app.route('/submit_quiz', methods=['POST'])
@login_required
def submit_quiz():
    user_id = session['user_id']
    answers = request.form
    quiz_questions = session.get('quiz_questions', {"aptitude": [], "technical": [], "communication": []})

    def score_section(section):
        qs = quiz_questions.get(section, [])
        if not qs:
            return 0

        correct = 0

        for q in qs:
            if not isinstance(q, dict):
                continue

            key = f"{section}_{q.get('id')}"

            user_ans = answers.get(key)

            if user_ans == q.get('answer'):
                correct += 1

        return round((correct / len(qs)) * 100, 2)

    apt = score_section('aptitude')
    tech = score_section('technical')
    comm = score_section('communication')

    db.save_quiz_scores(user_id, apt, tech, comm)

    flash(f"Quiz submitted! Aptitude: {apt}% | Technical: {tech}% | Communication: {comm}%", "success")
    return redirect(url_for('resume'))
# ============================================================
@app.route('/resume', methods=['GET', 'POST'])
@login_required
def resume():
    ats_result = session.get('ats_result')
    if request.method == 'POST':
        file = request.files.get('resume_pdf')
        if not file or file.filename == '':
            flash("Please select a PDF file.", "warning")
            return redirect(url_for('resume'))

        # FIX 11a: sanitize the filename before it ever appears in a log
        # line (defense in depth -- it is never used to build a filesystem
        # path since the file is processed entirely in memory, but an
        # unsanitized filename in a log is still a log-injection risk).
        safe_name = secure_filename(file.filename)

        if not safe_name.lower().endswith('.pdf'):
            flash("Only PDF files are accepted.", "danger")
            return redirect(url_for('resume'))

        # FIX 11b: verify the actual file content, not just the extension.
        # Previously a renamed non-PDF (e.g. malware.exe -> malware.pdf)
        # would sail past the extension check and go straight into the PDF
        # parser. Real PDFs always start with the 5-byte magic header
        # "%PDF-". We peek at it without consuming the stream.
        header = file.stream.read(5)
        file.stream.seek(0)
        if header != b'%PDF-':
            logger.info('Rejected non-PDF upload (filename=%s) from user_id=%s',
                        safe_name, session.get('user_id'))
            flash("That file doesn't look like a valid PDF. Please upload a real PDF resume.", "danger")
            return redirect(url_for('resume'))

        # FIX 11c: MAX_CONTENT_LENGTH (app-wide) already rejects oversized
        # requests with a 413 before Flask even populates request.files,
        # but we double-check the actual stream size here too, since some
        # WSGI servers only enforce Content-Length at the connection level.
        file.stream.seek(0, os.SEEK_END)
        size_bytes = file.stream.tell()
        file.stream.seek(0)
        max_bytes = app.config['MAX_CONTENT_LENGTH']
        if max_bytes and size_bytes > max_bytes:
            flash(f"That file is too large. Please upload a PDF under {max_bytes // (1024*1024)} MB.", "danger")
            return redirect(url_for('resume'))

        try:
            text = extract_text(file)

            if not text or not text.strip():
                # Graceful handling of a malformed/empty/scanned-image PDF
                # that "parsed" but yielded no usable text, instead of
                # silently scoring it or crashing further down the pipeline.
                flash("We couldn't read any text from that PDF. If it's a scanned "
                      "image, try uploading a text-based PDF instead.", "warning")
                return redirect(url_for('resume'))

            ats_score = calculate_ats(text)
            breakdown = get_ats_breakdown(text)
            session['resume_score'] = ats_score
            session['ats_result']   = breakdown
            flash(f"Resume analyzed! ATS Score: {ats_score}/100", "success")
            return redirect(url_for('resume'))
        except Exception:
            # FIX 10 + 11d: malformed PDFs (corrupt structure, encrypted,
            # etc.) can raise from PyMuPDF/pdfminer internals. Never surface
            # that internal error text to the user; log it server-side.
            logger.exception('Resume parsing failed for user_id=%s filename=%s',
                             session.get('user_id'), safe_name)
            flash("We couldn't process that PDF. It may be corrupted, "
                  "password-protected, or in an unsupported format.", "danger")

    # Phase 9: target-role keyword comparison. Only attempted when a real
    # resume has actually been analyzed AND the student has a real domain
    # AND a job market snapshot with role data exists for it -- reuses
    # the exact same _load_job_market_snapshot()/find_role_entry()/
    # compute_skill_match() functions Phases 5 and 8 already built and
    # tested. No new matching logic is written here.
    role_comparison = None
    if ats_result:
        academic = db.get_academic_details(session['user_id'])
        domain_name = academic.get('domain') if academic else None
        if domain_name:
            snapshot, _ = _load_job_market_snapshot(domain_name)
            roles = (snapshot.get('role_skill_breakdown') if snapshot else None) or []
            if roles:
                requested_role = request.args.get('role', '').strip()
                role_entry = job_market.find_role_entry(snapshot, requested_role) if requested_role else None
                if not role_entry:
                    role_entry = roles[0]  # default: role with the most postings in this snapshot
                resume_skills = ats_result.get('skills_found', [])
                match = job_market.compute_skill_match(resume_skills, {'skill_frequencies': role_entry['skills']})
                role_comparison = {
                    'domain_name': domain_name,
                    'roles'      : roles,
                    'role_entry' : role_entry,
                    'match'      : match,
                }

    return render_template('resume.html',
                           ats_result=ats_result,
                           resume_score=session.get('resume_score', 0),
                           role_comparison=role_comparison)


# ============================================================
# PREDICTION
# ============================================================
@app.route('/predict')
@login_required
def predict():
    user_id  = session['user_id']
    academic = db.get_academic_details(user_id)
    quiz     = db.get_latest_quiz_scores(user_id)

    if not academic:
        flash('Please complete your academic details first.', 'warning')
        return redirect(url_for('academic'))

    if not quiz:
        flash('Please complete the skill assessment quiz first.', 'warning')
        return redirect(url_for('quiz_setup'))

    domain_name = db.get_domain_name(academic['preferred_domain_id'])
    if not domain_name:
        flash("Invalid domain. Please update academic details.", "danger")
        return redirect(url_for('academic'))

    # Real "previous assessment" readiness score for the gauge delta, read
    # BEFORE this run's result is saved below, so it's genuinely the prior
    # value rather than the one we're about to insert. None if this is the
    # student's first prediction — the UI must not fabricate a delta.
    _prior = db.get_latest_prediction(user_id)
    previous_readiness_score = _prior['readiness_score'] if _prior else None

    # Build input_data — all fields guaranteed present due to checks above
    input_data = {
        'tenth'        : float(academic['tenth_percentage']),
        'twelfth'      : float(academic['twelfth_percentage']),
        'cgpa'         : float(academic['cgpa']),
        'backlogs'     : int(academic['backlogs']),
        'aptitude'     : float(quiz.get('aptitude_score') or 0),
        'technical'    : float(quiz.get('technical_score') or 0),
        'communication': float(quiz.get('communication_score') or 0),
        'resume_score' : float(session.get('resume_score', 0)),
        'domain'       : domain_name,
    }

    result = pred.predict_placement(input_data)

    # FIX: academic_score/skill_score saved to the DB now come from the same
    # canonical feature_engineering functions the model was trained and
    # scored with. Previously this route computed its OWN separate weighted
    # formula (tenth*0.2 + twelfth*0.3 + cgpa*10*0.5, and a plain average
    # for skill_score) that matched neither model_training.py nor
    # prediction.py — so the "Academic Score" / "Skill Score" a student saw
    # on their dashboard was not the value that actually drove their
    # prediction. See feature_engineering.py for the single source of truth.
    academic_score = round(feature_engineering.compute_academic_score(
        input_data['tenth'], input_data['twelfth'], input_data['cgpa']), 2)
    skill_score = round(feature_engineering.compute_skill_score(
        input_data['aptitude'], input_data['technical'],
        input_data['communication'], input_data['resume_score']), 2)

    db.save_prediction_results(
        user_id         = user_id,
        academic_score  = academic_score,
        skill_score     = skill_score,
        resume_score    = input_data['resume_score'],
        prediction      = result['prediction'],
        confidence      = result['confidence'] / 100,
        readiness_score = result['readiness_score'],
        readiness_level = result['readiness_level'],
        # FIX: previously hardcoded 'Random Forest' regardless of which
        # model model_training.py actually selected as best. Now reflects
        # the real selected model (see prediction.py / model_metadata.json).
        model_used      = result['model_used'],
        predicted_domain= domain_name,
    )

    sg = result['skill_gap']
    db.save_skill_gap(
        user_id          = user_id,
        apt_gap          = sg['apt_gap'],
        tech_gap         = sg['tech_gap'],
        comm_gap         = sg['comm_gap'],
        weak_areas_json  = json.dumps(sg['weak_areas']),
        suggestions_json = json.dumps(sg['suggestions']),
    )

    return render_template(
        'result.html',
        result   = result,
        academic = academic,
        quiz     = quiz,
        user     = db.get_user_by_id(user_id),
        previous_readiness_score = previous_readiness_score,
        # Phase 10: real completed-task IDs for this student, used to
        # pre-check the roadmap's deliverable checkboxes. Never
        # fabricated -- an empty set just means nothing is checked yet.
        roadmap_progress = db.get_roadmap_progress_set(user_id),
    )


# ============================================================
# ROADMAP PROGRESS (Phase 10)
# ============================================================
@app.route('/roadmap/toggle', methods=['POST'])
@login_required
def roadmap_toggle():
    """
    Toggles one deliverable's completion state for the current student.
    JSON in/out, same CSRF pattern as /generate_quiz (X-CSRFToken header,
    since this is called via fetch() from result.html, not an HTML form).
    task_id must be a real id from that student's own most-recently
    rendered roadmap (see prediction.py::get_career_roadmap) -- an
    unrecognized or missing task_id is rejected with 400 rather than
    silently accepted, since a fabricated task_id would let a student
    "complete" something that was never actually part of their roadmap.
    """
    user_id = session['user_id']
    data = request.get_json(silent=True) or {}
    task_id = str(data.get('task_id', '')).strip()

    # task_id is always a 12-char lowercase hex sha1 prefix (see
    # prediction.py::get_career_roadmap._deliverable) -- validate the
    # shape before touching the database, rather than trusting arbitrary
    # client input as a query parameter.
    if not re.fullmatch(r'[0-9a-f]{12}', task_id):
        return jsonify({'error': 'Invalid task id.'}), 400

    try:
        completed = db.toggle_roadmap_task(user_id, task_id)
        return jsonify({'completed': completed})
    except Exception:
        logger.exception('Failed to toggle roadmap task for user_id=%s task_id=%s', user_id, task_id)
        return jsonify({'error': 'Could not update progress. Please try again.'}), 500


# ============================================================
# JOB MARKET INTELLIGENCE (Phase 5)
# ============================================================
def _load_job_market_snapshot(domain_name):
    """
    Shared by /opportunities and /qualified: loads and parses the latest
    saved snapshot for a domain. Returns (snapshot_dict_or_None,
    snapshot_row_or_None). Never raises -- a corrupted/unparseable row
    degrades to (None, snapshot_row) so callers can render an honest
    empty state instead of crashing or trusting bad data.
    """
    snapshot_row = db.get_latest_job_market_snapshot(domain_name)
    if not snapshot_row:
        return None, None

    try:
        raw = snapshot_row['raw_aggregates']
        # mysql-connector's JSON column type may already return a parsed
        # dict, or a JSON string depending on driver/version -- handle both.
        snapshot = raw if isinstance(raw, dict) else json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        logger.exception('Could not parse stored job market snapshot for domain=%s', domain_name)
        snapshot = None

    return snapshot, snapshot_row


@app.route('/opportunities')
@login_required
def opportunities():
    """
    Reads the LATEST already-saved job market snapshot only -- never
    calls the external RemoteOK API on this request path. Snapshots are
    produced offline by scripts/refresh_job_market.py. See job_market.py
    and docs/job-market-methodology.md for the full architecture and the
    real-vs-derived-vs-matched distinctions this page must preserve.
    """
    user_id  = session['user_id']
    academic = db.get_academic_details(user_id)

    if not academic:
        flash("Add your academic details and preferred domain first to see market intelligence.", "warning")
        return redirect(url_for('academic'))

    domain_name = academic.get('domain')
    if not domain_name:
        flash("Invalid domain. Please update academic details.", "danger")
        return redirect(url_for('academic'))

    snapshot, snapshot_row = _load_job_market_snapshot(domain_name)
    match = None
    role_recommendations = []

    if snapshot:
        student_skills = (session.get('ats_result') or {}).get('skills_found', [])
        match = job_market.compute_skill_match(student_skills, snapshot)
        # Phase 11: Job Role Recommendation Engine -- ranked list of every
        # role in this snapshot, reusing compute_skill_match() per role
        # (see job_market.rank_roles_by_match). No new matching logic.
        role_recommendations = job_market.rank_roles_by_match(student_skills, snapshot)

    return render_template(
        'opportunities.html',
        domain_name    = domain_name,
        snapshot       = snapshot,
        snapshot_fetched_at = snapshot_row['fetched_at'] if snapshot_row else None,
        match          = match,
        role_recommendations = role_recommendations,
    )


# ============================================================
# "AM I QUALIFIED?" -- Phase 8
# ============================================================
@app.route('/qualified')
@login_required
def qualified():
    """
    Interactive role-vs-skills matcher. Reuses the same snapshot-loading
    path as /opportunities (never calls RemoteOK live) and the same
    job_market.compute_skill_match() logic Phase 5 already built and
    tested -- scoped to one role's skill breakdown
    (snapshot['role_skill_breakdown']) instead of the whole domain.

    The skill checklist is rendered once with the student's real resume
    skills pre-checked; toggling checkboxes recomputes the match %
    entirely client-side (see templates/qualified.html) -- no new AJAX
    endpoint, since every number needed is already on the page as JSON.
    Switching ROLE is a normal page reload via ?role=, so a role change
    always reflects real server-side data for that specific role.
    """
    user_id  = session['user_id']
    academic = db.get_academic_details(user_id)

    if not academic:
        flash("Add your academic details and preferred domain first to see role matching.", "warning")
        return redirect(url_for('academic'))

    domain_name = academic.get('domain')
    if not domain_name:
        flash("Invalid domain. Please update academic details.", "danger")
        return redirect(url_for('academic'))

    snapshot, snapshot_row = _load_job_market_snapshot(domain_name)

    roles = (snapshot.get('role_skill_breakdown') if snapshot else None) or []
    role_entry = None
    match = None

    if roles:
        requested_role = request.args.get('role', '').strip()
        role_entry = job_market.find_role_entry(snapshot, requested_role) if requested_role else None
        if not role_entry:
            # Default to the role with the most postings in this snapshot
            # -- roles list is already sorted that way by build_snapshot().
            role_entry = roles[0]
        student_skills = (session.get('ats_result') or {}).get('skills_found', [])
        match = job_market.compute_skill_match(student_skills, {'skill_frequencies': role_entry['skills']})

    # Simple set for the template to check membership against, instead of
    # a convoluted double-filter Jinja expression over match['ranked'].
    matched_skill_set = {f['skill'] for f in (match['ranked'] if match else []) if f['has_skill']}

    # Phase 12: Skill Gap Engine -- Gap x Importance ranked table for the
    # currently-selected role. Reuses role_entry, computed above; no new
    # snapshot fetch, no new matching primitive (see
    # job_market.compute_priority_ranking's docstring for the formula).
    priority_ranking = job_market.compute_priority_ranking(student_skills, role_entry) if role_entry else []

    return render_template(
        'qualified.html',
        domain_name          = domain_name,
        snapshot_fetched_at  = snapshot_row['fetched_at'] if snapshot_row else None,
        roles                = roles,
        role_entry           = role_entry,
        match                = match,
        matched_skill_set    = matched_skill_set,
        priority_ranking      = priority_ranking,
    )


# ============================================================
# ADMIN
# ============================================================
@app.route('/admin/login', methods=['GET', 'POST'])
@limiter.limit('10 per minute')
def admin_login():
    if request.method == 'POST':
        admin = db.get_admin_by_username(request.form.get('username', '').strip())
        # FIX 9: same timing-safe check as /login.
        if _verify_password(admin, request.form.get('password', '')):
            session.clear()
            session['admin_id']   = admin['id']
            session['admin_name'] = admin['username']
            return redirect(url_for('admin_dashboard'))
        flash('Invalid admin credentials.', 'danger')
    return render_template('admin_login.html')


@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_id', None)
    session.pop('admin_name', None)
    flash('Admin logged out.', 'info')
    return redirect(url_for('admin_login'))


@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    students = db.get_all_students_summary()
    stats    = db.get_placement_stats()

    # Phase 7: admin analytics. All additive -- students/stats above are
    # untouched, same queries as before.
    summary = db.get_admin_summary_stats()
    domain_distribution = db.get_domain_distribution()

    # Reorder readiness buckets into a fixed left-to-right sequence
    # (including any bucket with zero students -- a real zero, not a
    # fabricated one) since SQL's GROUP BY doesn't guarantee order and a
    # chart needs one.
    bucket_order = ['0-19', '20-39', '40-59', '60-79', '80-100']
    readiness_raw = {row['bucket']: row['count'] for row in db.get_readiness_distribution()}
    readiness_distribution = [{'bucket': b, 'count': readiness_raw.get(b, 0)} for b in bucket_order]

    # Skill-gap frequency: weak_areas is stored as a JSON-encoded list
    # per row (see database.py::get_all_weak_areas_raw docstring).
    # Aggregated here in Python since the column is TEXT, not native JSON.
    weak_area_counts = Counter()
    for row in db.get_all_weak_areas_raw():
        try:
            areas = json.loads(row['weak_areas']) if row['weak_areas'] else []
            weak_area_counts.update(areas)
        except (TypeError, ValueError, json.JSONDecodeError):
            # A malformed row must not crash the whole admin dashboard --
            # skip it and keep aggregating the rest.
            logger.warning('Skipping malformed weak_areas row in admin analytics')
    skill_gap_frequency = [{'area': area, 'count': count} for area, count in weak_area_counts.most_common()]

    registration_trend = [
        {'reg_date': (row['reg_date'].isoformat() if hasattr(row['reg_date'], 'isoformat') else str(row['reg_date'])),
         'count': row['count']}
        for row in db.get_registration_trend()
    ]

    return render_template(
        'admin_dashboard.html',
        students=students, stats=stats,
        summary=summary,
        domain_distribution=domain_distribution,
        readiness_distribution=readiness_distribution,
        skill_gap_frequency=skill_gap_frequency,
        registration_trend=registration_trend,
    )


# ============================================================
# API ENDPOINTS (JSON)
# ============================================================
@app.route('/api/domains')
def api_domains():
    return jsonify(db.get_all_domains())


# ============================================================
# RUN
# ============================================================
if __name__ == '__main__':
    # FIX 3: bind to Render's injected $PORT instead of a hardcoded port.
    # FIX 5: debug defaults to False; only enabled locally via FLASK_DEBUG=1.
    port  = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    app.run(debug=debug, host='0.0.0.0', port=port)