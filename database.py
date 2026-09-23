"""
database.py
-----------
MySQL-only database layer.

FIX APPLIED:
  get_all_students_summary() now uses subqueries to pull only the LATEST
  quiz_score and prediction_result per user, eliminating duplicate rows in
  the admin dashboard when a student has taken multiple quiz attempts.
"""

import os
import logging
from dotenv import load_dotenv
import mysql.connector

load_dotenv()
logger = logging.getLogger('placementai.database')

DB_CONFIG = {
    'host'    : os.getenv('DB_HOST', 'localhost'),
    'user'    : os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', ''),
    'database': os.getenv('DB_NAME', 'placement_db'),
    'port'    : int(os.getenv('DB_PORT', 3306))
}

logger.info('Database backend: MySQL (host=%s, db=%s)', DB_CONFIG['host'], DB_CONFIG['database'])


def get_connection():
    return mysql.connector.connect(**DB_CONFIG)


def execute_query(query, params=None, fetch=False, fetchone=False):
    conn   = get_connection()
    cursor = conn.cursor(dictionary=True)
    params = params or ()

    try:
        cursor.execute(query, params)

        if fetch:
            return cursor.fetchall()
        elif fetchone:
            return cursor.fetchone()
        else:
            conn.commit()
            return cursor.lastrowid

    except Exception:
        # FIX (security audit): the raw exception still propagates to the
        # caller (app.py) so callers can decide how to handle it, but we no
        # longer print it -- callers are now responsible for logging via
        # `logger.exception(...)` and showing the user a generic message
        # rather than flashing this exception's text directly (see app.py).
        conn.rollback()
        raise

    finally:
        cursor.close()
        conn.close()


# =========================
# USER OPERATIONS
# =========================
def create_user(full_name, email, password_hash, register_no, department, batch_year):
    q = """INSERT INTO users (full_name, email, password, register_no, department, batch_year)
           VALUES (%s, %s, %s, %s, %s, %s)"""
    return execute_query(q, (full_name, email, password_hash, register_no, department, batch_year))


def get_user_by_email(email):
    return execute_query("SELECT * FROM users WHERE email = %s", (email,), fetchone=True)


def get_user_by_id(user_id):
    return execute_query("SELECT * FROM users WHERE id = %s", (user_id,), fetchone=True)


# =========================
# ACADEMIC DETAILS
# =========================
def save_academic_details(user_id, tenth, twelfth, cgpa, backlogs=0, domain_id=1):
    q = """INSERT INTO academic_details
           (user_id, tenth_percentage, twelfth_percentage, cgpa, backlogs, preferred_domain_id)
           VALUES (%s, %s, %s, %s, %s, %s)
           ON DUPLICATE KEY UPDATE
           tenth_percentage=%s, twelfth_percentage=%s, cgpa=%s,
           backlogs=%s, preferred_domain_id=%s"""
    return execute_query(q, (user_id, tenth, twelfth, cgpa, backlogs, domain_id,
                             tenth, twelfth, cgpa, backlogs, domain_id))


def get_academic_details(user_id):
    q = """SELECT a.*, d.domain_name AS domain
           FROM academic_details a
           LEFT JOIN domains d ON a.preferred_domain_id = d.id
           WHERE a.user_id = %s
           ORDER BY a.updated_at DESC LIMIT 1"""
    return execute_query(q, (user_id,), fetchone=True)


def get_all_domains():
    return execute_query("SELECT * FROM domains ORDER BY id", fetch=True)


def get_domain_name(domain_id):
    q      = "SELECT domain_name FROM domains WHERE id = %s"
    result = execute_query(q, (domain_id,), fetchone=True)
    return result['domain_name'] if result else None


# =========================
# QUIZ OPERATIONS
# =========================
def save_quiz_scores(user_id, aptitude, technical, communication):
    total = round((aptitude + technical + communication) / 3, 2)
    q = """
    INSERT INTO quiz_scores
        (user_id, aptitude_score, technical_score, communication_score, total_score)
    VALUES (%s, %s, %s, %s, %s)
    """
    return execute_query(q, (user_id, aptitude, technical, communication, total))


def get_latest_quiz_scores(user_id):
    q = """SELECT * FROM quiz_scores WHERE user_id = %s
           ORDER BY attempt_date DESC LIMIT 1"""
    return execute_query(q, (user_id,), fetchone=True)


# =========================
# PREDICTION OPERATIONS
# =========================
def save_prediction_results(user_id, academic_score, skill_score, resume_score,
                             prediction, confidence, readiness_score, readiness_level,
                             model_used, predicted_domain=None):
    q = """INSERT INTO prediction_results
           (user_id, academic_score, skill_score, resume_score,
            prediction, confidence_score, readiness_score, readiness_level,
            model_used, predicted_domain)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
    return execute_query(q, (user_id, academic_score, skill_score, resume_score,
                             prediction, confidence, readiness_score, readiness_level,
                             model_used, predicted_domain))


def get_latest_prediction(user_id):
    q = """SELECT * FROM prediction_results WHERE user_id = %s
           ORDER BY predicted_at DESC LIMIT 1"""
    return execute_query(q, (user_id,), fetchone=True)


def get_previous_prediction(user_id):
    """
    The prediction immediately before the current one (2nd most recent row),
    used only to show a genuine "+N since last assessment" delta on the
    readiness gauge. Returns None if the student has fewer than 2 predictions
    on record -- the UI must not fabricate a delta in that case.
    """
    q = """SELECT * FROM prediction_results WHERE user_id = %s
           ORDER BY predicted_at DESC LIMIT 1 OFFSET 1"""
    return execute_query(q, (user_id,), fetchone=True)


# =========================
# SKILL GAP
# =========================
def save_skill_gap(user_id, apt_gap, tech_gap, comm_gap, weak_areas_json, suggestions_json):
    q = """INSERT INTO skill_gap_analysis
           (user_id, apt_gap, tech_gap, comm_gap, weak_areas, suggestions)
           VALUES (%s, %s, %s, %s, %s, %s)"""
    return execute_query(q, (user_id, apt_gap, tech_gap, comm_gap, weak_areas_json, suggestions_json))


# =========================
# ADMIN
# =========================
def get_admin_by_username(username):
    return execute_query("SELECT * FROM admins WHERE username = %s", (username,), fetchone=True)


def get_all_students_summary():
    """
    FIX: Subqueries ensure only the LATEST quiz attempt and latest prediction
    are joined per user, preventing duplicate rows in the admin dashboard.
    """
    q = """
    SELECT
        u.id, u.full_name, u.register_no, u.department, u.batch_year,
        a.tenth_percentage, a.twelfth_percentage, a.cgpa, a.backlogs,
        d.domain_name AS domain,
        q.aptitude_score, q.technical_score, q.communication_score, q.total_score,
        p.prediction, p.confidence_score, p.readiness_score, p.readiness_level
    FROM users u
    LEFT JOIN academic_details a ON u.id = a.user_id
    LEFT JOIN domains d ON a.preferred_domain_id = d.id
    LEFT JOIN (
        SELECT qs.*
        FROM quiz_scores qs
        INNER JOIN (
            SELECT user_id, MAX(id) AS max_id
            FROM quiz_scores
            GROUP BY user_id
        ) latest_q ON qs.id = latest_q.max_id
    ) q ON u.id = q.user_id
    LEFT JOIN (
        SELECT pr.*
        FROM prediction_results pr
        INNER JOIN (
            SELECT user_id, MAX(id) AS max_id
            FROM prediction_results
            GROUP BY user_id
        ) latest_p ON pr.id = latest_p.max_id
    ) p ON u.id = p.user_id
    ORDER BY u.created_at DESC
    """
    return execute_query(q, fetch=True)


def get_placement_stats():
    q = """
    SELECT prediction, COUNT(*) AS count,
           AVG(confidence_score) AS avg_confidence,
           AVG(readiness_score)  AS avg_readiness
    FROM prediction_results
    GROUP BY prediction
    """
    return execute_query(q, fetch=True)


# ============================================================
# ADMIN ANALYTICS (Phase 7)
# ============================================================
# NOTE ON "LATEST PREDICTION PER STUDENT": a student can re-run their
# readiness estimate multiple times (see docs/security.md's documented
# /predict residual finding). The pre-existing get_placement_stats()
# above counts every prediction ROW, so a student who reran 5 times
# counts 5 times. That behavior is untouched (not in scope for this
# phase). The NEW aggregate queries below deliberately use each
# student's LATEST prediction only, via ROW_NUMBER(), so a single
# heavy re-run user cannot skew "Average Readiness" or "Placement Rate."
# This is a real statistical difference worth being precise about, not
# just a style choice -- see docs for the same reasoning applied here.

def get_admin_summary_stats():
    """
    One row of headline numbers for the admin dashboard's stat cards.
    avg_ats_score / ats_sample_size only consider predictions where a
    real resume was uploaded (resume_score > 0) -- students who skipped
    resume upload have resume_score=0, and including those would
    understate the average for students who actually uploaded one.
    Sample sizes are returned alongside every averaged metric so the UI
    can disclose them rather than presenting a possibly-small-sample
    average as if it were reliable.
    """
    q = """
    WITH latest_predictions AS (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY predicted_at DESC) AS rn
        FROM prediction_results
    )
    SELECT
        (SELECT COUNT(*) FROM users) AS total_students,
        (SELECT COUNT(DISTINCT user_id) FROM quiz_scores) AS assessments_completed,
        (SELECT COUNT(*) FROM latest_predictions WHERE rn = 1) AS students_with_prediction,
        (SELECT ROUND(AVG(readiness_score), 1) FROM latest_predictions WHERE rn = 1) AS avg_readiness,
        (SELECT ROUND(SUM(prediction = 'Placed') / COUNT(*) * 100, 1)
           FROM latest_predictions WHERE rn = 1) AS placement_rate_pct,
        (SELECT ROUND(AVG(resume_score), 1) FROM latest_predictions WHERE rn = 1 AND resume_score > 0) AS avg_ats_score,
        (SELECT COUNT(*) FROM latest_predictions WHERE rn = 1 AND resume_score > 0) AS ats_sample_size
    """
    return execute_query(q, fetchone=True)


def get_domain_distribution():
    """Student count per domain (their saved preferred_domain_id), including domains with 0 students."""
    q = """
    SELECT d.domain_name AS domain, COUNT(a.id) AS student_count
    FROM domains d
    LEFT JOIN academic_details a ON a.preferred_domain_id = d.id
    GROUP BY d.id, d.domain_name
    ORDER BY student_count DESC
    """
    return execute_query(q, fetch=True)


def get_readiness_distribution():
    """
    Bucketed count of each student's LATEST readiness_score. Bucket
    labels are fixed 20-point bands; ordering into a fixed left-to-right
    sequence (including empty buckets) is handled by the caller in
    app.py, since SQL's GROUP BY on a derived string doesn't guarantee
    an order and a chart needs one.
    """
    q = """
    WITH latest_predictions AS (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY predicted_at DESC) AS rn
        FROM prediction_results
    )
    SELECT
        CASE
            WHEN readiness_score < 20 THEN '0-19'
            WHEN readiness_score < 40 THEN '20-39'
            WHEN readiness_score < 60 THEN '40-59'
            WHEN readiness_score < 80 THEN '60-79'
            ELSE '80-100'
        END AS bucket,
        COUNT(*) AS count
    FROM latest_predictions
    WHERE rn = 1
    GROUP BY bucket
    """
    return execute_query(q, fetch=True)


def get_all_weak_areas_raw():
    """
    Raw weak_areas TEXT column from every skill_gap_analysis row (each a
    JSON-encoded list of strings, e.g. '["Aptitude", "Communication"]').
    The column is TEXT, not a native JSON type, so frequency aggregation
    happens in Python (app.py) rather than via SQL JSON functions here --
    kept as a thin raw-fetch function, consistent with this file's role
    as a pure data-access layer.
    """
    q = "SELECT weak_areas FROM skill_gap_analysis"
    return execute_query(q, fetch=True)


def get_registration_trend():
    """Daily registration counts, chronological. Real counts only -- no interpolation for gap days."""
    q = """
    SELECT DATE(created_at) AS reg_date, COUNT(*) AS count
    FROM users
    GROUP BY DATE(created_at)
    ORDER BY reg_date ASC
    """
    return execute_query(q, fetch=True)


# ============================================================
# ROADMAP PROGRESS (Phase 10)
# ============================================================
def get_roadmap_progress_set(user_id):
    """Returns the set of task_ids this student has marked complete."""
    q = "SELECT task_id FROM roadmap_progress WHERE user_id = %s"
    rows = execute_query(q, (user_id,), fetch=True)
    return {row['task_id'] for row in rows}


def is_roadmap_task_complete(user_id, task_id):
    q = "SELECT id FROM roadmap_progress WHERE user_id = %s AND task_id = %s"
    return execute_query(q, (user_id, task_id), fetchone=True) is not None


def toggle_roadmap_task(user_id, task_id):
    """
    Marks a task complete if it wasn't, or un-marks it if it was.
    Returns the new completion state (True/False) so the caller (the
    /roadmap/toggle route) can report it back without a second query.
    """
    if is_roadmap_task_complete(user_id, task_id):
        execute_query("DELETE FROM roadmap_progress WHERE user_id = %s AND task_id = %s", (user_id, task_id))
        return False
    else:
        execute_query("INSERT INTO roadmap_progress (user_id, task_id) VALUES (%s, %s)", (user_id, task_id))
        return True


# ============================================================
# JOB MARKET INTELLIGENCE (Phase 5)
# ============================================================
def save_job_market_snapshot(domain, source, fetched_at, posting_count, raw_aggregates_json):
    """
    Insert one snapshot row. Called only by scripts/refresh_job_market.py
    -- never by a request handler in app.py (the running app only reads
    snapshots, it never triggers a fetch+write itself). `raw_aggregates_json`
    must already be a JSON-serialized string (see json.dumps in the
    refresh script).
    """
    q = """INSERT INTO job_market_snapshots
           (domain, source, fetched_at, posting_count, raw_aggregates)
           VALUES (%s, %s, %s, %s, %s)"""
    return execute_query(q, (domain, source, fetched_at, posting_count, raw_aggregates_json))


def get_latest_job_market_snapshot(domain):
    """
    Returns the most recent snapshot row for a domain, or None if no
    refresh has ever been run for it. Callers (app.py) must handle the
    None case as an honest empty state -- never fabricate a snapshot.
    """
    q = """SELECT * FROM job_market_snapshots
           WHERE domain = %s
           ORDER BY fetched_at DESC LIMIT 1"""
    return execute_query(q, (domain,), fetchone=True)
