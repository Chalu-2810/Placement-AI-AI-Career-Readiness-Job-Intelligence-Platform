"""
tests/test_core_workflows.py
------------------------------
Phase 13: closes specific, real gaps found by auditing the existing test
suite against the original spec section 18 checklist (registration, quiz
loading/scoring, ATS empty-resume/score-range, ML missing-input
handling) -- NOT a general "add more tests" pass. Everything already
well-covered elsewhere (auth, CSRF, feature consistency, skill matching,
role ranking, roadmap generation) is deliberately not re-tested here.
"""

import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault('SECRET_KEY', 'test-only-secret-key-not-for-production')
os.environ.setdefault('SESSION_COOKIE_SECURE', '0')

import pytest
from werkzeug.security import generate_password_hash

import app as app_module
import database as db
from utils.ats_score import calculate_ats


@pytest.fixture
def client(monkeypatch):
    app_module.app.config['TESTING'] = True
    app_module.app.config['WTF_CSRF_ENABLED'] = True
    app_module.app.config['RATELIMIT_ENABLED'] = False
    app_module.limiter.reset()
    # /quiz/setup (used just to fetch a CSRF token in several tests below)
    # itself calls get_academic_details/get_all_domains -- default them to
    # harmless values here so tests that don't care about academic state
    # don't each need to repeat this mock. Tests that DO care override it.
    monkeypatch.setattr(db, 'get_academic_details', lambda uid: None)
    monkeypatch.setattr(db, 'get_all_domains', lambda: [])
    with app_module.app.test_client() as c:
        yield c
    app_module.app.config['RATELIMIT_ENABLED'] = False
    app_module.limiter.reset()


def _csrf(client, url):
    html = client.get(url).get_data(as_text=True)
    return html.split('name="csrf-token" content="')[1].split('"')[0]


def _login(client, email, password, follow_redirects=True):
    token = _csrf(client, '/login')
    return client.post('/login', data={'email': email, 'password': password, 'csrf_token': token}, follow_redirects=follow_redirects)


# ================================================================
# REGISTRATION (spec section 18: "registration, login, unauthorized access")
# ================================================================
def test_registration_success_creates_user_and_redirects_to_login(client, monkeypatch):
    monkeypatch.setattr(db, 'get_all_domains', lambda: [])
    created = {}
    def fake_create_user(**kw):
        created.update(kw)
        return 1
    monkeypatch.setattr(db, 'create_user', fake_create_user)

    token = _csrf(client, '/register')
    resp = client.post('/register', data={
        'full_name': 'Priya Sharma', 'email': 'priya@example.com', 'register_no': '21CS099',
        'department': 'CSE', 'batch_year': '2025', 'password': 'SecurePass123',
        'confirm_password': 'SecurePass123', 'csrf_token': token,
    }, follow_redirects=True)

    assert resp.status_code == 200
    assert 'Sign in' in resp.get_data(as_text=True)
    assert created['email'] == 'priya@example.com'
    assert created['full_name'] == 'Priya Sharma'
    assert created['batch_year'] == 2025
    assert created['password_hash'] != 'SecurePass123'
    assert created['password_hash'].startswith('pbkdf2:') or created['password_hash'].startswith('scrypt:')


def test_registration_rejects_invalid_email_format(client, monkeypatch):
    monkeypatch.setattr(db, 'get_all_domains', lambda: [])
    token = _csrf(client, '/register')
    resp = client.post('/register', data={
        'full_name': 'X', 'email': 'not-an-email', 'register_no': '1', 'department': 'CSE',
        'batch_year': '2025', 'password': 'SecurePass123', 'confirm_password': 'SecurePass123',
        'csrf_token': token,
    }, follow_redirects=True)
    assert 'Invalid email address' in resp.get_data(as_text=True)


def test_registration_rejects_short_password(client, monkeypatch):
    monkeypatch.setattr(db, 'get_all_domains', lambda: [])
    token = _csrf(client, '/register')
    resp = client.post('/register', data={
        'full_name': 'X', 'email': 'x@example.com', 'register_no': '1', 'department': 'CSE',
        'batch_year': '2025', 'password': 'short', 'confirm_password': 'short', 'csrf_token': token,
    }, follow_redirects=True)
    assert 'at least 6 characters' in resp.get_data(as_text=True)


def test_registration_rejects_mismatched_passwords(client, monkeypatch):
    monkeypatch.setattr(db, 'get_all_domains', lambda: [])
    token = _csrf(client, '/register')
    resp = client.post('/register', data={
        'full_name': 'X', 'email': 'x@example.com', 'register_no': '1', 'department': 'CSE',
        'batch_year': '2025', 'password': 'SecurePass123', 'confirm_password': 'DifferentPass456',
        'csrf_token': token,
    }, follow_redirects=True)
    assert 'do not match' in resp.get_data(as_text=True)


def test_registration_rejects_missing_required_fields(client, monkeypatch):
    monkeypatch.setattr(db, 'get_all_domains', lambda: [])
    token = _csrf(client, '/register')
    resp = client.post('/register', data={
        'full_name': '', 'email': 'x@example.com', 'register_no': '', 'department': '',
        'batch_year': '2025', 'password': 'SecurePass123', 'confirm_password': 'SecurePass123',
        'csrf_token': token,
    }, follow_redirects=True)
    assert 'fill in all required fields' in resp.get_data(as_text=True)


def test_registration_duplicate_email_shows_generic_message_not_raw_db_error(client, monkeypatch):
    monkeypatch.setattr(db, 'get_all_domains', lambda: [])
    def fake_create_user(**kw):
        raise Exception("Duplicate entry 'x@example.com' for key 'users.email'")
    monkeypatch.setattr(db, 'create_user', fake_create_user)

    token = _csrf(client, '/register')
    resp = client.post('/register', data={
        'full_name': 'X', 'email': 'x@example.com', 'register_no': '1', 'department': 'CSE',
        'batch_year': '2025', 'password': 'SecurePass123', 'confirm_password': 'SecurePass123',
        'csrf_token': token,
    }, follow_redirects=True)
    body = resp.get_data(as_text=True)
    assert 'already be in use' in body
    assert 'Duplicate entry' not in body
    assert 'users.email' not in body


def test_registration_rejects_non_numeric_batch_year(client, monkeypatch):
    monkeypatch.setattr(db, 'get_all_domains', lambda: [])
    token = _csrf(client, '/register')
    resp = client.post('/register', data={
        'full_name': 'X', 'email': 'x@example.com', 'register_no': '1', 'department': 'CSE',
        'batch_year': 'not-a-year', 'password': 'SecurePass123', 'confirm_password': 'SecurePass123',
        'csrf_token': token,
    }, follow_redirects=True)
    assert 'Batch year must be a number' in resp.get_data(as_text=True)


# ================================================================
# ATS: empty resume, score range (spec section 18: "empty resume, score range")
# ================================================================
def test_ats_score_stays_within_0_to_100_for_varied_inputs():
    assert 0 <= calculate_ats('') <= 100
    assert 0 <= calculate_ats('a') <= 100
    assert 0 <= calculate_ats('x ' * 5000) <= 100
    rich_resume = (
        "Education: B.Tech Computer Science. Skills: Python, SQL, React, AWS, Docker, "
        "Machine Learning, Git, Java, JavaScript, MongoDB, Kubernetes, Excel, Tableau. "
        "Projects: Built and developed a full-stack application. Led a team of 5 engineers. "
        "Experience: Software Engineer Intern. Contact: test@example.com +91-9876543210. "
        "Achieved 30% improvement in performance, optimized database queries, implemented CI/CD."
    )
    assert 0 <= calculate_ats(rich_resume) <= 100


def test_ats_score_is_higher_for_a_richer_resume_than_a_sparse_one():
    sparse = "Hi I am looking for a job."
    rich = (
        "Education: B.Tech. Skills: Python, SQL, React, AWS. Projects: Built a dashboard. "
        "Experience: Intern. Contact: test@example.com +91-9876543210. "
        "Developed and led a project achieving measurable results."
    )
    assert calculate_ats(rich) > calculate_ats(sparse)


def test_resume_upload_empty_extracted_text_shows_honest_message_not_a_score(client, monkeypatch):
    monkeypatch.setattr(app_module, 'extract_text', lambda f: '   ')

    monkeypatch.setattr(db, 'get_user_by_email', lambda e: {
        'id': 1, 'full_name': 'T', 'email': 'x@example.com', 'password': generate_password_hash('pw123456'),
    })
    _login(client, 'x@example.com', 'pw123456', follow_redirects=False)

    token2 = _csrf(client, '/resume')
    data = {'csrf_token': token2, 'resume_pdf': (io.BytesIO(b'%PDF-1.4\n%%EOF'), 'resume.pdf')}
    resp = client.post('/resume', data=data, content_type='multipart/form-data', follow_redirects=True)
    body = resp.get_data(as_text=True)
    assert "couldn" in body.lower() and "read any text" in body.lower()
    assert 'ATS Score:' not in body


# ================================================================
# ML: missing input handling (spec section 18: "missing input, domain handling")
# ================================================================
def test_predict_redirects_to_academic_when_no_academic_details(client, monkeypatch):
    monkeypatch.setattr(db, 'get_academic_details', lambda uid: None)
    monkeypatch.setattr(db, 'get_latest_quiz_scores', lambda uid: None)
    with client.session_transaction() as s:
        s['user_id'] = 1

    resp = client.get('/predict', follow_redirects=False)
    assert resp.status_code == 302
    assert '/academic' in resp.headers['Location']


def test_predict_shows_warning_flash_when_no_academic_details(client, monkeypatch):
    monkeypatch.setattr(db, 'get_academic_details', lambda uid: None)
    monkeypatch.setattr(db, 'get_latest_quiz_scores', lambda uid: None)
    with client.session_transaction() as s:
        s['user_id'] = 1

    resp = client.get('/predict', follow_redirects=True)
    assert 'complete your academic details first' in resp.get_data(as_text=True).lower()


def test_predict_redirects_to_quiz_setup_when_academic_exists_but_no_quiz(client, monkeypatch):
    monkeypatch.setattr(db, 'get_academic_details', lambda uid: {
        'domain': 'Data Science', 'preferred_domain_id': 1,
        'tenth_percentage': 80, 'twelfth_percentage': 80, 'cgpa': 8.0, 'backlogs': 0,
    })
    monkeypatch.setattr(db, 'get_latest_quiz_scores', lambda uid: None)
    with client.session_transaction() as s:
        s['user_id'] = 1

    resp = client.get('/predict', follow_redirects=False)
    assert resp.status_code == 302
    assert '/quiz/setup' in resp.headers['Location']


def test_predict_redirects_to_academic_when_domain_is_invalid(client, monkeypatch):
    monkeypatch.setattr(db, 'get_academic_details', lambda uid: {
        'domain': None, 'preferred_domain_id': 999,
        'tenth_percentage': 80, 'twelfth_percentage': 80, 'cgpa': 8.0, 'backlogs': 0,
    })
    monkeypatch.setattr(db, 'get_latest_quiz_scores', lambda uid: {
        'aptitude_score': 70, 'technical_score': 70, 'communication_score': 70, 'total_score': 70,
    })
    monkeypatch.setattr(db, 'get_domain_name', lambda did: None)
    with client.session_transaction() as s:
        s['user_id'] = 1

    resp = client.get('/predict', follow_redirects=False)
    assert resp.status_code == 302
    assert '/academic' in resp.headers['Location']


# ================================================================
# QUIZ: loading, scoring, timer/session behavior (spec section 18)
# ================================================================
def test_quiz_page_redirects_when_no_quiz_generated_yet(client):
    with client.session_transaction() as s:
        s['user_id'] = 1

    resp = client.get('/quiz', follow_redirects=False)
    assert resp.status_code == 302
    assert '/quiz/setup' in resp.headers['Location']


def test_generate_quiz_success_loads_real_questions_into_session(client, monkeypatch):
    monkeypatch.setattr(db, 'get_domain_name', lambda did: 'Data Science')
    with client.session_transaction() as s:
        s['user_id'] = 1

    token = _csrf(client, '/quiz/setup')
    resp = client.post('/generate_quiz', json={'domain': 1}, headers={'X-CSRFToken': token})
    assert resp.status_code == 200
    assert resp.get_json() == {'status': 'ok'}

    with client.session_transaction() as s:
        quiz = s.get('quiz_questions')
    assert quiz is not None
    assert len(quiz['aptitude']) == 10
    assert len(quiz['technical']) == 10
    assert len(quiz['communication']) == 10
    for q in quiz['aptitude']:
        assert q.get('id') and q.get('question') and q.get('options') and q.get('answer')


def test_generate_quiz_rejects_missing_domain(client):
    with client.session_transaction() as s:
        s['user_id'] = 1
    token = _csrf(client, '/quiz/setup')
    resp = client.post('/generate_quiz', json={}, headers={'X-CSRFToken': token})
    assert resp.status_code == 400
    assert 'error' in resp.get_json()


def test_generate_quiz_rejects_unknown_domain(client, monkeypatch):
    monkeypatch.setattr(db, 'get_domain_name', lambda did: 'Not A Real Domain')
    with client.session_transaction() as s:
        s['user_id'] = 1
    token = _csrf(client, '/quiz/setup')
    resp = client.post('/generate_quiz', json={'domain': 999}, headers={'X-CSRFToken': token})
    assert resp.status_code == 400


def test_quiz_page_renders_after_successful_generation(client, monkeypatch):
    monkeypatch.setattr(db, 'get_domain_name', lambda did: 'Data Science')
    with client.session_transaction() as s:
        s['user_id'] = 1
    token = _csrf(client, '/quiz/setup')
    client.post('/generate_quiz', json={'domain': 1}, headers={'X-CSRFToken': token})

    resp = client.get('/quiz')
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert 'sec-aptitude' in body
    assert 'sec-technical' in body
    assert 'sec-communication' in body
    assert body.count("type='radio'") + body.count('type="radio"') == 120


def test_submit_quiz_scores_all_correct_answers_as_100_percent(client, monkeypatch):
    monkeypatch.setattr(db, 'get_domain_name', lambda did: 'Data Science')
    saved = {}
    monkeypatch.setattr(db, 'save_quiz_scores', lambda uid, apt, tech, comm: saved.update(
        {'apt': apt, 'tech': tech, 'comm': comm}))

    with client.session_transaction() as s:
        s['user_id'] = 1
    token = _csrf(client, '/quiz/setup')
    client.post('/generate_quiz', json={'domain': 1}, headers={'X-CSRFToken': token})

    with client.session_transaction() as s:
        quiz = s['quiz_questions']

    form_data = {'csrf_token': token}
    for section in ('aptitude', 'technical', 'communication'):
        for q in quiz[section]:
            form_data[f"{section}_{q['id']}"] = q['answer']

    resp = client.post('/submit_quiz', data=form_data, follow_redirects=False)
    assert resp.status_code == 302
    assert saved == {'apt': 100.0, 'tech': 100.0, 'comm': 100.0}


def test_submit_quiz_scores_all_wrong_answers_as_0_percent(client, monkeypatch):
    monkeypatch.setattr(db, 'get_domain_name', lambda did: 'Data Science')
    saved = {}
    monkeypatch.setattr(db, 'save_quiz_scores', lambda uid, apt, tech, comm: saved.update(
        {'apt': apt, 'tech': tech, 'comm': comm}))

    with client.session_transaction() as s:
        s['user_id'] = 1
    token = _csrf(client, '/quiz/setup')
    client.post('/generate_quiz', json={'domain': 1}, headers={'X-CSRFToken': token})

    with client.session_transaction() as s:
        quiz = s['quiz_questions']

    form_data = {'csrf_token': token}
    for section in ('aptitude', 'technical', 'communication'):
        for q in quiz[section]:
            form_data[f"{section}_{q['id']}"] = '__definitely_not_a_real_answer__'

    client.post('/submit_quiz', data=form_data, follow_redirects=False)
    assert saved == {'apt': 0.0, 'tech': 0.0, 'comm': 0.0}


def test_submit_quiz_scores_partial_correctness_accurately(client, monkeypatch):
    monkeypatch.setattr(db, 'get_domain_name', lambda did: 'Data Science')
    saved = {}
    monkeypatch.setattr(db, 'save_quiz_scores', lambda uid, apt, tech, comm: saved.update(
        {'apt': apt, 'tech': tech, 'comm': comm}))

    with client.session_transaction() as s:
        s['user_id'] = 1
    token = _csrf(client, '/quiz/setup')
    client.post('/generate_quiz', json={'domain': 1}, headers={'X-CSRFToken': token})

    with client.session_transaction() as s:
        quiz = s['quiz_questions']

    form_data = {'csrf_token': token}
    aptitude_qs = quiz['aptitude']
    for i, q in enumerate(aptitude_qs):
        form_data[f"aptitude_{q['id']}"] = q['answer'] if i < len(aptitude_qs) // 2 else 'wrong'

    client.post('/submit_quiz', data=form_data, follow_redirects=False)
    expected = round((len(aptitude_qs) // 2) / len(aptitude_qs) * 100, 2)
    assert saved['apt'] == expected


def test_submit_quiz_with_no_answers_submitted_scores_zero_not_a_crash(client, monkeypatch):
    monkeypatch.setattr(db, 'get_domain_name', lambda did: 'Data Science')
    saved = {}
    monkeypatch.setattr(db, 'save_quiz_scores', lambda uid, apt, tech, comm: saved.update(
        {'apt': apt, 'tech': tech, 'comm': comm}))

    with client.session_transaction() as s:
        s['user_id'] = 1
    token = _csrf(client, '/quiz/setup')
    client.post('/generate_quiz', json={'domain': 1}, headers={'X-CSRFToken': token})

    resp = client.post('/submit_quiz', data={'csrf_token': token}, follow_redirects=False)
    assert resp.status_code == 302
    assert saved == {'apt': 0.0, 'tech': 0.0, 'comm': 0.0}


def test_submit_quiz_redirects_to_resume_after_scoring(client, monkeypatch):
    monkeypatch.setattr(db, 'get_domain_name', lambda did: 'Data Science')
    monkeypatch.setattr(db, 'save_quiz_scores', lambda uid, apt, tech, comm: None)
    with client.session_transaction() as s:
        s['user_id'] = 1
    token = _csrf(client, '/quiz/setup')
    client.post('/generate_quiz', json={'domain': 1}, headers={'X-CSRFToken': token})

    resp = client.post('/submit_quiz', data={'csrf_token': token}, follow_redirects=False)
    assert resp.status_code == 302
    assert '/resume' in resp.headers['Location']
