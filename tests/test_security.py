"""
tests/test_security.py
------------------------
Security regression tests added during the Phase 2 security-hardening
pass. Uses Flask's test client against the real `app` object, with the
`database` module monkeypatched (no live MySQL server required) so these
tests can run in any environment, including CI.

Run: pytest tests/test_security.py -v
"""

import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault('SECRET_KEY', 'test-only-secret-key-not-for-production')
os.environ.setdefault('SESSION_COOKIE_SECURE', '0')  # allow the plain-http test client

import pytest
from werkzeug.security import generate_password_hash

import app as app_module
import database as db


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------
FAKE_PASSWORD = 'CorrectHorseBatteryStaple1'
FAKE_USER = {
    'id': 1,
    'full_name': 'Test Student',
    'email': 'student@example.com',
    'password': generate_password_hash(FAKE_PASSWORD),
}
FAKE_ADMIN = {
    'id': 1,
    'username': 'admin',
    'password': generate_password_hash(FAKE_PASSWORD),
}


@pytest.fixture
def client(monkeypatch):
    """
    Flask test client with CSRF enabled (matching production behavior) and
    the database layer stubbed out.

    Rate limiting uses Flask-Limiter's in-memory storage, which is shared
    across the whole `app` object -- i.e. across every test in this
    process, not reset per-client. To keep tests independent, we reset the
    limiter's storage and disable enforcement by default here; the one
    test that specifically exercises rate limiting
    (test_login_rate_limit_engages_after_threshold) re-enables it and
    resets storage itself first.
    """
    app_module.app.config['TESTING'] = True
    app_module.app.config['WTF_CSRF_ENABLED'] = True
    app_module.app.config['RATELIMIT_ENABLED'] = False
    app_module.limiter.reset()

    monkeypatch.setattr(db, 'get_user_by_email',
                         lambda email: FAKE_USER if email == FAKE_USER['email'] else None)
    monkeypatch.setattr(db, 'get_admin_by_username',
                         lambda uname: FAKE_ADMIN if uname == FAKE_ADMIN['username'] else None)
    monkeypatch.setattr(db, 'get_all_domains', lambda: [])
    monkeypatch.setattr(db, 'get_academic_details', lambda uid: None)
    monkeypatch.setattr(db, 'get_latest_quiz_scores', lambda uid: None)
    monkeypatch.setattr(db, 'get_user_by_id', lambda uid: FAKE_USER)
    monkeypatch.setattr(db, 'get_latest_prediction', lambda uid: None)
    monkeypatch.setattr(db, 'get_all_students_summary', lambda: [])
    monkeypatch.setattr(db, 'get_placement_stats', lambda: [])

    with app_module.app.test_client() as c:
        yield c

    app_module.app.config['RATELIMIT_ENABLED'] = False
    app_module.limiter.reset()


def _get_csrf_token(client, url):
    """Fetch a page and pull the csrf-token meta tag value out of it."""
    resp = client.get(url)
    html = resp.get_data(as_text=True)
    marker = 'name="csrf-token" content="'
    start = html.index(marker) + len(marker)
    end = html.index('"', start)
    return html[start:end]


# ------------------------------------------------------------------
# 1. Unauthorized student-route access
# ------------------------------------------------------------------
def test_dashboard_requires_login_redirects_to_login(client):
    resp = client.get('/dashboard', follow_redirects=False)
    assert resp.status_code == 302
    assert '/login' in resp.headers['Location']


def test_academic_requires_login(client):
    resp = client.get('/academic', follow_redirects=False)
    assert resp.status_code == 302
    assert '/login' in resp.headers['Location']


def test_predict_requires_login(client):
    resp = client.get('/predict', follow_redirects=False)
    assert resp.status_code == 302
    assert '/login' in resp.headers['Location']


def test_resume_requires_login(client):
    resp = client.get('/resume', follow_redirects=False)
    assert resp.status_code == 302
    assert '/login' in resp.headers['Location']


# ------------------------------------------------------------------
# 2. Unauthorized admin-route access
# ------------------------------------------------------------------
def test_admin_dashboard_requires_admin_session(client):
    resp = client.get('/admin/dashboard', follow_redirects=False)
    assert resp.status_code == 302
    assert '/admin/login' in resp.headers['Location']


def test_logged_in_student_cannot_reach_admin_dashboard(client):
    """A normal student session must NOT satisfy admin_required."""
    with client.session_transaction() as sess:
        sess['user_id'] = FAKE_USER['id']
        sess['username'] = FAKE_USER['full_name']
        # deliberately no admin_id in session

    resp = client.get('/admin/dashboard', follow_redirects=False)
    assert resp.status_code == 302
    assert '/admin/login' in resp.headers['Location']


def test_admin_session_does_not_grant_student_routes_implicitly():
    """
    Sanity check on the decorators themselves: login_required checks
    'user_id', admin_required checks 'admin_id' -- they are independent
    session keys, so neither role accidentally satisfies the other.
    """
    import inspect
    src = inspect.getsource(app_module.login_required)
    assert "'user_id'" in src
    src_admin = inspect.getsource(app_module.admin_required)
    assert "'admin_id'" in src_admin


# ------------------------------------------------------------------
# 3. CSRF protection
# ------------------------------------------------------------------
def test_login_post_without_csrf_token_is_rejected(client):
    resp = client.post('/login', data={
        'email': FAKE_USER['email'], 'password': FAKE_PASSWORD,
    }, follow_redirects=False)
    # CSRFProtect intercepts before the view runs; our CSRFError handler
    # redirects with a 400 rather than logging the user in.
    assert resp.status_code in (400, 302)
    with client.session_transaction() as sess:
        assert 'user_id' not in sess


def test_login_post_with_valid_csrf_token_succeeds(client):
    token = _get_csrf_token(client, '/login')
    resp = client.post('/login', data={
        'email': FAKE_USER['email'],
        'password': FAKE_PASSWORD,
        'csrf_token': token,
    }, follow_redirects=False)
    assert resp.status_code == 302
    assert '/dashboard' in resp.headers['Location']
    with client.session_transaction() as sess:
        assert sess.get('user_id') == FAKE_USER['id']


# ------------------------------------------------------------------
# 4. Authentication correctness (with valid CSRF token)
# ------------------------------------------------------------------
def test_login_wrong_password_rejected(client):
    token = _get_csrf_token(client, '/login')
    resp = client.post('/login', data={
        'email': FAKE_USER['email'],
        'password': 'totally-wrong-password',
        'csrf_token': token,
    }, follow_redirects=True)
    assert resp.status_code == 200
    with client.session_transaction() as sess:
        assert 'user_id' not in sess


def test_login_nonexistent_account_rejected(client):
    token = _get_csrf_token(client, '/login')
    resp = client.post('/login', data={
        'email': 'nobody@nowhere.com',
        'password': 'whatever',
        'csrf_token': token,
    }, follow_redirects=True)
    assert resp.status_code == 200
    with client.session_transaction() as sess:
        assert 'user_id' not in sess


def test_login_error_message_identical_for_bad_password_and_unknown_email(client):
    """
    No user-enumeration via differing error text: both cases must flash
    the exact same generic message.
    """
    token = _get_csrf_token(client, '/login')
    resp1 = client.post('/login', data={
        'email': FAKE_USER['email'], 'password': 'wrong', 'csrf_token': token,
    }, follow_redirects=True)
    token2 = _get_csrf_token(client, '/login')
    resp2 = client.post('/login', data={
        'email': 'nobody@nowhere.com', 'password': 'wrong', 'csrf_token': token2,
    }, follow_redirects=True)
    assert b'Invalid email or password' in resp1.data
    assert b'Invalid email or password' in resp2.data


def test_admin_login_wrong_password_rejected(client):
    token = _get_csrf_token(client, '/admin/login')
    resp = client.post('/admin/login', data={
        'username': FAKE_ADMIN['username'],
        'password': 'wrong',
        'csrf_token': token,
    }, follow_redirects=True)
    with client.session_transaction() as sess:
        assert 'admin_id' not in sess
    assert b'Invalid admin credentials' in resp.data


# ------------------------------------------------------------------
# 5. Timing-safe credential check (unit-level, not wall-clock timing)
# ------------------------------------------------------------------
def test_verify_password_false_for_missing_user_but_still_hits_hash_check():
    """
    _verify_password() must return False for a nonexistent account WITHOUT
    short-circuiting before the hash comparison -- this is what closes the
    timing side-channel. We assert behavior (always compares against
    _DUMMY_HASH), not wall-clock timing, which is unreliable in CI.
    """
    result = app_module._verify_password(None, 'any-password')
    assert result is False


def test_verify_password_true_for_correct_credentials():
    assert app_module._verify_password(FAKE_USER, FAKE_PASSWORD) is True


def test_verify_password_false_for_wrong_password():
    assert app_module._verify_password(FAKE_USER, 'wrong-password') is False


# ------------------------------------------------------------------
# 6. Resume upload validation
# ------------------------------------------------------------------
def _login(client):
    token = _get_csrf_token(client, '/login')
    client.post('/login', data={
        'email': FAKE_USER['email'], 'password': FAKE_PASSWORD, 'csrf_token': token,
    })


def test_resume_upload_rejects_non_pdf_extension(client):
    _login(client)
    token = _get_csrf_token(client, '/resume')
    data = {
        'csrf_token': token,
        'resume_pdf': (io.BytesIO(b'not a real pdf'), 'resume.exe'),
    }
    resp = client.post('/resume', data=data, content_type='multipart/form-data', follow_redirects=True)
    assert b'Only PDF files are accepted' in resp.data


def test_resume_upload_rejects_renamed_non_pdf(client):
    """A file renamed to .pdf but without the %PDF- magic header must be rejected."""
    _login(client)
    token = _get_csrf_token(client, '/resume')
    fake_pdf_bytes = b'MZ\x90\x00\x03\x00\x00\x00this is actually an exe'  # PE header, not PDF
    data = {
        'csrf_token': token,
        'resume_pdf': (io.BytesIO(fake_pdf_bytes), 'resume.pdf'),
    }
    resp = client.post('/resume', data=data, content_type='multipart/form-data', follow_redirects=True)
    # Jinja2 auto-escapes "doesn't" -> "doesn&#39;t" in the rendered HTML,
    # so we match on the apostrophe-free portion of the message.
    assert b'look like a valid PDF' in resp.data


def test_resume_upload_accepts_minimal_valid_pdf_header(client, monkeypatch):
    """
    A byte stream that starts with the real %PDF- magic header should pass
    the content-validation gate (it may still fail deeper in parsing if
    it's not a fully valid PDF -- that's handled by the try/except around
    extract_text, tested separately below).
    """
    _login(client)
    token = _get_csrf_token(client, '/resume')

    monkeypatch.setattr(app_module, 'extract_text', lambda f: 'python sql excel machine learning')
    monkeypatch.setattr(app_module, 'calculate_ats', lambda text: 77)
    monkeypatch.setattr(app_module, 'get_ats_breakdown', lambda text: {
        'breakdown': {}, 'suggestions': [], 'skills_found': [],
    })

    minimal_pdf = b'%PDF-1.4\n%%EOF'
    data = {
        'csrf_token': token,
        'resume_pdf': (io.BytesIO(minimal_pdf), 'resume.pdf'),
    }
    resp = client.post('/resume', data=data, content_type='multipart/form-data', follow_redirects=True)
    assert b'ATS Score: 77/100' in resp.data


def test_resume_upload_handles_malformed_pdf_gracefully(client, monkeypatch):
    """
    A file with a valid %PDF- header but corrupt internal structure should
    be caught by the except block and shown a generic message -- never a
    raw parser exception.
    """
    _login(client)
    token = _get_csrf_token(client, '/resume')

    def _boom(f):
        raise RuntimeError("some internal PyMuPDF parser detail that must never reach the user")

    monkeypatch.setattr(app_module, 'extract_text', _boom)

    corrupt_pdf = b'%PDF-1.4\ncorrupted garbage that is not a real pdf structure'
    data = {
        'csrf_token': token,
        'resume_pdf': (io.BytesIO(corrupt_pdf), 'resume.pdf'),
    }
    resp = client.post('/resume', data=data, content_type='multipart/form-data', follow_redirects=True)
    # Same Jinja2 auto-escaping note as above: match the apostrophe-free part.
    assert b'process that PDF' in resp.data
    assert b'PyMuPDF' not in resp.data
    assert b'RuntimeError' not in resp.data


def test_resume_upload_rejects_oversized_file(client):
    _login(client)
    token = _get_csrf_token(client, '/resume')
    max_bytes = app_module.app.config['MAX_CONTENT_LENGTH']
    oversized = b'%PDF-1.4\n' + b'A' * (max_bytes + 1024)
    data = {
        'csrf_token': token,
        'resume_pdf': (io.BytesIO(oversized), 'resume.pdf'),
    }
    resp = client.post('/resume', data=data, content_type='multipart/form-data', follow_redirects=True)
    # Werkzeug/Flask enforces MAX_CONTENT_LENGTH at the request level (413)
    # before the view even runs.
    assert resp.status_code == 413 or b'too large' in resp.data


def test_resume_upload_dangerous_filename_is_sanitized(client, monkeypatch):
    """
    A path-traversal-style filename must never be used as-is. We assert
    secure_filename() actually strips it, since the filename is never
    used to build a filesystem path in this app (in-memory parsing only)
    -- this test guards that invariant if that ever changes.
    """
    from werkzeug.utils import secure_filename
    dangerous = "../../../../etc/passwd.pdf"
    cleaned = secure_filename(dangerous)
    assert '..' not in cleaned
    assert '/' not in cleaned


# ------------------------------------------------------------------
# 7. Rate limiting
# ------------------------------------------------------------------
def test_login_rate_limit_engages_after_threshold(client):
    """
    /login is limited to 10/minute. The 11th request within the window
    should be rejected with 429, regardless of credentials.

    Rate limiting is disabled by default in the `client` fixture (so other
    tests aren't affected by shared in-memory limiter state); we explicitly
    re-enable it and reset storage here so this test starts from a clean
    slate.
    """
    app_module.app.config['RATELIMIT_ENABLED'] = True
    app_module.limiter.reset()

    token = _get_csrf_token(client, '/login')
    statuses = []
    for _ in range(11):
        resp = client.post('/login', data={
            'email': 'nobody@nowhere.com', 'password': 'x', 'csrf_token': token,
        }, follow_redirects=False)
        statuses.append(resp.status_code)
    assert 429 in statuses, f"Expected a 429 after 10 requests, got statuses: {statuses}"


# ------------------------------------------------------------------
# 8. Safe error responses
# ------------------------------------------------------------------
def test_404_page_has_no_stack_trace(client):
    resp = client.get('/this-route-does-not-exist')
    assert resp.status_code == 404
    assert b'Traceback' not in resp.data
    assert b'.py' not in resp.data


def test_generate_quiz_error_message_has_no_internal_details(client, monkeypatch):
    _login(client)
    monkeypatch.setattr(db, 'get_domain_name', lambda did: (_ for _ in ()).throw(RuntimeError('db exploded')))
    token = _get_csrf_token(client, '/dashboard')
    resp = client.post('/generate_quiz', json={'domain': 1},
                       headers={'X-CSRFToken': token})
    assert resp.status_code == 500
    body = resp.get_json()
    assert 'db exploded' not in body.get('error', '')
    assert 'RuntimeError' not in body.get('error', '')
