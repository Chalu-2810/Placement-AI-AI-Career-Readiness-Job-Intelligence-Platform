"""
tests/test_admin_analytics.py
-------------------------------
Tests for Phase 7 (Admin Analytics): the new database.py aggregate
queries and the extended /admin/dashboard route behavior.

Real MySQL isn't available in this test environment, so:
  - database.py query-shape tests mock `database.execute_query` itself
    and assert the SQL text references the right tables/clauses (a
    smoke check that the query is at least querying the right things),
    consistent with how DB-dependent code is tested elsewhere in this
    project (see tests/test_security.py's `database` monkeypatching).
  - Route-level tests monkeypatch the db.get_* functions directly (the
    same pattern used throughout tests/test_security.py) and inspect
    the real rendered HTML/embedded JSON, so the actual Jinja logic in
    app.py::admin_dashboard() -- bucket ordering, zero-fill, malformed
    weak_areas handling -- is exercised for real, not just asserted.
"""

import json
import os
import re
import sys
import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault('SECRET_KEY', 'test-only-secret-key-not-for-production')

import pytest
from werkzeug.security import generate_password_hash

import app as app_module
import database as db

FAKE_ADMIN_PASSWORD = 'AdminPass123!'
FAKE_ADMIN = {'id': 1, 'username': 'admin', 'password': generate_password_hash(FAKE_ADMIN_PASSWORD)}

ALL_DOMAINS = [
    'Data Science', 'Web Development', 'Cybersecurity', 'Cloud Computing',
    'DevOps', 'Internet of Things (IoT)', 'Blockchain Technology',
    'Mobile App Development',
]


@pytest.fixture
def admin_client(monkeypatch):
    app_module.app.config['TESTING'] = True
    app_module.app.config['RATELIMIT_ENABLED'] = False
    app_module.limiter.reset()
    monkeypatch.setattr(db, 'get_admin_by_username', lambda u: FAKE_ADMIN if u == 'admin' else None)

    with app_module.app.test_client() as c:
        r = c.get('/admin/login')
        token = r.get_data(as_text=True).split('name="csrf-token" content="')[1].split('"')[0]
        c.post('/admin/login', data={'username': 'admin', 'password': FAKE_ADMIN_PASSWORD, 'csrf_token': token})
        yield c

    app_module.app.config['RATELIMIT_ENABLED'] = False
    app_module.limiter.reset()


def _extract_json_var(body, var_name):
    """Pull `const <var_name> = [...]` out of the rendered <script> block and parse it as JSON."""
    match = re.search(rf'const {var_name} = (\[.*?\]);', body, re.S)
    assert match, f"couldn't find `const {var_name} = [...]` in rendered page"
    return json.loads(match.group(1))


# ------------------------------------------------------------------
# Query-shape smoke checks (no live DB -- mock execute_query itself)
# ------------------------------------------------------------------
def test_get_admin_summary_stats_query_references_expected_tables(monkeypatch):
    captured = {}
    def fake_execute_query(query, params=None, fetch=False, fetchone=False):
        captured['query'] = query
        return {'total_students': 0}
    monkeypatch.setattr(db, 'execute_query', fake_execute_query)

    db.get_admin_summary_stats()
    q = captured['query']
    assert 'ROW_NUMBER()' in q  # latest-prediction-per-student, not every rerun
    assert 'prediction_results' in q
    assert 'resume_score > 0' in q  # ATS average excludes skipped resumes
    assert 'quiz_scores' in q


def test_get_domain_distribution_uses_left_join_so_zero_count_domains_included(monkeypatch):
    captured = {}
    def fake_execute_query(query, params=None, fetch=False, fetchone=False):
        captured['query'] = query
        return []
    monkeypatch.setattr(db, 'execute_query', fake_execute_query)

    db.get_domain_distribution()
    assert 'LEFT JOIN' in captured['query']  # domains with 0 students must still appear


def test_get_readiness_distribution_uses_fixed_buckets_and_latest_prediction(monkeypatch):
    captured = {}
    def fake_execute_query(query, params=None, fetch=False, fetchone=False):
        captured['query'] = query
        return []
    monkeypatch.setattr(db, 'execute_query', fake_execute_query)

    db.get_readiness_distribution()
    q = captured['query']
    assert 'ROW_NUMBER()' in q
    for bucket in ["'0-19'", "'20-39'", "'40-59'", "'60-79'", "'80-100'"]:
        assert bucket in q


def test_get_registration_trend_groups_by_date(monkeypatch):
    captured = {}
    def fake_execute_query(query, params=None, fetch=False, fetchone=False):
        captured['query'] = query
        return []
    monkeypatch.setattr(db, 'execute_query', fake_execute_query)

    db.get_registration_trend()
    assert 'DATE(created_at)' in captured['query']
    assert 'ORDER BY reg_date ASC' in captured['query']


# ------------------------------------------------------------------
# Route-level: readiness bucket ordering + zero-fill (real Jinja logic)
# ------------------------------------------------------------------
def test_readiness_buckets_are_reordered_and_zero_filled(admin_client, monkeypatch):
    monkeypatch.setattr(db, 'get_all_students_summary', lambda: [])
    monkeypatch.setattr(db, 'get_placement_stats', lambda: [])
    monkeypatch.setattr(db, 'get_admin_summary_stats', lambda: {
        'total_students': 3, 'assessments_completed': 2, 'students_with_prediction': 2,
        'avg_readiness': 55.0, 'placement_rate_pct': 50.0, 'avg_ats_score': None, 'ats_sample_size': 0,
    })
    monkeypatch.setattr(db, 'get_domain_distribution', lambda: [{'domain': d, 'student_count': 0} for d in ALL_DOMAINS])
    # Deliberately out of order and missing buckets -- the route must fix both.
    monkeypatch.setattr(db, 'get_readiness_distribution', lambda: [
        {'bucket': '80-100', 'count': 1}, {'bucket': '40-59', 'count': 2},
    ])
    monkeypatch.setattr(db, 'get_all_weak_areas_raw', lambda: [])
    monkeypatch.setattr(db, 'get_registration_trend', lambda: [])

    resp = admin_client.get('/admin/dashboard')
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)

    readiness_data = _extract_json_var(body, 'readinessData')
    buckets_in_order = [row['bucket'] for row in readiness_data]
    assert buckets_in_order == ['0-19', '20-39', '40-59', '60-79', '80-100']
    counts_by_bucket = {row['bucket']: row['count'] for row in readiness_data}
    assert counts_by_bucket['40-59'] == 2
    assert counts_by_bucket['80-100'] == 1
    assert counts_by_bucket['0-19'] == 0  # zero-filled, not omitted
    assert counts_by_bucket['20-39'] == 0
    assert counts_by_bucket['60-79'] == 0


# ------------------------------------------------------------------
# Route-level: skill-gap frequency aggregation, including malformed rows
# ------------------------------------------------------------------
def test_skill_gap_frequency_aggregates_and_skips_malformed_rows_without_crashing(admin_client, monkeypatch):
    monkeypatch.setattr(db, 'get_all_students_summary', lambda: [])
    monkeypatch.setattr(db, 'get_placement_stats', lambda: [])
    monkeypatch.setattr(db, 'get_admin_summary_stats', lambda: {
        'total_students': 0, 'assessments_completed': 0, 'students_with_prediction': 0,
        'avg_readiness': None, 'placement_rate_pct': None, 'avg_ats_score': None, 'ats_sample_size': 0,
    })
    monkeypatch.setattr(db, 'get_domain_distribution', lambda: [])
    monkeypatch.setattr(db, 'get_readiness_distribution', lambda: [])
    monkeypatch.setattr(db, 'get_all_weak_areas_raw', lambda: [
        {'weak_areas': json.dumps(['Aptitude', 'Communication'])},
        {'weak_areas': json.dumps(['Communication'])},
        {'weak_areas': json.dumps(['Communication', 'Resume Quality'])},
        {'weak_areas': None},                    # edge case: null
        {'weak_areas': 'not valid json {{'},      # edge case: malformed
    ])
    monkeypatch.setattr(db, 'get_registration_trend', lambda: [])

    resp = admin_client.get('/admin/dashboard')
    assert resp.status_code == 200  # malformed row must not crash the page
    body = resp.get_data(as_text=True)
    assert 'Traceback' not in body and 'JSONDecodeError' not in body

    skill_gap_data = _extract_json_var(body, 'skillGapData')
    counts = {row['area']: row['count'] for row in skill_gap_data}
    assert counts['Communication'] == 3
    assert counts['Aptitude'] == 1
    assert counts['Resume Quality'] == 1
    # Ranked strongest-first.
    assert skill_gap_data[0]['area'] == 'Communication'


# ------------------------------------------------------------------
# Route-level: full population with real-shaped data
# ------------------------------------------------------------------
def test_admin_dashboard_summary_cards_use_real_computed_values(admin_client, monkeypatch):
    monkeypatch.setattr(db, 'get_all_students_summary', lambda: [])
    monkeypatch.setattr(db, 'get_placement_stats', lambda: [])
    monkeypatch.setattr(db, 'get_admin_summary_stats', lambda: {
        'total_students': 42, 'assessments_completed': 30, 'students_with_prediction': 25,
        'avg_readiness': 61.4, 'placement_rate_pct': 68.0, 'avg_ats_score': 74.2, 'ats_sample_size': 18,
    })
    monkeypatch.setattr(db, 'get_domain_distribution', lambda: [{'domain': 'Data Science', 'student_count': 10}])
    monkeypatch.setattr(db, 'get_readiness_distribution', lambda: [])
    monkeypatch.setattr(db, 'get_all_weak_areas_raw', lambda: [])
    monkeypatch.setattr(db, 'get_registration_trend', lambda: [
        {'reg_date': datetime.date(2026, 1, 1), 'count': 4},
    ])

    resp = admin_client.get('/admin/dashboard')
    body = resp.get_data(as_text=True)
    assert '42' in body       # total_students
    assert '30' in body       # assessments_completed
    assert '61.4' in body     # avg_readiness
    assert '68.0' in body or '68' in body  # placement_rate_pct
    assert '74.2' in body     # avg_ats_score
    assert 'n=18' in body     # ATS sample size disclosed
    assert 'n=25' in body     # readiness/placement-rate sample size disclosed

    # Registration trend date serialized as a plain ISO string, not a raw
    # Python date repr, and safely embedded in the page's JSON.
    reg_data = _extract_json_var(body, 'regData')
    assert reg_data[0]['reg_date'] == '2026-01-01'


def test_admin_dashboard_handles_fully_empty_state(admin_client, monkeypatch):
    monkeypatch.setattr(db, 'get_all_students_summary', lambda: [])
    monkeypatch.setattr(db, 'get_placement_stats', lambda: [])
    monkeypatch.setattr(db, 'get_admin_summary_stats', lambda: {
        'total_students': 0, 'assessments_completed': 0, 'students_with_prediction': 0,
        'avg_readiness': None, 'placement_rate_pct': None, 'avg_ats_score': None, 'ats_sample_size': 0,
    })
    monkeypatch.setattr(db, 'get_domain_distribution', lambda: [{'domain': d, 'student_count': 0} for d in ALL_DOMAINS])
    monkeypatch.setattr(db, 'get_readiness_distribution', lambda: [])
    monkeypatch.setattr(db, 'get_all_weak_areas_raw', lambda: [])
    monkeypatch.setattr(db, 'get_registration_trend', lambda: [])

    resp = admin_client.get('/admin/dashboard')
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert 'No skill gap records yet' in body
    assert 'No registrations recorded yet' in body
    # Placement-distribution/avg-score charts are conditional on `stats`
    # (untouched Phase-3 behavior) -- must not render with empty stats.
    assert 'id="pieChart"' not in body


def test_admin_dashboard_requires_admin_session_still_enforced(monkeypatch):
    """Regression guard: Phase 7 additions must not weaken the existing admin auth check."""
    app_module.app.config['TESTING'] = True
    with app_module.app.test_client() as c:
        resp = c.get('/admin/dashboard', follow_redirects=False)
    assert resp.status_code == 302
    assert '/admin/login' in resp.headers['Location']
