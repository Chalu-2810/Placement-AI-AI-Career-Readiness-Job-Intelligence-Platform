"""
tests/test_qualified.py
-------------------------
Tests for Phase 8 ("Am I Qualified?"): job_market.py's per-role skill
breakdown extension and the new /qualified route.

Uses the same fixture as tests/test_job_market.py -- no live HTTP calls.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import job_market
from job_market import build_snapshot, find_role_entry, compute_skill_match

FIXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures')


def _load_fixture(name):
    with open(os.path.join(FIXTURES_DIR, name)) as f:
        return json.load(f)


def _ds_snapshot():
    fixture = _load_fixture('remoteok_sample.json')
    ds_postings = [p for p in fixture if p.get('position') in ('Data Analyst', 'Senior Data Scientist', 'BI Analyst')]
    return build_snapshot('Data Science', ds_postings)


# ------------------------------------------------------------------
# role_skill_breakdown aggregation (job_market.py)
# ------------------------------------------------------------------
def test_role_skill_breakdown_contains_one_entry_per_distinct_role_title():
    snap = _ds_snapshot()
    roles = {r['role'] for r in snap['role_skill_breakdown']}
    assert roles == {'Data Analyst', 'Senior Data Scientist', 'BI Analyst'}


def test_role_skill_breakdown_posting_counts_match_real_fixture_postings():
    snap = _ds_snapshot()
    by_role = {r['role']: r for r in snap['role_skill_breakdown']}
    # Each of the 3 DS-domain fixture postings has a distinct title -> 1 posting each.
    assert by_role['Data Analyst']['posting_count'] == 1
    assert by_role['Senior Data Scientist']['posting_count'] == 1
    assert by_role['BI Analyst']['posting_count'] == 1


def test_role_skill_breakdown_is_scoped_per_role_not_domain_wide():
    """
    The whole point of Phase 8: skill lists must be per-role, not just
    the domain's skill_frequencies relabeled. 'Tableau' only appears in
    the BI Analyst posting -- it must NOT show up under Data Analyst.
    """
    snap = _ds_snapshot()
    by_role = {r['role']: {s['skill'] for s in r['skills']} for r in snap['role_skill_breakdown']}
    assert 'tableau' in by_role['BI Analyst']
    assert 'tableau' not in by_role['Data Analyst']
    assert 'machine learning' in by_role['Senior Data Scientist']
    assert 'machine learning' not in by_role['BI Analyst']


def test_role_skill_breakdown_sorted_by_posting_count_descending():
    postings = [
        {'id': '1', 'position': 'Popular Role', 'tags': ['sql'], 'description': 'sql', 'location': 'Remote'},
        {'id': '2', 'position': 'Popular Role', 'tags': ['sql'], 'description': 'sql', 'location': 'Remote'},
        {'id': '3', 'position': 'Rare Role', 'tags': ['python'], 'description': 'python', 'location': 'Remote'},
    ]
    snap = build_snapshot('Data Science', postings)
    assert snap['role_skill_breakdown'][0]['role'] == 'Popular Role'
    assert snap['role_skill_breakdown'][0]['posting_count'] == 2


def test_role_skill_breakdown_empty_for_zero_postings():
    snap = build_snapshot('Data Science', [])
    assert snap['role_skill_breakdown'] == []


def test_existing_snapshot_keys_unchanged_by_phase_8_addition():
    """Regression guard: adding role_skill_breakdown must not alter any Phase 5 key."""
    snap = _ds_snapshot()
    for key in ('domain', 'source', 'fetched_at', 'posting_count', 'skill_frequencies',
                'locations', 'sample_roles', 'salary', 'note'):
        assert key in snap


# ------------------------------------------------------------------
# find_role_entry
# ------------------------------------------------------------------
def test_find_role_entry_returns_correct_entry():
    snap = _ds_snapshot()
    entry = find_role_entry(snap, 'BI Analyst')
    assert entry is not None
    assert entry['role'] == 'BI Analyst'


def test_find_role_entry_returns_none_for_unknown_role():
    snap = _ds_snapshot()
    assert find_role_entry(snap, 'Nonexistent Role Title') is None


def test_find_role_entry_handles_snapshot_without_the_new_key():
    """
    Backward compatibility: a snapshot JSON blob saved by a pre-Phase-8
    refresh run wouldn't have 'role_skill_breakdown' at all. Must not
    crash -- treated as no roles available.
    """
    old_style_snapshot = {'domain': 'Data Science', 'skill_frequencies': []}  # no role_skill_breakdown key
    assert find_role_entry(old_style_snapshot, 'Data Analyst') is None


# ------------------------------------------------------------------
# Role-scoped matching reuses compute_skill_match (no duplicate logic)
# ------------------------------------------------------------------
def test_role_scoped_match_percentage_bounds():
    snap = _ds_snapshot()
    entry = find_role_entry(snap, 'Data Analyst')
    result = compute_skill_match(['sql', 'python', 'power bi', 'excel'], {'skill_frequencies': entry['skills']})
    assert result['match_percentage'] is not None
    assert 0 <= result['match_percentage'] <= 100


def test_role_scoped_match_no_student_skills_is_none_not_zero():
    snap = _ds_snapshot()
    entry = find_role_entry(snap, 'Data Analyst')
    result = compute_skill_match([], {'skill_frequencies': entry['skills']})
    assert result['match_percentage'] is None


# ------------------------------------------------------------------
# /qualified route
# ------------------------------------------------------------------
@pytest.fixture
def app_client(monkeypatch):
    os.environ.setdefault('SECRET_KEY', 'test-only-secret-key-not-for-production')
    import app as app_module
    app_module.app.config['TESTING'] = True
    app_module.app.config['RATELIMIT_ENABLED'] = False
    app_module.limiter.reset()
    with app_module.app.test_client() as c:
        yield c
    app_module.app.config['RATELIMIT_ENABLED'] = False
    app_module.limiter.reset()


def test_qualified_route_requires_login(app_client):
    resp = app_client.get('/qualified', follow_redirects=False)
    assert resp.status_code == 302
    assert '/login' in resp.headers['Location']


def test_qualified_route_empty_state_when_no_snapshot(app_client, monkeypatch):
    import database as db
    monkeypatch.setattr(db, 'get_academic_details', lambda uid: {'domain': 'Data Science', 'preferred_domain_id': 1})
    monkeypatch.setattr(db, 'get_latest_job_market_snapshot', lambda domain: None)

    with app_client.session_transaction() as s:
        s['user_id'] = 1

    resp = app_client.get('/qualified')
    assert resp.status_code == 200
    assert 'Not enough role data yet' in resp.get_data(as_text=True)


def test_qualified_route_redirects_if_no_academic_details(app_client, monkeypatch):
    import database as db
    monkeypatch.setattr(db, 'get_academic_details', lambda uid: None)

    with app_client.session_transaction() as s:
        s['user_id'] = 1

    resp = app_client.get('/qualified', follow_redirects=False)
    assert resp.status_code == 302
    assert '/academic' in resp.headers['Location']


def test_qualified_route_default_role_is_most_posted(app_client, monkeypatch):
    import database as db
    postings = [
        {'id': '1', 'position': 'Common Role', 'tags': ['sql'], 'description': 'sql role', 'location': 'Remote'},
        {'id': '2', 'position': 'Common Role', 'tags': ['sql'], 'description': 'sql role', 'location': 'Remote'},
        {'id': '3', 'position': 'Rare Role', 'tags': ['python'], 'description': 'python role', 'location': 'Remote'},
    ]
    snap = build_snapshot('Data Science', postings)
    monkeypatch.setattr(db, 'get_academic_details', lambda uid: {'domain': 'Data Science', 'preferred_domain_id': 1})
    monkeypatch.setattr(db, 'get_latest_job_market_snapshot', lambda domain: {
        'fetched_at': snap['fetched_at'], 'raw_aggregates': json.dumps(snap),
    })

    with app_client.session_transaction() as s:
        s['user_id'] = 1

    resp = app_client.get('/qualified')
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert 'Skills for Common Role' in body  # 2 postings > 1, should be the default


def test_qualified_route_role_query_param_switches_role(app_client, monkeypatch):
    import database as db
    fixture = _load_fixture('remoteok_sample.json')
    ds_postings = [p for p in fixture if p.get('position') in ('Data Analyst', 'Senior Data Scientist', 'BI Analyst')]
    snap = build_snapshot('Data Science', ds_postings)
    monkeypatch.setattr(db, 'get_academic_details', lambda uid: {'domain': 'Data Science', 'preferred_domain_id': 1})
    monkeypatch.setattr(db, 'get_latest_job_market_snapshot', lambda domain: {
        'fetched_at': snap['fetched_at'], 'raw_aggregates': json.dumps(snap),
    })

    with app_client.session_transaction() as s:
        s['user_id'] = 1

    resp = app_client.get('/qualified?role=BI Analyst')
    body = resp.get_data(as_text=True)
    assert 'Skills for BI Analyst' in body
    assert 'Tableau' in body


def test_qualified_route_unknown_role_param_falls_back_to_default(app_client, monkeypatch):
    import database as db
    fixture = _load_fixture('remoteok_sample.json')
    ds_postings = [p for p in fixture if p.get('position') in ('Data Analyst', 'Senior Data Scientist', 'BI Analyst')]
    snap = build_snapshot('Data Science', ds_postings)
    monkeypatch.setattr(db, 'get_academic_details', lambda uid: {'domain': 'Data Science', 'preferred_domain_id': 1})
    monkeypatch.setattr(db, 'get_latest_job_market_snapshot', lambda domain: {
        'fetched_at': snap['fetched_at'], 'raw_aggregates': json.dumps(snap),
    })

    with app_client.session_transaction() as s:
        s['user_id'] = 1

    resp = app_client.get('/qualified?role=Totally Made Up Role')
    assert resp.status_code == 200  # must not crash on an unrecognized role param
    body = resp.get_data(as_text=True)
    assert 'Skills for' in body  # fell back to a real role, not blank


def test_qualified_route_never_fabricates_posting_count(app_client, monkeypatch):
    """
    The 'matches N analyzed job postings' claim must always equal the
    real posting_count computed by build_snapshot() for that role.
    """
    import database as db
    postings = [
        {'id': str(i), 'position': 'Data Analyst', 'tags': ['sql'], 'description': 'sql role', 'location': 'Remote'}
        for i in range(4)
    ]
    snap = build_snapshot('Data Science', postings)
    monkeypatch.setattr(db, 'get_academic_details', lambda uid: {'domain': 'Data Science', 'preferred_domain_id': 1})
    monkeypatch.setattr(db, 'get_latest_job_market_snapshot', lambda domain: {
        'fetched_at': snap['fetched_at'], 'raw_aggregates': json.dumps(snap),
    })

    with app_client.session_transaction() as s:
        s['user_id'] = 1

    resp = app_client.get('/qualified')
    body = resp.get_data(as_text=True)
    assert 'matches <strong style="color:var(--casing-ink)">4</strong>' in body


def test_qualified_route_handles_corrupted_snapshot_gracefully(app_client, monkeypatch):
    import database as db
    monkeypatch.setattr(db, 'get_academic_details', lambda uid: {'domain': 'Data Science', 'preferred_domain_id': 1})
    monkeypatch.setattr(db, 'get_latest_job_market_snapshot', lambda domain: {
        'fetched_at': '2026-01-01T00:00:00+00:00', 'raw_aggregates': 'not valid json {{',
    })

    with app_client.session_transaction() as s:
        s['user_id'] = 1

    resp = app_client.get('/qualified')
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert 'Not enough role data yet' in body
    assert 'JSONDecodeError' not in body and 'Traceback' not in body
