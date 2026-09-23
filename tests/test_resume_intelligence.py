"""
tests/test_resume_intelligence.py
------------------------------------
Tests for Phase 9 (Resume Intelligence — target-role keyword
comparison). This phase adds no new matching logic: it wires the
/resume route to the exact same job_market functions Phases 5 and 8
already built and tested (_load_job_market_snapshot, find_role_entry,
compute_skill_match). These tests focus on the NEW wiring and its edge
cases, not on re-testing that underlying logic (already covered by
tests/test_job_market.py and tests/test_qualified.py).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault('SECRET_KEY', 'test-only-secret-key-not-for-production')

import pytest

from job_market import build_snapshot

FIXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures')


def _load_fixture(name):
    with open(os.path.join(FIXTURES_DIR, name)) as f:
        return json.load(f)


def _ds_snapshot():
    fixture = _load_fixture('remoteok_sample.json')
    ds_postings = [p for p in fixture if p.get('position') in ('Data Analyst', 'Senior Data Scientist', 'BI Analyst')]
    return build_snapshot('Data Science', ds_postings)


@pytest.fixture
def client(monkeypatch):
    import app as app_module
    app_module.app.config['TESTING'] = True
    app_module.app.config['RATELIMIT_ENABLED'] = False
    app_module.limiter.reset()
    with app_module.app.test_client() as c:
        yield c
    app_module.app.config['RATELIMIT_ENABLED'] = False
    app_module.limiter.reset()


def _fake_ats_result(skills):
    return {
        'breakdown': {'Skills Found': 16, 'Key Sections': 18, 'Length': 15, 'Contact Info': 20, 'Action Verbs': 14},
        'suggestions': ['Add measurable outcomes'],
        'skills_found': skills,
    }


# ------------------------------------------------------------------
# No comparison attempted unless a real resume was analyzed
# ------------------------------------------------------------------
def test_no_role_comparison_shown_without_a_resume(client, monkeypatch):
    import database as db
    monkeypatch.setattr(db, 'get_academic_details', lambda uid: {'domain': 'Data Science', 'preferred_domain_id': 1})
    monkeypatch.setattr(db, 'get_latest_job_market_snapshot', lambda domain: {
        'fetched_at': '2026-01-01T00:00:00+00:00', 'raw_aggregates': json.dumps(_ds_snapshot()),
    })

    with client.session_transaction() as s:
        s['user_id'] = 1
        # deliberately no 'ats_result' in session

    resp = client.get('/resume')
    assert resp.status_code == 200
    assert 'Resume →' not in resp.get_data(as_text=True)


# ------------------------------------------------------------------
# Honest empty states when domain/snapshot data is missing
# ------------------------------------------------------------------
def test_honest_message_when_no_academic_domain_set(client, monkeypatch):
    import database as db
    monkeypatch.setattr(db, 'get_academic_details', lambda uid: None)

    with client.session_transaction() as s:
        s['user_id'] = 1
        s['ats_result'] = _fake_ats_result(['sql'])

    resp = client.get('/resume')
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert 'No target-role comparison available' in body
    assert 'Traceback' not in body


def test_honest_message_when_no_snapshot_exists_for_domain(client, monkeypatch):
    import database as db
    monkeypatch.setattr(db, 'get_academic_details', lambda uid: {'domain': 'Data Science', 'preferred_domain_id': 1})
    monkeypatch.setattr(db, 'get_latest_job_market_snapshot', lambda domain: None)

    with client.session_transaction() as s:
        s['user_id'] = 1
        s['ats_result'] = _fake_ats_result(['sql'])

    resp = client.get('/resume')
    assert resp.status_code == 200
    assert 'No target-role comparison available' in resp.get_data(as_text=True)


def test_no_crash_on_corrupted_snapshot(client, monkeypatch):
    import database as db
    monkeypatch.setattr(db, 'get_academic_details', lambda uid: {'domain': 'Data Science', 'preferred_domain_id': 1})
    monkeypatch.setattr(db, 'get_latest_job_market_snapshot', lambda domain: {
        'fetched_at': '2026-01-01T00:00:00+00:00', 'raw_aggregates': 'not valid json {{',
    })

    with client.session_transaction() as s:
        s['user_id'] = 1
        s['ats_result'] = _fake_ats_result(['sql'])

    resp = client.get('/resume')
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert 'No target-role comparison available' in body
    assert 'JSONDecodeError' not in body


# ------------------------------------------------------------------
# The comparison, when it exists, uses real resume skills + real role data
# ------------------------------------------------------------------
def test_role_comparison_uses_resumes_real_skills(client, monkeypatch):
    import database as db
    snapshot = _ds_snapshot()
    monkeypatch.setattr(db, 'get_academic_details', lambda uid: {'domain': 'Data Science', 'preferred_domain_id': 1})
    monkeypatch.setattr(db, 'get_latest_job_market_snapshot', lambda domain: {
        'fetched_at': snapshot['fetched_at'], 'raw_aggregates': json.dumps(snapshot),
    })

    with client.session_transaction() as s:
        s['user_id'] = 1
        s['ats_result'] = _fake_ats_result(['sql', 'python'])  # matches Data Analyst's skill list

    resp = client.get('/resume')
    body = resp.get_data(as_text=True)
    assert 'Resume →' in body
    assert 'Keyword match' in body
    assert '✓ Sql' in body or '✓ SQL' in body.upper() or 'Sql' in body  # matched skill shown as "Strong"


def test_role_query_param_switches_comparison_role(client, monkeypatch):
    import database as db
    snapshot = _ds_snapshot()
    monkeypatch.setattr(db, 'get_academic_details', lambda uid: {'domain': 'Data Science', 'preferred_domain_id': 1})
    monkeypatch.setattr(db, 'get_latest_job_market_snapshot', lambda domain: {
        'fetched_at': snapshot['fetched_at'], 'raw_aggregates': json.dumps(snapshot),
    })

    with client.session_transaction() as s:
        s['user_id'] = 1
        s['ats_result'] = _fake_ats_result(['sql'])

    resp = client.get('/resume?role=BI Analyst')
    body = resp.get_data(as_text=True)
    assert 'Resume → BI Analyst' in body
    assert 'Tableau' in body  # BI-Analyst-specific skill missing from this resume's skill list


def test_unknown_role_param_falls_back_without_crashing(client, monkeypatch):
    import database as db
    snapshot = _ds_snapshot()
    monkeypatch.setattr(db, 'get_academic_details', lambda uid: {'domain': 'Data Science', 'preferred_domain_id': 1})
    monkeypatch.setattr(db, 'get_latest_job_market_snapshot', lambda domain: {
        'fetched_at': snapshot['fetched_at'], 'raw_aggregates': json.dumps(snapshot),
    })

    with client.session_transaction() as s:
        s['user_id'] = 1
        s['ats_result'] = _fake_ats_result(['sql'])

    resp = client.get('/resume?role=Totally Made Up Title')
    assert resp.status_code == 200
    assert 'Resume →' in resp.get_data(as_text=True)  # fell back to a real role, not blank/crashed


def test_role_comparison_never_fabricates_posting_count(client, monkeypatch):
    import database as db
    postings = [
        {'id': str(i), 'position': 'Data Analyst', 'tags': ['sql'], 'description': 'sql role', 'location': 'Remote'}
        for i in range(3)
    ]
    snapshot = build_snapshot('Data Science', postings)
    monkeypatch.setattr(db, 'get_academic_details', lambda uid: {'domain': 'Data Science', 'preferred_domain_id': 1})
    monkeypatch.setattr(db, 'get_latest_job_market_snapshot', lambda domain: {
        'fetched_at': snapshot['fetched_at'], 'raw_aggregates': json.dumps(snapshot),
    })

    with client.session_transaction() as s:
        s['user_id'] = 1
        s['ats_result'] = _fake_ats_result(['sql'])

    resp = client.get('/resume')
    body = resp.get_data(as_text=True)
    assert 'based on\n            3 analyzed postings' in body or 'based on' in body and '3 analyzed posting' in body


def test_no_matched_skills_shows_honest_message_not_blank(client, monkeypatch):
    import database as db
    snapshot = _ds_snapshot()
    monkeypatch.setattr(db, 'get_academic_details', lambda uid: {'domain': 'Data Science', 'preferred_domain_id': 1})
    monkeypatch.setattr(db, 'get_latest_job_market_snapshot', lambda domain: {
        'fetched_at': snapshot['fetched_at'], 'raw_aggregates': json.dumps(snapshot),
    })

    with client.session_transaction() as s:
        s['user_id'] = 1
        s['ats_result'] = _fake_ats_result([])  # resume with zero detected skills

    resp = client.get('/resume')
    body = resp.get_data(as_text=True)
    assert "None of this role's top skills were detected" in body


def test_existing_resume_upload_flow_unaffected(client, monkeypatch):
    """Regression guard: Phase 9 must not break the pre-existing upload/empty-state behavior."""
    import database as db
    monkeypatch.setattr(db, 'get_academic_details', lambda uid: None)

    with client.session_transaction() as s:
        s['user_id'] = 1
        # no ats_result -> should show the original upload/empty-state page

    resp = client.get('/resume')
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert 'Resume tips for ATS' in body
    assert 'Choose PDF file' in body
