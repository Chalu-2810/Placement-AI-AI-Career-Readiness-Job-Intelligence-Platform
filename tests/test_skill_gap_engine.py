"""
tests/test_skill_gap_engine.py
---------------------------------
Tests for Phase 12 (Skill Gap Engine — Gap x Importance): job_market.py's
compute_priority_ranking() and its rendering on /qualified.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault('SECRET_KEY', 'test-only-secret-key-not-for-production')

import pytest

from job_market import build_snapshot, find_role_entry, compute_priority_ranking


def _role_entry(postings):
    snap = build_snapshot('Data Science', postings)
    return find_role_entry(snap, 'Data Analyst')


# ------------------------------------------------------------------
# Formula correctness
# ------------------------------------------------------------------
def test_current_is_binary_100_when_skill_present_in_student_skills():
    postings = [{'id': '1', 'position': 'Data Analyst', 'tags': ['sql'], 'description': 'sql role', 'location': 'Remote'}]
    entry = _role_entry(postings)
    ranking = compute_priority_ranking(['sql'], entry)
    assert ranking[0]['current'] == 100.0


def test_current_is_binary_0_when_skill_absent():
    postings = [{'id': '1', 'position': 'Data Analyst', 'tags': ['sql'], 'description': 'sql role', 'location': 'Remote'}]
    entry = _role_entry(postings)
    ranking = compute_priority_ranking([], entry)
    assert ranking[0]['current'] == 0.0


def test_target_equals_real_demand_percentage():
    postings = [
        {'id': '1', 'position': 'Data Analyst', 'tags': ['sql', 'python'], 'description': 'sql python', 'location': 'Remote'},
        {'id': '2', 'position': 'Data Analyst', 'tags': ['sql'], 'description': 'sql only', 'location': 'Remote'},
        {'id': '3', 'position': 'Data Analyst', 'tags': ['sql'], 'description': 'sql only', 'location': 'Remote'},
        {'id': '4', 'position': 'Data Analyst', 'tags': ['sql'], 'description': 'sql only', 'location': 'Remote'},
    ]
    entry = _role_entry(postings)  # sql in 4/4 postings = 100%, python in 1/4 = 25%
    ranking = {r['skill']: r for r in compute_priority_ranking([], entry)}
    assert ranking['sql']['target'] == 100.0
    assert ranking['python']['target'] == 25.0


def test_gap_is_target_minus_current_floored_at_zero():
    postings = [{'id': '1', 'position': 'Data Analyst', 'tags': ['sql'], 'description': 'sql role', 'location': 'Remote'}]
    entry = _role_entry(postings)
    # Student HAS sql (current=100), target=100 -> gap must be 0, never negative.
    ranking = compute_priority_ranking(['sql'], entry)
    assert ranking[0]['gap'] == 0.0
    assert ranking[0]['gap'] >= 0  # never negative by construction


def test_priority_equals_gap_times_importance():
    postings = [
        {'id': '1', 'position': 'Data Analyst', 'tags': ['sql'], 'description': 'sql role', 'location': 'Remote'},
        {'id': '2', 'position': 'Data Analyst', 'tags': ['python'], 'description': 'python role', 'location': 'Remote'},
    ]
    entry = _role_entry(postings)  # sql: 1/2=50% target, python: 1/2=50% target
    ranking = {r['skill']: r for r in compute_priority_ranking([], entry)}
    # gap=50, importance=50/100=0.5 -> priority = 50*0.5 = 25.0
    assert ranking['sql']['priority'] == 25.0
    assert ranking['python']['priority'] == 25.0


# ------------------------------------------------------------------
# Ranking order and labels
# ------------------------------------------------------------------
def test_ranked_by_priority_descending():
    postings = [
        {'id': '1', 'position': 'Data Analyst', 'tags': ['sql', 'python', 'excel'], 'description': 'sql python excel role', 'location': 'Remote'},
        {'id': '2', 'position': 'Data Analyst', 'tags': ['sql', 'python'], 'description': 'sql python role', 'location': 'Remote'},
        {'id': '3', 'position': 'Data Analyst', 'tags': ['sql'], 'description': 'sql role', 'location': 'Remote'},
    ]
    entry = _role_entry(postings)
    ranking = compute_priority_ranking([], entry)  # no skills -> gap=target for everyone
    priorities = [r['priority'] for r in ranking]
    assert priorities == sorted(priorities, reverse=True)
    assert ranking[0]['skill'] == 'sql'  # highest demand (3/3=100%) among gaps -> highest priority


def test_priority_label_high_for_top_tier():
    postings = [
        {'id': '1', 'position': 'Data Analyst', 'tags': ['sql'], 'description': 'sql role', 'location': 'Remote'},
        {'id': '2', 'position': 'Data Analyst', 'tags': ['sql'], 'description': 'sql role', 'location': 'Remote'},
        {'id': '3', 'position': 'Data Analyst', 'tags': ['python'], 'description': 'python role', 'location': 'Remote'},
    ]
    entry = _role_entry(postings)
    ranking = compute_priority_ranking([], entry)
    by_skill = {r['skill']: r for r in ranking}
    assert by_skill['sql']['priority_label'] == 'HIGH'  # the top-priority skill is always HIGH by the >=66% rule


def test_priority_label_low_when_no_gap():
    postings = [{'id': '1', 'position': 'Data Analyst', 'tags': ['sql'], 'description': 'sql role', 'location': 'Remote'}]
    entry = _role_entry(postings)
    ranking = compute_priority_ranking(['sql'], entry)  # student already has the only skill -> zero priority everywhere
    assert ranking[0]['priority_label'] == 'Low'


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------
def test_empty_role_entry_returns_empty_list():
    assert compute_priority_ranking(['sql'], None) == []
    assert compute_priority_ranking(['sql'], {}) == []
    assert compute_priority_ranking(['sql'], {'skills': [], 'posting_count': 0}) == []


def test_zero_posting_count_does_not_crash_divide_by_zero():
    entry = {'role': 'X', 'posting_count': 0, 'skills': [{'skill': 'sql', 'count': 0}]}
    ranking = compute_priority_ranking([], entry)
    assert ranking[0]['target'] == 0.0
    assert ranking[0]['priority'] == 0.0


def test_all_values_within_expected_bounds():
    postings = [
        {'id': str(i), 'position': 'Data Analyst', 'tags': ['sql', 'python', 'excel'],
         'description': 'sql python excel role', 'location': 'Remote'}
        for i in range(5)
    ]
    entry = _role_entry(postings)
    ranking = compute_priority_ranking(['sql'], entry)
    for r in ranking:
        assert 0 <= r['current'] <= 100
        assert 0 <= r['target'] <= 100
        assert 0 <= r['gap'] <= 100
        assert 0 <= r['priority'] <= 100
        assert r['priority_label'] in ('HIGH', 'Medium', 'Low')


# ------------------------------------------------------------------
# /qualified route rendering
# ------------------------------------------------------------------
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


def test_qualified_route_shows_skill_gap_table(client, monkeypatch):
    import database as db
    postings = [
        {'id': '1', 'position': 'Data Analyst', 'tags': ['sql', 'python'], 'description': 'sql python role', 'location': 'Remote'},
        {'id': '2', 'position': 'Data Analyst', 'tags': ['sql'], 'description': 'sql role', 'location': 'Remote'},
    ]
    snap = build_snapshot('Data Science', postings)
    monkeypatch.setattr(db, 'get_academic_details', lambda uid: {'domain': 'Data Science', 'preferred_domain_id': 1})
    monkeypatch.setattr(db, 'get_latest_job_market_snapshot', lambda domain: {
        'fetched_at': snap['fetched_at'], 'raw_aggregates': json.dumps(snap),
    })

    with client.session_transaction() as s:
        s['user_id'] = 1
        s['ats_result'] = {'skills_found': ['sql']}

    resp = client.get('/qualified')
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert 'Skill gap analysis' in body
    assert 'Priority = Gap' in body


def test_qualified_route_no_table_when_no_role_data(client, monkeypatch):
    import database as db
    monkeypatch.setattr(db, 'get_academic_details', lambda uid: {'domain': 'Data Science', 'preferred_domain_id': 1})
    monkeypatch.setattr(db, 'get_latest_job_market_snapshot', lambda domain: None)

    with client.session_transaction() as s:
        s['user_id'] = 1

    resp = client.get('/qualified')
    assert resp.status_code == 200
    assert 'Skill gap analysis' not in resp.get_data(as_text=True)
    assert 'Not enough role data yet' in resp.get_data(as_text=True)
