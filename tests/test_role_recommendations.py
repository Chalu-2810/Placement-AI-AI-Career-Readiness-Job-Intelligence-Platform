"""
tests/test_role_recommendations.py
------------------------------------
Tests for Phase 11 (Job Role Recommendation Engine): job_market.py's
rank_roles_by_match() and its rendering on /opportunities. This phase
adds no new matching logic -- rank_roles_by_match() composes and sorts
results from the same compute_skill_match() already built and tested in
Phase 5, so these tests focus on the ranking/composition behavior, not
re-testing the underlying match math.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault('SECRET_KEY', 'test-only-secret-key-not-for-production')

import pytest

from job_market import build_snapshot, rank_roles_by_match

FIXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures')


def _load_fixture(name):
    with open(os.path.join(FIXTURES_DIR, name)) as f:
        return json.load(f)


def _ds_snapshot():
    fixture = _load_fixture('remoteok_sample.json')
    ds_postings = [p for p in fixture if p.get('position') in ('Data Analyst', 'Senior Data Scientist', 'BI Analyst')]
    return build_snapshot('Data Science', ds_postings)


# ------------------------------------------------------------------
# rank_roles_by_match() -- composition/sorting
# ------------------------------------------------------------------
def test_rank_roles_returns_one_entry_per_role_with_match_and_posting_count():
    snap = _ds_snapshot()
    ranked = rank_roles_by_match(['sql', 'python'], snap)
    assert len(ranked) == 3
    for r in ranked:
        assert 'role' in r and 'posting_count' in r and 'match' in r
        assert 'match_percentage' in r['match']
        assert 'ranked' in r['match']


def test_rank_roles_sorted_by_match_percentage_descending_when_student_has_skills():
    snap = _ds_snapshot()
    ranked = rank_roles_by_match(['sql', 'python'], snap)
    percentages = [r['match']['match_percentage'] for r in ranked]
    assert percentages == sorted(percentages, reverse=True)
    # Data Analyst's skills are exactly {python, sql, data visualization, power bi} --
    # matching 2/4 should out-rank BI Analyst's {sql, tableau, power bi} matching 1/3.
    assert ranked[0]['role'] == 'Data Analyst'


def test_rank_roles_falls_back_to_posting_count_order_with_no_student_skills():
    postings = [
        {'id': '1', 'position': 'Popular Role', 'tags': ['sql'], 'description': 'sql', 'location': 'Remote'},
        {'id': '2', 'position': 'Popular Role', 'tags': ['sql'], 'description': 'sql', 'location': 'Remote'},
        {'id': '3', 'position': 'Rare Role', 'tags': ['python'], 'description': 'python', 'location': 'Remote'},
    ]
    snap = build_snapshot('Data Science', postings)
    ranked = rank_roles_by_match([], snap)
    assert all(r['match']['match_percentage'] is None for r in ranked)  # honest None, not a fabricated tie-break
    assert ranked[0]['role'] == 'Popular Role'  # 2 postings > 1
    assert ranked[0]['posting_count'] == 2


def test_rank_roles_empty_for_snapshot_with_no_role_breakdown():
    assert rank_roles_by_match(['sql'], {'role_skill_breakdown': []}) == []
    assert rank_roles_by_match(['sql'], {}) == []  # missing key entirely -- must not crash


def test_rank_roles_match_percentage_always_bounded():
    snap = _ds_snapshot()
    from utils.resume import ALL_SKILLS
    everything = list(ALL_SKILLS.keys())
    ranked = rank_roles_by_match(everything, snap)
    for r in ranked:
        assert 0 <= r['match']['match_percentage'] <= 100


def test_rank_roles_uses_compute_skill_match_not_duplicate_logic():
    """
    Cross-check: rank_roles_by_match()'s per-role match must be identical
    to calling compute_skill_match() directly on that role's skill list --
    proves no separate/diverging matching logic was written for Phase 11.
    """
    from job_market import compute_skill_match, find_role_entry
    snap = _ds_snapshot()
    student_skills = ['sql', 'python']
    ranked = rank_roles_by_match(student_skills, snap)

    for r in ranked:
        entry = find_role_entry(snap, r['role'])
        direct = compute_skill_match(student_skills, {'skill_frequencies': entry['skills']})
        assert r['match'] == direct


# ------------------------------------------------------------------
# /opportunities route: ranked list rendering
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


def test_opportunities_shows_ranked_matches_with_real_skills(client, monkeypatch):
    import database as db
    snap = _ds_snapshot()
    monkeypatch.setattr(db, 'get_academic_details', lambda uid: {'domain': 'Data Science', 'preferred_domain_id': 1})
    monkeypatch.setattr(db, 'get_latest_job_market_snapshot', lambda domain: {
        'fetched_at': snap['fetched_at'], 'raw_aggregates': json.dumps(snap),
    })

    with client.session_transaction() as s:
        s['user_id'] = 1
        s['ats_result'] = {'skills_found': ['sql', 'python']}

    resp = client.get('/opportunities')
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert 'Best career matches' in body
    assert '1. Data Analyst' in body  # top-ranked role shown first, numbered
    assert '%' in body


def test_opportunities_ranked_matches_show_posting_count_fallback_without_skills(client, monkeypatch):
    import database as db
    snap = _ds_snapshot()
    monkeypatch.setattr(db, 'get_academic_details', lambda uid: {'domain': 'Data Science', 'preferred_domain_id': 1})
    monkeypatch.setattr(db, 'get_latest_job_market_snapshot', lambda domain: {
        'fetched_at': snap['fetched_at'], 'raw_aggregates': json.dumps(snap),
    })

    with client.session_transaction() as s:
        s['user_id'] = 1

    resp = client.get('/opportunities')
    body = resp.get_data(as_text=True)
    assert 'Best career matches' in body
    assert 'Upload your resume' in body


def test_opportunities_empty_snapshot_shows_honest_message(client, monkeypatch):
    import database as db
    monkeypatch.setattr(db, 'get_academic_details', lambda uid: {'domain': 'Data Science', 'preferred_domain_id': 1})
    monkeypatch.setattr(db, 'get_latest_job_market_snapshot', lambda domain: None)

    with client.session_transaction() as s:
        s['user_id'] = 1

    resp = client.get('/opportunities')
    assert resp.status_code == 200
    assert 'No market snapshot yet' in resp.get_data(as_text=True)


def test_opportunities_route_requires_login_still_enforced(client):
    """Regression guard: Phase 11 additions must not weaken existing auth."""
    resp = client.get('/opportunities', follow_redirects=False)
    assert resp.status_code == 302
    assert '/login' in resp.headers['Location']
