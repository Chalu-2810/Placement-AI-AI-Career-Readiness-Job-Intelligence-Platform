"""
tests/test_job_market.py
--------------------------
Deterministic tests for job_market.py (Phase 5). No live HTTP calls --
all "fetches" use recorded fixture JSON or mocked `requests.get`.

Covers: successful ingestion, malformed/empty responses, duplicate
postings, skill extraction, skill-frequency aggregation, student/market
matching, match-percentage bounds, snapshot timestamps, stale/empty
snapshot handling, unavailable salary/location data, the authenticated
route, and API failure handling.
"""

import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import requests

import job_market
from job_market import (
    fetch_raw_postings, classify_domain, build_snapshot,
    refresh_all_domains, compute_skill_match,
)

FIXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures')


def _load_fixture(name):
    with open(os.path.join(FIXTURES_DIR, name)) as f:
        return json.load(f)


# ------------------------------------------------------------------
# Fake requests.Response for mocking requests.get
# ------------------------------------------------------------------
class FakeResponse:
    def __init__(self, json_data, status_code=200, raise_error=None):
        self._json_data = json_data
        self.status_code = status_code
        self._raise_error = raise_error

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self._raise_error:
            raise self._raise_error


# ------------------------------------------------------------------
# 1. Successful ingestion
# ------------------------------------------------------------------
def test_fetch_raw_postings_filters_legal_notice_and_parses_real_postings(monkeypatch):
    fixture = _load_fixture('remoteok_sample.json')
    monkeypatch.setattr(job_market.requests, 'get', lambda *a, **kw: FakeResponse(fixture))

    postings = fetch_raw_postings()
    assert len(postings) == 5  # 6 fixture entries minus the 1 legal-notice object
    assert all('position' in p and 'id' in p for p in postings)
    assert not any(p.get('legal') for p in postings)


def test_refresh_all_domains_end_to_end_from_fixture(monkeypatch):
    fixture = _load_fixture('remoteok_sample.json')
    monkeypatch.setattr(job_market.requests, 'get', lambda *a, **kw: FakeResponse(fixture))

    snapshots = refresh_all_domains()
    assert set(snapshots.keys()) == set(job_market.JOB_ROLES.keys())  # every domain gets a snapshot, even if empty

    ds = snapshots['Data Science']
    assert ds['posting_count'] == 3  # Data Analyst, Senior Data Scientist, BI Analyst
    assert ds['source'] == 'RemoteOK'

    web = snapshots['Web Development']
    assert web['posting_count'] == 2  # Frontend Developer, Full Stack Developer


# ------------------------------------------------------------------
# 2. Malformed API response
# ------------------------------------------------------------------
def test_fetch_raw_postings_rejects_non_list_response(monkeypatch):
    monkeypatch.setattr(job_market.requests, 'get', lambda *a, **kw: FakeResponse({'not': 'a list'}))
    with pytest.raises(ValueError):
        fetch_raw_postings()


# ------------------------------------------------------------------
# 3. Empty response
# ------------------------------------------------------------------
def test_fetch_raw_postings_handles_empty_list(monkeypatch):
    monkeypatch.setattr(job_market.requests, 'get', lambda *a, **kw: FakeResponse([]))
    postings = fetch_raw_postings()
    assert postings == []


def test_build_snapshot_handles_zero_postings_without_crashing():
    snap = build_snapshot('Data Science', [])
    assert snap['posting_count'] == 0
    assert snap['skill_frequencies'] == []
    assert snap['locations'] == []
    assert snap['sample_roles'] == []
    assert snap['salary']['available'] is False


# ------------------------------------------------------------------
# 4. Duplicate jobs
# ------------------------------------------------------------------
def test_fetch_raw_postings_dedupes_by_id(monkeypatch):
    fixture = _load_fixture('remoteok_sample.json')
    duplicated = fixture + [fixture[1]]  # repeat the first real posting (id 1001)
    monkeypatch.setattr(job_market.requests, 'get', lambda *a, **kw: FakeResponse(duplicated))

    postings = fetch_raw_postings()
    ids = [p['id'] for p in postings]
    assert len(ids) == len(set(ids)), "duplicate posting id must not appear twice"
    assert len(postings) == 5  # same as the non-duplicated fixture


# ------------------------------------------------------------------
# 5 + 6. Skill extraction / skill-frequency aggregation
# ------------------------------------------------------------------
def test_skill_extraction_and_aggregation_matches_known_fixture_content():
    fixture = _load_fixture('remoteok_sample.json')
    ds_postings = [p for p in fixture if p.get('position') in ('Data Analyst', 'Senior Data Scientist', 'BI Analyst')]

    snapshot = build_snapshot('Data Science', ds_postings)
    skills_by_name = {s['skill']: s['count'] for s in snapshot['skill_frequencies']}

    # 'sql' appears in Data Analyst description+tags, Senior Data Scientist description, BI Analyst tags+description -> 3
    assert skills_by_name.get('sql') == 3
    # 'python' appears in Data Analyst (tags+desc) and Senior Data Scientist (tags+desc) -> 2
    assert skills_by_name.get('python') == 2
    # 'power bi' appears in both Data Analyst ("...Power BI and Excel...") and BI Analyst -> 2
    assert skills_by_name.get('power bi') == 2


def test_classify_domain_assigns_correct_domain_and_none_for_unrelated_text():
    posting = {'position': 'Data Analyst', 'tags': ['sql', 'python'], 'description': 'SQL and Python analytics role'}
    assert classify_domain(posting) == 'Data Science'

    unrelated = {'position': 'Office Manager', 'tags': [], 'description': 'Manage the office and schedule meetings'}
    assert classify_domain(unrelated) is None


# ------------------------------------------------------------------
# 7 + 8. Student/market skill matching + bounds
# ------------------------------------------------------------------
def test_compute_skill_match_basic_overlap():
    snapshot = build_snapshot('Data Science', _load_fixture('remoteok_sample.json')[1:4])  # DS postings
    student_skills = ['sql', 'python']  # has 2 of the top skills, missing e.g. power bi / machine learning

    result = compute_skill_match(student_skills, snapshot)
    assert result['has_student_skills'] is True
    assert result['matched_count'] >= 1
    assert 0 <= result['match_percentage'] <= 100
    matched_names = {r['skill'] for r in result['ranked'] if r['has_skill']}
    assert 'sql' in matched_names
    assert 'python' in matched_names


def test_compute_skill_match_no_student_skills_returns_none_percentage_not_zero():
    """
    Absence of resume skills must not be silently reported as a 0% match
    (which would read as a real negative signal) -- match_percentage must
    be None so the UI can show an honest 'we don't know yet' state.
    """
    snapshot = build_snapshot('Data Science', _load_fixture('remoteok_sample.json')[1:4])
    result = compute_skill_match([], snapshot)
    assert result['has_student_skills'] is False
    assert result['match_percentage'] is None


def test_compute_skill_match_percentage_bounds_never_exceed_100_or_go_negative():
    snapshot = build_snapshot('Data Science', _load_fixture('remoteok_sample.json')[1:4])
    # Student "knows" every possible skill in the taxonomy -- should cap at 100, never overflow.
    from utils.resume import ALL_SKILLS
    everything = list(ALL_SKILLS.keys())
    result = compute_skill_match(everything, snapshot)
    assert result['match_percentage'] is not None
    assert 0 <= result['match_percentage'] <= 100


def test_compute_skill_match_handles_empty_snapshot_skills_gracefully():
    empty_snapshot = build_snapshot('Data Science', [])
    result = compute_skill_match(['python', 'sql'], empty_snapshot)
    assert result['considered_count'] == 0
    assert result['match_percentage'] is None  # nothing to compare against -- not a fabricated 0/100


# ------------------------------------------------------------------
# 9. Snapshot timestamp
# ------------------------------------------------------------------
def test_snapshot_has_iso_timestamp_close_to_now():
    snap = build_snapshot('Data Science', [])
    fetched = datetime.fromisoformat(snap['fetched_at'])
    now = datetime.now(timezone.utc)
    assert abs((now - fetched).total_seconds()) < 10


# ------------------------------------------------------------------
# 11. Unavailable salary/location data (RemoteOK limitation -- never fabricated)
# ------------------------------------------------------------------
def test_salary_unavailable_below_sample_threshold():
    # Fixture's Data Science bucket has only 2 postings with real salary numbers -- below the 5-sample floor.
    ds_postings = _load_fixture('remoteok_sample.json')[1:4]
    snap = build_snapshot('Data Science', ds_postings)
    assert snap['salary']['available'] is False
    assert snap['salary']['sample_size'] == 2  # 1001 and 1002 have salary; 1003 has null


def test_salary_available_when_sample_threshold_met():
    postings = [
        {'id': str(i), 'position': 'Data Analyst', 'tags': ['sql'], 'description': 'sql role',
         'location': 'Remote', 'salary_min': 50000 + i * 1000, 'salary_max': 70000 + i * 1000}
        for i in range(6)
    ]
    snap = build_snapshot('Data Science', postings)
    assert snap['salary']['available'] is True
    assert snap['salary']['sample_size'] == 6
    assert snap['salary']['min'] <= snap['salary']['median'] <= snap['salary']['max']


def test_locations_reflect_only_what_source_actually_provided():
    postings = [{'id': '1', 'position': 'Data Analyst', 'tags': [], 'description': '', 'location': ''}]
    snap = build_snapshot('Data Science', postings)
    # Blank location must not be silently invented as a real location entry.
    assert snap['locations'] == []


# ------------------------------------------------------------------
# 12. Authenticated route
# ------------------------------------------------------------------
def test_opportunities_route_requires_login():
    os.environ.setdefault('SECRET_KEY', 'test-only-secret-key-not-for-production')
    import app as app_module
    app_module.app.config['TESTING'] = True
    client = app_module.app.test_client()
    resp = client.get('/opportunities', follow_redirects=False)
    assert resp.status_code == 302
    assert '/login' in resp.headers['Location']


def test_opportunities_route_shows_empty_state_when_no_snapshot_exists(monkeypatch):
    os.environ.setdefault('SECRET_KEY', 'test-only-secret-key-not-for-production')
    import app as app_module
    import database as db

    app_module.app.config['TESTING'] = True
    monkeypatch.setattr(db, 'get_academic_details', lambda uid: {
        'tenth_percentage': 80, 'twelfth_percentage': 80, 'cgpa': 8.0, 'backlogs': 0,
        'preferred_domain_id': 1, 'domain': 'Data Science',
    })
    monkeypatch.setattr(db, 'get_latest_job_market_snapshot', lambda domain: None)

    client = app_module.app.test_client()
    with client.session_transaction() as sess:
        sess['user_id'] = 1

    resp = client.get('/opportunities')
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert 'No market snapshot yet' in body


def test_opportunities_route_renders_real_snapshot_data(monkeypatch):
    os.environ.setdefault('SECRET_KEY', 'test-only-secret-key-not-for-production')
    import app as app_module
    import database as db

    app_module.app.config['TESTING'] = True
    fixture = _load_fixture('remoteok_sample.json')
    ds_postings = [p for p in fixture if p.get('position') in ('Data Analyst', 'Senior Data Scientist', 'BI Analyst')]
    snapshot = build_snapshot('Data Science', ds_postings)

    monkeypatch.setattr(db, 'get_academic_details', lambda uid: {
        'tenth_percentage': 80, 'twelfth_percentage': 80, 'cgpa': 8.0, 'backlogs': 0,
        'preferred_domain_id': 1, 'domain': 'Data Science',
    })
    monkeypatch.setattr(db, 'get_latest_job_market_snapshot', lambda domain: {
        'fetched_at': snapshot['fetched_at'],
        'raw_aggregates': json.dumps(snapshot),
    })

    client = app_module.app.test_client()
    with client.session_transaction() as sess:
        sess['user_id'] = 1
        sess['ats_result'] = {'skills_found': ['sql', 'python']}

    resp = client.get('/opportunities')
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert 'Data Science' in body
    assert 'Market snapshot as of' in body
    assert 'Sql' in body or 'sql' in body.lower()


def test_opportunities_route_handles_corrupted_snapshot_json_gracefully(monkeypatch):
    os.environ.setdefault('SECRET_KEY', 'test-only-secret-key-not-for-production')
    import app as app_module
    import database as db

    app_module.app.config['TESTING'] = True
    monkeypatch.setattr(db, 'get_academic_details', lambda uid: {
        'tenth_percentage': 80, 'twelfth_percentage': 80, 'cgpa': 8.0, 'backlogs': 0,
        'preferred_domain_id': 1, 'domain': 'Data Science',
    })
    monkeypatch.setattr(db, 'get_latest_job_market_snapshot', lambda domain: {
        'fetched_at': '2026-01-01T00:00:00+00:00',
        'raw_aggregates': 'THIS IS NOT VALID JSON {{{',
    })

    client = app_module.app.test_client()
    with client.session_transaction() as sess:
        sess['user_id'] = 1

    resp = client.get('/opportunities')
    # Must not 500 / leak a raw parser exception -- degrades to the empty state.
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert 'No market snapshot yet' in body
    assert 'JSONDecodeError' not in body


# ------------------------------------------------------------------
# 13. API failure handling (refresh script level)
# ------------------------------------------------------------------
def test_refresh_all_domains_propagates_network_failure(monkeypatch):
    def _raise(*a, **kw):
        raise requests.ConnectionError("simulated network failure")
    monkeypatch.setattr(job_market.requests, 'get', _raise)

    with pytest.raises(requests.RequestException):
        refresh_all_domains()


def test_refresh_script_main_returns_nonzero_on_network_failure(monkeypatch):
    import scripts.refresh_job_market as refresh_script

    def _raise(*a, **kw):
        raise requests.ConnectionError("simulated network failure")
    monkeypatch.setattr(job_market.requests, 'get', _raise)

    exit_code = refresh_script.main()
    assert exit_code == 1


def test_refresh_script_main_returns_nonzero_on_malformed_response(monkeypatch):
    import scripts.refresh_job_market as refresh_script
    monkeypatch.setattr(job_market.requests, 'get', lambda *a, **kw: FakeResponse({'bad': 'shape'}))

    exit_code = refresh_script.main()
    assert exit_code == 1
