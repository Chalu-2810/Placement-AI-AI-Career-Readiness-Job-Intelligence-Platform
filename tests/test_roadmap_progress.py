"""
tests/test_roadmap_progress.py
--------------------------------
Tests for Phase 10 (Personalized Career Roadmap upgrade): the
Focus/Deliverables restructuring of prediction.py::get_career_roadmap(),
its stable content-derived task IDs, and the new roadmap-completion
persistence (database.py + app.py's /roadmap/toggle route).
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault('SECRET_KEY', 'test-only-secret-key-not-for-production')

import pytest

import prediction as pred
import database as db

TASK_ID_RE = re.compile(r'^[0-9a-f]{12}$')


# ------------------------------------------------------------------
# get_career_roadmap() structure
# ------------------------------------------------------------------
def test_roadmap_has_focus_and_deliverables_per_stage():
    roadmap = pred.get_career_roadmap('Data Science', [])
    for stage in ('30_day_plan', '60_day_plan', '90_day_plan'):
        assert stage in roadmap
        assert 'focus' in roadmap[stage]
        assert 'deliverables' in roadmap[stage]
        assert isinstance(roadmap[stage]['focus'], list)
        assert isinstance(roadmap[stage]['deliverables'], list)
        for item in roadmap[stage]['deliverables']:
            assert 'id' in item and 'text' in item
            assert TASK_ID_RE.match(item['id'])


def test_roadmap_task_ids_are_deterministic_for_identical_input():
    r1 = pred.get_career_roadmap('Data Science', ['Aptitude'])
    r2 = pred.get_career_roadmap('Data Science', ['Aptitude'])
    ids1 = [d['id'] for stage in r1.values() for d in stage['deliverables']]
    ids2 = [d['id'] for stage in r2.values() for d in stage['deliverables']]
    assert ids1 == ids2


def test_roadmap_task_ids_change_when_weak_areas_change():
    """
    Content-derived IDs are the whole point: when the actual roadmap
    content changes (different weak areas), old progress correctly does
    NOT carry over to different tasks.
    """
    r1 = pred.get_career_roadmap('Data Science', ['Aptitude'])
    r2 = pred.get_career_roadmap('Data Science', [])
    ids1 = {d['id'] for d in r1['30_day_plan']['deliverables']}
    ids2 = {d['id'] for d in r2['30_day_plan']['deliverables']}
    assert ids1 != ids2


def test_roadmap_task_ids_differ_across_domains():
    r1 = pred.get_career_roadmap('Data Science', [])
    r2 = pred.get_career_roadmap('Web Development', [])
    ids1 = {d['id'] for d in r1['30_day_plan']['deliverables']}
    ids2 = {d['id'] for d in r2['30_day_plan']['deliverables']}
    assert ids1.isdisjoint(ids2)


# ------------------------------------------------------------------
# All 4 weak_areas values actually affect the roadmap (FIX: Resume
# Quality was previously silently ignored)
# ------------------------------------------------------------------
def test_all_four_weak_areas_individually_affect_roadmap_content():
    baseline = pred.get_career_roadmap('Data Science', [])
    for weak_area in ('Aptitude', 'Technical Skills', 'Communication', 'Resume Quality'):
        varied = pred.get_career_roadmap('Data Science', [weak_area])
        baseline_ids = {d['id'] for stage in baseline.values() for d in stage['deliverables']}
        varied_ids = {d['id'] for stage in varied.values() for d in stage['deliverables']}
        assert baseline_ids != varied_ids, f"'{weak_area}' had no effect on the roadmap -- regression of the Phase 10 fix"


def test_resume_quality_gap_produces_resume_focused_30_day_deliverable():
    roadmap = pred.get_career_roadmap('Data Science', ['Resume Quality'])
    texts = [d['text'] for d in roadmap['30_day_plan']['deliverables']]
    assert any('resume' in t.lower() for t in texts)


def test_no_weak_areas_still_produces_a_complete_roadmap():
    roadmap = pred.get_career_roadmap('Data Science', [])
    for stage in roadmap.values():
        assert len(stage['deliverables']) >= 3
        assert len(stage['focus']) >= 1


def test_unknown_domain_falls_back_to_default_without_crashing():
    roadmap = pred.get_career_roadmap('Nonexistent Domain', ['Aptitude'])
    assert '30_day_plan' in roadmap
    assert roadmap['30_day_plan']['deliverables']


# ------------------------------------------------------------------
# database.py progress functions (mocked execute_query -- no live DB)
# ------------------------------------------------------------------
def test_get_roadmap_progress_set_returns_task_id_set(monkeypatch):
    monkeypatch.setattr(db, 'execute_query', lambda *a, **kw: [{'task_id': 'aaa111222333'}, {'task_id': 'bbb444555666'}])
    result = db.get_roadmap_progress_set(1)
    assert result == {'aaa111222333', 'bbb444555666'}


def test_toggle_roadmap_task_marks_complete_when_not_already(monkeypatch):
    calls = []
    def fake_execute_query(query, params=None, fetch=False, fetchone=False):
        calls.append((query, params))
        if 'SELECT id FROM roadmap_progress' in query:
            return None  # not currently complete
        return 1
    monkeypatch.setattr(db, 'execute_query', fake_execute_query)

    result = db.toggle_roadmap_task(1, 'aaa111222333')
    assert result is True
    assert any('INSERT INTO roadmap_progress' in q for q, p in calls)


def test_toggle_roadmap_task_unmarks_when_already_complete(monkeypatch):
    calls = []
    def fake_execute_query(query, params=None, fetch=False, fetchone=False):
        calls.append((query, params))
        if 'SELECT id FROM roadmap_progress' in query:
            return {'id': 1}  # currently complete
        return 1
    monkeypatch.setattr(db, 'execute_query', fake_execute_query)

    result = db.toggle_roadmap_task(1, 'aaa111222333')
    assert result is False
    assert any('DELETE FROM roadmap_progress' in q for q, p in calls)


# ------------------------------------------------------------------
# /roadmap/toggle route
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


def _get_csrf_token(client, url='/login'):
    html = client.get(url).get_data(as_text=True)
    return html.split('name="csrf-token" content="')[1].split('"')[0]


def test_roadmap_toggle_requires_login_even_with_valid_csrf(client):
    """Isolates auth enforcement from CSRF enforcement: a valid token with no session must still redirect to login."""
    token = _get_csrf_token(client)
    resp = client.post('/roadmap/toggle', json={'task_id': 'aaa111222333'},
                       headers={'X-CSRFToken': token}, follow_redirects=False)
    assert resp.status_code == 302
    assert '/login' in resp.headers['Location']


def test_roadmap_toggle_rejects_missing_csrf_token(client):
    with client.session_transaction() as s:
        s['user_id'] = 1
    resp = client.post('/roadmap/toggle', json={'task_id': 'aaa111222333'}, follow_redirects=False)
    assert resp.status_code == 400  # global CSRF handler, same as /generate_quiz's established behavior


def test_roadmap_toggle_rejects_malformed_task_id(client):
    token = _get_csrf_token(client)
    with client.session_transaction() as s:
        s['user_id'] = 1
    resp = client.post('/roadmap/toggle', json={'task_id': 'not a real hash!!'}, headers={'X-CSRFToken': token})
    assert resp.status_code == 400
    assert resp.get_json()['error']


def test_roadmap_toggle_round_trip_checks_then_unchecks(client, monkeypatch):
    token = _get_csrf_token(client)
    with client.session_transaction() as s:
        s['user_id'] = 1

    store = set()
    def fake_toggle(user_id, task_id):
        if task_id in store:
            store.discard(task_id)
            return False
        store.add(task_id)
        return True
    monkeypatch.setattr(db, 'toggle_roadmap_task', fake_toggle)

    task_id = 'a1b2c3d4e5f6'
    r1 = client.post('/roadmap/toggle', json={'task_id': task_id}, headers={'X-CSRFToken': token})
    assert r1.status_code == 200
    assert r1.get_json() == {'completed': True}

    r2 = client.post('/roadmap/toggle', json={'task_id': task_id}, headers={'X-CSRFToken': token})
    assert r2.status_code == 200
    assert r2.get_json() == {'completed': False}


def test_roadmap_toggle_db_error_returns_generic_message_not_raw_exception(client, monkeypatch):
    token = _get_csrf_token(client)
    with client.session_transaction() as s:
        s['user_id'] = 1

    def _boom(user_id, task_id):
        raise RuntimeError("some internal DB detail that must never reach the client")
    monkeypatch.setattr(db, 'toggle_roadmap_task', _boom)

    resp = client.post('/roadmap/toggle', json={'task_id': 'a1b2c3d4e5f6'}, headers={'X-CSRFToken': token})
    assert resp.status_code == 500
    body = resp.get_json()
    assert 'RuntimeError' not in str(body)
    assert 'internal DB detail' not in str(body)


# ------------------------------------------------------------------
# predict_placement() regression: roadmap key still present, new shape
# ------------------------------------------------------------------
def test_predict_placement_roadmap_key_has_new_shape():
    sample = dict(tenth=85, twelfth=90, cgpa=8.5, backlogs=0, aptitude=75, technical=80,
                  communication=60, resume_score=82, domain='Data Science')
    result = pred.predict_placement(sample)
    assert 'roadmap' in result
    assert 'focus' in result['roadmap']['30_day_plan']
    assert 'deliverables' in result['roadmap']['30_day_plan']
