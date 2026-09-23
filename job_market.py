"""
job_market.py
--------------
Job Market Intelligence pipeline (Phase 5).

Architecture (matches the approved plan):

    RemoteOK (external, real postings)
        -> fetch_raw_postings()          [one HTTP call per refresh run]
        -> classify_domain()             [local, deterministic keyword match]
        -> build_snapshot()              [skill extraction + aggregation]
        -> database.save_job_market_snapshot()   [persisted, timestamped]
        -> app.py reads the LATEST SNAPSHOT ONLY  [never calls RemoteOK per-request]

Three concepts this module deliberately keeps separate (see
docs/job-market-methodology.md):

  1. Placement Readiness  -- prediction.py, synthetic-trained ML model.
  2. Job Market Intelligence -- THIS module, real external postings,
     snapshot-dated, never described as "live."
  3. Skill Match -- transparent set-overlap between a student's own
     extracted skills and this snapshot's skill-frequency table. No ML,
     no embeddings, no semantic matching.

Reuses utils/resume.py's existing skill taxonomy (detect_skills) for
BOTH resume parsing and job-posting skill extraction, so there is one
skill vocabulary in the whole app, not two incompatible ones.

Source: RemoteOK (https://remoteok.com/api), chosen for Phase 5 because
it requires no API key/signup and has no restrictive terms-of-service
found for this kind of use (see docs/job-market-methodology.md for the
full comparison against Adzuna, which is documented as a FUTURE option
pending a separate ToS decision -- not implemented here).

RemoteOK limitation, stated plainly: it is a remote-jobs board. This
pipeline does NOT invent city-level India statistics, salary
distributions, or location rankings the source doesn't actually provide
-- see build_snapshot()'s location/salary handling below.
"""

import re
import logging
from collections import Counter
from datetime import datetime, timezone

import requests

from utils.resume import detect_skills
from prediction import DOMAIN_DATA, JOB_ROLES

logger = logging.getLogger('placementai.job_market')

REMOTEOK_API_URL = 'https://remoteok.com/api'
REQUEST_TIMEOUT_SECONDS = 15
# RemoteOK's own docs ask API consumers to identify themselves with a
# real User-Agent rather than a generic script UA.
REQUEST_HEADERS = {'User-Agent': 'PlacementAI-JobMarketRefresh/1.0 (student portfolio project)'}

SOURCE_NAME = 'RemoteOK'

# How many of the most-frequently-mentioned skills in a snapshot count as
# "the identified market skills" for the match-percentage calculation.
# Deliberately small and documented (not hidden) -- see
# docs/job-market-methodology.md for why 15 was chosen.
TOP_SKILLS_CONSIDERED = 15


# ============================================================
# DOMAIN CLASSIFICATION KEYWORDS
# ============================================================
# Built from the SAME domain vocabulary already used everywhere else in
# the app (prediction.py's JOB_ROLES / DOMAIN_DATA) -- not a second,
# diverging taxonomy. A posting is classified into a domain if its
# combined title+tags+description text contains at least one of these
# keywords; ties/multiple matches go to whichever domain has the most
# keyword hits.
def _build_domain_keywords():
    keywords = {}
    for domain, roles in JOB_ROLES.items():
        kws = set()
        for role in roles:
            kws.add(role.lower())
        for skill in DOMAIN_DATA.get(domain, {}).get('skills', []):
            # Strip parenthetical/slash noise like "Pandas/NumPy" -> also add split terms
            cleaned = re.sub(r'[()/]', ' ', skill).lower()
            kws.add(cleaned.strip())
            for part in cleaned.split():
                if len(part) > 2:
                    kws.add(part.strip())
        keywords[domain] = kws
    return keywords


DOMAIN_KEYWORDS = _build_domain_keywords()


# ============================================================
# FETCH (one call per refresh run -- never called per user request)
# ============================================================
def fetch_raw_postings():
    """
    Fetch the current RemoteOK job feed. Returns a list of posting dicts.

    Raises requests.RequestException on network failure or a non-2xx
    response -- callers (the refresh script) are responsible for
    catching this and logging/aborting gracefully. This function never
    fabricates a result on failure.
    """
    resp = requests.get(REMOTEOK_API_URL, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT_SECONDS)
    resp.raise_for_status()
    data = resp.json()

    if not isinstance(data, list):
        raise ValueError(f'Unexpected RemoteOK response shape: expected a list, got {type(data).__name__}')

    # RemoteOK's feed starts with a non-job "legal notice" object (has no
    # 'id'/'position' field). Filter to objects that actually look like
    # job postings rather than assuming a fixed index.
    postings = [p for p in data if isinstance(p, dict) and p.get('id') and p.get('position')]

    # Defensive de-duplication by posting id -- a repeated entry (source
    # glitch, pagination overlap, etc.) must never double-count a single
    # real posting's skills/salary/location in the aggregates below.
    seen_ids = set()
    deduped = []
    for p in postings:
        pid = p['id']
        if pid not in seen_ids:
            seen_ids.add(pid)
            deduped.append(p)

    return deduped


# ============================================================
# CLASSIFY postings into the app's 8 existing domains
# ============================================================
def classify_domain(posting: dict):
    """
    Returns the best-matching domain name for a posting, or None if no
    domain keyword appears at all (the posting is simply excluded from
    every snapshot -- never force-assigned to a wrong domain).
    """
    text = ' '.join([
        str(posting.get('position', '')),
        ' '.join(posting.get('tags', []) or []),
        str(posting.get('description', ''))[:2000],  # cap: classification doesn't need the full text
    ]).lower()

    best_domain, best_hits = None, 0
    for domain, keywords in DOMAIN_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw and kw in text)
        if hits > best_hits:
            best_domain, best_hits = domain, hits

    return best_domain if best_hits > 0 else None


# ============================================================
# BUILD SNAPSHOT for one domain from its classified postings
# ============================================================
def build_snapshot(domain: str, postings: list):
    """
    Deterministic aggregation only -- every number here traces directly
    to the `postings` list passed in. No estimation, no interpolation.
    """
    posting_count = len(postings)

    skill_counter = Counter()
    location_counter = Counter()
    role_titles = Counter()
    salaries = []
    role_texts = {}  # role title -> list of combined posting texts (Phase 8: per-role skill breakdown)

    for p in postings:
        text = ' '.join([
            str(p.get('position', '')),
            ' '.join(p.get('tags', []) or []),
            str(p.get('description', '')),
        ])
        for skill in detect_skills(text).keys():
            skill_counter[skill] += 1

        loc = str(p.get('location') or '').strip()
        if loc:
            location_counter[loc] += 1

        pos = str(p.get('position') or '').strip()
        if pos:
            role_titles[pos] += 1
            role_texts.setdefault(pos, []).append(text)

        smin, smax = p.get('salary_min'), p.get('salary_max')
        try:
            smin, smax = float(smin), float(smax)
            if smin > 0 and smax > 0:
                salaries.append((smin + smax) / 2)
        except (TypeError, ValueError):
            pass  # salary not provided for this posting -- skip, don't fabricate

    max_freq = max(skill_counter.values()) if skill_counter else 0

    def _demand_label(count):
        # Deterministic, relative to the strongest skill IN THIS SNAPSHOT
        # -- not an absolute claim about the whole job market. Documented
        # in docs/job-market-methodology.md.
        if max_freq == 0:
            return 'Unknown'
        ratio = count / max_freq
        if ratio >= 0.66:
            return 'High demand'
        elif ratio >= 0.33:
            return 'Medium demand'
        return 'Low demand'

    skill_frequencies = [
        {'skill': skill, 'count': count, 'demand_label': _demand_label(count)}
        for skill, count in skill_counter.most_common()
    ]

    # Salary: only included if the source actually provided usable numbers
    # for a meaningful slice of postings. Otherwise explicitly marked
    # unavailable -- never estimated.
    salary_summary = None
    if len(salaries) >= 5:  # small-sample guard; documented threshold
        salary_summary = {
            'available'    : True,
            'sample_size'  : len(salaries),
            'min'          : round(min(salaries), 2),
            'max'          : round(max(salaries), 2),
            'median'       : round(sorted(salaries)[len(salaries) // 2], 2),
        }
    else:
        salary_summary = {'available': False, 'sample_size': len(salaries)}

    # Phase 8 ("Am I Qualified?"): skill frequency computed PER ROLE TITLE,
    # not just per domain -- so a role-specific match % is honestly about
    # that role's own postings, not the whole domain relabeled. Additive:
    # every other key above is unchanged from Phase 5.
    role_skill_breakdown = []
    for role, count in role_titles.items():
        role_skill_counter = Counter()
        for text in role_texts.get(role, []):
            for skill in detect_skills(text).keys():
                role_skill_counter[skill] += 1
        role_skill_breakdown.append({
            'role'         : role,
            'posting_count': count,
            'skills'       : [{'skill': s, 'count': c} for s, c in role_skill_counter.most_common()],
        })
    role_skill_breakdown.sort(key=lambda r: r['posting_count'], reverse=True)

    return {
        'domain'           : domain,
        'source'           : SOURCE_NAME,
        'fetched_at'       : datetime.now(timezone.utc).isoformat(),
        'posting_count'    : posting_count,
        'skill_frequencies': skill_frequencies,
        'locations'        : [{'location': loc, 'count': c} for loc, c in location_counter.most_common(10)],
        'sample_roles'     : [role for role, _ in role_titles.most_common(6)],
        'role_skill_breakdown': role_skill_breakdown,
        'salary'           : salary_summary,
        'note'             : (
            'RemoteOK is a remote-jobs board. Location data reflects only '
            'what RemoteOK postings report (often "Remote" with no further '
            'detail) -- this is not a claim about India-wide or regional '
            'hiring trends.'
        ),
    }


def refresh_all_domains():
    """
    Fetches RemoteOK ONCE, classifies every posting into one of the app's
    8 domains, and builds one snapshot dict per domain (including domains
    with zero matched postings -- an honest empty snapshot, not a skipped
    one). Returns {domain_name: snapshot_dict}.
    """
    raw_postings = fetch_raw_postings()

    buckets = {domain: [] for domain in JOB_ROLES.keys()}
    unclassified = 0
    for posting in raw_postings:
        domain = classify_domain(posting)
        if domain:
            buckets[domain].append(posting)
        else:
            unclassified += 1

    logger.info('RemoteOK refresh: %d postings fetched, %d unclassified, buckets=%s',
                len(raw_postings), unclassified, {d: len(p) for d, p in buckets.items()})

    return {domain: build_snapshot(domain, postings) for domain, postings in buckets.items()}


# ============================================================
# ROLE RECOMMENDATION ENGINE (Phase 11)
# ============================================================
def rank_roles_by_match(student_skills, snapshot: dict):
    """
    Ranks every role in this snapshot's role_skill_breakdown by how well
    it matches the student's real skills -- the "Job Role Recommendation
    Engine": a ranked, explainable list, not an AI-generated
    recommendation. Reuses compute_skill_match() once per role; adds no
    new matching logic, only composition/sorting.

    Sort order: by match_percentage descending when the student has
    real extracted skills to compare; falls back to posting_count
    descending when they don't (match_percentage is None for everyone
    in that case, so ranking by "would-be-best-match" is meaningless --
    showing the best-represented roles first is the more honest default).

    Returns a list of:
        {'role': str, 'posting_count': int, 'match': <compute_skill_match() result dict>}
    Empty list if the snapshot has no role_skill_breakdown (e.g. a
    pre-Phase-8 snapshot, or zero classified postings for this domain).
    """
    roles = snapshot.get('role_skill_breakdown') or []
    ranked = []
    for entry in roles:
        match = compute_skill_match(student_skills, {'skill_frequencies': entry['skills']})
        ranked.append({'role': entry['role'], 'posting_count': entry['posting_count'], 'match': match})

    has_student_skills = bool({s.strip().lower() for s in (student_skills or []) if s and s.strip()})
    if has_student_skills:
        ranked.sort(key=lambda r: (r['match']['match_percentage'] or 0), reverse=True)
    else:
        ranked.sort(key=lambda r: r['posting_count'], reverse=True)

    return ranked


def find_role_entry(snapshot: dict, role_name: str):
    """
    Returns the role_skill_breakdown entry for an exact role name, or
    None if that role isn't in this snapshot. Case-sensitive exact match
    on purpose -- role names come from the same snapshot's own
    role_skill_breakdown list (rendered as <option> values), so there is
    no free-text input to normalize/fuzzy-match against.
    """
    for entry in (snapshot.get('role_skill_breakdown') or []):
        if entry['role'] == role_name:
            return entry
    return None


# ============================================================
# SKILL MATCH -- student's own skills vs. this snapshot
# ============================================================
def compute_skill_match(student_skills, snapshot: dict):
    """
    student_skills: iterable of lowercase skill-name strings (from
                    utils.resume.detect_skills(), e.g. session['ats_result']
                    ['skills_found'] after a resume upload).
    snapshot: a snapshot dict as returned by build_snapshot() /
              stored in job_market_snapshots.raw_aggregates.

    Returns:
        {
          'has_student_skills': bool,
          'ranked'            : [ {skill, count, demand_label, has_skill}, ... ]  # top TOP_SKILLS_CONSIDERED
          'match_percentage'  : float | None,   # None if no student skills to compare
          'matched_count'     : int,
          'considered_count'  : int,
          'priority_skills'   : [ {skill, count, demand_label}, ... ]  # top missing, by demand
        }

    Pure set overlap -- no ML, no embeddings, no fuzzy/semantic matching.
    """
    student_set = {s.strip().lower() for s in (student_skills or []) if s and s.strip()}
    top = (snapshot.get('skill_frequencies') or [])[:TOP_SKILLS_CONSIDERED]

    ranked = []
    matched = []
    missing = []
    for entry in top:
        has_it = entry['skill'] in student_set
        row = {**entry, 'has_skill': has_it}
        ranked.append(row)
        (matched if has_it else missing).append(row)

    considered_count = len(top)
    match_percentage = (
        round(len(matched) / considered_count * 100, 1) if (student_set and considered_count) else None
    )

    return {
        'has_student_skills': bool(student_set),
        'ranked'            : ranked,
        'match_percentage'  : match_percentage,
        'matched_count'     : len(matched),
        'considered_count'  : considered_count,
        'priority_skills'   : missing[:5],
    }


# ============================================================
# SKILL GAP ENGINE -- Gap x Importance (Phase 12)
# ============================================================
def compute_priority_ranking(student_skills, role_entry: dict):
    """
    Per-skill Skill Gap Analysis for one role, ranked by
    Priority = Gap x Importance -- the original spec's exact formula.

    Honesty note on "Current": this app only ever detects skill
    PRESENCE (utils/resume.py's keyword taxonomy), never a graduated
    proficiency level. Inventing a fake 0-100 proficiency score per
    skill would be fabrication, so Current is deliberately binary: 100
    if the skill was detected in the student's resume/profile, else 0.
    This is disclosed in the UI, not hidden behind a falsely-precise number.

    Target: the REAL demand percentage for that skill within this one
    role's own postings -- i.e. what fraction of this role's analyzed
    postings actually mention it. Grounded entirely in
    role_entry['skills'] (already-real data from build_snapshot()), not
    estimated.

    Gap = max(Target - Current, 0) -- a skill the student already has
    at or above the target demand contributes no gap (never negative).

    Importance = Target / 100 -- a skill's own real demand percentage
    doubles as its importance weight, so Priority = Gap x Importance
    naturally ranks "big gap AND widely demanded" skills highest,
    exactly the spec's intent.

    Priority label (HIGH/Medium/Low) uses the SAME relative-to-max-in-
    this-list methodology already established for demand_label() in
    build_snapshot() (>=66% of the top priority score = HIGH, >=33% =
    Medium, else Low) -- one consistent thresholding approach across the
    whole job_market module, not two diverging ones.

    Returns a list of:
        {skill, current, target, gap, priority, priority_label}
    sorted by priority descending. Empty list if role_entry has no skills.
    """
    student_set = {s.strip().lower() for s in (student_skills or []) if s and s.strip()}
    skills = (role_entry or {}).get('skills') or []
    posting_count = (role_entry or {}).get('posting_count') or 0

    rows = []
    for entry in skills:
        target = round((entry['count'] / posting_count) * 100, 1) if posting_count else 0.0
        current = 100.0 if entry['skill'] in student_set else 0.0
        gap = round(max(target - current, 0), 1)
        importance = target / 100
        priority = round(gap * importance, 1)
        rows.append({
            'skill'   : entry['skill'],
            'current' : current,
            'target'  : target,
            'gap'     : gap,
            'priority': priority,
        })

    max_priority = max((r['priority'] for r in rows), default=0)

    def _priority_label(p):
        if max_priority == 0:
            return 'Low'
        ratio = p / max_priority
        if ratio >= 0.66:
            return 'HIGH'
        elif ratio >= 0.33:
            return 'Medium'
        return 'Low'

    for r in rows:
        r['priority_label'] = _priority_label(r['priority'])

    rows.sort(key=lambda r: r['priority'], reverse=True)
    return rows
