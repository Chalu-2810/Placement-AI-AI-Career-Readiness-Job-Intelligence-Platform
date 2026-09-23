# PlacementAI — Job Market Intelligence Methodology

This document covers Phase 5: real external job-market data, kept
structurally separate from the synthetic-trained placement model
(`docs/ml-methodology.md`) and from the security posture
(`docs/security.md`).

## The three concepts this feature keeps separate

1. **Placement Readiness** — `prediction.py`, a model trained on a
   100% synthetic, formula-generated dataset (see `docs/ml-methodology.md`).
   A model estimate, never presented as a real-world probability.
2. **Job Market Intelligence** — *this* module (`job_market.py`), real
   external job postings, snapshot-dated, never described as live.
3. **Skill Match** — transparent set-overlap between a student's own
   extracted skills and a snapshot's skill-frequency table. No ML.

These are never combined into one number. The results page
(`result.html`) links to the Job Market Intelligence page
(`opportunities.html`) as a related-but-separate view — it does not fold
market data into the readiness score.

---

## 1. Data source: RemoteOK

**Why RemoteOK for this phase:** no API key, no signup, no rate-limit
paperwork, and no restrictive terms-of-service found for this kind of
non-commercial/portfolio use. That certainty was the deciding factor —
see §8 below on why Adzuna, despite better India-specific and
salary-histogram endpoints, was deliberately *not* implemented this
phase.

**Endpoint:** `https://remoteok.com/api` — a single GET request returns
the current public job feed as a JSON array. The first element of that
array is a non-job "legal notice" object, not a posting; `fetch_raw_postings()`
filters to objects that actually have `id` and `position` fields rather
than assuming a fixed array index.

**Known limitation, stated plainly:** RemoteOK is a *remote-jobs* board.
This has real consequences the UI does not paper over:
- Locations are almost universally "Remote" (sometimes "Remote (US)"
  etc.) — there is no India-city-level or regional data in this source.
  The Job Market Intelligence page shows exactly what the source
  provides and nothing more; if a snapshot's `locations` list is empty
  or just says "Remote," that's the honest answer, not a bug.
- Salary is optional per-posting and often absent. A snapshot only
  reports a salary range if **at least 5 postings** in that domain's
  bucket disclosed real numbers — below that, the page shows "not
  enough data" rather than a number computed from 1–2 postings.
- Because it's remote/global-tilted, RemoteOK postings skew toward
  companies hiring internationally, not an India-specific hiring
  picture. The UI never claims "Indian hiring trends."

## 2. Refresh mechanism

```
RemoteOK (external, real postings)
    -> fetch_raw_postings()         [one HTTP call]
    -> classify_domain() per posting [local, deterministic]
    -> build_snapshot() per domain   [skill extraction + aggregation]
    -> database.save_job_market_snapshot()  [persisted, timestamped]
```

Run via `python scripts/refresh_job_market.py` — **not a Flask route,
not user-triggerable.** There is no way for a visitor, authenticated or
not, to cause a RemoteOK fetch on demand. The running app
(`app.py`'s `/opportunities` route) only ever calls
`database.get_latest_job_market_snapshot(domain)`, which reads the most
recent already-saved row — it never calls RemoteOK itself. This is what
keeps the app's external API usage small, predictable, and independent
of traffic volume, and is why the "Market snapshot as of `<date>`" label
is meaningful rather than decorative: it's genuinely the last time the
external script ran, not "now."

**Suggested cadence:** weekly, via cron or a manually-triggered run.
RemoteOK's feed doesn't change meaningfully hour-to-hour, and a
predictable, infrequent refresh keeps this comfortably within any
reasonable request budget.

## 3. Snapshot architecture — derived aggregates only

`job_market_snapshots` (additive table, `schema.sql`) stores one row per
domain per refresh: `domain, source, fetched_at, posting_count,
raw_aggregates (JSON)`. **Raw job-posting description text is not
persisted long-term** — only the derived numbers (`skill_frequencies`,
`locations`, `sample_roles`, `salary` summary) computed at refresh time.
This is both the simpler design and the more conservative one: it avoids
any question of whether re-publishing original posting text long-term
is appropriate, regardless of source.

## 4. Skill extraction — reused, not reinvented

`job_market.py` imports `detect_skills()` directly from
`utils/resume.py` — the **same** skill taxonomy already used for resume
ATS scoring. A posting's title + tags + description text is run through
this exact function during snapshot building. This means a student's
resume-extracted skills and a job posting's extracted skills are
guaranteed to use the same vocabulary — "sql" means the same thing on
both sides — rather than maintaining two taxonomies that could silently
drift apart (the same "single source of truth" principle established in
Phase 1 for feature engineering).

## 5. Domain classification

Each posting is classified into one of the app's existing 8 domains (the
same `JOB_ROLES`/`DOMAIN_DATA` vocabulary already used everywhere else in
the app — not a new taxonomy) by counting keyword hits from that
domain's role titles + skill list against the posting's combined text.
A posting is assigned to whichever domain has the *most* keyword hits;
if no domain has any hit at all, the posting is excluded from every
snapshot rather than force-assigned to the wrong one.

## 6. Skill-demand analytics

Within a domain's bucket of classified postings, `Counter`-based
frequency counting produces `skill_frequencies`: how many postings in
*this snapshot* mentioned each skill. A `demand_label` (High / Medium /
Low) is assigned **relative to the most-mentioned skill in that same
snapshot** — `count / max_count`: ≥66% → High, ≥33% → Medium, else Low.
This is stated explicitly in the UI ("Demand labels are relative to the
most-mentioned skill in this snapshot... not an absolute market-wide
claim") because it is a relative, snapshot-local ranking, not a claim
about the entire real-world job market.

## 7. Skill matching calculation

`compute_skill_match(student_skills, snapshot)`:

1. Take the snapshot's top **15** most-mentioned skills (`TOP_SKILLS_CONSIDERED`
   — a documented, fixed cutoff, not tuned per student).
2. For each, check whether it's in the student's own skill set (from
   `session['ats_result']['skills_found']`, populated by the existing
   resume-upload pipeline — the *same* `detect_skills()` output).
3. `match_percentage = matched_count / considered_count * 100` — pure
   set overlap, always between 0 and 100 by construction.
4. If the student has **no** extracted skills yet (no resume uploaded),
   `match_percentage` is `None`, not `0` — an absent profile is not the
   same claim as "you have none of these skills," and the UI shows an
   honest "upload your resume to see this" prompt instead of a
   misleadingly negative number.

No embeddings, no vector search, no LLM call, no fuzzy/semantic
matching — exact, case-insensitive string membership in the same fixed
taxonomy used everywhere else in the app.

## 8. Per-role skill matching — "Am I Qualified?" (Phase 8)

Phase 5's skill matching is domain-level: one skill-frequency table for
the whole domain bucket (e.g. every "Data Science" posting combined).
Phase 8 adds a **role-level** breakdown so a claim like "82% match for
Data Analyst" is actually about Data Analyst postings specifically, not
the whole domain relabeled.

`build_snapshot()` gained one additive key, `role_skill_breakdown`: for
each distinct posting title (role) seen in that domain's postings, a
skill-frequency list computed *only from that role's own postings*, plus
the real posting count for that exact title. No existing key changed —
`skill_frequencies`, `sample_roles`, `locations`, etc. are all untouched
Phase 5 behavior; `role_skill_breakdown` is a new, separate field.

`/qualified` (new route) lets a student pick a role from that list and
see a live match against its skill breakdown, reusing
`job_market.compute_skill_match()` — the exact same tested function
Phase 5 built, just scoped to `{'skill_frequencies': role_entry['skills']}`
instead of the whole snapshot. No new matching logic was written.

**The "matches N analyzed job postings" claim is always
`role_entry['posting_count']`** — the real count of postings with that
exact title in the current snapshot, never estimated or rounded up.
Because RemoteOK's free feed yields small per-role samples (often 1–3
postings per title), that count is always shown directly next to the
role name in the picker and next to the match result — small samples are
disclosed, not hidden, consistent with this project's approach elsewhere
(e.g. §7's `n=` disclosures).

**Toggling skills recomputes the match % entirely client-side** — the
role's skill-mention counts are embedded in the page once as JSON;
checking/unchecking a box just recounts them in JavaScript. Switching
*role* is a normal page reload via `?role=`, so a role change always
reflects real server-side data for that role, never a client-side guess.
No new AJAX endpoint, no new backend state.

## 9. Resume-to-role keyword comparison (Phase 9)

The Resume Intelligence page (`/resume`) shows a "Resume → `<Role>`"
comparison once a resume has actually been analyzed: keyword match %,
a Strong list (skills the resume has that the role's postings mention),
and a Missing list (top skills the role's postings mention that the
resume doesn't have).

This adds **zero new matching logic**. It calls the same
`_load_job_market_snapshot()` / `find_role_entry()` /
`compute_skill_match()` functions Phases 5 and 8 already built and
tested, using the resume's real `skills_found` list (from the existing
ATS pipeline, `utils/resume.py`) as the "student skills" input instead
of `/qualified`'s toggle checkboxes. The only difference from
`/qualified` is *which* skill list feeds the comparison — resume-
extracted (read-only) here, vs. interactively toggled there.

The comparison section only renders when all three are true: a resume
has been analyzed this session, the student has a saved academic domain,
and a job market snapshot with `role_skill_breakdown` exists for that
domain. Any missing piece shows an honest explanatory message instead of
a blank or fabricated section — never silently omitted without
explanation, and never a guessed number.

## 10. Job Role Recommendation Engine — ranked matches (Phase 11)

`/opportunities`'s "Best career matches" panel shows every role in the
current snapshot's `role_skill_breakdown`, ranked by match % — the
"Job Role Recommendation Engine" the original project spec calls for,
explicitly as a **ranked, explainable list, not an AI-generated
recommendation.**

`job_market.rank_roles_by_match()` adds zero new matching logic: it
calls `compute_skill_match()` once per role (the same function Phases 5,
8, and 9 already use) and sorts the results. When the student has real
extracted skills, ranking is by match % descending. When they don't
(`match_percentage` is `None` for every role in that case — see §7),
ranking falls back to posting count descending, since ordering by a
"would-be match" that can't actually be computed would be misleading; a
direct "upload your resume to rank by match" prompt is shown instead.

Each ranked entry shows up to 4 Strong (✓) and 3 Missing (⚠) skills —
the same `ranked`/`priority_skills` fields `compute_skill_match()`
already returns, just truncated for a compact list view rather than the
full breakdown `/qualified` and `/resume` show for one role at a time.

## 11. Skill Gap Engine — Gap × Importance (Phase 12)

`/qualified`'s "Skill gap analysis" table implements the original
project spec's exact formula: **Priority = Gap × Importance**, per
skill, for the currently-selected role.

**Current** is deliberately binary (100 if the student's resume has the
skill, 0 if not) — this app only ever detects skill *presence*
(`utils/resume.py`'s keyword taxonomy), never a graduated proficiency
level. A fake "72/100 Python skill" would be fabrication; a disclosed
binary signal is honest. The UI states this directly, not just this doc.

**Target** is the real percentage of that *role's own* postings (from
`role_skill_breakdown`, not the whole domain) that mention the skill —
grounded entirely in already-real snapshot data, computed nowhere else,
estimated nowhere.

**Gap** = `max(Target − Current, 0)` — a skill the student already meets
or exceeds the target for contributes zero gap, never a negative one.

**Importance** = `Target / 100` — a skill's own real demand percentage
doubles as its importance weight, so skills that are both a large gap
*and* widely demanded rank highest, which is the whole point of the
formula.

**Priority labels** (HIGH/Medium/Low) reuse the exact same relative-to-
max-in-this-list thresholding already established for `demand_label()`
in `build_snapshot()` (§6) — one consistent methodology across the
module, not two different ones for similar-looking labels.

## 12. Adzuna — documented future option, not implemented

Adzuna offers materially better data for this feature: real India
coverage, a `histogram` endpoint (actual salary distributions), a
`geodata` endpoint (regional counts), and `top_companies`. It was
evaluated during Phase 5 planning and **deliberately not implemented**:

> Adzuna's Terms of Service state that non-publisher use by "commercial,
> government or academic organisations... is permitted subject to a 14
> day trial period," and that data "may not be used in its original
> format or in aggregation... to deliver any ongoing work or research...
> without written consent."

For a portfolio project meant to run continuously (not just a 14-day
trial), that requires either (a) written permission from Adzuna, or (b)
scoping the integration explicitly as "captured during a development
trial," refreshed rarely, and disclosed as such. Neither was decided as
part of Phase 5 — this is flagged as a **future upgrade path requiring a
separate decision**, not a technical task. If pursued later, the
architecture in §2–3 above (refresh script → snapshot table → app reads
snapshot only) is designed to accept a second source with minimal
change: `job_market.py` would gain a second fetch function, and
`build_snapshot()`'s aggregation logic is already source-agnostic.

## 13. What's real vs. derived vs. matched vs. estimated — summary table

| Shown on the page | Type | Source |
|---|---|---|
| Posting count | Real (counted) | RemoteOK, this snapshot |
| Skill frequency counts | Derived (aggregated) | Counted from real postings, this snapshot |
| Demand label (High/Medium/Low) | Derived (relative ranking) | Computed from the same snapshot's own max count |
| Sample role titles | Real (extracted) | Distinct titles from real postings |
| Locations | Real (as reported) | Exactly what RemoteOK postings state — often just "Remote" |
| Salary range | Real (aggregated), or explicitly unavailable | Only shown with n≥5 real disclosed values |
| Match percentage | Derived (calculated) | Student's real extracted skills ∩ this snapshot's top skills |
| Role-specific skill list (`/qualified`) | Real (extracted, per role) | Skills detected only in that exact role title's postings |
| "Matches N postings" (`/qualified`, `/resume`) | Real (counted) | Exact posting count for that role title in this snapshot |
| Resume → Role keyword match (`/resume`) | Derived (calculated) | Resume's real extracted skills ∩ that role's skill list |
| Best career matches ranking (`/opportunities`) | Derived (calculated + sorted) | Same per-role match, computed for every role and sorted |
| Skill gap Current (`/qualified`) | Real (binary presence) | Detected in resume or not — never a graduated proficiency guess |
| Skill gap Target (`/qualified`) | Real (counted) | % of that role's own postings mentioning the skill |
| Skill gap Priority (`/qualified`) | Derived (calculated) | Gap × Importance, both computed from the above |
| Placement readiness score | **Model estimate**, not market data | `prediction.py`, synthetic-trained model — never shown on this page |

## 14. Limitations

- RemoteOK's remote-jobs focus means this is not a representative sample
  of the broader (especially on-campus/India-local) job market — stated
  directly in the UI's location section.
- Small per-domain sample sizes (a single RemoteOK feed pull, filtered
  to one of 8 domains) limit statistical confidence — this is why the
  salary section has an explicit `n≥5` floor and locations/roles are
  shown as raw counts, never percentages of some larger implied
  population.
- Keyword-based skill matching inherits every limitation of
  `utils/resume.py`'s taxonomy — synonyms or abbreviations outside that
  fixed list are simply not detected, on either the student or the
  market side. Consistent, not perfect.
- Snapshots go stale between refreshes; the page always shows the
  snapshot's real `fetched_at` timestamp so this is visible, never
  hidden behind an implied "live" claim.
