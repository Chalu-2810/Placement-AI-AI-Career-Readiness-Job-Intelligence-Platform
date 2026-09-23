# PlacementAI — Security Hardening & Repository Hygiene

This document covers the Phase 2 security audit: what was found, what was
fixed, what was deliberately left alone and why, and what remains as a
known residual risk. Companion to `docs/ml-methodology.md` (Phase 1).

---

## 1. Secrets audit

**Method:** the uploaded project included an accidentally-bundled `.git`
folder. It was inspected directly (object hashes only — no secret values
were ever printed) rather than assumed clean.

**Findings:**
- The `.git` history contained **zero commits** (`git rev-list --all
  --count` → 0). Only `.env.example` was ever staged.
- `git hash-object .env` was computed and compared against the blob hash
  already in the git index — they matched **because `.env` and
  `.env.example` are byte-identical**. In other words, the "real" `.env`
  shipped in the upload is not actually a real credential file; it's a
  copy of the placeholder template. No real secret was ever exposed via
  git, staged or committed.
- **No history rewriting was necessary or performed.** If this repo is
  later connected to a real GitHub remote with actual production
  credentials ever committed to `.env`, treat that as a rotation event
  (rotate `SECRET_KEY`, `DB_PASSWORD`, etc.) and use `git filter-repo`
  (not `git filter-branch`) to scrub history — those exact commands are
  intentionally **not** included here since no destructive rewrite was
  needed for the current state, and running one against a repo that
  doesn't need it is itself risky. Ask for this separately if it's ever
  needed against a real remote.

**Recommendation regardless:** never commit a real `.env`. Rotate
`SECRET_KEY` and database credentials before any real deployment, since
this document (and the repo) may end up in a public place.

---

## 2. `.gitignore`

Rewritten. Notable decisions, not just a checklist:

- `.env` / `.env.*` are ignored, with an explicit `!.env.example`
  negation — a blanket `.env.*` pattern would otherwise also swallow the
  tracked template file, which is not a secret and should stay in the
  repo.
- **`models/*.pkl` is now tracked (previously ignored).** `Procfile` is
  just `web: gunicorn app:app` — there is no build/release step that runs
  `model_training.py` on deploy — and `README.md` explicitly tells users
  "pre-trained models are included, skip training if `models/` already
  has `.pkl` files." Ignoring the `.pkl` files contradicted the app's own
  deployment story: a fresh clone/deploy would have no model artifacts at
  all and `/predict` would crash (model/scaler load as `None`). Since the
  instructions for this phase explicitly say to keep artifacts tracked
  "if genuinely required by the application," and they are, this was
  fixed rather than left as a checklist item.
- `dataset/*.csv` stays ignored — it's deterministically regenerable via
  `python generate_dataset.py` (fixed seed), so no runtime risk in
  leaving it out.
- Added: `.venv/`, `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`,
  `htmlcov/`, `.coverage`.

---

## 3. Repository pollution

The zip you uploaded contained a full Windows `venv/` and a `.git/`
folder at the project root — both stripped from the working copy used for
this project (never committed anywhere; see §1). No pollution remains in
the delivered archive. `.gitignore` now prevents recurrence for `venv/`,
`__pycache__/`, `.pytest_cache/`, IDE folders, and OS files (already
mostly present, extended per §2).

---

## 4. CSRF protection

**Before:** zero CSRF protection. No `flask-wtf`, no tokens anywhere.

**After:** `flask_wtf.CSRFProtect` enabled globally on the `app` object,
which protects every POST/PUT/PATCH/DELETE request by default —
including any route added later, with no per-route opt-in required.

- Every HTML form that POSTs now carries a hidden
  `<input type="hidden" name="csrf_token" value="{{ csrf_token() }}"/>`:
  `login.html`, `registration.html`, `admin_login.html`, `academic.html`,
  `quiz.html`, `resume.html`.
- The one non-form POST — `/generate_quiz`, called via `fetch()` from
  `quiz_setup.html` — sends the token via the `X-CSRFToken` header
  instead, read from a `<meta name="csrf-token">` tag added to
  `base.html`. Flask-WTF checks this header automatically for JSON/AJAX
  requests.
- A `CSRFError` handler (`handle_csrf_error`) returns a clean redirect
  with a user-facing flash message instead of Flask-WTF's default raw
  400 page.
- **Verified by tests:** `test_login_post_without_csrf_token_is_rejected`
  and `test_login_post_with_valid_csrf_token_succeeds` in
  `tests/test_security.py` confirm both the rejection and the
  happy path end-to-end through the real Flask app.
- No route uses `@csrf.exempt` anywhere in the codebase — confirmed by
  grep, not just by design intent.

---

## 5. Authentication security

| Item | Status |
|---|---|
| Password hashing | Werkzeug's `generate_password_hash`/`check_password_hash` (salted, industry standard). Unchanged — already correct. |
| Login failure messages | Already generic ("Invalid email or password", "Invalid admin credentials") — no account-existence leakage via message text. Unchanged. |
| **Timing side-channel** | **Fixed.** `check_password_hash()` was previously only called when a matching row existed, so a login attempt for a nonexistent account returned near-instantly while a wrong-password attempt on a real account took measurably longer (an actual hash comparison) — this timing gap lets an attacker enumerate valid emails/usernames without ever seeing a different error message. `_verify_password()` now always performs a hash comparison, against a fixed dummy hash when no account was found, closing the gap. |
| Session fixation | Flask's default sessions are entirely client-side and cryptographically signed (no server-side session ID to "fixate" the way traditional session-ID-based auth is vulnerable to) — the classic session-fixation attack doesn't directly apply to this architecture. `session.clear()` was still added at the top of both login handlers as defense in depth, so any pre-existing session data (e.g. leftover quiz state) can't carry over into a freshly authenticated session. |
| Logout | Clears the relevant session keys. Unchanged — already correct. |
| Authorization checks | `login_required` checks `'user_id' in session`; `admin_required` checks `'admin_id' in session` — two independent keys, so a student session can never satisfy `admin_required`. Verified by `test_logged_in_student_cannot_reach_admin_dashboard` and `test_admin_session_does_not_grant_student_routes_implicitly`. |

---

## 6. Rate limiting

`Flask-Limiter`, keyed by remote address.

| Scope | Limit | Rationale |
|---|---|---|
| Global default | 200/day, 50/hour per IP | Backstop against generic abuse without affecting normal use. |
| `/login` | 10/minute | Slows credential-stuffing/brute-force without blocking a real user who mistypes a password a few times. |
| `/register` | 10/minute | Slows automated account-creation spam. |
| `/admin/login` | 10/minute | Same rationale as `/login`; admin credentials are higher-value. |

**Ordering matters and was fixed during testing.** `Limiter` is now
initialized *before* `CSRFProtect` in `app.py`. Both hook into Flask via
`before_request`, and Flask stops at the first handler that returns a
response — so whichever registers first effectively runs first. With the
original ordering, a script hammering `/login` without a CSRF token would
have every request absorbed by the CSRF check (400) and would *never*
reach the rate limiter, so the 429 would never fire. Verified by
`test_login_rate_limit_engages_after_threshold`, which sends 11 requests
and asserts a 429 appears.

**Known limitation:** storage defaults to in-memory
(`RATELIMIT_STORAGE_URI=memory://`), which is per-process. Under a
multi-worker Gunicorn deployment, each worker has its own counter, so the
effective limit is `configured_limit × worker_count`. For a real
multi-worker production deployment, point `RATELIMIT_STORAGE_URI` at a
shared backend (e.g. Redis) via the environment variable already wired
for this.

---

## 7. Resume upload security

| Check | Before | After |
|---|---|---|
| Extension check | `.pdf` suffix only | Unchanged (still required) |
| **Content validation** | **None** — a renamed non-PDF went straight to the parser | **Added.** First 5 bytes checked against the real PDF magic header `%PDF-` before any parsing happens. |
| Max file size | None enforced anywhere | `MAX_CONTENT_LENGTH` (10 MB, configurable via `MAX_UPLOAD_MB` env var) rejects oversized requests at the Flask/Werkzeug level with a 413, handled by a custom error handler with a friendly flash message. The route also double-checks the actual stream size as defense in depth. |
| Filename handling | Original filename never touched | `secure_filename()` applied before the name is ever used in a log line. The file is parsed entirely in memory (`fitz.open(stream=...)` / pdfminer on the stream) and never written to disk, so there is no path-traversal surface today — the sanitization is defense in depth for if that ever changes. |
| Malformed PDF | Raised exception text was flashed directly to the user (e.g. raw PyMuPDF error string) | Caught, logged server-side via `logger.exception(...)`, generic message shown to the user. |
| Empty/unreadable PDF | Not distinguished from a normal empty resume | Now explicitly detected (empty extracted text) and given a specific, actionable message (possible scanned/image-based PDF). |

**Test coverage** (`tests/test_security.py`):
- `test_resume_upload_rejects_non_pdf_extension` — `.exe` rejected
- `test_resume_upload_rejects_renamed_non_pdf` — `.pdf`-named file with a
  PE (`MZ`) header rejected by the magic-byte check
- `test_resume_upload_accepts_minimal_valid_pdf_header` — a real `%PDF-`
  header passes the gate (deeper parsing mocked)
- `test_resume_upload_handles_malformed_pdf_gracefully` — a parser
  exception never leaks its text to the response
- `test_resume_upload_rejects_oversized_file` — over `MAX_CONTENT_LENGTH`
  → 413
- `test_resume_upload_dangerous_filename_is_sanitized` — path-traversal
  filename stripped by `secure_filename()`

---

## 8. SQL / database layer

Re-confirmed (already correct in the Phase 1 audit, re-verified here):
every query in `database.py` uses parameterized placeholders
(`mysql-connector-python`'s `%s` style via `cursor.execute(query,
params)`); grepped explicitly for `.format(` and f-string-built SQL —
none found. Connections are opened per-call and always closed in a
`finally` block; `execute_query()` rolls back on any exception before
re-raising. Credentials load from environment variables via
`python-dotenv`, never hardcoded. No ORM was introduced — not warranted,
and out of scope for this pass.

**Changed:** the raw exception is still re-raised to the caller (so
callers can decide what to show the user — see §9), but `database.py` no
longer `print()`s it first; that responsibility now belongs entirely to
the caller's `logger.exception(...)` call.

---

## 9. Error handling & info disclosure

Every `except Exception` in the request path (`app.py`, `database.py`)
was reviewed individually rather than mechanically stripped:

| Location | Before | After |
|---|---|---|
| `register()` | `flash(f'Registration failed: {e}')` — leaked raw exception text (e.g. a MySQL duplicate-key message revealing column names) | Field validation moved earlier (clear messages for missing fields / bad batch year) so the try/except only wraps genuine DB failures; generic message + `logger.exception(...)` |
| `academic()` | `flash(f'Error saving details: {e}')` | Split into a `(KeyError, ValueError)` branch (clear "fill in all fields" message) and a generic `Exception` branch (logged, generic message) |
| `generate_quiz()` | `return jsonify({"error": str(e)}), 500` — raw exception text returned in the JSON API response | Generic JSON error message; real exception logged server-side. Verified by `test_generate_quiz_error_message_has_no_internal_details`. |
| `resume()` | `flash(f"Error parsing resume: {e}")` | See §7 — generic message, logged server-side |
| Global 404/500 | Flask defaults (500 could leak a traceback if `debug=True` was ever accidentally left on) | Custom handlers render `templates/error.html`; 500 handler explicitly calls `logger.exception(...)` so the real cause is never lost, just never shown to the client. Verified by `test_404_page_has_no_stack_trace`. |

None of these `except Exception` blocks were removed outright — each
still catches real, expected failure modes (duplicate email on
registration, a corrupt PDF, a bad domain ID) and now handles them with
an honest, non-leaking message plus a server-side log line for real
debugging.

---

## 10. Debug code removal

All `print()` calls and one `traceback.print_exc()` in the request path
were removed and replaced with Python's `logging` module
(`logging.basicConfig` configured once in `app.py`, level controlled by
`LOG_LEVEL` env var, defaulting to `INFO`). This includes the very noisy
per-question `print("DEBUG:", ...)` inside the `submit_quiz()` scoring
loop, which printed a line for every single quiz question on every
submission. `database.py` no longer prints `"[DB] Backend: MYSQL"` on
import — that's now `logger.info(...)`.

No `pdb`/`breakpoint()` calls were present in the codebase to begin with.

---

## 11. Security headers & Flask configuration

| Setting | Value | Notes |
|---|---|---|
| `SECRET_KEY` | Required from `os.environ['SECRET_KEY']`, no fallback | App fails to start if unset — fail-fast beats a silently-guessable default. |
| `SESSION_COOKIE_HTTPONLY` | `True` | JS cannot read the session cookie (mitigates XSS-driven cookie theft). |
| `SESSION_COOKIE_SAMESITE` | `'Lax'` | Blocks the cookie being sent on cross-site POSTs (the main CSRF delivery vector) while still working for normal top-level navigation. |
| `SESSION_COOKIE_SECURE` | `True` by default, `SESSION_COOKIE_SECURE=0` env var to disable for local HTTP dev | Defaults safe for production; doesn't break `localhost` development once explicitly turned off. |
| `PERMANENT_SESSION_LIFETIME` | 8 hours | Caps how long a session stays valid. |
| `MAX_CONTENT_LENGTH` | 10 MB (`MAX_UPLOAD_MB` env var) | See §7. |
| `debug` | `False` unless `FLASK_DEBUG=1` | Unchanged from the earlier deployment-correctness pass — the Werkzeug debugger (which allows arbitrary code execution from the error page) never runs in production. |

Custom error pages (`error.html`) are used for 404/500/413 instead of
Flask/Werkzeug defaults, so a production error never shows a Python
traceback, file path, or library name to an end user.

---

## 12. Authorization matrix

Every route in `app.py`, enumerated directly from the route table (not
from memory):

| Route | Methods | Access level | Enforced by |
|---|---|---|---|
| `/` | GET | Public | — (redirects to login) |
| `/register` | GET, POST | Public | — (rate-limited 10/min) |
| `/login` | GET, POST | Public | — (rate-limited 10/min) |
| `/logout` | GET | Public (no-op if not logged in) | — |
| `/dashboard` | GET | **Student** | `@login_required` |
| `/academic` | GET, POST | **Student** | `@login_required` |
| `/quiz/setup` | GET | **Student** | `@login_required` |
| `/generate_quiz` | POST | **Student** | `@login_required` + CSRF header |
| `/quiz` | GET | **Student** | `@login_required` |
| `/submit_quiz` | POST | **Student** | `@login_required` + CSRF token |
| `/resume` | GET, POST | **Student** | `@login_required` + CSRF token (POST) |
| `/predict` | GET | **Student** | `@login_required` — see residual risk below |
| `/opportunities` | GET | **Student** | `@login_required` |
| `/qualified` | GET | **Student** | `@login_required` |
| `/roadmap/toggle` | POST | **Student** | `@login_required` + CSRF header (JSON/fetch, same pattern as `/generate_quiz`) |
| `/admin/login` | GET, POST | Public | — (rate-limited 10/min) |
| `/admin/logout` | GET | Public (no-op if not logged in) | — |
| `/admin/dashboard` | GET | **Admin** | `@admin_required` |
| `/api/domains` | GET | Public (read-only reference data — domain names/IDs only, no student PII) | — |

Verified with real requests through the Flask test client (not just
decorator inspection): unauthenticated access to every student route
redirects to `/login`; an authenticated student session cannot reach
`/admin/dashboard`; unauthenticated access to `/admin/dashboard`
redirects to `/admin/login`.

---

## 13. Automated security tests

`tests/test_security.py` — 25 tests, all passing, run against the real
`app` object via Flask's test client with `database` monkeypatched
(no live MySQL required):

- Unauthorized student-route access (4 tests)
- Unauthorized admin-route access (3 tests)
- CSRF rejection / acceptance (2 tests)
- Authentication correctness + no user-enumeration via error text (4 tests)
- Timing-safe credential check, unit-level (3 tests)
- Resume upload validation (6 tests — see §7)
- Rate limiting (1 test)
- Safe error responses / no stack traces or internal details leaked (2 tests)

Combined with the 5 pre-existing ML tests
(`tests/test_feature_consistency.py`), the full suite is **30/30
passing**, and was additionally run twice with `pytest-randomly` to
confirm no ordering dependencies between tests (both runs: 30/30).

---

## 14. Residual risks (honest, not swept under the rug)

- **`/predict` is a GET route that writes to the database.** It's only
  ever reached via `<a href>` links in the app's own templates, never a
  form, so it wasn't in scope for CSRF token protection the way POST
  forms were — but a GET route with a side effect is still vulnerable to
  cross-site request forgery via GET (e.g. an auto-loading `<img>` tag on
  another site while a student is logged in could silently trigger a
  prediction re-run). **Impact is limited**: it only overwrites the
  *victim's own* prediction record; there's no cross-user data exposure
  or privilege escalation. Recommended fix for a future pass: convert
  this to a POST-only route with a small form/button (like the others),
  which would then be covered by the existing CSRF protection
  automatically. Left as a known risk rather than fixed silently in this
  pass, since it means changing 6 template links into forms — a larger
  change than the rest of this security pass.
- **Rate limiting is per-process (in-memory storage).** See §6 — fine for
  a single-worker deployment, needs a shared backend (Redis) for a real
  multi-worker production deployment.
- **No account lockout after repeated failed logins**, only rate
  limiting by IP. An attacker distributing attempts across many IPs
  isn't slowed by the current rate limiter. Account-level lockout/backoff
  is a reasonable future addition if this becomes a real concern.
- **No HSTS or other response security headers** (`Strict-Transport-
  Security`, `X-Content-Type-Options`, `Content-Security-Policy`, etc.)
  were added. These are usually best set at the reverse-proxy/CDN layer
  (e.g. Render's edge, or Nginx) rather than in the Flask app itself, and
  doing so well requires knowing the final deployment topology — flagging
  as a deployment-time task rather than guessing at values here.
- **Dependency versions aren't pinned to exact versions** (`>=` in
  `requirements.txt`), so a future `pip install` could pull in a newer
  major version with different behavior. Not changed in this pass since
  it wasn't part of the security audit scope, but worth a `pip freeze`
  before any real production deploy.
