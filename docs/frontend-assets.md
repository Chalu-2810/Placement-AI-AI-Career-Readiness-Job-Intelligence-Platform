# PlacementAI — Frontend Assets (for local visual QA)

All visual assets load from public CDNs at request time — nothing is
bundled or vendored. This is intentional (per Phase 3 scope: no build
step, no new frontend tooling). All six asset requirements are covered
by the six URLs below, all referenced in `templates/base.html`.

## Exact URLs currently in use

| Asset | URL | Loaded as |
|---|---|---|
| Bootstrap 5.3.0 (CSS) | `https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css` | `<link rel="stylesheet">` |
| Bootstrap 5.3.0 (JS bundle) | `https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js` | `<script>` (end of body) |
| Chart.js 4.4.0 | `https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js` | `<script>` (end of body) |
| Font Awesome 6.4.0 | `https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css` | `<link rel="stylesheet">` |
| Fraunces + IBM Plex Sans + IBM Plex Mono | `https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap` | `<link rel="stylesheet">`, plus two `<link rel="preconnect">` to `fonts.googleapis.com` and `fonts.gstatic.com` |

Domains involved: `cdn.jsdelivr.net`, `cdnjs.cloudflare.com`,
`fonts.googleapis.com`, `fonts.gstatic.com`.

## Local development

These are ordinary public CDNs — any machine with normal internet access
(the constraint that blocked rendering in the sandbox does not apply
locally) will load all six assets with no extra setup. Nothing needs to
change to QA the app on your machine.

**If you ever need a fully offline dev environment** (e.g. on a flight,
or behind a restrictive corporate proxy), the lowest-risk option is
`npm install bootstrap@5.3.0 chart.js@4.4.0 @fortawesome/fontawesome-free@6.4.0`
into a `static/vendor/` folder and downloading the three Google Fonts
`.woff2` files once, then swapping the six URLs in `base.html` for local
`{{ url_for('static', ...) }}` paths. This is **not implemented** —
it's a bigger change (new build/vendoring step) than "document the
assets," and per your instructions I haven't touched the design or
added components. Flag if you want it done as its own follow-up.

## Regression status
`pytest` → **30/30 passing**, unchanged from Phase 3.
`docs/security.md` still documents `/predict` (GET, state-changing) as
a known residual risk — unchanged in this response.
