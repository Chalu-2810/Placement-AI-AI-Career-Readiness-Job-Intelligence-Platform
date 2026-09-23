# PlacementAI — Waypoint Design System

## Why "Waypoint"

PlacementAI's job is to tell someone where they stand on a journey and
what to do next. That's a navigation problem, not a dashboard problem —
so the visual language is built from **instruments and charts**: the
tools people have always used to know where they are and where they're
headed. A career readiness score reads like a bearing on a compass, not
a KPI tile. A roadmap reads like a route plotted on a chart, not three
generic cards labeled 30/60/90.

This is a deliberate departure from the "sidebar + white KPI cards +
gradient hero" pattern the brief explicitly asked to avoid, and from the
three AI-generic defaults (warm-cream-and-terracotta, near-black-and-neon,
broadsheet-hairlines) — none of which fit a product about *navigating
toward a goal*.

---

## 1. Color

Named, not just hex — every token describes what it's *for*.

| Token | Hex | Use |
|---|---|---|
| `--paper` | `#EEF2F0` | Page background — cool, blue-green-tinted "chart paper," not warm cream |
| `--panel` | `#FFFFFF` | Raised surfaces (panels, inputs) |
| `--ink` | `#16233A` | Primary text, headlines — deep indigo-navy, not black |
| `--ink-soft` | `#4B5A70` | Secondary text, captions |
| `--ink-faint` | `#8A97A8` | Placeholder text, disabled states |
| `--line` | `#D8DFDC` | Hairline borders, chart rules |
| `--line-strong` | `#B8C2BE` | Dividers that need more presence |
| `--brass` | `#BE8C33` | Primary accent — instrument brass. Calls to action, focus states, the gauge needle |
| `--brass-deep` | `#96691F` | Brass hover/active |
| `--chart-teal` | `#1F5C52` | Secondary accent — deep chart-ink teal. Links, secondary emphasis |
| `--signal-good` | `#2E7D5B` | Positive signal — strengths, "on track," success states |
| `--signal-gap` | `#B5533C` | Attention signal — gaps, priorities (brick/rust, not terracotta-orange) |
| `--signal-missing` | `#8A3B32` | Used sparingly — missing/critical only |
| `--casing` | `#0F1B2B` | Dark "instrument casing" — nav bar, hero panels, footer |
| `--casing-line` | `#2A3B52` | Borders/dividers on dark casing surfaces |

Rule: color signals meaning (good/gap/missing/accent), never decoration.
No gradients except one restrained radial glow behind the bearing gauge.

---

## 2. Typography

| Role | Face | Notes |
|---|---|---|
| Display | **Fraunces** (serif, via Google Fonts) | Headlines, the readiness number itself, section titles. Used at weight 400–600, optical sizing on. Characterful without being decorative — reads like chart lettering, not a startup logotype. |
| Body | **IBM Plex Sans** | All running text, labels, buttons, nav. Technical, legible, unglamorous — the "instrument panel" voice. |
| Data / mono | **IBM Plex Mono** | Every number that represents a *measurement*: scores, percentages, gauge readouts, timer, table figures. Numerals in monospace read as instrument data, not decoration. |

Scale (rem): 3.5 / 2.5 / 1.75 / 1.25 / 1 / 0.875 / 0.75 — display headings
use Fraunces, everything else Plex Sans/Mono.

---

## 3. Spacing & shape

- Spacing scale: 4 / 8 / 12 / 16 / 24 / 32 / 48 / 64px. Panels breathe —
  minimum 24px internal padding, 32–48px between major sections.
- **Radius is small and consistent, not "rounded-4 everywhere."**
  Panels: 4px. Buttons/inputs/chips: 3px. Nothing pill-shaped except
  status badges. Sharp-ish corners read as "instrument," not "app."
- Shadows are near-flat: a single 1px hairline border does most of the
  work; shadow is reserved for genuinely elevated elements (dropdowns,
  the gauge glow) — `0 8px 24px rgba(15,27,43,0.08)` at most.

---

## 4. Signature element — the Bearing Gauge

The one thing this product should be remembered by: **readiness is shown
as a semicircular instrument dial**, not a circular progress ring or a
number-in-a-card. Calibrated ticks from 0–100, three named zones
(Building / Developing / Ready), a brass needle at the current score, and
a thin ghost mark showing the previous score so movement is visible at a
glance. Implemented once as `templates/_gauge.html` and reused on the
dashboard and the prediction/results page — one motif, two appearances,
not two different widgets.

---

## 5. Components

- **Console nav** — dark casing bar, not a sidebar. Wordmark left,
  journey-oriented links center (`Overview / Assess / Resume / Skills /
  Opportunities`), account control right. Collapses to a bottom tab bar
  on mobile, not a hamburger drawer, since there are only 5 destinations.
- **Panel** — the base container (`.panel`). Replaces Bootstrap `.card`
  as the primary visual vocabulary: 1px `--line` border, 4px radius,
  `--panel` background, no shadow at rest.
- **Instrument readout** (`.readout`) — a labeled mono-numeral value used
  for every score (ATS, confidence, quiz sub-scores). Label above in
  small caps `--ink-soft`, value in Plex Mono, `--ink`.
- **Signal chip** (`.chip`, `.chip--good/gap/missing`) — replaces
  Bootstrap badges for skill/status states. Small rectangular tag, 3px
  radius, colored border + text, no filled backgrounds except on dark
  casing surfaces.
- **Insight block** (`.insight`) — editorial-style entries for AI Career
  Insights: a left rule in the signal color, a small-caps label
  (STRENGTH / GAP / NEXT MOVE — not numbered, since these are
  categorical, not a sequence), body copy in Plex Sans.
- **Journey timeline** (`.journey`) — vertical rule with milestone dots
  for the career roadmap, which *is* a genuine sequence (30/60/90 days →
  target), so ordering markers are justified here specifically.
- **Skill landscape** (`.skill-band`) — three horizontal bands (Strong /
  Developing / Priority Gap) instead of a table; each skill is a chip
  with a thin inline bar showing current vs. target.
- **Buttons** — solid brass for primary actions, outline ink for
  secondary, text-only teal for tertiary/links. No color-per-context
  buttons (`btn-success`/`btn-danger` per action) — the action's
  importance sets the style, not its associated backend route.
- **Forms** — flat, bordered, brass focus ring
  (`box-shadow: 0 0 0 3px rgba(190,140,51,0.25)`), labels above fields in
  small caps `--ink-soft`.

---

## 6. States

- **Empty** — every empty section explains what it unlocks and gives one
  clear action, in the interface's voice ("Your resume is missing" +
  what it unlocks + one button), never a bare "No data."
- **Loading** — the multi-step checklist pattern (`✓ / ● / ○`) for
  anything with a real multi-stage process (prediction, resume parsing);
  a simple inline spinner for single-step waits (quiz generation).
- **Error** — states what happened and how to fix it, in plain language,
  no backend/stack details (already enforced server-side in Phase 2 —
  this phase makes sure the *presentation* matches: calm, not alarmed).

---

## 7. Responsive behavior

- Breakpoints follow Bootstrap's grid (still used for layout only, not
  visual language): ≥1200 desktop, 992–1199 laptop, 768–991 tablet,
  <768 mobile.
- Console nav → bottom tab bar under 768px (5 destinations fit a tab
  bar cleanly; a hamburger would hide the journey structure the nav is
  meant to communicate).
- The bearing gauge scales its SVG viewBox rather than being cropped;
  side-by-side panels (Strongest Signal / Biggest Gap) stack vertically
  under 768px.
- Tables (admin) scroll horizontally within a bordered panel rather than
  collapsing into unreadable stacked cards.

---

## 8. Accessibility

- All interactive elements keep a visible focus ring (`:focus-visible`,
  brass, 3px, never `outline: none` without a replacement).
- Color is never the only signal: chips carry a label and (where
  relevant) an icon-free text prefix ("Gap:", "✓"), not color alone.
- Contrast: body text `--ink` (#16233A) on `--paper`/`--panel`
  (#EEF2F0/#FFFFFF) exceeds WCAG AA for normal text; `--ink-soft` used
  only for secondary/caption text at larger sizes.
- `prefers-reduced-motion: reduce` disables the gauge needle transition,
  progress-bar fill animation, and hover-lift transforms.
- Forms use real `<label for>` associations (already present in the
  existing markup) — preserved, not regressed, by this redesign.
