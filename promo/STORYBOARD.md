---
format: 1920x1080
duration: 30s
message: "Find out if your strategy survives the prop firm — before you pay for the evaluation"
arc: Hook → Problem → Product intro → Mechanism → The one that kills accounts → Proof → Honesty + CTA
audience: retail futures traders on NinjaTrader 8 about to buy a prop-firm evaluation
mode: autonomous
music: tense driving minimal electronic, percussive pulse, no melody, dark
---

## Video direction

**No voiceover.** Reveals are paced to the **music's percussive pulse and the cut
rhythm** instead of a spoken line — the same discipline applies: nothing is
dumped at t=0, each piece lands on its own beat, and the back half of every
frame still carries reveals. Read every "cue" below as a beat, not a word.

**Palette system** (roles from `frame.md`, never invented):

- ground — `ink-black` `#0b0b0d` in **every** frame; there is no second register.
- ink — `cream` `#e7e9ea` for statements; `cream-muted` `#8b9096` for support.
- accent — `fire-orange` = PropSim's signal **green** `#34d399`: a path that
  survived, an answer, the wordmark.
- `signal-red` `#f2635f` — **only** where something breaches or dies. Never
  decorative.
- `signal-amber` `#f0b429` — **only** a caveat the tool refuses to hide (a
  sampled close, an unverified value). Never decorative.
- Hairlines `border-dark` `#232529`. Flat plane: no shadow, no radius, no
  gradient ground.

**Type** — statements in the display ramp (lowercase, negative-tracked, never
uppercase). **Every number, ticker, path and date in mono**, `tabular-nums`.
Mono chrome is uppercase and tracked.

**Motion grammar** — `power3` long-tail settles; smooth over bouncy, no
overshoot anywhere (this product is not playful). Curves and geometry are real
SVG that **draws itself** (`svg-path-draw`); type arrives by hard-cut beat slam
or per-word stagger, never by fade-up alone. Frame-to-frame continuity is
carried by the equity-path motif: it is drawn in Frame 1, multiplied in Frame 2,
dimmed to ground in Frame 3, and returns as the closing ground in Frame 7.

**Rhythm / held frames** — Frames 1, 2, 4, 5, 6 develop across their full
duration. **Frame 3 and Frame 7 are the held breathers**: content resolves early
and then reads still, which is what makes the two questions and the honesty line
land. A held read beats bad motion — no drift, no breathing, no back-half push.

**Negative list** — never: up/down arrows, glass cards, laptop/cloud metaphors,
raster imagery of any kind, bokeh or purple-blue "AI" gradients, brand logos of
the prop firms (plain mono text only), rounded corners, drop shadows, a second
accent hue used for emphasis. Never `repeat`/`yoyo`, `Math.random`, or
`Date.now`. Both motion failure modes are banned: **slideshow** (front-load then
freeze) and **screensaver** (everything floating independently).

**Caption band** — captions are disabled, but the bottom ~17% stays clear in
every frame for bottom-edge consistency.

## Frame 1 — You found out after you paid

- scene: One confident green equity curve draws upward, a red floor snaps on beneath it, the curve dies on contact
- duration: 4s
- transition_in: cut
- status: outline
- src: compositions/frames/01-hook-found-out.html
- type: hook
- blueprint: kinetic-type-beats (Adapt)
- focal: none — drawn geometry is the hero
- roles: n/a (no captured assets)
- asset_candidates: (drawn in-composition — no image files)

Adapt: keep the beat-slam signature, but the first two beats are **geometry, not
words** — the shape of the failure lands before any text does.

Scene 1 (0.0–1.4s): bare `ink-black` field, one hairline baseline. A single
`fire-orange` equity path **self-draws** left→right (`svg-path-draw`), climbing
with the small pullbacks of a real curve — it looks like it is working. Occupies
the lower-middle band, ~62% of frame width; nothing else on screen.

Scene 2 (1.4–2.5s): a `signal-red` floor hairline **hard-cuts** on beneath the
curve (no fade — the rule was always there, you just could not see it). The path
keeps drawing, rolls over, and plunges; at the instant it touches the floor the
stroke turns `signal-red` and the draw stops dead. A single red tick marks the
contact point.

Scene 3 (2.5–4.0s): the dead curve holds. Two short lines slam into the upper
third, left-anchored, one per beat (`kinetic-beat-slam`): **"you found out"** /
**"after you paid."** Asymmetric 65/35, three depth layers (floor hairline /
curve / type). Settles and reads **still**.

## Frame 2 — Ten thousand times first

- scene: The single path multiplies into thousands fanning from one origin — green above, red below — as a counter runs to 10,000
- duration: 4s
- transition_in: cut
- status: outline
- src: compositions/frames/02-ten-thousand.html
- type: pain_point
- blueprint: dataviz-countup (Adapt)
- focal: none — the probability cone is the hero
- roles: n/a (no captured assets)
- asset_candidates: (drawn in-composition — no image files)

Adapt: keep the count-up signature, drop the ring — the counter is chrome and
**the cone is the data**. The cut from Frame 1 is velocity-matched: the dead
path is still on screen at t=0 and the fan grows out of it.

Scene 1 (0.0–1.1s): the Frame 1 path sits dead center-left. From its single
origin, paths **expand outward** in staggered waves
(`center-outward-expansion`) — ~200 hairline strokes, `fire-orange` above the
origin line, `signal-red` below, each drawing along its own trajectory. Layered
depth by opacity: the outer envelope faintest.

Scene 2 (1.1–2.7s): the fan densifies wave over wave until the probability cone
reads as one volume. In the upper right, a mono counter **counts up** 0 → 10,000
(`counting-dynamic-scale`), `tabular-nums`, the digits locked so nothing
reflows. Full-width strip composition, the cone filling ~70% of frame.

Scene 3 (2.7–4.0s): the cone completes. Mono label lands under the counter —
**"paths · not one run"** — and the whole frame holds; at most **subtle
jitter** (`sine-wave-loop`, low amplitude) on the outer strokes keeps the volume
alive. No push, no drift.

## Frame 3 — Two questions

- scene: The cone dims to a ground; the PropSim wordmark lands and the two questions hard-cut in beneath it
- duration: 4s
- transition_in: cut
- status: outline
- src: compositions/frames/03-two-questions.html
- type: product_intro
- blueprint: kinetic-type-beats (Reproduce)
- focal: none — type only
- roles: n/a (no captured assets)
- asset_candidates: (drawn in-composition — no image files)

**Held breather #1.** The product is introduced by the two questions it answers
and nothing else.

Scene 1 (0.0–0.9s): the cone drops to ~16% opacity and becomes the ground — the
same geometry, demoted. **PropSim** spring-pops into the centre in the display
ramp, lowercase, `cream` (`spring-pop-entrance`, smooth settle, no overshoot).

Scene 2 (0.9–2.0s): first question **hard-cuts** in beneath the wordmark —
**"would it pass?"** — no fade, on the beat.

Scene 3 (2.0–3.0s): second question hard-cuts in directly below —
**"would it pay?"** Centered, ~50% of frame, both questions on the same left
edge so they read as a pair.

Scene 4 (3.0–4.0s): a `fire-orange` **keyword glow** (`asr-keyword-glow`) lands
once on **pass** and once on **pay**, an attack-decay-rest envelope on each.
Then everything holds **completely still**.

## Frame 4 — Your trades, not a model

- scene: A dense stream of tick marks flows in and condenses into a solid block of real round-trip trades; a folder path resolves beneath
- duration: 4s
- transition_in: cut
- status: outline
- src: compositions/frames/04-your-trades.html
- type: feature_showcase
- blueprint: grid-card-assemble (Adapt)
- focal: none — the condensing tape is the hero
- roles: n/a (no captured assets)
- asset_candidates: (drawn in-composition — no image files)

Adapt: keep the staggered self-assembly signature, but the items assemble **out
of a stream** rather than popping onto a grid — raw tape compressing into
structure.

Scene 1 (0.0–1.2s): a dense horizontal stream of hairline tick marks enters from
the left edge at speed, carrying a directional **motion-blur streak**
(`motion-blur-streak`), `cream-muted`, filling a full-width strip across the
middle band.

Scene 2 (1.2–2.5s): the stream **collapses inward** and the ticks resolve into a
compact block of discrete round-trip rows in mono — entry, exit, direction, P&L —
arriving in a staggered cascade, `fire-orange` on the winners and `signal-red` on
the losers, at the real proportion of a mixed record (not a flattering one).
Asymmetric 60/40, block held left of centre.

Scene 3 (2.5–4.0s): a mono path **types on** beneath the block with a caret
(`type-on with caret`) — **`db/NinjaTrader.sqlite`** — and the statement lands in
the upper third: **"your trades. not a model."** Holds.

## Frame 5 — The floor moves at the close. The breach doesn't.

- scene: Split stage — the same equity path on both sides; left checks only the daily close and survives, right tests unrealized equity in real time and breaches mid-day
- duration: 5s
- transition_in: cut
- status: outline
- src: compositions/frames/05-the-floor.html
- type: feature_showcase
- blueprint: comparison-split (Reproduce)
- focal: none — the two mirrored panels are the hero
- roles: n/a (no captured assets)
- asset_candidates: (drawn in-composition — no image files)

The technical beat. Both panels must show **visibly identical** path geometry —
the whole point is that only the test differs.

Scene 1 (0.0–1.1s): the stage splits on a 1px vertical hairline. Two panels
enter from opposite wings with mirrored `rotationY` tilts
(`split-tilt-cards`) and flatten to face-on. Each carries the **same** equity
path and the **same** floor hairline. Split-screen, panels ~46% each.

Scene 2 (1.1–2.4s): left panel only. `signal-amber` dots stamp onto the path at
each **end-of-day close** — four samples, nothing between them. Every sample sits
above the floor. A `fire-orange` pill spring-pops on the panel's inner edge:
**"survives"**. Mono label above: **"checked at the close"**.

Scene 3 (2.4–3.7s): right panel only. The live trace runs **continuously** along
the identical path; the instant it crosses the floor mid-day a `signal-red`
marker fires at the crossing, the trace goes red from that point on, and a red
pill snaps on the inner edge: **"breached"**. Mono label above: **"tested on
unrealized equity, in real time"**.

Scene 4 (3.7–5.0s): both panels hold side by side. One line lands centred on the
divider, `cream`: **"same trades. one of these is real."** Still — no push, no
drift, the comparison does the work.

## Frame 6 — Sourced, dated, and counted

- scene: A grid of rule chips assembles — 5 firms, 148 account variants — each stamped with the date it was read; a trials counter ticks alongside
- duration: 4.5s
- transition_in: cut
- status: outline
- src: compositions/frames/06-sourced-dated.html
- type: social_proof
- blueprint: grid-card-assemble (Reproduce)
- focal: none — the assembling rule grid is the hero
- roles: n/a (no captured assets)
- asset_candidates: (drawn in-composition — no image files)

Firm names are set as **plain mono text chips — never logos, never brand
colours.** They are a factual statement of what the rule table covers.

Scene 1 (0.0–1.2s): five firm chips **self-assemble** in a staggered cascade
across the upper band, mono, `cream` on hairline-bordered rectangles: **my funded
futures · apex · topstep · take profit trader · lucid trading**. Rule-of-thirds,
chips on the upper line.

Scene 2 (1.2–2.5s): beneath them a grid of 148 small variant ticks fills in
staggered waves (`center-outward-expansion`), `cream-hint`, while a mono counter
**counts up** to **148** at its left — `tabular-nums`. Density: the grid carries
~45% of the canvas, the one deliberately dense frame.

Scene 3 (2.5–3.5s): a dim `cream-muted` retrieved-date stamps under each firm
chip in a **per-word staggered reveal** — the dates are small, unglamorous, and
the point.

Scene 4 (3.5–4.5s): the statement lands lower-left: **"every value carries where
it came from."** Holds still.

## Frame 7 — Not a prediction

- scene: Everything clears to the bare cone at low opacity; the honesty line holds alone, then the wordmark and the install line land
- duration: 4.5s
- transition_in: cut
- status: outline
- src: compositions/frames/07-not-a-prediction.html
- type: cta
- blueprint: titlecard-reveal (Adapt)
- focal: none — type over the returning cone
- roles: n/a (no captured assets)
- asset_candidates: (drawn in-composition — no image files)

**Held breather #2, and the only frame with a real exit.** Adapt: the calm
landing beat, but the card that lands is a disclaimer before it is a CTA — the
refusal is the brand.

Scene 1 (0.0–1.3s): everything clears. The Frame 2 cone returns at ~10% as the
ground, drawn not popped. One line holds alone, centred, in the display ramp,
`cream`: **"a simulated pass rate is not a prediction about your account."**
Nothing else. Centered, ~55% of frame.

Scene 2 (1.3–2.6s): the line recedes to `cream-muted` and drops to the lower
third; **PropSim** spring-pops into the centre in `fire-orange`
(`spring-pop-entrance`, smooth settle).

Scene 3 (2.6–4.5s): a mono line **types on** beneath the wordmark —
**`github.com/jalv92/PropSim`** — with a dim mono rail under it: **free · mit ·
runs on your machine**. The lockup holds still, then the cone ground fades to
`ink-black` on the final beat — the video's only exit.
