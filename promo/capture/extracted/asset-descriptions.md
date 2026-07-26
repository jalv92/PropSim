# Asset inventory

No assets were captured — this is the no-capture path. PropSim has no marketing
site; its brand signals were read directly from the product's own UI
(`../dashboard.html`) and README.

Every visual in this video is drawn procedurally in the composition (SVG paths,
CSS, canvas). There are no image files to stage, and no frame should reference
one.

Ten AI concept frames were generated during art direction and reviewed
(`tmp/imagegen/propsim/f01..f10.png` at the workspace root). They are exploration
only and do not ship. What they settled:

- **Adopted** — the probability cone (f01), the hard drawdown floor with paths
  bending under it (f02), the red/green outcome histogram with an amber marker
  (f07), the donut gauge on a hairline grid (f10), the hash-chained ledger
  blocks (f06), ticks condensing into structure (f05).
- **Rejected** — literal up/down arrows (f03), glass card stacks (f04),
  laptop-and-cloud privacy metaphor (f09): stock clichés.
- **Lesson, not asset** — the icon sheet (f08) rendered mushy and duplicated;
  icons must be inline SVG strokes, single weight, never raster.
