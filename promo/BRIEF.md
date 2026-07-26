---
workflow: product-launch-video
flow: automation
storyboard: no
message: "Find out if your strategy survives the prop firm — before you pay for the evaluation"
destination: youtube
aspect: 1920x1080
language: en
audience: retail futures traders on NinjaTrader 8 who are about to buy a prop-firm evaluation
length: 30s
angle: proof
---

## Intent

A 30-second promo for **PropSim** — a local Windows/Python tool that reads a
trader's own NinjaTrader 8 database and answers two questions against the real
published rules of five prop firms: *would this pass the evaluation, and would
it pay out once funded?*

Fast, hard-cut, high-energy. The motion IS the argument: thousands of simulated
equity paths fanning out, most of them dying on a drawdown floor. Quant-terminal
aesthetic, near-black, no stock-photo warmth, no corporate gloss. It should feel
like an instrument, not an ad.

**No voiceover** — kinetic typography carries the message and reads faster than
speech at this cut rate. Music only.

## Assets

None. Every visual is drawn procedurally (SVG/CSS/canvas) in the composition —
no raster imagery. Concept exploration lives in `tmp/imagegen/propsim/` at the
workspace root and informed the art direction, but none of those PNGs ship.

## Customizations

- Palette and type are lifted verbatim from the product's own UI
  (`dashboard.html`): canvas `#0b0b0d`, panel `#121316`, hairline `#232529`,
  ink `#e7e9ea`, dim `#8b9096`, green `#34d399`, red `#f2635f`, amber `#f0b429`,
  blue `#5aa9e6`, monospace for every number. The video must look like the tool.
- Count-ups on the hard numbers: 10,000 sims, 148 account variants, 5 firms.
- The signature shot is the probability cone — many thin paths fanning from one
  origin, green above / red below — reprised as the closing ground.

## Notes

- Honesty is the product. Never imply PropSim predicts profit or guarantees a
  pass. The closing card must carry "A simulated pass rate is not a prediction."
- The differentiator to dramatize is the **drawdown floor tested on unrealized
  equity in real time**, not at the close — a simulator that only checks the
  close lets dead accounts survive. That is the one technical beat worth a frame.
- Free and MIT-licensed; runs entirely on the trader's machine, nothing
  uploaded. Worth one beat, not more.
- Avoid: literal up/down arrows, glass card stacks, laptop-and-cloud metaphors,
  any stock motion-graphics cliché. Explored and rejected during concept.
