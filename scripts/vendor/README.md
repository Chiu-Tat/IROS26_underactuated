# Vendored JS libraries

Inlined into `arm_control_sim.html` by `scripts/make_control_html.py` so the
generated file stays fully self-contained (works offline, no CDN).

| file | source | version | license |
|---|---|---|---|
| `three.min.js` | https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js | r128 | MIT |
| `OrbitControls.js` | https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js | r128 | MIT |

r128 is the last three.js line that ships UMD builds (`THREE.*` globals),
which is what makes single-file inlining trivial. Do not upgrade to an
ESM-only release without changing the embedding strategy.
