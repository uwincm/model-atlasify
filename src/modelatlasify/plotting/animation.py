"""HTML pages for browsing a run's composite maps.

``model-images/<area>/<variable>.html`` sits next to ``<variable>/`` and plays
its PNGs in time order as a looping animation. ``model-images.html``, next to
``model-images/``, is a table of the latest map of every variable (rows) and
plot area (columns), each linking to its animation. The pages are
self-contained (inline CSS and JavaScript, images referenced by relative
path), so they work from ``file://`` as well as from a web server.
"""

from __future__ import annotations

import html
import json
import re
from datetime import datetime
from pathlib import Path

from modelatlasify.core.areas import FULL_AREA
from modelatlasify.core.variables import all_variables, get_variable

#: The valid time in a composite map's file name (see composite_path).
_TIME = re.compile(r"\.(\d{8}T\d{6}Z)\.png$")


def _label(name: str) -> str:
    m = _TIME.search(name)
    if not m:
        return name
    return f"{datetime.strptime(m[1], '%Y%m%dT%H%M%SZ'):%Y-%m-%d %H:%M} UTC"


def write_animation(var_dir: Path) -> Path | None:
    """Write the animation page for the composite maps in ``var_dir``.

    Frames are every PNG in the directory, so maps left by earlier runs into
    the same output are included. Returns the page, or None if there are no
    frames.
    """
    frames = sorted(var_dir.glob("*.png"))  # the time stamp in the name sorts them
    if not frames:
        return None
    long_name = _long_name(var_dir.name)
    area, run = var_dir.parent.name, var_dir.parent.parent.parent.name
    title = f"{long_name} — {area} — {run}"
    data = [{"src": f"{var_dir.name}/{p.name}", "label": _label(p.name)} for p in frames]
    page = (_TEMPLATE.replace("__TITLE__", html.escape(title))
            .replace("__FRAMES__", json.dumps(data, indent=1).replace("</", "<\\/")))
    path = var_dir.parent / f"{var_dir.name}.html"
    path.write_text(page, encoding="utf-8")
    return path


def _long_name(variable: str) -> str:
    try:
        return get_variable(variable).long_name
    except KeyError:
        return variable


def write_summary(images_dir: Path) -> Path | None:
    """Write ``model-images.html`` next to ``images_dir`` (``<run>/model-images``).

    One column per plot area (``full`` first, then the others by name), one
    row per variable (in the order the variables are registered). Each cell
    shows the variable's latest map in that area and links to its animation.
    Returns the page, or None if there are no maps.
    """
    if not images_dir.is_dir():
        return None
    cells: dict[tuple[str, str], Path] = {}  # (area, variable) -> latest frame
    for area_dir in images_dir.iterdir():
        if not area_dir.is_dir():
            continue
        for var_dir in area_dir.iterdir():
            frames = sorted(var_dir.glob("*.png")) if var_dir.is_dir() else []
            if frames:
                cells[(area_dir.name, var_dir.name)] = frames[-1]
    if not cells:
        return None
    areas = sorted({a for a, _ in cells}, key=lambda a: (a != FULL_AREA, a))
    order = {name: k for k, name in enumerate(all_variables())}
    variables = sorted({v for _, v in cells}, key=lambda v: (order.get(v, len(order)), v))

    esc = html.escape
    rows = ["<tr><th></th>" + "".join(f'<th scope="col">{esc(a)}</th>' for a in areas) + "</tr>"]
    for v in variables:
        row = [f'<th scope="row">{esc(_long_name(v))}<span class="code">{esc(v)}</span></th>']
        for a in areas:
            frame = cells.get((a, v))
            if frame is None:
                row.append('<td class="none">&mdash;</td>')
                continue
            src = f"{images_dir.name}/{a}/{v}/{frame.name}"
            row.append(
                f'<td><a href="{esc(f"{images_dir.name}/{a}/{v}.html")}" '
                f'title="Animate {esc(_long_name(v))} ({esc(a)})">'
                f'<img src="{esc(src)}" width="300" alt="{esc(_long_name(v))}, {esc(a)}" '
                f'loading="lazy"></a><div class="time">{esc(_label(frame.name))}</div></td>')
        rows.append("<tr>" + "".join(row) + "</tr>")
    title = f"Composite maps — {images_dir.parent.name}"
    page = (_SUMMARY_TEMPLATE.replace("__TITLE__", esc(title))
            .replace("__ROWS__", "\n".join(rows)))
    path = images_dir.parent / f"{images_dir.name}.html"
    path.write_text(page, encoding="utf-8")
    return path


_SUMMARY_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { --bg: #ffffff; --fg: #1d1d1f; --muted: #6b6b70; --line: #d0d0d5; --head: #f2f2f5; }
  @media (prefers-color-scheme: dark) {
    :root { --bg: #18181b; --fg: #ececf0; --muted: #a0a0a8; --line: #3a3a40; --head: #26262b; }
  }
  * { box-sizing: border-box; }
  body { margin: 0; padding: 12px 16px; background: var(--bg); color: var(--fg);
         font: 14px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif; }
  h1 { font-size: 18px; font-weight: 600; margin: 0 0 4px; }
  p { color: var(--muted); margin: 0 0 12px; }
  .wrap { overflow-x: auto; }
  table { border-collapse: collapse; }
  th, td { border: 1px solid var(--line); padding: 6px; vertical-align: middle; }
  th { background: var(--head); font-weight: 600; }
  th[scope=row] { text-align: left; max-width: 180px; }
  .code { display: block; color: var(--muted); font: 12px ui-monospace, monospace; font-weight: 400; }
  td { text-align: center; }
  td img { display: block; width: 300px; height: auto; }
  td a:hover img { outline: 2px solid var(--muted); }
  .time { color: var(--muted); font-size: 12px; margin-top: 2px; }
  .none { color: var(--muted); }
</style>
</head>
<body>
<h1>__TITLE__</h1>
<p>Latest map of each variable in each plot area. Click a map to animate the run.</p>
<div class="wrap">
<table>
__ROWS__
</table>
</div>
</body>
</html>
"""


_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { --bg: #ffffff; --fg: #1d1d1f; --muted: #6b6b70; --line: #d0d0d5; --btn: #f2f2f5; }
  @media (prefers-color-scheme: dark) {
    :root { --bg: #18181b; --fg: #ececf0; --muted: #a0a0a8; --line: #3a3a40; --btn: #26262b; }
  }
  * { box-sizing: border-box; }
  body { margin: 0; padding: 12px 16px; background: var(--bg); color: var(--fg);
         font: 14px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif; }
  h1 { font-size: 16px; font-weight: 600; margin: 0 0 4px; }
  nav { margin: 0 0 8px; }
  a { color: inherit; }
  #view { display: flex; justify-content: center; }
  #frame { max-width: 100%; max-height: calc(100vh - 150px); object-fit: contain; }
  #controls { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin-top: 8px; }
  button { background: var(--btn); color: var(--fg); border: 1px solid var(--line);
           border-radius: 6px; padding: 4px 10px; font: inherit; cursor: pointer; min-width: 40px; }
  button:hover { border-color: var(--muted); }
  #slider { flex: 1 1 200px; min-width: 120px; }
  label { color: var(--muted); display: inline-flex; align-items: center; gap: 4px; }
  input[type=number] { width: 4.5em; background: var(--btn); color: var(--fg);
                       border: 1px solid var(--line); border-radius: 4px; padding: 2px 4px; }
  #info { font-variant-numeric: tabular-nums; margin-top: 6px; }
  .hint { color: var(--muted); font-size: 12px; margin-top: 4px; }
</style>
</head>
<body>
<h1>__TITLE__</h1>
<nav><a href="../../model-images.html">&larr; All composite maps</a></nav>
<div id="view"><img id="frame" alt=""></div>
<div id="info"></div>
<div id="controls">
  <button id="first" title="First (Home)">&#x23EE;</button>
  <button id="prev" title="Previous (&larr;)">&#x25C0;</button>
  <button id="play" title="Play/pause (space)">&#x25B6;</button>
  <button id="next" title="Next (&rarr;)">&#x25B6;&#xFE0E;|</button>
  <button id="last" title="Last (End)">&#x23ED;</button>
  <input id="slider" type="range" min="0" value="0" step="1" aria-label="Frame">
  <label>fps <input id="fps" type="number" min="0.5" max="30" step="0.5" value="4"></label>
  <label>end pause (s) <input id="dwell" type="number" min="0" max="10" step="0.5" value="1"></label>
  <label><input id="loop" type="checkbox" checked> loop</label>
</div>
<div class="hint">Keys: &larr;/&rarr; step, space play/pause, Home/End first/last.</div>
<script>
const FRAMES = __FRAMES__;
const img = document.getElementById("frame"), info = document.getElementById("info");
const slider = document.getElementById("slider"), playBtn = document.getElementById("play");
const fps = document.getElementById("fps"), dwell = document.getElementById("dwell");
const loop = document.getElementById("loop");
const cache = FRAMES.map(f => { const im = new Image(); im.src = f.src; return im; });
let i = 0, timer = null;
slider.max = FRAMES.length - 1;

function show(n) {
  i = (n + FRAMES.length) % FRAMES.length;
  img.src = cache[i].src;
  img.alt = FRAMES[i].label;
  slider.value = i;
  info.textContent = `${i + 1} / ${FRAMES.length} \\u00B7 ${FRAMES[i].label}`;
}
function delay() {
  const step = 1000 / Math.max(parseFloat(fps.value) || 4, 0.1);
  return i === FRAMES.length - 1 ? step + 1000 * Math.max(parseFloat(dwell.value) || 0, 0) : step;
}
function tick() {
  if (i === FRAMES.length - 1 && !loop.checked) { stop(); return; }
  show(i + 1);
  timer = setTimeout(tick, delay());
}
function play() {
  if (timer || FRAMES.length < 2) return;
  if (i === FRAMES.length - 1 && !loop.checked) show(0);
  playBtn.innerHTML = "&#x23F8;";
  timer = setTimeout(tick, delay());
}
function stop() { clearTimeout(timer); timer = null; playBtn.innerHTML = "&#x25B6;"; }
function toggle() { timer ? stop() : play(); }
function step(n) { stop(); show(n); }

playBtn.onclick = toggle;
document.getElementById("first").onclick = () => step(0);
document.getElementById("last").onclick = () => step(FRAMES.length - 1);
document.getElementById("prev").onclick = () => step(i - 1);
document.getElementById("next").onclick = () => step(i + 1);
slider.oninput = () => step(parseInt(slider.value, 10));
document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT" && e.target.type === "number") return;
  if (e.key === "ArrowLeft") step(i - 1);
  else if (e.key === "ArrowRight") step(i + 1);
  else if (e.key === "Home") step(0);
  else if (e.key === "End") step(FRAMES.length - 1);
  else if (e.key === " ") toggle();
  else return;
  e.preventDefault();
});
show(0);
</script>
</body>
</html>
"""
