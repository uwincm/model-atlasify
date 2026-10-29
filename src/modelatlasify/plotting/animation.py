"""Looping HTML animation of one variable's composite maps.

``model-images/<area>/<variable>.html`` sits next to ``<variable>/`` and plays
its PNGs in time order. The page is self-contained (inline CSS and
JavaScript, frames referenced by relative path), so it works from ``file://``
as well as from a web server.
"""

from __future__ import annotations

import html
import json
import re
from datetime import datetime
from pathlib import Path

from modelatlasify.core.variables import get_variable

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
    try:
        long_name = get_variable(var_dir.name).long_name
    except KeyError:
        long_name = var_dir.name
    area, run = var_dir.parent.name, var_dir.parent.parent.parent.name
    title = f"{long_name} — {area} — {run}"
    data = [{"src": f"{var_dir.name}/{p.name}", "label": _label(p.name)} for p in frames]
    page = (_TEMPLATE.replace("__TITLE__", html.escape(title))
            .replace("__FRAMES__", json.dumps(data, indent=1).replace("</", "<\\/")))
    path = var_dir.parent / f"{var_dir.name}.html"
    path.write_text(page, encoding="utf-8")
    return path


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
  h1 { font-size: 16px; font-weight: 600; margin: 0 0 8px; }
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
