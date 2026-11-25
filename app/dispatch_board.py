"""
Dispatch board: a SIMULATION of how the model's output could be used.

Routing (type -> team) and queue order (severity, then arrival) are fixed rules
applied to the model's output. Nothing in this file is learned.
Images with type confidence below the threshold go to a human reviewer instead.
"""
import base64
import json

import cv2
import streamlit.components.v1 as components

# key, team name, icon, defect types it handles
TEAMS = [
    ("structural", "Structural / composite repair", "🔩", ["crack", "craze_crack"]),
    ("laminate", "Composite laminate repair", "🧩", ["delamination"]),
    ("leading_edge", "Leading-edge protection", "🌬️", ["erosion"]),
    ("lightning", "Lightning protection system", "⚡", ["lightning_damage"]),
    ("coating", "Coating / surface maintenance", "🎨", ["surface_damage"]),
]
TEAM_OF = {c: key for key, _, _, classes in TEAMS for c in classes}
TEAM_NAME = {c: name for _, name, _, classes in TEAMS for c in classes}
HUMAN = "human"
MAX_VISIBLE = 6          # cards shown per team lane
MAX_VISIBLE_HUMAN = 2    # cards shown in the human review box


def thumbnail(rgb, size=96):
    small = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf.tobytes()).decode("ascii")


def make_ticket(seq, image_id, rgb, pred, conf, level, low_conf):
    return {"id": seq, "image": image_id, "thumb": thumbnail(rgb), "pred": pred, "conf": float(conf),
            "level": int(level), "lane": HUMAN if conf < low_conf else TEAM_OF[pred]}


def lanes(tickets):
    """Priority queue per lane: highest severity first, then first come first served."""
    out = {key: [] for key, *_ in TEAMS}
    out[HUMAN] = []
    for t in tickets:
        out[t["lane"]].append(t)
    for key in out:
        out[key].sort(key=lambda t: (-t["level"], t["id"]))
    return out


def snapshot(tickets):
    return {t["id"]: [lane, i] for lane, ts in lanes(tickets).items() for i, t in enumerate(ts)}


def render(tickets, anim, sev_color, nice, low_conf, height=780):
    queues = lanes(tickets)
    data = {
        "teams": [{"key": k, "name": n, "icon": i} for k, n, i, _ in TEAMS],
        "queues": {k: [{"id": t["id"], "image": t["image"], "thumb": t["thumb"], "type": nice(t["pred"]),
                        "conf": round(t["conf"] * 100), "level": t["level"], "lane": k} for t in v]
                   for k, v in queues.items()},
        "prev": {str(k): v for k, v in (anim or {}).get("prev", {}).items()},
        "new": (anim or {}).get("new", []),
        "sevColor": {str(k): v for k, v in sev_color.items()},
        "maxVisible": MAX_VISIBLE, "maxVisibleHuman": MAX_VISIBLE_HUMAN,
        "lowConf": round(low_conf * 100),
    }
    payload = json.dumps(data).replace("</", "<\\/")
    components.html(TEMPLATE.replace("__DATA__", payload), height=height, scrolling=False)


TEMPLATE = r"""
<style>
:root{--dark:#0F2A36;--teal:#1F7A8C;--amber:#F2A541;--red:#C8553D;--line:#A9BDC4;--muted:#5B7480}
*{box-sizing:border-box;font-family:"Source Sans Pro","Source Sans 3",system-ui,-apple-system,"Segoe UI",sans-serif}
body{margin:0;background:transparent;color:var(--dark)}
#wrap{position:relative;padding:0 2px 8px}
#wires{position:absolute;left:0;top:0;width:100%;height:100%;pointer-events:none;overflow:visible;z-index:0}
.top{position:relative;height:214px}
.box{position:relative;z-index:1;background:#fff;border:1px solid #DCE5E8;border-radius:14px;box-shadow:0 1px 3px rgba(15,42,54,.06)}
#router{position:absolute;left:50%;top:22px;transform:translateX(-50%);width:270px;padding:14px 16px;text-align:center;
        background:linear-gradient(120deg,#0F2A36,#1F7A8C);color:#fff;border:none}
#router .h{font-weight:700;font-size:1.02rem}
#router .s{font-size:.8rem;color:#D5E4E8;margin-top:4px;line-height:1.35}
#human{position:absolute;right:0;top:0;width:290px;height:204px;min-height:0;padding:10px 12px;border:2px dashed #E0A84F;background:#FFFBF3}
.lanehead{display:flex;align-items:center;gap:8px;font-weight:700;font-size:.9rem;line-height:1.15}
.lanehead .ic{font-size:1.25rem}
.lanehead .n{margin-left:auto;font-weight:600;font-size:.75rem;color:var(--muted);white-space:nowrap}
.hint{font-size:.72rem;color:var(--muted);margin:2px 0 6px}
.teams{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-top:44px}
.lane{padding:10px 10px 8px;min-height:470px}
.lane .lanehead{min-height:38px;padding-bottom:8px;border-bottom:1px solid #E6EEF0;margin-bottom:8px}
.cards{position:relative}
.card{position:relative;display:flex;align-items:center;gap:8px;height:56px;margin-bottom:8px;padding:5px 7px;
      background:#fff;border:1px solid #DCE5E8;border-radius:10px;will-change:transform}
.card img{width:44px;height:44px;border-radius:7px;object-fit:cover;flex:none}
.card .m{min-width:0;flex:1}
.card .t{font-weight:700;font-size:.8rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.card .s{font-size:.7rem;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.card .lv{flex:none;font-weight:700;font-size:.75rem;color:#fff;border-radius:6px;padding:3px 6px}
.card .pos{position:absolute;top:-7px;left:-6px;background:var(--dark);color:#fff;font-size:.62rem;font-weight:700;
           border-radius:99px;padding:1px 6px}
.card.first{border-color:var(--teal);box-shadow:0 0 0 1px var(--teal) inset}
.more{font-size:.75rem;color:var(--muted);text-align:center;padding:2px}
.empty{font-size:.75rem;color:#9AAEB6;text-align:center;padding:18px 4px;border:1px dashed #DCE5E8;border-radius:10px}
#human .empty{padding:10px 4px}
path.w{fill:none;stroke:var(--line);stroke-width:1.6}
path.dot{stroke:#E0A84F;stroke-dasharray:3 5;stroke-width:1.6}
path.pulse{stroke:var(--amber);stroke-width:3.2;stroke-dasharray:9 7;animation:dash .7s linear infinite}
@keyframes dash{to{stroke-dashoffset:-16}}
text.lbl{font-size:11px;fill:var(--muted)}
.lane.glow{animation:glow 1.4s ease-out}
@keyframes glow{0%{box-shadow:0 0 0 0 rgba(242,165,65,.0)}25%{box-shadow:0 0 0 4px rgba(242,165,65,.55)}100%{box-shadow:0 0 0 0 rgba(242,165,65,0)}}
</style>
<div id="wrap">
  <svg id="wires"></svg>
  <div class="top">
    <div id="router" class="box">
      <div class="h">Model output &rarr; routing rule</div>
      <div class="s">type &rarr; team &middot; severity &rarr; queue priority<br>confidence &lt; <span id="lc"></span>% &rarr; human review</div>
    </div>
    <div id="human" class="box lane" data-lane="human">
      <div class="lanehead"><span class="ic">👷</span><span>Human review</span><span class="n" id="n-human"></span></div>
      <div class="hint">Low confidence: an inspector checks the type, then forwards it (dotted lines)</div>
      <div class="cards" id="c-human"></div>
    </div>
  </div>
  <div class="teams" id="teams"></div>
</div>
<script>
const D = __DATA__;
const STEP = 64;   // card height 56 + margin 8
const $ = s => document.querySelector(s);
$('#lc').textContent = D.lowConf;

function cardHTML(t, idx, human){
  const col = D.sevColor[String(t.level)];
  const sub = human ? `guess ${t.conf}% · ${t.image}` : `${t.image} · ${t.conf}%`;
  // start position for the first paint, so nothing flashes before the animation begins
  let style = '';
  const p = D.prev[String(t.id)];
  if(D.new.includes(t.id)) style = 'opacity:0';
  else if(p && p[0] === t.lane && p[1] !== idx) style = `transform:translateY(${(p[1]-idx)*STEP}px)`;
  return `<div class="card${idx===0?' first':''}" style="${style}" data-id="${t.id}" data-lane="${t.lane}" data-idx="${idx}">
     <span class="pos">${idx===0?'next':'#'+(idx+1)}</span>
     <img src="data:image/jpeg;base64,${t.thumb}">
     <div class="m"><div class="t">${t.type}</div><div class="s">${sub}</div></div>
     <span class="lv" style="background:${col}">L${t.level}</span></div>`;
}
function fill(container, list, max, human){
  if(!list.length){ container.innerHTML = `<div class="empty">${human?'nothing to review':'queue empty'}</div>`; return; }
  container.innerHTML = list.slice(0,max).map((t,i)=>cardHTML(t,i,human)).join('') +
    (list.length>max ? `<div class="more">+${list.length-max} more in queue</div>` : '');
}
function count(list){
  if(!list.length) return '0 jobs';
  return `${list.length} job${list.length>1?'s':''} · top L${list[0].level}`;
}
// build lanes
$('#teams').innerHTML = D.teams.map(tm => `
  <div class="box lane" id="lane-${tm.key}" data-lane="${tm.key}">
    <div class="lanehead"><span class="ic">${tm.icon}</span><span>${tm.name}</span></div>
    <div class="hint" id="n-${tm.key}"></div>
    <div class="cards" id="c-${tm.key}"></div>
  </div>`).join('');
D.teams.forEach(tm => { fill($('#c-'+tm.key), D.queues[tm.key], D.maxVisible, false); $('#n-'+tm.key).textContent = count(D.queues[tm.key]); });
fill($('#c-human'), D.queues.human, D.maxVisibleHuman, true);
$('#n-human').textContent = count(D.queues.human);

// wires
function rel(el){ const w = $('#wrap').getBoundingClientRect(), r = el.getBoundingClientRect();
  return {l:r.left-w.left, t:r.top-w.top, r:r.right-w.left, b:r.bottom-w.top, cx:(r.left+r.right)/2-w.left, cy:(r.top+r.bottom)/2-w.top}; }
function drawWires(){
  const svg = $('#wires'); const R = rel($('#router')), H = rel($('#human'));
  const firstLane = rel($('#lane-'+D.teams[0].key));
  const bus1 = firstLane.t - 30, bus2 = firstLane.t - 14;
  let s = `<defs>
    <marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#A9BDC4"/></marker>
    <marker id="ad" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#E0A84F"/></marker></defs>`;
  D.teams.forEach(tm => {
    const L = rel($('#lane-'+tm.key));
    s += `<path id="w-${tm.key}" class="w" marker-end="url(#a)" d="M${R.cx},${R.b} V${bus1} H${L.cx-10} V${L.t}"/>`;
    s += `<path class="w dot" marker-end="url(#ad)" d="M${H.cx},${H.b} V${bus2} H${L.cx+10} V${L.t}"/>`;
  });
  s += `<path id="w-human" class="w" marker-end="url(#a)" d="M${R.r},${R.cy} H${H.l}"/>`;
  s += `<text class="lbl" x="${(R.r+H.l)/2}" y="${R.cy-8}" text-anchor="middle">conf &lt; ${D.lowConf}%</text>`;
  s += `<text class="lbl" x="${R.cx+8}" y="${bus1-6}">automatic routing</text>`;
  s += `<text class="lbl" x="${H.cx+8}" y="${bus2+0}" dx="0" dy="-22" fill="#B07A1E">forward after review</text>`;
  svg.innerHTML = s;
}
drawWires();
window.addEventListener('resize', drawWires);

// animation: existing cards slide to their new queue position, new cards fly in from the router
function pulse(lane){
  const p = document.getElementById('w-'+lane); if(p){ p.classList.add('pulse'); setTimeout(()=>p.classList.remove('pulse'), 1600); }
  const box = lane==='human' ? $('#human') : $('#lane-'+lane);
  if(box){ box.classList.remove('glow'); void box.offsetWidth; box.classList.add('glow'); }
}
function animate(){
  const isNew = new Set(D.new);
  document.querySelectorAll('.card').forEach(el => {
    const id = el.dataset.id, p = D.prev[id];
    if(isNew.has(+id) || !p) return;
    const now = +el.dataset.idx;
    if(p[0] === el.dataset.lane && p[1] !== now){
      el.style.transition = 'none'; el.style.transform = `translateY(${(p[1]-now)*STEP}px)`;
      setTimeout(() => { el.style.transition = 'transform .7s cubic-bezier(.2,.8,.2,1)'; el.style.transform = ''; }, 200);
    }
  });
  const R = $('#router').getBoundingClientRect();
  D.new.forEach((id, k) => {
    const el = document.querySelector(`.card[data-id="${id}"]`);
    const lane = Object.keys(D.queues).find(key => D.queues[key].some(t => t.id === id));
    setTimeout(() => pulse(lane), 200 + k*500);
    if(!el) return;                       // landed below the visible part of the queue
    const r = el.getBoundingClientRect();
    const dx = (R.left+R.width/2) - (r.left+r.width/2), dy = (R.top+R.height/2) - (r.top+r.height/2);
    el.style.transition = 'none'; el.style.opacity = '0';
    el.style.transform = `translate(${dx}px,${dy}px) scale(.45)`;
    setTimeout(() => {
      el.style.transition = 'transform .9s cubic-bezier(.2,.8,.2,1), opacity .35s ease-out';
      el.style.opacity = '1'; el.style.transform = '';
    }, 250 + k*500);
  });
}
if(D.new.length) requestAnimationFrame(() => requestAnimationFrame(animate));
</script>
"""
