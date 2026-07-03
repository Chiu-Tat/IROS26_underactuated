"""Generate a standalone HTML animation of the dual-arm platform being controlled.

Kinematic animation: each joint is driven from a start to a goal pose one motor at
a time; while a motor is actuated we compute the *actual* spillover-controller coil
currents (Eq. spillover) for tracking the rotating field at that motor in the arm's
current pose. The result is written as a self-contained HTML file (embedded data +
vanilla-canvas renderer, no external dependencies) showing the moving arms, the
eight coils, and the live current in each coil.

    python scripts/make_control_html.py   ->  arm_control_sim.html
"""
import sys
import json
import math
import pathlib

import numpy as np
np.seterr(all="ignore")

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import matplotlib  # noqa (case_study imports it); force headless
matplotlib.use("Agg")
from case_study_dualarm import (
    A_planar, spillover_S, current, arm, ALL_COILS, COILS_IDX, RADII, DRIVE_K,
    XB, SEP, YAW, BASE_Z,
)

CYCLES = 3          # visible field rotations per joint motion (illustrative)
FRAMES_PER_SEG = 70
CURRENT_LIMIT = 17.0
LAM_ANIM = 2.5e-5   # current regularization for displayed currents (keeps <=17 A)

# per-arm (alpha, beta, grip) start -> goal; grip 1 = open, 0 = closed
START = dict(a1=[35, -45, 1.0], a2=[35, -45, 1.0])
GOAL = dict(a1=[-15, 25, 0.0], a2=[-15, 25, 0.0])
# segment -> (motor index 0..5, which arm 'a1'/'a2', which var 0=alpha 1=beta 2=grip)
SEQUENCE = [(0, "a1", 0), (1, "a1", 1), (2, "a1", 2),
            (3, "a2", 0), (4, "a2", 1), (5, "a2", 2)]


def smooth(p):
    return p * p * (3 - 2 * p)


def poses_and_currents():
    coils = ALL_COILS[COILS_IDX]
    state = {k: list(v) for k, v in START.items()}
    frames = []
    for seg, (m, ak, var) in enumerate(SEQUENCE):
        v0, v1 = state[ak][var], GOAL[ak][var]
        for f in range(FRAMES_PER_SEG):
            p = f / (FRAMES_PER_SEG - 1)
            state[ak][var] = v0 + (v1 - v0) * smooth(p)
            a1, b1, g1 = state["a1"]
            a2, b2, g2 = state["a2"]
            mA = arm((XB, SEP, BASE_Z), YAW, a1, b1)
            mB = arm((XB, -SEP, BASE_Z), YAW, a2, b2)
            motors = mA + mB
            # actual controller currents for the active motor at this pose
            if var == 2:                       # gripper motor uses its own position
                curr = np.zeros(len(coils))    # (still compute below)
            A = [A_planar(pt, coils) for pt in motors]
            S = spillover_S(A, m, LAM_ANIM)
            theta = 2 * math.pi * CYCLES * p
            curr = current(S, DRIVE_K * RADII[m], theta)
            frames.append({
                "a1": [[round(float(v), 5) for v in pt] for pt in mA],   # 3-D
                "a2": [[round(float(v), 5) for v in pt] for pt in mB],
                "g1": round(g1, 3), "g2": round(g2, 3),
                "cur": [round(float(c), 3) for c in curr],
                "m": m, "seg": seg,
            })
    return frames, coils


BOX_IMAX = 15.0     # current bound used for the reachable-field (box) MFW


def build_html(frames, coils):
    data = {
        # full dipole parameters so the MFW can be recomputed per frame in JS
        "coil_params": [[round(float(x), 6) for x in c] for c in coils],
        "coil_ids": [i + 1 for i in COILS_IDX],
        "bmin": list(RADII),
        "box_imax": BOX_IMAX,
        "frames": frames,
        "ilim": CURRENT_LIMIT,
    }
    payload = json.dumps(data)
    return _HTML_TEMPLATE.replace("/*DATA*/", payload)


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Dual-arm selective control simulation</title>
<style>
  body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#0f1420;color:#e6edf3}
  #wrap{display:flex;flex-wrap:wrap;gap:12px;padding:14px}
  canvas{background:#141b2d;border-radius:10px}
  #side{min-width:300px;flex:1}
  h1{font-size:17px;margin:2px 0 8px}
  .sub{color:#9fb0c3;font-size:13px;margin-bottom:10px}
  #controls{display:flex;align-items:center;gap:10px;margin:8px 0}
  button{background:#2b6cb0;color:#fff;border:0;border-radius:6px;padding:6px 12px;cursor:pointer;font-size:14px}
  button:hover{background:#3182ce}
  #slider{flex:1}
  .tag{display:inline-block;padding:2px 8px;border-radius:5px;font-weight:600}
  .lbl{color:#9fb0c3;font-size:12px}
  #mfwwrap{padding:0 14px 18px}
  #mfwrow{display:flex;gap:8px;flex-wrap:wrap}
  #mfwrow canvas{border-radius:8px}
</style></head>
<body>
<div id="wrap">
  <div>
    <h1>Selective control of a 6-motor dual-arm platform</h1>
    <div class="sub">Top view. Each joint is actuated one motor at a time; only the
      active joint moves while the others stay fixed. Coil fill = current
      (red&nbsp;+, blue&nbsp;&minus;).</div>
    <canvas id="scene" width="600" height="600"></canvas>
    <div id="controls">
      <button id="play">Pause</button>
      <input id="slider" type="range" min="0" value="0">
      <span class="lbl" id="clock"></span>
    </div>
    <div>Now driving: <span class="tag" id="active"></span></div>
  </div>
  <div id="side">
    <h1>Coil currents (A)</h1>
    <div class="sub">Eight active coils. Limit &plusmn;<span id="ilimtxt"></span>&nbsp;A.</div>
    <canvas id="bars" width="360" height="470"></canvas>
  </div>
</div>
<div id="mfwwrap">
  <h1>Magnetic-feasible workspace (MFW) of each motor &mdash; real time</h1>
  <div class="sub">Field plane (B<sub>x</sub>, B<sub>y</sub>) at each motor. Shaded =
    MFW (reachable field for |i|&le;<span id="boxtxt"></span>&nbsp;A, recomputed each
    frame from the coil geometry). Green = minimum-actuation circle b<sub>min</sub>.
    Dot &amp; trail = the actual field produced by the applied current. A motor
    rotates only when its field traces <em>around</em> its b<sub>min</sub> circle.</div>
  <div id="mfwrow"></div>
</div>
<script>
const D = /*DATA*/;
const scene = document.getElementById('scene'), sx = scene.getContext('2d');
const bars = document.getElementById('bars'), bx = bars.getContext('2d');
const slider = document.getElementById('slider'), playBtn = document.getElementById('play');
const activeEl = document.getElementById('active'), clockEl = document.getElementById('clock');
document.getElementById('ilimtxt').textContent = D.ilim;
document.getElementById('boxtxt').textContent = D.box_imax;
slider.max = D.frames.length - 1;
const MOTOR_COL = ['#3aa0ff','#66b8ff','#a9d4ff','#ff5a5a','#ff8a8a','#ffb3b3'];
const CP = D.coil_params;

// ---------- field model: dipole actuation matrix + reachable-field zonotope ----
function dipoleCol(cp, pos){ // [Bx,By] per amp for one coil at pos
  const dx=pos[0]-cp[3], dy=pos[1]-cp[4], dz=pos[2]-cp[5];
  const r=Math.sqrt(dx*dx+dy*dy+dz*dz)+1e-9, r3=r*r*r, r5=r3*r*r;
  const dot=cp[0]*dx+cp[1]*dy+cp[2]*dz, c=1e-7;
  return [c*(3*dx*dot/r5 - cp[0]/r3), c*(3*dy*dot/r5 - cp[1]/r3)];
}
function motorPos(fi, j){ const f=D.frames[fi]; return (j<3?f.a1:f.a2)[j%3]; }
function actMatrix(pos){ return CP.map(cp=>dipoleCol(cp, pos)); }   // 8 columns
function fieldAt(fi, j){ // actual field at motor j from the applied current
  const cols=actMatrix(motorPos(fi,j)), cur=D.frames[fi].cur;
  let Bx=0, By=0; for(let k=0;k<cols.length;k++){Bx+=cols[k][0]*cur[k];By+=cols[k][1]*cur[k];}
  return [Bx,By];
}
function zono(gens){ // vertices of {sum c_k g_k : c_k in [-1,1]}
  let g=gens.map(v=>(v[1]<0||(v[1]===0&&v[0]<0))?[-v[0],-v[1]]:[v[0],v[1]]);
  g.sort((a,b)=>Math.atan2(a[1],a[0])-Math.atan2(b[1],b[0]));
  let c=[0,0]; g.forEach(v=>{c[0]-=v[0];c[1]-=v[1];});
  let vs=[[c[0],c[1]]];
  g.forEach(v=>{c=[c[0]+2*v[0],c[1]+2*v[1]];vs.push([c[0],c[1]]);});
  g.forEach(v=>{c=[c[0]-2*v[0],c[1]-2*v[1]];vs.push([c[0],c[1]]);});
  return vs;
}

// ---------- scene (arms + coils) ----------
let xs=[], ys=[];
CP.forEach(c=>{xs.push(c[3]);ys.push(c[4]);});
D.frames.forEach(f=>{f.a1.concat(f.a2).forEach(p=>{xs.push(p[0]);ys.push(p[1]);});});
const pad=0.02, xmin=Math.min(...xs)-pad, xmax=Math.max(...xs)+pad,
      ymin=Math.min(...ys)-pad, ymax=Math.max(...ys)+pad;
const W=scene.width, H=scene.height, span=Math.max(xmax-xmin,ymax-ymin);
const TX=x=>(x-xmin)/span*(W-40)+20, TY=y=>H-((y-ymin)/span*(H-40)+20);

function curColor(i){
  const t=Math.max(-1,Math.min(1,i/D.ilim));
  if(t>=0){const k=Math.round(255*(1-t));return `rgb(255,${k},${k})`;}
  const k=Math.round(255*(1+t));return `rgb(${k},${k},255)`;
}
function drawGripper(ctx,base,tip,grip,col){
  const dx=tip[0]-base[0], dy=tip[1]-base[1], L=Math.hypot(dx,dy)||1;
  const ux=dx/L, uy=dy/L, px=-uy, py=ux, spread=(0.2+grip*0.8)*0.010, len=0.014;
  [1,-1].forEach(s=>{
    const rx=tip[0]+px*s*spread*0.4, ry=tip[1]+py*s*spread*0.4;
    const ex=rx+ux*len+px*s*spread, ey=ry+uy*len+py*s*spread;
    ctx.beginPath();ctx.moveTo(TX(rx),TY(ry));ctx.lineTo(TX(ex),TY(ey));
    ctx.strokeStyle=col;ctx.lineWidth=4;ctx.lineCap='round';ctx.stroke();
  });
}
function drawArm(ctx,j,grip,ai,active){
  ctx.strokeStyle='#7a8aa0';ctx.lineWidth=7;ctx.lineCap='round';
  ctx.beginPath();ctx.moveTo(TX(j[0][0]),TY(j[0][1]));
  ctx.lineTo(TX(j[1][0]),TY(j[1][1]));ctx.lineTo(TX(j[2][0]),TY(j[2][1]));ctx.stroke();
  ctx.fillStyle='#586074';ctx.fillRect(TX(j[0][0])-13,TY(j[0][1])-13,26,26);
  for(let k=0;k<3;k++){
    const mi=ai*3+k, isAct=(mi===active);
    ctx.beginPath();ctx.arc(TX(j[k][0]),TY(j[k][1]),k<2?10:8,0,7);
    ctx.fillStyle=MOTOR_COL[mi];ctx.fill();
    ctx.lineWidth=isAct?4:1.5;ctx.strokeStyle=isAct?'#ffe600':'#0f1420';ctx.stroke();
  }
  drawGripper(ctx,j[1],j[2],grip,MOTOR_COL[ai*3+2]);
  ctx.fillStyle='#cfe0f0';ctx.font='12px sans-serif';
  for(let k=0;k<3;k++)ctx.fillText('M'+(ai*3+k+1),TX(j[k][0])+11,TY(j[k][1])-9);
}
function drawScene(fi){
  const f=D.frames[fi]; sx.clearRect(0,0,W,H);
  CP.forEach((c,i)=>{
    sx.beginPath();sx.arc(TX(c[3]),TY(c[4]),15,0,7);
    sx.fillStyle=curColor(f.cur[i]);sx.fill();
    sx.lineWidth=2;sx.strokeStyle='#26324a';sx.stroke();
    sx.fillStyle='#0f1420';sx.font='bold 11px sans-serif';sx.textAlign='center';
    sx.fillText('C'+D.coil_ids[i],TX(c[3]),TY(c[4])-19);
    sx.fillStyle='#e6edf3';sx.fillText(f.cur[i].toFixed(1),TX(c[3]),TY(c[4])+4);
    sx.textAlign='left';
  });
  drawArm(sx,f.a1,f.g1,0,f.m); drawArm(sx,f.a2,f.g2,1,f.m);
}

// ---------- coil-current bars ----------
function drawBars(fi){
  const f=D.frames[fi], n=CP.length, W2=bars.width, H2=bars.height;
  bx.clearRect(0,0,W2,H2);
  const midY=H2/2, bw=(W2-40)/n, scale=(H2/2-24)/D.ilim;
  bx.strokeStyle='#3a475e';bx.beginPath();bx.moveTo(20,midY);bx.lineTo(W2-10,midY);bx.stroke();
  bx.strokeStyle='#5a2b2b';bx.setLineDash([4,4]);
  [1,-1].forEach(s=>{bx.beginPath();bx.moveTo(20,midY-s*D.ilim*scale);bx.lineTo(W2-10,midY-s*D.ilim*scale);bx.stroke();});
  bx.setLineDash([]);
  for(let i=0;i<n;i++){
    const x=24+i*bw, v=f.cur[i], h=v*scale;
    bx.fillStyle=curColor(v); bx.fillRect(x,midY-Math.max(h,0),bw-6,Math.abs(h));
    bx.fillStyle='#9fb0c3';bx.font='11px sans-serif';bx.textAlign='center';
    bx.fillText('C'+D.coil_ids[i],x+(bw-6)/2,H2-6);
    bx.fillStyle='#e6edf3';bx.fillText(v.toFixed(1),x+(bw-6)/2,midY-h-(h>=0?4:-12));
    bx.textAlign='left';
  }
  bx.fillStyle='#9fb0c3';bx.font='11px sans-serif';
  bx.fillText('+'+D.ilim,2,midY-D.ilim*scale+4);bx.fillText('-'+D.ilim,4,midY+D.ilim*scale+4);
}

// ---------- MFW panels (one per motor) ----------
const PS=185, mfwRow=document.getElementById('mfwrow'), panels=[], pctx=[], titles=[];
for(let j=0;j<6;j++){
  const wrap=document.createElement('div');
  const t=document.createElement('div'); t.className='lbl'; t.style.textAlign='center'; t.style.marginBottom='3px';
  const cv=document.createElement('canvas'); cv.width=PS; cv.height=PS;
  wrap.appendChild(t); wrap.appendChild(cv); mfwRow.appendChild(wrap);
  panels.push(cv); pctx.push(cv.getContext('2d')); titles.push(t);
}
// fixed per-panel field scale (max reachable extent over the run)
const pScale=[];
for(let j=0;j<6;j++){
  let mx=D.bmin[j]*1.4;
  for(let fi=0; fi<D.frames.length; fi+=5){
    let s=0; actMatrix(motorPos(fi,j)).forEach(c=>s+=Math.hypot(c[0],c[1])*D.box_imax);
    if(s>mx)mx=s;
  }
  pScale[j]=mx*1.06;
}
const trails=[[],[],[],[],[],[]];
function drawMFW(fi){
  const act=D.frames[fi].m;
  for(let j=0;j<6;j++){
    const ctx=pctx[j], R=PS/2-12, C=PS/2, sc=pScale[j];
    const MX=x=>C+x/sc*R, MY=y=>C-y/sc*R;
    ctx.clearRect(0,0,PS,PS);
    ctx.strokeStyle='#243049';ctx.lineWidth=1;
    ctx.beginPath();ctx.moveTo(8,C);ctx.lineTo(PS-8,C);ctx.moveTo(C,8);ctx.lineTo(C,PS-8);ctx.stroke();
    // MFW zonotope
    const gens=actMatrix(motorPos(fi,j)).map(c=>[c[0]*D.box_imax,c[1]*D.box_imax]);
    const vs=zono(gens);
    ctx.beginPath(); vs.forEach((v,i)=>{const X=MX(v[0]),Y=MY(v[1]); i?ctx.lineTo(X,Y):ctx.moveTo(X,Y);});
    ctx.closePath(); ctx.fillStyle='rgba(84,140,220,0.16)'; ctx.fill();
    ctx.strokeStyle='#5a8fd6'; ctx.lineWidth=1.3; ctx.stroke();
    // b_min circle
    ctx.beginPath(); ctx.arc(C,C,D.bmin[j]/sc*R,0,7);
    ctx.strokeStyle='#3fd18a'; ctx.lineWidth=2; ctx.stroke();
    // field trail + live dot
    const tr=trails[j];
    ctx.strokeStyle=MOTOR_COL[j]; ctx.lineWidth=2; ctx.beginPath();
    tr.forEach((p,i)=>{const X=MX(p[0]),Y=MY(p[1]); i?ctx.lineTo(X,Y):ctx.moveTo(X,Y);}); ctx.stroke();
    const B=fieldAt(fi,j);
    ctx.beginPath(); ctx.arc(MX(B[0]),MY(B[1]),4.5,0,7);
    ctx.fillStyle=MOTOR_COL[j]; ctx.fill(); ctx.strokeStyle='#fff'; ctx.lineWidth=1.2; ctx.stroke();
    // zoomed inset: field vs b_min circle (the selectivity detail)
    const IN=64, ix=PS-IN-5, iy=PS-IN-5, ic=IN/2, iR=IN/2-5, isc=D.bmin[j]*2.6;
    ctx.save();
    ctx.fillStyle='rgba(9,14,26,0.94)'; ctx.fillRect(ix,iy,IN,IN);
    ctx.strokeStyle='#39465e'; ctx.lineWidth=1; ctx.strokeRect(ix,iy,IN,IN);
    ctx.beginPath(); ctx.rect(ix,iy,IN,IN); ctx.clip();
    const IX=x=>ix+ic+x/isc*iR, IY=y=>iy+ic-y/isc*iR;
    ctx.beginPath(); ctx.arc(ix+ic,iy+ic,D.bmin[j]/isc*iR,0,7);
    ctx.strokeStyle='#3fd18a'; ctx.lineWidth=1.6; ctx.stroke();
    ctx.strokeStyle=MOTOR_COL[j]; ctx.lineWidth=1.6; ctx.beginPath();
    tr.forEach((p,i)=>{const X=IX(p[0]),Y=IY(p[1]); i?ctx.lineTo(X,Y):ctx.moveTo(X,Y);}); ctx.stroke();
    ctx.beginPath(); ctx.arc(IX(B[0]),IY(B[1]),3,0,7);
    ctx.fillStyle=MOTOR_COL[j]; ctx.fill(); ctx.strokeStyle='#fff'; ctx.lineWidth=1; ctx.stroke();
    ctx.restore();
    ctx.fillStyle='#8598ad'; ctx.font='8px sans-serif'; ctx.textAlign='left';
    ctx.fillText('b_min zoom', ix+2, iy-3);
    const rot=(j===act);
    titles[j].innerHTML='M'+(j+1)+(rot?' &mdash; <b style="color:#ffe600">rotating</b>':' &mdash; held');
    panels[j].style.outline=rot?'2px solid #ffe600':'1px solid #223';
  }
}

let cur=0, playing=true;
function updateTrails(fi){
  for(let j=0;j<6;j++){ trails[j].push(fieldAt(fi,j)); if(trails[j].length>28)trails[j].shift(); }
}
function render(){
  updateTrails(cur); drawScene(cur); drawBars(cur); drawMFW(cur);
  slider.value=cur;
  const m=D.frames[cur].m;
  activeEl.textContent='Motor '+(m+1);
  activeEl.style.background=MOTOR_COL[m]; activeEl.style.color='#0f1420';
  clockEl.textContent='frame '+cur+' / '+(D.frames.length-1);
}
function loop(){ if(playing){cur=(cur+1)%D.frames.length;render();} setTimeout(loop,45); }
slider.addEventListener('input',()=>{cur=+slider.value; trails.forEach(t=>t.length=0); render();});
playBtn.addEventListener('click',()=>{playing=!playing;playBtn.textContent=playing?'Pause':'Play';});
render();loop();
</script>
</body></html>
"""


def main():
    frames, coils = poses_and_currents()
    html = build_html(frames, coils)
    out = ROOT / "arm_control_sim.html"
    out.write_text(html)
    peak = max(max(abs(c) for c in fr["cur"]) for fr in frames)
    print(f"wrote {out}  ({len(frames)} frames, peak current {peak:.1f} A)")


if __name__ == "__main__":
    main()
