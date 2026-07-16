"""Generate a standalone interactive *3D* simulation of the dual-arm platform.

Live physics port of case study II (``case_study_dualarm.py``) to a single
self-contained HTML file rendered with three.js (vendored in
``scripts/vendor/``, inlined -- no CDN, works offline):

* the dipole field model, the spillover-minimizing tracking controller
  ``i = S(p) b_m`` and every magnet's rotational dynamics (inertia, viscous
  damping, b_min Coulomb friction, 1:50 gearbox) are re-implemented in JS and
  integrated in real time (RK4), so the arms move *because the physics moves
  them* -- held joints creep exactly as far as spillover allows;
* 3D scene: the eight solenoids of the operating set (posed via
  ``COIL_POSES``, winding glow = live current), both arms, spinning
  input-magnet markers, rotating local-field arrows, orbit/zoom camera;
* monitors: live per-coil current bars, and one MFW panel per motor showing
  the reachable-field zonotope A(p)*CFW under the box current constraint
  |i|_inf <= BOX_IMAX together with the minimal actuation circle b_min, the
  minimal supporting distance (MSD) and the circle-enclosure selectivity test;
* a parity check computed here with the *Python* model (``A_planar`` /
  ``spillover_S``) is embedded and re-evaluated by the JS model at load, so a
  broken port is flagged in the page footer.

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
    OMEGA, JR, CR, EPS, GEAR, XB, SEP, YAW, BASE_Z, L1, L2, DZ,
    COIL_POSES, CORE_R, WIND_R, CORE_TIP, WIND_LEN,
    ARM_COLORS, MOTOR_COLORS,
)

VENDOR = ROOT / "scripts" / "vendor"
CURRENT_LIMIT = 17.0   # hardware per-coil limit (A), shown on the bars
BOX_IMAX = 15.0        # current bound used for the reachable-field (box) MFW
LAM_SIM = 3e-6         # current regularization, the case-study operating value.
                       # The old kinematic animation used 2.5e-5 (lower currents),
                       # but with real magnet dynamics that weakens spillover
                       # suppression enough that driving M2/M5 spins neighbouring
                       # magnets past their friction threshold (joints creep away
                       # from their goals). 3e-6 keeps every held joint still.
GRIP_DEG = 30.0        # gripper output travel (deg) mapped to grip 1 -> 0

# per-arm (alpha, beta, grip) start -> goal; grip 1 = open, 0 = closed
START = dict(a1=[35, -45, 1.0], a2=[35, -45, 1.0])
GOAL = dict(a1=[-15, 25, 0.0], a2=[-15, 25, 0.0])
# auto sequence: (motor index 0..5, which arm 'a1'/'a2', var 0=alpha 1=beta 2=grip).
# Order is a planning freedom (cf. the paper's planner): driving the elbows
# (M2/M5, b_min = 30 mT, the most field-hungry motors) first, while both arms
# are still at the START pose, keeps the peak current at 15.6 A (<= 17 A limit)
# with lam = 3e-6. The naive arm-by-arm order needs 26.8 A driving M5 once
# arm 1 is parked at its goal.
SEQUENCE = [(1, "a1", 1), (4, "a2", 1), (0, "a1", 0),
            (3, "a2", 0), (2, "a1", 2), (5, "a2", 2)]


def parity_currents(coils):
    """Controller currents at the START pose from the *Python* physics.

    Motor 1 actuated, field phase theta = 0, lam = LAM_SIM. The JS port must
    reproduce these numbers to machine precision.
    """
    mA = arm((XB, SEP, BASE_Z), YAW, START["a1"][0], START["a1"][1])
    mB = arm((XB, -SEP, BASE_Z), YAW, START["a2"][0], START["a2"][1])
    A = [A_planar(p, coils) for p in mA + mB]
    S = spillover_S(A, 0, LAM_SIM)
    return [float(c) for c in current(S, DRIVE_K * RADII[0], 0.0)]


def build_html():
    coils = ALL_COILS[COILS_IDX]
    data = {
        "coil_params": [[float(x) for x in c] for c in coils],
        "coil_ids": [i + 1 for i in COILS_IDX],
        "coil_poses": {str(i + 1): COIL_POSES[i + 1] for i in COILS_IDX},
        "coil_geom": {"core_r": CORE_R, "wind_r": WIND_R,
                      "core_tip": CORE_TIP, "wind_len": WIND_LEN},
        "bmin": list(RADII),
        "ilim": CURRENT_LIMIT,
        "box_imax": BOX_IMAX,
        "lambda": LAM_SIM,
        "drive_k": DRIVE_K,
        "omega": OMEGA,
        "grip_deg": GRIP_DEG,
        "dyn": {"jr": JR, "cr": CR, "eps": EPS, "gear": GEAR},
        "geom": {"xb": XB, "sep": SEP, "yaw": YAW, "base_z": BASE_Z,
                 "l1": L1, "l2": L2, "dz": DZ},
        "start": START, "goal": GOAL, "sequence": SEQUENCE,
        "arm_colors": list(ARM_COLORS), "motor_colors": list(MOTOR_COLORS),
        "check": {"m": 0, "cur": parity_currents(coils)},
    }
    html = _HTML_TEMPLATE
    html = html.replace("//__THREE_SRC__", (VENDOR / "three.min.js").read_text(encoding="utf-8"))
    html = html.replace("//__ORBIT_SRC__", (VENDOR / "OrbitControls.js").read_text(encoding="utf-8"))
    html = html.replace("/*__DATA__*/null", json.dumps(data))
    return html


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Dual-arm selective control &mdash; 3D live simulation</title>
<style>
  body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#0f1420;color:#e6edf3}
  h1{font-size:22.95px;margin:2px 0 6px}
  h2{font-size:20.25px;margin:2px 0 6px}
  .sub{color:#9fb0c3;font-size:17.55px;margin-bottom:8px;max-width:1250px}
  #page{padding:12px 14px 18px}
  #ctrls{display:flex;align-items:center;gap:7px;flex-wrap:wrap;margin:8px 0}
  button{background:#2b3c55;color:#e6edf3;border:0;border-radius:6px;padding:6px 11px;cursor:pointer;font-size:17.55px}
  button:hover{background:#3a4f6f}
  button.primary{background:#2b6cb0}button.primary:hover{background:#3182ce}
  button.on{background:#2b6cb0;outline:1px solid #7db4e8}
  button.mbtn.on{outline:2px solid #ffe600}
  select,input[type=range]{accent-color:#3182ce}
  select{background:#1b2436;color:#e6edf3;border:1px solid #34435e;border-radius:5px;padding:4px;font-size:17.55px}
  .sep{width:1px;height:33px;background:#2c3950;margin:0 3px}
  .lbl{color:#9fb0c3;font-size:16.2px}
  .tag{display:inline-block;padding:2px 8px;border-radius:5px;font-weight:600;font-size:17.55px}
  #seginfo{color:#cfe0f0;font-size:17.55px;margin:2px 0 8px;min-height:26px}
  #mainrow{display:flex;gap:12px;flex-wrap:wrap}
  #view3d{position:relative;flex:2 1 620px;height:560px;min-width:480px;background:#141b2d;border-radius:10px;overflow:hidden}
  #view3d canvas{display:block}
  #legend{position:absolute;left:10px;bottom:8px;font-size:15.525px;color:#9fb0c3;pointer-events:none;
          background:rgba(15,20,32,.55);padding:4px 8px;border-radius:6px;line-height:1.5}
  .chip{display:inline-block;width:13.5px;height:13.5px;border-radius:3px;margin:0 4.5px 0 9px}
  #side{flex:1 1 580px;min-width:570px;display:flex;flex-direction:column;gap:10px}
  .panel{background:#141b2d;border-radius:10px;padding:10px 12px}
  canvas.mon{background:#101626;border-radius:8px}
  table{border-collapse:collapse;font-size:16.875px;width:100%}
  th{color:#9fb0c3;font-weight:600;text-align:left;padding:2px 6px;border-bottom:1px solid #2c3950}
  td{padding:2.5px 6px;border-bottom:1px solid #1d2740;font-variant-numeric:tabular-nums}
  tr.act td{background:#26314b}
  #stats{font-size:16.875px;color:#cfe0f0;line-height:1.6}
  #mfwwrap{margin-top:14px}
  #mfwrow{display:flex;gap:8px;flex-wrap:wrap}
  #mfwrow canvas{background:#141b2d;border-radius:8px}
  #foot{margin-top:10px;color:#7b8ba0;font-size:15.525px}
</style></head>
<body>
<div id="page">
  <h1>Dual-arm selective control &mdash; 3D simulation</h1>
  <div class="sub">8 coils control 6 motors. Drag to orbit; scroll to zoom.</div>

  <div id="ctrls">
    <button id="playBtn" class="primary">Pause</button>
    <button id="resetBtn">Reset</button>
    <span class="sep"></span>
    <button id="autoBtn" class="on">Auto</button>
    <span class="lbl">Motor:</span>
    <button class="mbtn" data-m="0">M1</button><button class="mbtn" data-m="1">M2</button>
    <button class="mbtn" data-m="2">M3</button><button class="mbtn" data-m="3">M4</button>
    <button class="mbtn" data-m="4">M5</button><button class="mbtn" data-m="5">M6</button>
    <button id="dirBtn">dir: CCW</button>
    <button id="stopBtn">Stop</button>
    <span class="sep"></span>
    <span class="lbl">speed &times;</span>
    <select id="speedSel"><option>1</option><option selected>2</option><option>4</option><option>8</option></select>
    <span class="lbl">&lambda;</span>
    <input id="lamSlider" type="range" min="-6" max="-4" step="0.05" style="width:110px">
    <span class="lbl" id="lamVal"></span>
  </div>
  <div id="seginfo"></div>

  <div id="mainrow">
    <div id="view3d">
      <div id="legend">
        <span id="jointLegend"></span><br>
        <span class="chip" style="background:#b86b45"></span>coil current<br>
        yellow = active &middot; arrow = B &middot; tick = angle
      </div>
    </div>
    <div id="side">
      <div class="panel">
        <h2>Coil currents (A)</h2>
        <div class="sub" style="margin-bottom:6px">Limit &plusmn;<span id="ilimtxt"></span>&nbsp;A.</div>
        <canvas id="bars" class="mon" width="540" height="240"></canvas>
      </div>
      <div class="panel">
        <h2>Motors <span class="lbl">(active: <span class="tag" id="activeTag">&mdash;</span>)</span></h2>
        <table>
          <thead><tr><th>M</th><th>Joint</th><th>Output</th><th>Turns</th><th>|B| mT</th><th>MSD/b<sub>min</sub></th></tr></thead>
          <tbody id="jt"></tbody>
        </table>
        <div id="stats"></div>
      </div>
    </div>
  </div>

  <div id="mfwwrap">
    <h2>MFW selectivity windows</h2>
    <div class="sub">Zoom near b<sub>min</sub> at &plusmn;<span id="boxtxt"></span>&nbsp;A.
      Green = enclosed; dot/trail = applied field.</div>
    <div id="mfwrow"></div>
  </div>

  <div id="foot"><span id="parity"></span></div>
</div>

<script>
//__THREE_SRC__
</script>
<script>
//__ORBIT_SRC__
</script>
<script>
'use strict';
const D = /*__DATA__*/null;
const CP=D.coil_params, IDS=D.coil_ids, NC=CP.length, NM=6;
const GEOM=D.geom, DYN=D.dyn, SEQ=D.sequence;
const MOTOR_COL=D.motor_colors, ARM_HEX=D.arm_colors;
const TEXT_SCALE=1.35;
const rad=d=>d*Math.PI/180, deg=r=>r*180/Math.PI;
const clamp01=x=>Math.max(0,Math.min(1,x));
const legendChip=(j)=>'<span class="chip" style="background:'+MOTOR_COL[j]+'"></span>M'+(j+1);
document.getElementById('jointLegend').innerHTML=
  'Arm 1: '+[0,1,2].map(legendChip).join(' ')+
  ' &nbsp; Arm 2: '+[3,4,5].map(legendChip).join(' ');
document.querySelectorAll('.mbtn').forEach(b=>{
  b.style.borderBottom='3px solid '+MOTOR_COL[+b.dataset.m];
});
document.getElementById('ilimtxt').textContent=D.ilim;
document.getElementById('boxtxt').textContent=D.box_imax;

// ================= linear algebra =================
function zeros(n){return new Float64Array(n);}
function matInv(M){                       // Gauss-Jordan with partial pivoting
  const n=M.length;
  const A=M.map((row,i)=>{const r=Array.from(row),e=new Array(n).fill(0);e[i]=1;return r.concat(e);});
  for(let c=0;c<n;c++){
    let p=c;for(let r=c+1;r<n;r++)if(Math.abs(A[r][c])>Math.abs(A[p][c]))p=r;
    const t=A[c];A[c]=A[p];A[p]=t;
    const d=A[c][c];for(let k=c;k<2*n;k++)A[c][k]/=d;
    for(let r=0;r<n;r++){if(r===c)continue;const f=A[r][c];if(!f)continue;
      for(let k=c;k<2*n;k++)A[r][k]-=f*A[c][k];}
  }
  return A.map(r=>r.slice(n));
}

// ================= physics (port of case_study_dualarm.py) =================
function dipoleCol(cp,pos){               // [Bx,By] per amp for one coil at pos
  const dx=pos[0]-cp[3], dy=pos[1]-cp[4], dz=pos[2]-cp[5];
  const r=Math.sqrt(dx*dx+dy*dy+dz*dz)+1e-9, r3=r*r*r, r5=r3*r*r;
  const dot=cp[0]*dx+cp[1]*dy+cp[2]*dz, c=1e-7;
  return [c*(3*dx*dot/r5-cp[0]/r3), c*(3*dy*dot/r5-cp[1]/r3)];
}
function computeA(pos){                   // planar actuation matrix, rows [Bx],[By]
  const bx=new Array(NC), by=new Array(NC);
  for(let k=0;k<NC;k++){const c=dipoleCol(CP[k],pos);bx[k]=c[0];by[k]=c[1];}
  return [bx,by];
}
function spilloverS(A,m,lam){             // i = S b_m minimizing off-target fields
  const M=[];for(let r=0;r<NC;r++){M.push(new Array(NC).fill(0));M[r][r]=lam;}
  for(let j=0;j<NM;j++){
    if(j===m)continue;
    const bx=A[j][0],by=A[j][1];
    for(let r=0;r<NC;r++)for(let c=0;c<NC;c++)M[r][c]+=bx[r]*bx[c]+by[r]*by[c];
  }
  const Mi=matInv(M), ax=A[m][0], ay=A[m][1];
  const P=[];                             // Minv @ Am^T   (NC x 2)
  for(let r=0;r<NC;r++){let px=0,py=0;
    for(let c=0;c<NC;c++){px+=Mi[r][c]*ax[c];py+=Mi[r][c]*ay[c];}
    P.push([px,py]);}
  let g00=0,g01=0,g10=0,g11=0;            // Am @ P        (2 x 2)
  for(let c=0;c<NC;c++){g00+=ax[c]*P[c][0];g01+=ax[c]*P[c][1];
                        g10+=ay[c]*P[c][0];g11+=ay[c]*P[c][1];}
  const det=g00*g11-g01*g10;
  const i00=g11/det,i01=-g01/det,i10=-g10/det,i11=g00/det;
  return P.map(p=>[p[0]*i00+p[1]*i10, p[0]*i01+p[1]*i11]);
}
function currentsOf(S,r,th){
  const cx=r*Math.cos(th), cy=r*Math.sin(th);
  return Float64Array.from(S,s=>s[0]*cx+s[1]*cy);
}
function armPts(bx,by,bz,yaw,alpha,beta){ // = arm() of the case study
  const t1=rad(alpha)-Math.PI/2+rad(yaw);
  const m2=[bx+GEOM.l1*Math.cos(t1), by+GEOM.l1*Math.sin(t1), bz-GEOM.dz];
  const t2=t1+rad(beta);
  return [[bx,by,bz], m2,
          [m2[0]+GEOM.l2*Math.cos(t2), m2[1]+GEOM.l2*Math.sin(t2), bz]];
}

// ---- state: input-magnet angles/speeds; joints derive from them via the gearbox
const phi=zeros(NM), w=zeros(NM), phi0=zeros(NM);
let theta=0, simT=0, dir=1, active=-1, mode='auto', seg=0, paused=false;
let speed=2, lam=D.lambda, lastCur=zeros(NC), peakCur=0;
let motors=[], Alist=[];

function jointVals(){                     // raw (unclamped) joint values
  const dout=j=>deg(phi[j]-phi0[j])/DYN.gear;
  return {a1:[D.start.a1[0]+dout(0), D.start.a1[1]+dout(1), D.start.a1[2]+dout(2)/D.grip_deg],
          a2:[D.start.a2[0]+dout(3), D.start.a2[1]+dout(4), D.start.a2[2]+dout(5)/D.grip_deg]};
}
function motorPositions(jv){
  return armPts(GEOM.xb, GEOM.sep,GEOM.base_z,GEOM.yaw,jv.a1[0],jv.a1[1])
 .concat(armPts(GEOM.xb,-GEOM.sep,GEOM.base_z,GEOM.yaw,jv.a2[0],jv.a2[1]));
}
function magnetAcc(j,ph,ww,Bx,By){        // = magnet_rhs() of the case study
  const mag=Math.hypot(Bx,By), ang=Math.atan2(By,Bx);
  return (mag*Math.sin(ang-ph)-DYN.cr*ww-D.bmin[j]*Math.tanh(ww/DYN.eps))/DYN.jr;
}
function rk4(j,Bx,By,h){
  const p=phi[j], v=w[j];
  const a1=magnetAcc(j,p,v,Bx,By);
  const v2=v+0.5*h*a1, a2=magnetAcc(j,p+0.5*h*v ,v2,Bx,By);
  const v3=v+0.5*h*a2, a3=magnetAcc(j,p+0.5*h*v2,v3,Bx,By);
  const v4=v+h*a3,     a4=magnetAcc(j,p+h*v3    ,v4,Bx,By);
  phi[j]=p+h/6*(v+2*v2+2*v3+v4);
  w[j]=v+h/6*(a1+2*a2+2*a3+a4);
}
function stepFrame(dt){                   // dt in sim-seconds
  motors=motorPositions(jointVals());
  Alist=motors.map(computeA);             // pose changes slowly: freeze A over the frame
  let S=null, rd=0;
  if(active>=0){S=spilloverS(Alist,active,lam);rd=D.drive_k*D.bmin[active];}
  const nSub=Math.max(1,Math.ceil(dt/0.0015)), h=dt/nSub;
  let cur=zeros(NC);
  for(let s=0;s<nSub;s++){
    if(S){theta+=dir*D.omega*h;cur=currentsOf(S,rd,theta);}
    for(let j=0;j<NM;j++){
      const bx=Alist[j][0], by=Alist[j][1];
      let Bx=0,By=0;for(let k=0;k<NC;k++){Bx+=bx[k]*cur[k];By+=by[k]*cur[k];}
      rk4(j,Bx,By,h);
    }
  }
  lastCur=cur; simT+=dt;
  for(let k=0;k<NC;k++)if(Math.abs(cur[k])>peakCur)peakCur=Math.abs(cur[k]);
}
function fieldAt(j){
  const bx=Alist[j][0], by=Alist[j][1];
  let X=0,Y=0;for(let k=0;k<NC;k++){X+=bx[k]*lastCur[k];Y+=by[k]*lastCur[k];}
  return [X,Y];
}

// ---- auto-sequence controller: drive each joint to its goal, one motor at a time
function autoCtrl(){
  if(seg>=SEQ.length){active=-1;return;}
  const s=SEQ[seg], m=s[0], ak=s[1], vi=s[2];
  const cur=jointVals()[ak][vi], tgt=D.goal[ak][vi];
  const tol=vi===2?0.02:0.4;
  if(Math.abs(tgt-cur)<=tol){seg++;active=-1;return;}
  active=m; dir=tgt>cur?1:-1;
}

// ================= zonotope MFW + MSD =================
function zonoVerts(g0){                   // boundary of {sum c_k g_k : |c_k|<=1}
  let g=g0.map(v=>(v[1]<0||(v[1]===0&&v[0]<0))?[-v[0],-v[1]]:[v[0],v[1]]);
  g.sort((a,b)=>Math.atan2(a[1],a[0])-Math.atan2(b[1],b[0]));
  let c=[0,0];g.forEach(v=>{c[0]-=v[0];c[1]-=v[1];});
  const vs=[[c[0],c[1]]];
  g.forEach(v=>{c=[c[0]+2*v[0],c[1]+2*v[1]];vs.push([c[0],c[1]]);});
  g.forEach(v=>{c=[c[0]-2*v[0],c[1]-2*v[1]];vs.push([c[0],c[1]]);});
  return vs;
}
function msd(gens){                       // minimal supporting distance (exact in 2D)
  let best=Infinity;
  for(const g of gens){
    const L=Math.hypot(g[0],g[1]);if(L<1e-15)continue;
    const nx=-g[1]/L, ny=g[0]/L;
    let s=0;for(const q of gens)s+=Math.abs(q[0]*nx+q[1]*ny);
    if(s<best)best=s;
  }
  return best;
}
function mfwGens(j){
  const bx=Alist[j][0], by=Alist[j][1], g=[];
  for(let k=0;k<NC;k++)g.push([bx[k]*D.box_imax, by[k]*D.box_imax]);
  return g;
}

// ================= three.js scene =================
const view=document.getElementById('view3d');
const renderer=new THREE.WebGLRenderer({antialias:true});
renderer.setPixelRatio(Math.min(devicePixelRatio,2));
view.insertBefore(renderer.domElement,view.firstChild);
const scene=new THREE.Scene();scene.background=new THREE.Color(0x121a2c);
const camera=new THREE.PerspectiveCamera(42,1,0.005,5);
camera.up.set(0,0,1);camera.position.set(0.235,-0.255,0.185);
const controls=new THREE.OrbitControls(camera,renderer.domElement);
controls.target.set(0,0,-0.05);controls.enableDamping=true;controls.dampingFactor=0.08;
controls.minDistance=0.06;controls.maxDistance=1.2;
function sizeView(){const wpx=view.clientWidth,hpx=view.clientHeight;
  renderer.setSize(wpx,hpx);camera.aspect=wpx/hpx;camera.updateProjectionMatrix();}
window.addEventListener('resize',sizeView);
scene.add(new THREE.HemisphereLight(0xdde7ff,0x0c1120,0.95));
const dl=new THREE.DirectionalLight(0xffffff,0.65);dl.position.set(0.3,-0.5,0.6);scene.add(dl);
const dl2=new THREE.DirectionalLight(0x88aaff,0.25);dl2.position.set(-0.4,0.3,0.3);scene.add(dl2);
const grid=new THREE.GridHelper(0.6,24,0x2a3653,0x1d2740);
grid.rotation.x=Math.PI/2;grid.position.z=-0.148;scene.add(grid);
const axes=new THREE.AxesHelper(0.035);axes.position.set(0.155,-0.155,-0.147);scene.add(axes);

const YUP=new THREE.Vector3(0,1,0), _dir=new THREE.Vector3();
function setCyl(mesh,p0,p1){
  const dx=p1[0]-p0[0],dy=p1[1]-p0[1],dz=p1[2]-p0[2];
  const L=Math.hypot(dx,dy,dz)||1e-9;
  mesh.position.set((p0[0]+p1[0])/2,(p0[1]+p1[1])/2,(p0[2]+p1[2])/2);
  mesh.scale.set(1,L,1);
  mesh.quaternion.setFromUnitVectors(YUP,_dir.set(dx/L,dy/L,dz/L));
}
function textSprite(txt,color,scale){
  const cv=document.createElement('canvas');cv.width=256;cv.height=96;
  const c=cv.getContext('2d');c.font='bold 52px sans-serif';
  c.textAlign='center';c.textBaseline='middle';
  c.lineWidth=9;c.strokeStyle='rgba(10,14,24,0.9)';c.strokeText(txt,128,48);
  c.fillStyle=color;c.fillText(txt,128,48);
  const sp=new THREE.Sprite(new THREE.SpriteMaterial(
    {map:new THREE.CanvasTexture(cv),transparent:true,depthTest:false}));
  sp.scale.set(scale*TEXT_SCALE,scale*TEXT_SCALE*96/256,1);return sp;
}
function stdMat(col){return new THREE.MeshStandardMaterial({color:col,roughness:.55,metalness:.15});}

// ---- coils (schematic solenoid: exposed core tip + copper winding barrel)
const coilMats=[];
IDS.forEach(id=>{
  const P=D.coil_poses[String(id)], G=D.coil_geom;
  const a=new THREE.Vector3(P[0][0],P[0][1],P[0][2]).normalize();
  const top=new THREE.Vector3(P[1][0],P[1][1],P[1][2]);
  const head=top.clone().addScaledVector(a,-G.core_tip);
  const back=head.clone().addScaledVector(a,-G.wind_len);
  const q=new THREE.Quaternion().setFromUnitVectors(YUP,a);
  const core=new THREE.Mesh(new THREE.CylinderGeometry(G.core_r,G.core_r,G.core_tip,24),
    new THREE.MeshStandardMaterial({color:0xd8d5ca,roughness:.45,metalness:.5}));
  core.position.copy(top.clone().add(head).multiplyScalar(0.5));core.quaternion.copy(q);scene.add(core);
  // translucent winding (as in the paper's layout figure) so the arms stay visible
  const wm=new THREE.MeshStandardMaterial({color:0xb86b45,roughness:.6,metalness:.25,
    transparent:true,opacity:0.82});
  const wind=new THREE.Mesh(new THREE.CylinderGeometry(G.wind_r,G.wind_r,G.wind_len,32),wm);
  wind.position.copy(head.clone().add(back).multiplyScalar(0.5));wind.quaternion.copy(q);scene.add(wind);
  const ring=new THREE.Mesh(new THREE.RingGeometry(G.core_r,G.wind_r,32),
    new THREE.MeshStandardMaterial({color:0x8a4a2c,side:THREE.DoubleSide,roughness:.7}));
  ring.position.copy(head);
  ring.quaternion.setFromUnitVectors(new THREE.Vector3(0,0,1),a);scene.add(ring);
  const lb=textSprite('C'+id,'#ffd9a8',0.028);
  lb.position.copy(top.clone().addScaledVector(a,0.013));scene.add(lb);
  coilMats.push(wm);
});

// ---- arms + per-motor visuals
const arms=[], motorVis=[], armLabels=[];
for(let ai=0;ai<2;ai++){
  const col=new THREE.Color(ARM_HEX[ai]);
  const gripCol=new THREE.Color(MOTOR_COL[ai*3+2]);
  const pale=col.clone().lerp(new THREE.Color(0xffffff),0.55);
  const A={
    base:new THREE.Mesh(new THREE.BoxGeometry(0.024,0.024,0.008),stdMat(0x4a4f5c)),
    pillar:new THREE.Mesh(new THREE.CylinderGeometry(0.006,0.006,1,16),stdMat(0x6a7080)),
    link1:new THREE.Mesh(new THREE.CylinderGeometry(0.006,0.006,1,16),stdMat(pale)),
    link2:new THREE.Mesh(new THREE.CylinderGeometry(0.006,0.006,1,16),stdMat(pale)),
    prongs:[new THREE.Mesh(new THREE.CylinderGeometry(0.0018,0.0018,1,10),stdMat(gripCol)),
            new THREE.Mesh(new THREE.CylinderGeometry(0.0018,0.0018,1,10),stdMat(gripCol))],
  };
  Object.values(A).flat().forEach(m=>scene.add(m));
  arms.push(A);
  const al=textSprite('Arm '+(ai+1),ai?'#ff8a8a':'#66b8ff',0.040);
  scene.add(al);armLabels.push(al);
  for(let k=0;k<3;k++){
    const j=ai*3+k, r=k<2?0.0085:0.0072, hh=k<2?0.012:0.010;
    const mat=stdMat(new THREE.Color(MOTOR_COL[j]));
    const cyl=new THREE.Mesh(new THREE.CylinderGeometry(r,r,hh,24),mat);
    cyl.quaternion.setFromUnitVectors(YUP,new THREE.Vector3(0,0,1));scene.add(cyl);
    const grp=new THREE.Group();                              // input-magnet angle tick
    const tick=new THREE.Mesh(new THREE.BoxGeometry(r*1.9,0.0024,0.0026),
      new THREE.MeshBasicMaterial({color:0xffffff}));
    tick.position.set(r*0.95,0,hh/2+0.0014);grp.add(tick);scene.add(grp);
    const arrow=new THREE.ArrowHelper(new THREE.Vector3(1,0,0),new THREE.Vector3(),0.02,
      MOTOR_COL[j],0.007,0.0045);scene.add(arrow);
    const ringA=new THREE.Mesh(new THREE.TorusGeometry(0.0125,0.0013,10,36),
      new THREE.MeshBasicMaterial({color:0xffe600}));
    ringA.visible=false;scene.add(ringA);
    const label=textSprite('M'+(j+1),MOTOR_COL[j],0.030);scene.add(label);
    motorVis.push({cyl,mat,grp,arrow,ringA,label});
  }
}
function updateScene(jv){
  for(let ai=0;ai<2;ai++){
    const m1=motors[ai*3], m2=motors[ai*3+1], m3=motors[ai*3+2], A=arms[ai];
    A.base.position.set(m1[0],m1[1],m1[2]-0.010);
    setCyl(A.pillar,[m1[0],m1[1],m1[2]-0.006],m1);
    setCyl(A.link1,m1,m2);setCyl(A.link2,m2,m3);
    const dx=m3[0]-m2[0],dy=m3[1]-m2[1];const L=Math.hypot(dx,dy)||1e-9;
    const d=[dx/L,dy/L,0], perp=[-d[1],d[0],0];
    const grip=clamp01(jv['a'+(ai+1)][2]), spread=0.002+0.004*grip;
    [[0,1],[1,-1]].forEach(ps=>{
      const s=ps[1];
      const root=[m3[0]+perp[0]*s*spread,m3[1]+perp[1]*s*spread,m3[2]];
      const tip=[root[0]+d[0]*0.013+perp[0]*s*0.003,root[1]+d[1]*0.013+perp[1]*s*0.003,root[2]];
      setCyl(A.prongs[ps[0]],root,tip);
    });
    const cx=(m1[0]+m2[0]+m3[0])/3,cy=(m1[1]+m2[1]+m3[1])/3;
    armLabels[ai].position.set(cx,cy,m1[2]+0.062);
  }
  for(let j=0;j<NM;j++){
    const p=motors[j], V=motorVis[j];
    V.cyl.position.set(p[0],p[1],p[2]);
    V.grp.position.set(p[0],p[1],p[2]);V.grp.rotation.z=phi[j];
    V.label.position.set(p[0],p[1],p[2]+0.021);
    const B=fieldAt(j), mag=Math.hypot(B[0],B[1]);
    if(mag>2e-4){
      V.arrow.visible=true;
      V.arrow.position.set(p[0],p[1],p[2]+0.011);
      V.arrow.setDirection(_dir.set(B[0]/mag,B[1]/mag,0));
      const L=Math.min(0.007+mag*0.55,0.05);
      V.arrow.setLength(L,L*0.34,L*0.20);
    }else V.arrow.visible=false;
    V.mat.emissive.setHex(j===active?0x554400:0x000000);
    V.ringA.visible=(j===active);V.ringA.position.set(p[0],p[1],p[2]);
  }
}
function updateCoilGlow(){
  for(let k=0;k<NC;k++){
    const t=Math.max(-1,Math.min(1,lastCur[k]/D.ilim));
    if(t>=0)coilMats[k].emissive.setRGB(0.85*t,0.06*t,0.06*t);
    else coilMats[k].emissive.setRGB(0.06*-t,0.10*-t,0.90*-t);
  }
}

// ================= monitors =================
const bars=document.getElementById('bars'), bctx=bars.getContext('2d');
function drawBars(){
  const W=bars.width,H=bars.height;bctx.clearRect(0,0,W,H);
  const midY=H/2,bw=(W-60)/NC,scale=(H/2-39)/D.ilim;
  bctx.strokeStyle='#3a475e';bctx.beginPath();bctx.moveTo(30,midY);bctx.lineTo(W-12,midY);bctx.stroke();
  bctx.strokeStyle='#5a2b2b';bctx.setLineDash([4,4]);
  [1,-1].forEach(s=>{bctx.beginPath();bctx.moveTo(30,midY-s*D.ilim*scale);
    bctx.lineTo(W-12,midY-s*D.ilim*scale);bctx.stroke();});
  bctx.setLineDash([]);
  for(let i=0;i<NC;i++){
    const x=36+i*bw,v=lastCur[i],h=v*scale;
    const t=Math.max(-1,Math.min(1,v/D.ilim));
    bctx.fillStyle=t>=0?'rgb(255,'+Math.round(255*(1-t))+','+Math.round(255*(1-t))+')'
                       :'rgb('+Math.round(255*(1+t))+','+Math.round(255*(1+t))+',255)';
    bctx.fillRect(x,midY-Math.max(h,0),bw-9,Math.abs(h));
    bctx.fillStyle='#9fb0c3';bctx.font=(11*TEXT_SCALE)+'px sans-serif';bctx.textAlign='center';
    bctx.fillText('C'+IDS[i],x+(bw-9)/2,H-9);
    bctx.fillStyle='#e6edf3';bctx.fillText(v.toFixed(1),x+(bw-9)/2,midY-h-(h>=0?6:-18));
    bctx.textAlign='left';
  }
  bctx.fillStyle='#9fb0c3';bctx.font=(10*TEXT_SCALE)+'px sans-serif';
  bctx.fillText('+'+D.ilim,1,midY-D.ilim*scale+6);bctx.fillText('-'+D.ilim,2,midY+D.ilim*scale+6);
}

// Compact b_min zoom panels
const PS=160, mfwRow=document.getElementById('mfwrow'), pctx=[], ptitle=[], pcv=[];
for(let j=0;j<NM;j++){
  const wrap=document.createElement('div');
  const t=document.createElement('div');t.className='lbl';
  t.style.textAlign='center';t.style.marginBottom='3px';
  const cv=document.createElement('canvas');cv.width=PS;cv.height=PS;
  wrap.appendChild(t);wrap.appendChild(cv);mfwRow.appendChild(wrap);
  pctx.push(cv.getContext('2d'));ptitle.push(t);pcv.push(cv);
}
const trails=[[],[],[],[],[],[]];
function drawMFW(){
  for(let j=0;j<NM;j++){
    const ctx=pctx[j], R=PS/2-14, C=PS/2, sc=D.bmin[j]*2.6;
    const gens=mfwGens(j);
    const MX=x=>C+x/sc*R, MY=y=>C-y/sc*R;
    ctx.clearRect(0,0,PS,PS);
    ctx.fillStyle='rgba(9,14,26,0.94)';ctx.fillRect(0,0,PS,PS);
    ctx.strokeStyle='#39465e';ctx.lineWidth=1;ctx.strokeRect(0.5,0.5,PS-1,PS-1);
    const m=msd(gens), ok=m>=D.bmin[j];
    ctx.beginPath();ctx.arc(C,C,D.bmin[j]/sc*R,0,7);
    ctx.strokeStyle=ok?'#3fd18a':'#ff5a5a';ctx.lineWidth=2.2;ctx.stroke();
    const tr=trails[j];
    ctx.strokeStyle=MOTOR_COL[j];ctx.lineWidth=2.2;ctx.beginPath();
    tr.forEach((p,i)=>{const X=MX(p[0]),Y=MY(p[1]);i?ctx.lineTo(X,Y):ctx.moveTo(X,Y);});ctx.stroke();
    const B=fieldAt(j);
    ctx.beginPath();ctx.arc(MX(B[0]),MY(B[1]),4,0,7);
    ctx.fillStyle=MOTOR_COL[j];ctx.fill();ctx.strokeStyle='#fff';ctx.lineWidth=1;ctx.stroke();
    ctx.fillStyle=ok?'#3fd18a':'#ff5a5a';ctx.font=(9*TEXT_SCALE)+'px sans-serif';ctx.textAlign='left';
    ctx.fillText('MSD/b_min '+(m/D.bmin[j]).toFixed(2),7,16);
    ctx.fillStyle='#8598ad';ctx.font=(8*TEXT_SCALE)+'px sans-serif';
    ctx.fillText('b_min',C+D.bmin[j]/sc*R+4,C-4);
    const rot=(j===active);
    ptitle[j].innerHTML='M'+(j+1)+(rot?' &mdash; <b style="color:#ffe600">active</b>':' &mdash; held');
    pcv[j].style.outline=rot?'2px solid #ffe600':'1px solid #223';
  }
}

// motors table + stats
const jtBody=document.getElementById('jt');
const JOINT_NAMES=['A1 J1 (α)','A1 J2 (β)','A1 grip','A2 J1 (α)','A2 J2 (β)','A2 grip'];
const jtCells=[];
for(let j=0;j<NM;j++){
  const tr=document.createElement('tr'), cells=[];
  for(let c=0;c<6;c++){const td=document.createElement('td');tr.appendChild(td);cells.push(td);}
  cells[0].textContent='M'+(j+1);cells[0].style.color=MOTOR_COL[j];cells[0].style.fontWeight='600';
  cells[1].textContent=JOINT_NAMES[j];
  jtBody.appendChild(tr);jtCells.push({tr,cells});
}
const statsEl=document.getElementById('stats'), segEl=document.getElementById('seginfo');
const activeTag=document.getElementById('activeTag');
function updateStatus(jv){
  for(let j=0;j<NM;j++){
    const g=(j%3)===2, v=g?jv['a'+(j<3?1:2)][2]:jv['a'+(j<3?1:2)][j%3];
    const cs=jtCells[j].cells;
    cs[2].textContent=g?(clamp01(v)*100).toFixed(0)+'% open':v.toFixed(1)+'°';
    cs[3].textContent=((phi[j]-phi0[j])/(2*Math.PI)).toFixed(2);
    const B=fieldAt(j);cs[4].textContent=(Math.hypot(B[0],B[1])*1e3).toFixed(1);
    const r=msd(mfwGens(j))/D.bmin[j];
    cs[5].textContent=r.toFixed(2);cs[5].style.color=r>=1?'#3fd18a':'#ff5a5a';
    jtCells[j].tr.className=(j===active)?'act':'';
  }
  if(active>=0){activeTag.textContent='M'+(active+1);
    activeTag.style.background=MOTOR_COL[active];activeTag.style.color='#0f1420';}
  else{activeTag.textContent='—';activeTag.style.background='transparent';
    activeTag.style.color='#9fb0c3';}
  statsEl.innerHTML='t '+simT.toFixed(1)+' s &middot; turns '
    +(theta/(2*Math.PI)).toFixed(1)+' &middot; peak '+peakCur.toFixed(1)+' A'
    +(peakCur>D.ilim?' <b style="color:#ff5a5a">over!</b>':'');
  let s;
  if(mode==='auto'){
    if(seg>=SEQ.length)s='Complete — reset to replay';
    else{const q=SEQ[seg],nm=['α','β','grip'][q[2]];
      const tv=q[2]===2?(D.goal[q[1]][q[2]]*100).toFixed(0)+'%':D.goal[q[1]][q[2]].toFixed(1)+'°';
      s='Auto '+(seg+1)+'/'+SEQ.length+' · M'+(q[0]+1)+' '+q[1].toUpperCase()
        +' '+nm+' → '+tv;}
  }else s=active>=0?'Manual · M'+(active+1)+' '+(dir>0?'CCW':'CW'):'Manual · idle';
  segEl.textContent=s;
}

// ================= UI wiring =================
const playBtn=document.getElementById('playBtn'), autoBtn=document.getElementById('autoBtn');
const dirBtn=document.getElementById('dirBtn'), lamSlider=document.getElementById('lamSlider');
const lamVal=document.getElementById('lamVal');
const mbtns=Array.from(document.querySelectorAll('.mbtn'));
function refreshBtns(){
  autoBtn.className=mode==='auto'?'on':'';
  mbtns.forEach(b=>b.className='mbtn'+((mode==='manual'&&active===+b.dataset.m)?' on':''));
}
playBtn.onclick=()=>{paused=!paused;playBtn.textContent=paused?'Resume':'Pause';};
document.getElementById('resetBtn').onclick=()=>{
  phi.fill(0);w.fill(0);phi0.fill(0);theta=0;simT=0;seg=0;peakCur=0;
  lastCur=zeros(NC);trails.forEach(t=>t.length=0);
  if(mode!=='manual')active=-1;refreshBtns();};
autoBtn.onclick=()=>{mode='auto';seg=0;active=-1;refreshBtns();};
mbtns.forEach(b=>b.onclick=()=>{
  const m=+b.dataset.m, wasManual=(mode==='manual');
  mode='manual';
  active=(wasManual&&active===m)?-1:m;      // second click on same motor stops it
  if(active>=0)dir=+dirBtn.dataset.d;
  refreshBtns();});
dirBtn.dataset.d='1';
dirBtn.onclick=()=>{
  const d=dirBtn.dataset.d==='1'?'-1':'1';dirBtn.dataset.d=d;
  dirBtn.textContent='dir: '+(d==='1'?'CCW':'CW');
  if(mode==='manual')dir=+d;};
document.getElementById('stopBtn').onclick=()=>{mode='manual';active=-1;refreshBtns();};
document.getElementById('speedSel').onchange=e=>{speed=+e.target.value;};
lamSlider.value=Math.log10(D.lambda);
function showLam(){lamVal.textContent=lam.toExponential(1);}
lamSlider.oninput=()=>{lam=Math.pow(10,+lamSlider.value);showLam();};
showLam();refreshBtns();

// ================= parity check vs the Python model =================
(function(){
  motors=motorPositions(jointVals());
  Alist=motors.map(computeA);
  const S=spilloverS(Alist,D.check.m,D.lambda);
  const c=currentsOf(S,D.drive_k*D.bmin[D.check.m],0);
  let e=0;for(let k=0;k<NC;k++)e=Math.max(e,Math.abs(c[k]-D.check.cur[k]));
  const el=document.getElementById('parity');
  el.textContent='Parity Δi: '+e.toExponential(1)+' A';
  el.style.color=e<1e-6?'#3fd18a':'#ff5a5a';
})();

// ================= main loop =================
let lastT=performance.now();
function loop(t){
  const dtReal=Math.min((t-lastT)/1000,0.05);lastT=t;
  if(!paused){
    if(mode==='auto')autoCtrl();
    stepFrame(dtReal*speed);
    for(let j=0;j<NM;j++){
      trails[j].push(fieldAt(j));
      if(trails[j].length>40)trails[j].shift();
    }
  }
  const jv=jointVals();
  updateScene(jv);updateCoilGlow();
  drawBars();drawMFW();updateStatus(jv);
  controls.update();renderer.render(scene,camera);
  requestAnimationFrame(loop);
}
sizeView();
requestAnimationFrame(loop);
</script>
</body></html>
"""


def main():
    html = build_html()
    out = ROOT / "arm_control_sim.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}  ({len(html)/1024:.0f} KB, three.js embedded)")


if __name__ == "__main__":
    main()
