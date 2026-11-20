// The radar (0-10 score or Δ vs base) and its titled panel.
// ---- radar core: the 0-10 score (opts.paper) or this page's Δ vs base ----
function drawRadar(spokes,polys,opts){
  // paper mode: fixed range RADAR.min..RADAR.max (0 at the centre), the outer ring is 10. The centre is the base on
  // the behaviour spokes; a value below it is DRAWN at the centre (rr clamps), unmarked.
  // No |Δ| rings and no significance band. Δ mode: an auto-ranged scale in score units (0 = base).
  const paper=!!(opts.paper&&RADAR);
  const R0=opts.R||88,padY=opts.padY||32;
  // side padding from the MEASURED spoke labels (13px, clipped to 16 characters): the widest label's far edge from the centre, beyond the ring
  const needFor=r=>{let need=0;spokes.forEach((sp,i)=>{const c=Math.cos(-Math.PI/2+i*2*Math.PI/spokes.length),w=textW(clip(sp.short,16),FS),x=(r+9)*c;
    need=Math.max(need,(Math.abs(c)<.35?Math.abs(x)+w/2:c>0?x+w:w-x)-r);});return Math.max(48,Math.ceil(need)+8);};
  // a radar that must fit a width (the drawer) gets a smaller ring, not smaller text
  let R=R0,pad=opts.pad||needFor(R);
  if(opts.maxW&&!opts.pad)for(let k=0;k<3&&2*(R+pad)>opts.maxW;k++){R=Math.max(60,Math.floor(opts.maxW/2)-pad);pad=needFor(R);}
  const W=2*(R+pad),foot=paper&&!opts.noNote?58:0,H=2*(R+padY)+foot;   // wide side pad (labels), slim top/bottom pad
  // Δ mode: the scale is rounded OUTWARD to a nice step so there are ~4 rings; the negative half is dropped when no value is below
  // 0 (it would be empty rings). opts.share: a share kept (0..1, centre = 0, outer ring = 1), not a Δ, so no base ring.
  let lo,hi,step=0;
  if(paper){
    // Values below the base (a behaviour spoke's 10 x negative Delta) are DRAWN: the centre is the lowest value any series needs, rounded
    // down to a nice step (never above 0), and the base (0) is a dashed ring. A health spoke is >= 0 and the cap is 10, as in the
    // paper's rule; only the centre moves. Steps: the smallest of 1 / 2 / 2.5 / 5 that gives at most 5 rings from centre to 10.
    hi=RADAR.max;let vmin=Math.min(0,RADAR.min);
    polys.forEach(p=>p.values.forEach(v=>{if(v!=null&&isFinite(v))vmin=Math.min(vmin,v);}));
    step=[1,2,2.5,5].find(st=>(hi-Math.floor(vmin/st+1e-9)*st)/st<=5.0001)||5;
    lo=Math.floor(vmin/step+1e-9)*step;
  }
  else if(opts.share){lo=0;hi=1;step=0.25;}
  else{let vmax=opts.zmax||0.2,vmin=0;
    polys.forEach(p=>p.values.forEach(v=>{if(v==null)return;vmax=Math.max(vmax,v);vmin=Math.min(vmin,v);}));
    step=niceStep((vmax-vmin)/4);while(Math.ceil(vmax/step-1e-9)-(vmin<0?Math.floor(vmin/step+1e-9):0)>5)step=niceStep(step*1.6);
    hi=Math.ceil(vmax/step-1e-9)*step;lo=vmin<0?Math.floor(vmin/step+1e-9)*step:0;}
  const rr=z=>Math.max(0,Math.min(1,(z-lo)/(hi-lo)))*R;
  const cx=R+pad,cy=R+padY,mid=rr(0),ang=i=>-Math.PI/2+i*2*Math.PI/spokes.length;
  const svg=el("svg",{viewBox:`0 0 ${W} ${H}`,width:W,height:H,class:"radar"}),ringLabs=[];
  if(paper){
    svg.appendChild(el("circle",{cx,cy,r:R,fill:"none",stroke:"var(--ring)","stroke-width":1.6,class:"outerring"}));   // 10
    const pla=ang(0)+Math.PI/spokes.length;
    const every=R*step/(hi-lo)<20?2:1;   // rings closer than 20px apart (the label sits on a diagonal): label every other one (13px labels would touch)
    for(let k=Math.ceil(lo/step-1e-9);k*step<hi-1e-9;k++){const z=+(k*step).toFixed(6),r=rr(z);if(z===lo&&lo<0||r<0.5)continue;   // the centre itself is no ring
      const base=Math.abs(z)<1e-9;
      const rc=el("circle",{cx,cy,r,fill:"none",stroke:base?"var(--ring)":"var(--grid2)","stroke-width":base?1.8:1,class:base?"basering":"gridring"});
      if(base)rc.setAttribute("stroke-dasharray","6 4");svg.appendChild(rc);
      if(Math.round((hi-z)/step)%every===0||base)ringLabs.push(el("text",{x:cx+Math.cos(pla)*r,y:cy+Math.sin(pla)*r+4,"text-anchor":"middle",fill:"var(--muted)","font-size":FS,class:"ringlab",
        stroke:"var(--halo)","stroke-width":2.4,"paint-order":"stroke"},[txt(base?"0":String(z))]));}
    if(lo<0)ringLabs.push(el("text",{x:cx,y:cy+16,"text-anchor":"middle",fill:"var(--muted)","font-size":FS,class:"ringlab centrelab",   // the centre's own value, below it (the 0 label is up-right)
      stroke:"var(--halo)","stroke-width":2.4,"paint-order":"stroke"},[txt(String(lo).replace("-","−"))]));
    ringLabs.push(el("text",{x:cx+Math.cos(pla)*R,y:cy+Math.sin(pla)*R+4,"text-anchor":"middle",fill:"var(--muted)","font-size":FS,class:"ringlab",
      stroke:"var(--halo)","stroke-width":2.4,"paint-order":"stroke"},[txt("10")]));
    if(!opts.noNote)(lo<0?["10 = base (coherence, capability)","dashed ring (0) = base (generality,","introspection, activations)"]:["10 = base (coherence, capability)","centre (0) = base (generality,","introspection, activations)"]).forEach((t,k)=>
      svg.appendChild(el("text",{x:W/2,y:H-40+k*17,"text-anchor":"middle",fill:"var(--muted)","font-size":FS,class:"ringnote"},[txt(t)])));
  }else{
  svg.appendChild(el("circle",{cx,cy,r:R,fill:"none",stroke:"var(--grid)"}));
  // faint concentric gridlines at round steps (about 4 in all), each labelled on a line half-way between two spokes
  const la=ang(0)+Math.PI/spokes.length,rings=[];
  for(let k=Math.ceil(lo/step-1e-9);k*step<=hi+1e-9;k++){const z=+(k*step).toFixed(6),r=rr(z);if(r>0.5&&r<R+0.5&&Math.abs(z)>1e-9)rings.push([z,r]);}
  rings.forEach(([z,r])=>{svg.appendChild(el("circle",{cx,cy,r,fill:"none",stroke:"var(--grid2)","stroke-width":1,class:"gridring"}));
    ringLabs.push(el("text",{x:cx+Math.cos(la)*r,y:cy+Math.sin(la)*r+3,"text-anchor":"middle",fill:"var(--muted)","font-size":FS,class:"ringlab",
      stroke:"var(--halo)","stroke-width":2.4,"paint-order":"stroke"},[txt(opts.share?String(z):fmt(z,step<0.1?2:1))]));});
  if(mid>0.5)svg.appendChild(el("circle",{cx,cy,r:mid,fill:"none",stroke:"var(--ring)","stroke-width":1.6,class:"zeroring"}));           // base (0), solid
  }
  spokes.forEach((sp,i)=>{const a=ang(i),x=cx+Math.cos(a)*R,y=cy+Math.sin(a)*R;
    const sel=opts.selLabel&&sp.key===opts.selLabel;
    svg.appendChild(el("line",{x1:cx,y1:cy,x2:x,y2:y,stroke:sel?"var(--accent)":"var(--grid)","stroke-width":sel?1.6:.8}));
    const lx=cx+Math.cos(a)*(R+9),ly=cy+Math.sin(a)*(R+9)+3;
    const anchor=Math.abs(Math.cos(a))<.35?"middle":(Math.cos(a)>0?"start":"end");
    const t=el("text",{x:lx,y:ly,"text-anchor":anchor,fill:sel?"var(--accent)":"var(--fg)","font-size":FS},[txt(clip(sp.short,16))]);
    if(opts.onSpoke){t.setAttribute("cursor","pointer");t.setAttribute("data-noenlarge","1");t.addEventListener("click",()=>opts.onSpoke(sp.key));}
    svg.appendChild(t);
  });
  const series=[];
  polys.forEach(p=>{
    const pts=spokes.map((sp,i)=>{const a=ang(i),v=p.values[i];const r=(v==null?mid:rr(v));return [cx+Math.cos(a)*r,cy+Math.sin(a)*r,v];});
    if(p.values.every(v=>v==null))return;
    // paper mode: a pending spoke (null) is left out of the outline, never drawn as 0 or as base
    const use=paper?pts.filter(q=>q[2]!=null):pts;
    const d=use.map((q,k)=>(k?"L":"M")+q[0].toFixed(1)+" "+q[1].toFixed(1)).join(" ")+(use.length>2||!paper?" Z":"");
    // one <g> per series (needed for data-method show/hide AND hover-fade)
    const g=el("g");if(p.method)g.setAttribute("data-method",p.method);
    if(p.dashed)g.setAttribute("data-flagged","1");
    // The visible fill must NOT capture the pointer (overlapping polygons would all trigger);
    // a wide transparent hit-STROKE along the line does, so you hover the line, not the area.
    const hit=el("path",{d,fill:"none",stroke:"transparent","stroke-width":11,"stroke-linejoin":"round","pointer-events":"stroke"});
    const poly=el("path",{d,fill:p.color,"fill-opacity":p.fill!=null?p.fill:.10,stroke:p.color,"stroke-width":p.width||3.2,"stroke-linejoin":"round","pointer-events":"none"});
    // a dashed outline marks an organism that falls below the expression threshold (as in the paper)
    if(p.dashed)poly.setAttribute("stroke-dasharray","7 4");
    if(p.tip){g.addEventListener("mousemove",e=>showTT(e,p.tip));g.addEventListener("mouseleave",hideTT);g.setAttribute("cursor","pointer");}
    g.appendChild(hit);g.appendChild(poly);
    // a value below 0 (below the base model) is clamped to the centre by rr, with no special marker: only the drawing clips,
    // the signed value stays in the tooltip and the payload.
    pts.forEach(q=>{if(q[2]==null)return;
      g.appendChild(el("circle",{cx:q[0],cy:q[1],r:5,fill:p.color,stroke:"var(--halo)","stroke-width":1.3}));});
    svg.appendChild(g);series.push(g);
  });
  ringLabs.forEach(t=>svg.appendChild(t));   // on top of the polygons (haloed), so a vertex never hides a ring's number
  if(series.length>1)fadeGroup(series);
  return natural(svg,W);
}
function panel(titleKey,titleShort,clickable,selected,svg){
  const d=document.createElement("div");d.className="rpanel"+(selected?" selpanel":"");
  if(clickable)d.dataset.bucket=titleKey;
  const t=document.createElement("div");t.className="rt"+(clickable?" clickable":"")+(selected?" sel":"");
  if(clickable)t.innerHTML=`${escHtml(cap(pretty(titleShort)))} <span class="drillhint">${selected?"✓ selected":"click to drill ›"}</span>`;else t.textContent=cap(pretty(titleShort));
  if(clickable)t.onclick=()=>selectBucket(titleKey);
  d.appendChild(t);d.appendChild(svg);return d;
}
