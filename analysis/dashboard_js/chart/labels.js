// Figure 1's point layout, both custom (not Inspect: this is page layout, with no upstream equivalent). `pos` holds one
// {p, x, y, tx, ty} per point in svg units (x/y drawn, tx/ty true); `F` is the plot frame {mL, mR, mT, mB, W, H}; ox/oy the
// base crosshair in svg units.
// jitterPoints moves x/y in place (visual only: tooltips use the true values) and keeps every point inside the plot and on its
// true side of the crosshair. `on` is the figure's "Jitter overlapping points" checkbox.
function jitterPoints(pos,r,F,ox,oy,on){
  const {mL,mR,mT,mB,W,H}=F;
  // Jitter (visual only; tooltips use the true values): points on ~the same spot are spread on a small ring, and every
  // point gets a tiny offset. Both are seeded functions of the organism id, so the layout is the same on every render.
  if(on){
    const par=pos.map((_,i)=>i),find=i=>par[i]===i?i:(par[i]=find(par[i]));
    for(let i=0;i<pos.length;i++)for(let j=i+1;j<pos.length;j++)
      if(Math.hypot(pos[i].x-pos[j].x,pos[i].y-pos[j].y)<1.6*r)par[find(i)]=find(j);
    const cl=new Map();pos.forEach((q,i)=>{const k=find(i);if(!cl.has(k))cl.set(k,[]);cl.get(k).push(q);});
    cl.forEach(m=>{if(m.length<2)return;
      const cx=m.reduce((t,q)=>t+q.x,0)/m.length,cy=m.reduce((t,q)=>t+q.y,0)/m.length;
      m.sort((u,v)=>seedUnit(u.p.id,1)-seedUnit(v.p.id,1));
      const a0=2*Math.PI*seedUnit(m[0].p.id,2),rho=Math.max(1.0,m.length*0.35)*r;
      m.forEach((q,k)=>{q.x=cx+rho*Math.cos(a0+2*Math.PI*k/m.length);q.y=cy+rho*Math.sin(a0+2*Math.PI*k/m.length);});});
    pos.forEach(q=>{const a=2*Math.PI*seedUnit(q.p.id,4),d=0.3*r*seedUnit(q.p.id,3);q.x+=d*Math.cos(a);q.y+=d*Math.sin(a);});
    // Jitter is a visual aid, not a measurement: a point never moves more than 1.5 marker radii from where it truly is.
    pos.forEach(q=>{const dx=q.x-q.tx,dy=q.y-q.ty,d=Math.hypot(dx,dy),cap=1.5*r;if(d>cap){q.x=q.tx+dx*cap/d;q.y=q.ty+dy*cap/d;}});
  }
  // A marker whose TRUE position is inside the plot is kept wholly inside it (jitter and zoom must not push half of it under
  // the clip edge); the paint is still clipped to the plot rect, so anything panned out of view stays out.
  pos.forEach(q=>{q.x=Math.max(mL+r+1,Math.min(W-mR-r-1,q.x));q.y=Math.max(mT+r+1,Math.min(H-mB-r-1,q.y));});
  // LAST, and never overridden: a point stays on its true side of the base crosshair (x=0 and y=0), at least 1px off the
  // line; a point truly ON a line keeps that coordinate unjittered. Reading a quadrant off this plot must be safe.
  pos.forEach(q=>{
    const sx=Math.abs(q.tx-ox)<1e-9?0:Math.sign(q.tx-ox),sy=Math.abs(q.ty-oy)<1e-9?0:Math.sign(q.ty-oy);
    q.x=sx>0?Math.max(q.x,ox+1):sx<0?Math.min(q.x,ox-1):q.tx;
    q.y=sy>0?Math.max(q.y,oy+1):sy<0?Math.min(q.y,oy-1):q.ty;});
}
// placePointLabels sets q.lab (the label box), q.name (the text drawn: the point's label, or its number) and q.full (a numbered
// point's full name, for the key) on every point, and returns the "untreated base" label box (bbest), the numbered points and the
// base label's font size. L = {r: marker radius, fs: label font size, ox, oy, baseIn: the crosshair is in view, caps: caption boxes}.
function placePointLabels(pos,F,L){
  const {mL,mR,mT,mB,W,H}=F,{r,fs,ox,oy,baseIn,caps}=L;
  const lh=fs*1.15,gap=r+3,hit=(a,b)=>Math.max(0,Math.min(a.x1,b.x1)-Math.max(a.x0,b.x0))*Math.max(0,Math.min(a.y1,b.y1)-Math.max(a.y0,b.y0));
  const bfs=Math.max(11,fs),blw=14*bfs*0.56+4,blh=bfs*1.2;
  pos.forEach((q,i)=>{q.i=i;});
  const mk=pos.map(q=>({x0:q.x-r-1,x1:q.x+r+1,y0:q.y-r-1,y1:q.y+r+1}));   // every marker (+1px): NO label may touch one
  mk.push({x0:ox-6,x1:ox+6,y0:oy-6,y1:oy+6});                              // ... nor the base diamond
  const obst=[...caps];                                                       // captions: avoided, but a soft constraint
  const cand=(q,lw,lhh,d,th)=>{const c=Math.cos(th),sn=Math.sin(th),ax=q.x+d*c,ay=q.y+d*sn;
    let x0=c>0.35?ax:c<-0.35?ax-lw:ax-lw/2,y0=sn>0.35?ay:sn<-0.35?ay-lhh:ay-lhh/2;
    x0=Math.max(mL+2,Math.min(W-mR-2-lw,x0));y0=Math.max(mT+2,Math.min(H-mB-2-lhh,y0));   // keep the label inside the plot
    return {x0,x1:x0+lw,y0,y1:y0+lhh};};
  // "untreated base": painted ABOVE the points (own layer, after them) at the nearest spot around the crosshair that touches
  // no marker and no caption; a leader line ties it to the diamond when it sits off to the side of a crowded origin.
  let bbest=null;
  for(const d of baseIn?[r+8,r+22,r+40,r+60,r+84,r+110,r+140,r+180]:[])for(let k=0;k<24&&!(bbest&&bbest.cost<0.5);k++){
    const box=cand({x:ox,y:oy},blw,blh,d,k*Math.PI/12);let cost=d*0.004;
    for(const o of mk)cost+=hit(box,o)*20;
    for(const o of obst)cost+=hit(box,o);
    if(!bbest||cost<bbest.cost)bbest={box,cost,d};}
  if(!bbest)bbest={box:{x0:-1e4,x1:-1e4,y0:-1e4,y1:-1e4},cost:0,d:0};
  mk.push(bbest.box);
  const labs=[bbest.box];
  // Point labels. Rings of candidate boxes grow around each marker until one touches NO marker and NO placed label (hard), and
  // as few captions as possible (soft); the leader line grows with the ring. Crowded points are placed first. If even the far
  // rings are full, the point gets a compact NUMBER instead, and the number is explained in a key under the plot: still every
  // point labelled, still no overlap. Custom (not Inspect): this is page layout, no upstream equivalent.
  const crowd=q=>pos.reduce((t,o)=>t+(Math.hypot(o.x-q.x,o.y-q.y)<5*r?1:0),0);
  const order=[...pos].sort((u,v)=>crowd(v)-crowd(u)||u.y-v.y||u.x-v.x);
  const MAXRING=22,clear=(box)=>!mk.some(o=>hit(box,o)>0.25)&&!labs.some(o=>hit(box,o)>0.25);
  // Leader lines are part of the layout: a leader runs from its marker to the nearest point of its label (none when the label is
  // adjacent), and a candidate is charged for every OTHER marker, label and leader that its leader crosses, and for every placed
  // leader that crosses its box, plus a little for the leader's length. The ring search keeps going outward while only crossing
  // spots are free, so a label stays near its marker without a line running through a neighbour. Custom: page layout.
  const leads=[],boxName=new Map();
  const leaderOf=(q,box)=>{const nx=Math.max(box.x0,Math.min(box.x1,q.x)),ny=Math.max(box.y0,Math.min(box.y1,q.y));
    return Math.hypot(nx-q.x,ny-q.y)>r+4?{x0:q.x,y0:q.y,x1:nx,y1:ny}:null;};
  const segBox=(g,b)=>{let t0=0,t1=1;const dx=g.x1-g.x0,dy=g.y1-g.y0;
    for(const [p,qq] of [[-dx,g.x0-b.x0],[dx,b.x1-g.x0],[-dy,g.y0-b.y0],[dy,b.y1-g.y0]]){
      if(p===0){if(qq<0)return false;}else{const t=qq/p;if(p<0){if(t>t1)return false;if(t>t0)t0=t;}else{if(t<t0)return false;if(t<t1)t1=t;}}}
    return t0<=t1;};
  const segSeg=(a,b)=>{const o=(ax,ay,bx,by,cx,cy)=>(bx-ax)*(cy-ay)-(by-ay)*(cx-ax);
    const d1=o(a.x0,a.y0,a.x1,a.y1,b.x0,b.y0),d2=o(a.x0,a.y0,a.x1,a.y1,b.x1,b.y1),d3=o(b.x0,b.y0,b.x1,b.y1,a.x0,a.y0),d4=o(b.x0,b.y0,b.x1,b.y1,a.x1,a.y1);
    return d1*d2<0&&d3*d4<0;};
  const place=(q,lw,lhh,maxRing)=>{let best=null;
    for(let ring=0;ring<maxRing&&!(best&&best.cost<0.5);ring++){
      const d=gap+ring*lhh*0.9;
      const dirs=ring===0?[0,Math.PI,-Math.PI/4,-3*Math.PI/4,Math.PI/4,3*Math.PI/4,-Math.PI/2,Math.PI/2]:Array.from({length:ring<4?16:32},(_,i)=>i*2*Math.PI/(ring<4?16:32));
      for(const th of dirs){const box=cand(q,lw,lhh,d,th);if(!clear(box))continue;
        let cost=ring*0.01;for(const o of obst)cost+=hit(box,o);
        // a label must read as belonging to ITS marker: charge it for every other marker that is as near to the box as its own
        const dpb=(u,b)=>Math.hypot(Math.max(b.x0-u.x,0,u.x-b.x1),Math.max(b.y0-u.y,0,u.y-b.y1)),dOwn=dpb(q,box);
        pos.forEach(o=>{if(o!==q&&dpb(o,box)<dOwn+3)cost+=.8;});
        const ld=leaderOf(q,box);
        if(ld){cost+=Math.hypot(ld.x1-ld.x0,ld.y1-ld.y0)*0.006;
          mk.forEach((o,j)=>{if(j!==q.i&&segBox(ld,o))cost+=1.5;});labs.forEach(o=>{if(segBox(ld,o))cost+=1.5;});leads.forEach(g=>{if(segSeg(ld,g))cost+=.8;});}
        leads.forEach(g=>{if(segBox(g,box))cost+=1.5;});
        // two labels with the same text (the two negative-prompt controls) must not sit together: a reader could not tell which is whose
        labs.forEach(o=>{if(boxName.get(o)===q.name&&Math.hypot((o.x0+o.x1)/2-(box.x0+box.x1)/2,(o.y0+o.y1)/2-(box.y0+box.y1)/2)<lw*.8+lhh*2)cost+=0.9;});
        if(!best||cost<best.cost)best={box,cost,ring};}}
    return best;};
  const numbered=[];
  for(const q of order){
    const name=pointLabel(q.p),lw=name.length*fs*0.56+4;
    q.name=name;let best=place(q,lw,lh,MAXRING);
    if(!best){const tag=String(numbered.length+1),nw=tag.length*fs*0.62+6;
      best=place(q,nw,lh,60)||{box:cand(q,nw,lh,gap,0)};   // a number is tiny: some ring has room
      numbered.push({q,tag});q.name=tag;q.full=niceOrg(q.p);}
    q.lab=best;labs.push(best.box);boxName.set(best.box,q.name);{const ld=leaderOf(q,best.box);if(ld)leads.push(ld);}}
  return {bbest,numbered,bfs};
}
