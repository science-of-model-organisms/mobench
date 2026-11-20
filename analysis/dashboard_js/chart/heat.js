// Heatmap grids, their cells and colour bars.
// One property cell of the effect matrix / a drill-down grid: Δ, significance, n and the tooltip body (`tip`: an extra line).
function axCell(o,k,tip){const c=DATA.axis_cells[o.id]?.[k];if(!c)return null;const r=axisRead(o.id,k);
  return {d:r?r.d:null,o:r?r.o:null,share:r?r.share:false,sig:r?r.sig:false,n:c.n,
    body:r?readTip(r,axisMean(o.id,k),axisBaseMean(o.id,k)):null,tip:tip||null};}
const MIS_TIP="A higher misalignment rate is the misaligned direction, so its Δ is shown as organism − base (+ = more misbehaviour) and coloured red. The cell text, where shown, is the raw rates.";
// A vertical diverging colour bar, drawn into `svg` at x..x+bw, spanning y0..y1.
function colorbar(svg,x,y0,y1,bw){   // diverging key: better (top) -> worse (bottom); the magnitude is per column
  const N=48,h=(y1-y0)/N;
  for(let i=0;i<N;i++){const t=1-(i/(N-1))*2;
    svg.appendChild(el("rect",{x,y:y0+i*h,width:bw,height:h+1,fill:diverge(t),stroke:"none"}));}
  svg.appendChild(el("rect",{x,y:y0,width:bw,height:y1-y0,fill:"none",stroke:"var(--line)"}));
  [["better",y0+6],["0",(y0+y1)/2+3],["worse",y1]].forEach(([lab,yy])=>
    svg.appendChild(el("text",{x:x+bw+4,y:yy-(lab==="worse"?2:0),"text-anchor":"start",fill:"var(--muted)","font-size":FS},[txt(lab)])));
}
// A heatmap column's colour scale: symmetric about 0, shared by every organism in the column, +/- its
// largest |oriented Δ| rounded up to 0.05 (floor 0.05). A share column (robustness) is 0-1.
function colScale(vals,share){if(share)return 1;let m=0;vals.forEach(v=>{if(v!=null)m=Math.max(m,Math.abs(v));});return Math.max(0.05,Math.ceil(m/0.05-1e-9)*0.05);}
function gridHeat(orgs,cols,cellOf,clickCols,opt={}){
  // opt.fitW (the drawer): the grid is drawn to FIT that width at 13px text - no colour bar or scale row, a shorter label column (names clipped,
  // full name on hover), steeper column labels, and the widest cell (<= 46px, a 13px "+0.12" needs ~38) that makes it fit.
  const fit=!!opt.fitW,rh=22,cbW=16,cbGut=fit?0:84;
  const mL=Math.min(fit?(opt.fitW<420?100:140):280,Math.max(fit?90:150,Math.ceil(7.2*Math.max(0,...orgs.map(o=>niceOrg(o).length)))+30));   // fit the longest short name and its colour dot
  const maxCh=fit?Math.floor((mL-30)/7.2):0;
  const ANG=(fit?70:40)*Math.PI/180,lw=cols.map(c=>pretty(c.short).length*7.8);
  const endAt=cw=>Math.max(mL+cols.length*cw+cbGut+8,...lw.map((w,j)=>Math.ceil(mL+j*cw+cw/2+w*Math.cos(ANG))+8));
  let cw=opt.cw||Math.max(46,Math.min(58,660/cols.length));
  if(fit){cw=46;while(cw>22&&endAt(cw)>opt.fitW)cw--;}
  // Column labels are rotated -40deg, so a long one overhangs the top and the right edge: size the margins
  // from the longest label (7.8px/char is a safe upper bound at 13px/600 in any fallback face) instead of
  // fixing them, or a long property name is clipped. Labels that fit keep the old 100px and width:
  // the width grows (with 8px to spare) only when a label's end passes the old right edge.
  const mT=Math.max(84,Math.ceil(Math.max(0,...lw)*Math.sin(ANG))+26);
  const W0=mL+cols.length*cw+cbGut+8,lEnd=Math.max(0,...lw.map((w,j)=>Math.ceil(mL+j*cw+cw/2+w*Math.cos(ANG))));
  const W=lEnd>W0?lEnd+8:W0,H=mT+orgs.length*rh+8+(opt.scaleRow===false||fit?0:24);
  const showVals=opt.values&&cw>=(fit?38:40);
  const svg=el("svg",{viewBox:`0 0 ${W} ${H}`,width:W,height:H});
  cols.forEach((c,j)=>{const x=mL+j*cw+cw/2;
    const t=el("text",{x,y:mT-8,transform:`rotate(-${fit?70:40} ${x} ${mT-8})`,"text-anchor":"start",
      fill:(clickCols&&state.bucket===c.key)?"var(--accent)":"var(--fg)","font-size":FS,"font-weight":clickCols?600:400},[txt(pretty(c.short))]);
    if(clickCols){t.setAttribute("cursor","pointer");t.setAttribute("data-noenlarge","1");t.addEventListener("click",()=>selectBucket(c.key));}
    svg.appendChild(t);
  });
  const CR=[];   // cell rects for row/col highlight
  const grid=orgs.map(o=>cols.map(c=>cellOf(o,c.key)));
  const M=cols.map((c,j)=>colScale(grid.map(r=>r[j]&&r[j].o),grid.some(r=>r[j]&&r[j].share)));
  orgs.forEach((o,i)=>{const y=mT+i*rh;
    rowLabel(svg,mL-8,y+rh/2+4,o,FS,{highlight:opt.highlight===o.id,maxCh});
    cols.forEach((c,j)=>{const x=mL+j*cw;const cell=grid[i][j];
      let fill="var(--cellna)",stroke="var(--halo)";
      const d=cell?cell.d:undefined,sig=cell?cell.sig:undefined;
      if(d!=null){fill=diverge(Math.max(-1,Math.min(1,cell.o/M[j])));}
      else if(cell){fill="none";stroke="var(--ref)";}
      const r=el("rect",{x:x+1,y:y+1,width:cw-2,height:rh-2,fill,stroke,"stroke-width":1,cursor:"pointer"});
      if(cell){const dl=d==null?"unmeasurable":(cell.body||"Δ "+fmt(d));
        r.addEventListener("mousemove",e=>showTT(e,`${orgHead(o)} · ${ph(c.short)}<br>${dl} · n=${cell.n??"?"}${cell.tip?"<br>"+cell.tip:""}`));
        r.addEventListener("mouseleave",hideTT);}
      svg.appendChild(r);CR.push({r,i,j});
      if(showVals&&(d!=null||(cell&&cell.txt))){
        svg.appendChild(el("text",{x:x+cw/2,y:y+rh/2+4,"text-anchor":"middle",fill:d!=null?inkOn(fill):"var(--fg)","font-size":FS,class:cell.txt?"celltxt":""},[txt(cell.txt!=null?cell.txt:fmt(d,2))]));}
      if(d!=null&&sig)svg.appendChild(el("circle",{cx:x+cw-4,cy:y+4,r:1.6,fill:inkOn(fill)}));
      // report-only differential-attrition marker (#254): a corner triangle in the cell's own contrast ink, so it reads on either theme's fill
      if(d!=null&&cell.attr)svg.appendChild(el("polygon",{points:`${x+1},${y+rh-1} ${x+12},${y+rh-1} ${x+1},${y+rh-12}`,fill:inkOn(fill),class:"attrmark"}));
    });
  });
  // opt.highlight (an organism id): its row is outlined, so a grid can say "this one" among its siblings (the drill-down drawer)
  const hi=orgs.findIndex(o=>o.id===opt.highlight);
  if(hi>=0)svg.appendChild(el("rect",{x:2,y:mT+hi*rh,width:mL+cols.length*cw-1,height:rh,fill:"none",stroke:"var(--accent)","stroke-width":2.2,rx:3,"pointer-events":"none",class:"hlrow"}));
  // hover a cell -> dim everything not in its row or column
  CR.forEach(o=>o.r.addEventListener("mouseenter",()=>CR.forEach(m=>m.r.classList.toggle("dim",m.i!==o.i&&m.j!==o.j))));
  svg.addEventListener("mouseleave",()=>CR.forEach(m=>m.r.classList.remove("dim")));
  if(opt.scaleRow!==false&&!fit){const yy=mT+orgs.length*rh+17;
    svg.appendChild(el("text",{x:mL-8,y:yy,"text-anchor":"end",fill:"var(--muted)","font-size":FS},[txt("colour scale")]));
    cols.forEach((c,j)=>{if(cw<46)return;svg.appendChild(el("text",{x:mL+j*cw+cw/2,y:yy,"text-anchor":"middle",fill:"var(--muted)","font-size":FS,class:"scalerow"},[txt(grid.some(r=>r[j]&&r[j].share)?"0\u20131":"\u00b1"+M[j].toFixed(2))]));});}
  if(!fit)colorbar(svg,mL+cols.length*cw+14,mT,mT+orgs.length*rh,cbW);
  const w=document.createElement("div");w.className="figwrap center heatscroll";w.appendChild(svg);return w;
}

// Vertical diverging colour bar into `svg`: value `hi` at top, `lo` at bottom, diverge(v).
function vRamp(svg,x,y0,y1,bw,lo,hi,labels){
  const N=48,h=(y1-y0)/N;
  for(let i=0;i<N;i++){const v=hi-(i/(N-1))*(hi-lo);
    svg.appendChild(el("rect",{x,y:y0+i*h,width:bw,height:h+1,fill:diverge(v),stroke:"none"}));}
  svg.appendChild(el("rect",{x,y:y0,width:bw,height:y1-y0,fill:"none",stroke:"var(--line)"}));
  const Y=v=>y0+(1-(v-lo)/(hi-lo))*(y1-y0);
  labels.forEach(([lab,v])=>svg.appendChild(el("text",{x:x+bw+4,y:Y(v)+3,"text-anchor":"start",fill:"var(--muted)","font-size":FS},[txt(lab)])));
}
// (unused legacy horizontal legend kept out; see vRamp above)
function rampLegend(lo,hi,labels){
  const W=300,H=42,barY=4,barH=14,mL=6,barW=W-2*mL,N=40;
  const svg=natural(el("svg",{viewBox:`0 0 ${W} ${H}`,width:W,height:H}),W);
  for(let i=0;i<N;i++){const t=i/(N-1),v=lo+(hi-lo)*t;
    svg.appendChild(el("rect",{x:mL+(i/N)*barW,y:barY,width:barW/N+0.6,height:barH,fill:diverge(v),stroke:"none"}));}
  svg.appendChild(el("rect",{x:mL,y:barY,width:barW,height:barH,fill:"none",stroke:"var(--line)"}));
  labels.forEach(([lab,t],k)=>svg.appendChild(el("text",{x:mL+t*barW,y:barY+barH+15,
    "text-anchor":k===0?"start":k===labels.length-1?"end":"middle",fill:"var(--muted)","font-size":FS},[txt(lab)])));
  const wrap=div("");wrap.style.marginTop="6px";wrap.appendChild(svg);return wrap;
}
