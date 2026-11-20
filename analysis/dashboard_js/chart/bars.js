// Paired diverging bars (a drill-down with fewer than 3 evals).
// Paired diverging bars — the drill body for a property with FEWER THAN 3 evals, where a radar
// would be degenerate (2 spokes = a line, 1 = a point). One row per organism, one bar per eval
// (colour = eval), diverging from a central zero: right = better, left = worse, on the same
// Δ scale as the radars. Each row carries data-method so applyDrillVis toggles it.
function drawPairedBars(spokes,orgs,opts){
  opts=opts||{};
  const rows=orgs.filter(o=>spokes.some(sp=>{const v=evalEff(o.id,sp.key);return v!=null&&isFinite(v);}));
  if(!rows.length)return empty("No measurable evals for this selection.");
  let dmax=0.2;
  rows.forEach(o=>spokes.forEach(sp=>{const v=evalEff(o.id,sp.key);if(v!=null&&isFinite(v))dmax=Math.max(dmax,Math.abs(v));}));
  const labelW=176,plotW=220,padR=16,rh=Math.max(30,spokes.length*15+10),bh=Math.min(12,(rh-8)/spokes.length-2);
  const W=labelW+plotW+padR,H=rows.length*rh+46;
  const cx=labelW+plotW/2,sx=d=>cx+Math.max(-1,Math.min(1,d/dmax))*(plotW/2);
  const svg=el("svg",{viewBox:`0 0 ${W} ${H}`,width:W,height:H,class:"pbars"});
  // gridlines at 0, plus faint ±dmax edge
  [[-dmax,"var(--grid2)"],[0,"var(--ring)"],[dmax,"var(--grid2)"]].forEach(([g,col])=>{
    const x=sx(g),dash=(g===0)?null:"3 3";
    const ln=el("line",{x1:x,y1:6,x2:x,y2:H-30,stroke:col,"stroke-width":g===0?1.6:1});
    if(dash)ln.setAttribute("stroke-dasharray",dash);svg.appendChild(ln);
  });
  svg.appendChild(el("text",{x:sx(0),y:H-18,"text-anchor":"middle",fill:"var(--faint)","font-size":FS},[txt("0")]));
  svg.appendChild(el("text",{x:labelW+6,y:H-2,"text-anchor":"start",fill:"var(--faint)","font-size":FS,"font-style":"italic"},[txt("← worse")]));
  svg.appendChild(el("text",{x:W-padR,y:H-2,"text-anchor":"end",fill:"var(--faint)","font-size":FS,"font-style":"italic"},[txt("better →")]));
  const rowNodes=[];   // {g, method} — for the method-grouped hover-fade below
  rows.forEach((o,ri)=>{
    const y0=ri*rh+8,g=el("g",{"data-method":serOf(o),cursor:"pointer"});
    rowLabel(g,labelW-8,y0+rh/2+4,o,FS);
    spokes.forEach((sp,si)=>{
      const v=evalEff(o.id,sp.key);const by=y0+2+si*(bh+2);
      if(v==null||!isFinite(v)){
        svg.appendChild(el("text",{x:cx+4,y:by+bh-1,fill:"var(--faint)","font-size":11},[txt("n/a")]));return;}
      const x=sx(v),x0=sx(0),bx=Math.min(x,x0),bw=Math.max(1.5,Math.abs(x-x0));
      g.appendChild(el("rect",{x:bx,y:by,width:bw,height:bh,rx:1.5,fill:evalColor(sp.key),"fill-opacity":.85}));
      g.appendChild(el("text",{x:v>=0?x+3:x-3,y:by+bh-1,"text-anchor":v>=0?"start":"end",fill:"var(--ring)","font-size":11},[txt((v>=0?"+":"")+v.toFixed(2))]));
    });
    g.addEventListener("mousemove",e=>showTT(e,`${orgHead(o)}<br>`+spokes.map(sp=>{const v=evalEff(o.id,sp.key);return `<span style="color:${evalColor(sp.key)}">■</span> ${ph(sp.short)}: ${v==null?"n/a":(v>=0?"+":"")+v.toFixed(2)}`;}).join("<br>")));
    g.addEventListener("mouseleave",hideTT);
    svg.appendChild(g);
    rowNodes.push({g,method:o.method});
  });
  // Hover-fade grouped by IMPLANT METHOD (bars are eval-coloured, so method is the axis to
  // separate): hovering any row lights every row of that method and dims the rest. Same .dim/
  // .hot classes and behaviour as fadeGroup, but keyed on method rather than one node.
  rowNodes.forEach(({g,method})=>{
    g.addEventListener("mouseenter",()=>rowNodes.forEach(o=>o.g.classList.add(o.method===method?"hot":"dim")));
    g.addEventListener("mouseleave",()=>rowNodes.forEach(o=>{o.g.classList.remove("dim");o.g.classList.remove("hot");}));
  });
  return natural(svg,W);
}
