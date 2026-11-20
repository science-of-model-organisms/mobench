// Figures no longer on the page (not called by render); kept so their removal is its own change.
function parcoords(container){
  const s=sec("Every organism across every property - parallel coordinates","parcoords");
  howto(s,`One polyline per organism across the properties (y = Δ vs base, oriented: + = better). Good for tracing an organism and spotting crossovers. <b>Hover</b> a line to highlight it.`);
  const orgs=treated(),axes=DATA.axes;if(orgs.length<1||axes.length<2){s.appendChild(empty("Nothing to show."));container.appendChild(s);return;}
  const W=Math.max(760,axes.length*140),H=440,mL=44,mR=130,mT=24,mB=54;
  let zmax=0.2;orgs.forEach(o=>axes.forEach(a=>{const z=ad(o.id,a.name);if(z!=null)zmax=Math.max(zmax,Math.abs(z));}));
  const X=j=>mL+j/(axes.length-1)*(W-mL-mR),Y=z=>mT+(1-(z+zmax)/(2*zmax))*(H-mT-mB);
  const svg=el("svg",{viewBox:`0 0 ${W} ${H}`,width:W,height:H});
  axes.forEach((a,j)=>{svg.appendChild(el("line",{x1:X(j),y1:mT,x2:X(j),y2:H-mB,stroke:"var(--grid)"}));
    svg.appendChild(el("text",{x:X(j),y:H-mB+16,"text-anchor":"middle",fill:"var(--fg)","font-size":11},[txt(pretty(a.short))]));});
  svg.appendChild(el("line",{x1:mL,y1:Y(0),x2:W-mR,y2:Y(0),stroke:"var(--axis)","stroke-dasharray":"4 3"}));
  orgs.forEach(o=>{const col=orgCol(o);
    const pts=axes.map((a,j)=>{const z=ad(o.id,a.name);return z==null?null:[X(j),Y(z)];}).filter(Boolean);
    if(pts.length<2)return;
    const path=el("path",{d:pts.map((q,i)=>(i?"L":"M")+q[0].toFixed(1)+" "+q[1].toFixed(1)).join(" "),fill:"none",stroke:col,"stroke-width":1.6,opacity:.5,cursor:"pointer"});
    path.addEventListener("mouseover",()=>{path.setAttribute("stroke-width",3.4);path.setAttribute("opacity",1);});
    path.addEventListener("mouseout",()=>{path.setAttribute("stroke-width",1.6);path.setAttribute("opacity",.5);});
    path.addEventListener("mousemove",e=>showTT(e,`<b>${ph(o.short)}</b> · ${ph(o.persona)}`));path.addEventListener("mouseleave",hideTT);
    svg.appendChild(path);});
  const w=div("figwrap center");w.appendChild(svg);s.appendChild(w);s.appendChild(spreadLegend());container.appendChild(s);
}

function bump(container){
  const s=sec("Does the winner change by property? - rank bump","bump");
  howto(s,`Each organism's <b>rank</b> (1 = best by Δ, at the top) within each property. Flat lines = a stable winner; crossing lines = "best method" depends on the property. Colour = ${colourBy()}.`);
  const orgs=treated(),axes=DATA.axes;if(orgs.length<2){s.appendChild(empty("Need ≥2 organisms."));container.appendChild(s);return;}
  const rank={};axes.forEach(a=>{const wz=orgs.map(o=>({id:o.id,z:ad(o.id,a.name)})).filter(r=>r.z!=null).sort((x,y)=>y.z-x.z);
    rank[a.name]={};wz.forEach((r,i)=>rank[a.name][r.id]=i+1);});
  const nmax=orgs.length,W=Math.max(760,axes.length*140),H=Math.max(340,nmax*20+80),mL=44,mR=150,mT=24,mB=54;
  const X=j=>mL+j/(axes.length-1)*(W-mL-mR),Y=r=>mT+(r-1)/Math.max(1,nmax-1)*(H-mT-mB);
  const svg=el("svg",{viewBox:`0 0 ${W} ${H}`,width:W,height:H});
  axes.forEach((a,j)=>{svg.appendChild(el("line",{x1:X(j),y1:mT,x2:X(j),y2:H-mB,stroke:"var(--grid)"}));
    svg.appendChild(el("text",{x:X(j),y:H-mB+16,"text-anchor":"middle",fill:"var(--fg)","font-size":11},[txt(pretty(a.short))]));});
  orgs.forEach(o=>{const col=orgCol(o);
    const pts=axes.map((a,j)=>{const r=rank[a.name][o.id];return r==null?null:[X(j),Y(r)];}).filter(Boolean);
    if(pts.length<2)return;
    const path=el("path",{d:pts.map((q,i)=>(i?"L":"M")+q[0].toFixed(1)+" "+q[1].toFixed(1)).join(" "),fill:"none",stroke:col,"stroke-width":1.8,opacity:.55,cursor:"pointer"});
    path.addEventListener("mouseover",()=>path.setAttribute("opacity",1));path.addEventListener("mouseout",()=>path.setAttribute("opacity",.55));
    path.addEventListener("mousemove",e=>showTT(e,`<b>${ph(o.short)}</b>`));path.addEventListener("mouseleave",hideTT);
    svg.appendChild(path);pts.forEach(q=>svg.appendChild(el("circle",{cx:q[0],cy:q[1],r:3,fill:col})));
    const last=pts[pts.length-1];svg.appendChild(el("text",{x:last[0]+6,y:last[1]+3,fill:"var(--fg)","font-size":9},[txt(clip(o.short,22))]));});
  const w=div("figwrap");w.appendChild(svg);s.appendChild(w);container.appendChild(s);
}

// ---- internal representation (white-box probes): REMOVED 2026-10-14 ----
// The bar chart of the six V0 `whitebox_probe_*` cells lived here. Those cells were retired as
// invalid (docs/evals/activations.md) and `load_rows` drops their historical rows, so the
// section had nothing left to draw. The truth-probe cells vote on the activations axis but have no section here.
