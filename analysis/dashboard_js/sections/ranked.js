// Ranked bars per property.
function rankedBars(container){
  const s=sec("Who wins each property - ranked","rankedBars");
  howto(s,`Organisms ranked best first by <b>&Delta; vs the same-size base</b>, in the score's own 0&ndash;1 units, on each property (<span style="color:${diverge(1)}">blue = better</span>, <span style="color:${diverge(-1)}">red = worse</span>). A <b>dot</b> on a bar = the 95% interval of &Delta; excludes 0 (see the forest below for the intervals). Misalignment is organism &minus; base, so + = more misbehaviour (red). Robustness is the share kept, not a &Delta;.`);
  const orgs=treated();if(!orgs.length){s.appendChild(empty("Nothing to show."));container.appendChild(s);return;}
  const grid=div("radargrid");
  DATA.axes.forEach(a=>{
    const rows=orgs.map(o=>({o,r:axisRead(o.id,a.name)})).filter(x=>x.r).sort((x,y)=>ctrlLast(x.o,y.o)||y.r.o-x.r.o);
    if(!rows.length)return;
    const rh=18,mL=190,mR=24,W=mL+186+mR,mT=28,H=mT+rows.length*rh+28,share=axShare(a.name);
    let dm=share?1:0.05;rows.forEach(x=>dm=Math.max(dm,Math.abs(x.r.d)));
    // share kept is 0-1: its panel runs 0..1 (no negative half, no zero = base line)
    const plotW=W-mL-mR,lo0=share?0:-dm,X=d=>mL+((d-lo0)/(dm-lo0))*plotW;
    const svg=natural(el("svg",{viewBox:`0 0 ${W} ${H}`,width:W,height:H}),W);
    if(!share)svg.appendChild(el("line",{x1:X(0),y1:mT-4,x2:X(0),y2:H-22,stroke:"var(--axis)","stroke-dasharray":"3 3"}));
    const _bars=[];rows.forEach((x,i)=>{const r=x.r,y=mT+i*rh,x0=X(0),x1=X(r.d);
      const rect=el("rect",{x:Math.min(x0,x1),y:y+2,width:Math.abs(x1-x0)||1,height:rh-5,rx:2,fill:diverge(Math.max(-1,Math.min(1,r.o/dm))),cursor:"pointer"});
      rect.addEventListener("mousemove",ev=>showTT(ev,`${orgHead(x.o)} · ${ph(a.short)}<br>${readTip(r,axisMean(x.o.id,a.name),axisBaseMean(x.o.id,a.name))}`));rect.addEventListener("mouseleave",hideTT);
      svg.appendChild(rect);
      if(r.sig)svg.appendChild(el("circle",{cx:x1,cy:y+2+(rh-5)/2,r:2.6,fill:"var(--fg)"}));
      _bars.push(rect);
      rowLabel(svg,mL-6,y+rh/2+4,x.o,FS);
    });
    if(_bars.length>1)fadeGroup(_bars);
    // x-axis label + ticks
    (share?[0,0.5,1]:[-dm,0,dm]).forEach(d=>svg.appendChild(el("text",{x:X(d),y:H-7,"text-anchor":"middle",fill:"var(--muted)","font-size":FS},[txt(d===0?"0":share?String(d):fmt(d,2))])));
    svg.appendChild(el("text",{x:(mL+W-mR)/2,y:mT-12,"text-anchor":"middle",fill:"var(--muted)","font-size":FS},[txt(share?"share kept":axBack(a.name)?"Δ vs base (+ = more misbehaviour)":"Δ vs base")]));
    grid.appendChild(panelBox(axisLabel(a.name,a.short)+(share?" (share kept)":""),svg));
  });
  s.appendChild(grid);container.appendChild(s);
}
