// Forest: Δ with a 95% interval per property.
function forest(container){
  const s=sec("Is it real? - &Delta; with a 95% interval","forest");
  howto(s,`Per property, each organism's <b>&Delta; vs the same-size base</b> (dot, score units) with its <b>95% confidence interval</b> (the horizontal <b>whisker</b>). If the whisker crosses the dashed <b>zero</b> line, the change is not statistically distinguishable from base. Sorted best first; colour = ${colourBy()}. Misalignment is organism &minus; base (+ = more misbehaviour); robustness is the share kept, not a &Delta; (no base comparison). <b>Pinch or &#8984;/Ctrl+scroll over a panel to zoom, drag to pan once zoomed</b> (or use + / &minus; / Reset zoom).`);
  const orgs=treated();if(!orgs.length){s.appendChild(empty("Nothing to show."));container.appendChild(s);return;}
  const grid=div("radargrid");const zooms=[];
  DATA.axes.forEach(a=>{
    const rows=orgs.map(o=>({o,r:axisRead(o.id,a.name)})).filter(x=>x.r).sort((x,y)=>ctrlLast(x.o,y.o)||y.r.o-x.r.o);
    if(!rows.length)return;
    const share=axShare(a.name);
    const rh=18,mL=190,mR=24,W=mL+186+mR,mT=30,H=mT+rows.length*rh+28;
    let dm=share?1:0.05;if(!share)rows.forEach(x=>dm=Math.max(dm,Math.abs(x.r.d),x.r.lo==null?0:Math.abs(x.r.lo),x.r.hi==null?0:Math.abs(x.r.hi)));
    // share kept is 0-1: its panel runs 0..1 (CI clipped to it), no zero = base line
    const plotW=W-mL-mR,base={x:[share?0:-dm,dm]},view={x:[...base.x]};
    const svg=natural(el("svg",{viewBox:`0 0 ${W} ${H}`,width:W,height:H}),W);
    const cid="zc"+(++_zid);
    const paint=()=>{
      hideTT();clearNode(svg);
      const lo0=view.x[0],hi0=view.x[1];
      const Xr=d=>mL+((d-lo0)/(hi0-lo0))*plotW,X=d=>share?Xr(Math.max(0,Math.min(1,d))):Xr(d);
      svg.appendChild(el("clipPath",{id:cid},[el("rect",{x:mL-6,y:mT-6,width:plotW+12,height:rows.length*rh+10})]));
      const plot=el("g",{"clip-path":`url(#${cid})`});
      if(!share)plot.appendChild(el("line",{x1:X(0),y1:mT-4,x2:X(0),y2:H-22,stroke:"var(--axis)","stroke-dasharray":"3 3"}));
      const _rows=[];
      rows.forEach((x,i)=>{const r=x.r,y=mT+i*rh+rh/2,col=orgCol(x.o);const g=el("g",{cursor:"pointer"});
        // The interval combines BOTH arms, so its df is the smaller of the two arms' seed counts (a
        // conservative Welch-Satterthwaite stand-in: a 1-seed base pins the pair to the normal even if the
        // treatment has 3 seeds). Computed once in axisRead.
        if(r.lo!=null)g.appendChild(el("line",{x1:X(r.lo),y1:y,x2:X(r.hi),y2:y,stroke:col,"stroke-width":1.8,opacity:.55,"stroke-linecap":"round"}));
        g.appendChild(el("circle",{cx:X(r.d),cy:y,r:3.8,fill:col,stroke:"var(--halo)","stroke-width":1}));
        g.addEventListener("mousemove",ev=>showTT(ev,`${orgHead(x.o)}<br>${ph(a.short)}<br>${readTip(r,axisMean(x.o.id,a.name),axisBaseMean(x.o.id,a.name))}`));g.addEventListener("mouseleave",hideTT);
        plot.appendChild(g);_rows.push(g);
        rowLabel(svg,mL-6,y+4,x.o,FS);
      });
      svg.appendChild(plot);
      if(_rows.length>1)fadeGroup(_rows);
      const unzoomed=view.x[0]===base.x[0]&&view.x[1]===base.x[1];
      (unzoomed?(share?[0,0.5,1]:[-dm,0,dm]):ticks(view.x)).forEach(d=>svg.appendChild(el("text",{x:X(d),y:H-7,"text-anchor":"middle",fill:"var(--muted)","font-size":FS},[txt(d===0?"0":share?String(+d.toFixed(3)):fmt(d,unzoomed?2:3))])));
      svg.appendChild(el("text",{x:(mL+W-mR)/2,y:12,"text-anchor":"middle",fill:"var(--muted)","font-size":FS},[txt(share?"share kept (95% CI)":axBack(a.name)?"Δ vs base (+ = more misbehaviour)":"Δ vs base")]));
    };
    paint();
    zooms.push(makeZoom(svg,W,{x:mL,y:mT-6,w:plotW,h:rows.length*rh+10},view,base,paint,"x"));
    grid.appendChild(panelBox(axisLabel(a.name,a.short)+(share?" (share kept)":""),svg));
  });
  const zr=div("zrow");zoomBar(zr,()=>zooms);s.appendChild(zr);s.appendChild(grid);
  s.appendChild(spreadLegend());container.appendChild(s);
}
