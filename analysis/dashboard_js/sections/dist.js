// Score distributions (ECDFs).
function distributions(container){
  const s=sec("Is the mean hiding a bimodal organism? - distributions","dist");
  howto(s,`For continuous-score evals, the per-sample score spread as an ECDF (cumulative step). Two organisms with the same mean but different shapes are different findings - this is where you see it. Colour = implant method.`);
  const evals=Object.keys(DATA.evals).filter(e=>DATA.evals[e].score_type==="continuous"&&DATA.evals[e].axis!=="unknown"&&treated().some(o=>(DATA.cells[o.id]||{})[e]?.hist));
  if(!evals.length){s.appendChild(empty("No per-sample histograms in this export (continuous evals)."));container.appendChild(s);return;}
  const grid=div("radargrid");
  evals.forEach(e=>{let edges=null;treated().forEach(o=>{const hh=(DATA.cells[o.id]||{})[e]?.hist;if(hh&&!edges)edges=hh.edges;});if(!edges)return;
    const W=340,H=236,mL=52,mR=18,mT=16,mB=56,lo=edges[0],hi=edges[edges.length-1];
    const X=v=>mL+((v-lo)/((hi-lo)||1))*(W-mL-mR),Y=c=>H-mB-c*(H-mT-mB);
    const svg=natural(el("svg",{viewBox:`0 0 ${W} ${H}`,width:W,height:H}),W);
    svg.appendChild(el("rect",{x:mL,y:mT,width:W-mL-mR,height:H-mT-mB,fill:"var(--plot)",stroke:"var(--line)"}));
    [0,0.5,1].forEach(c=>{svg.appendChild(el("line",{x1:mL,y1:Y(c),x2:W-mR,y2:Y(c),stroke:"var(--grid3)"}));
      svg.appendChild(el("text",{x:mL-6,y:Y(c)+4.5,"text-anchor":"end",fill:"var(--muted)","font-size":FS},[txt(c.toFixed(1))]));});
    [lo,(lo+hi)/2,hi].forEach(v=>svg.appendChild(el("text",{x:X(v),y:H-mB+20,"text-anchor":"middle",fill:"var(--muted)","font-size":FS},[txt(v.toFixed(1))])));
    svg.appendChild(el("text",{x:(mL+W-mR)/2,y:H-6,"text-anchor":"middle",fill:"var(--muted)","font-size":FS},[txt("score")]));
    svg.appendChild(el("text",{x:13,y:(mT+H-mB)/2,"text-anchor":"middle",fill:"var(--muted)","font-size":FS,transform:`rotate(-90 13 ${(mT+H-mB)/2})`},[txt("fraction ≤ score")]));
    const paths=[];
    treated().forEach(o=>{const hh=(DATA.cells[o.id]||{})[e]?.hist;if(!hh)return;const tot=hh.counts.reduce((s,v)=>s+v,0)||1;let cum=0;
      const pts=[[X(lo),Y(0)]];hh.counts.forEach((c,i)=>{cum+=c/tot;pts.push([X(hh.edges[i+1]),Y(cum)]);});
      const path=el("path",{d:pts.map((q,i)=>(i?"L":"M")+q[0].toFixed(1)+" "+q[1].toFixed(1)).join(" "),fill:"none",stroke:methodColor(o.method),"stroke-width":1.6,opacity:.55,cursor:"pointer"});
      path.addEventListener("mouseover",()=>{paths.forEach(pp=>pp.setAttribute("opacity",.12));path.setAttribute("opacity",1);path.setAttribute("stroke-width",3);});
      path.addEventListener("mouseout",()=>{paths.forEach(pp=>{pp.setAttribute("opacity",.55);pp.setAttribute("stroke-width",1.6);});});
      path.addEventListener("mousemove",ev=>showTT(ev,`${orgHead(o)} · ${ph(o.persona)}`));path.addEventListener("mouseleave",hideTT);
      svg.appendChild(path);paths.push(path);});
    grid.appendChild(panelBox(pretty(DATA.evals[e].display),svg));
  });
  s.appendChild(grid);s.appendChild(methodColorLegendAll());container.appendChild(s);
}
