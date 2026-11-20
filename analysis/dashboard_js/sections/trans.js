// Per-item transitions.
function transitionBars(container){
  const s=sec("Did it churn or just move? - per-item transitions","trans");
  howto(s,`Binary evals only. Each row is an organism; the bar centres on base - <span style="color:${diverge(1)}">gained</span> (base wrong → organism right; for lower-is-better evals "right" means the lower score) grows right, <span style="color:${diverge(-1)}">lost</span> grows left. A tiny net Δ with long bars means the implant churned many items, not moved a few.`);
  const evals=Object.keys(DATA.evals).filter(e=>treated().some(o=>(DATA.cells[o.id]||{})[e]?.trans));
  if(!evals.length){s.appendChild(empty("No per-item transition counts in this export (binary evals)."));container.appendChild(s);return;}
  const grid=div("radargrid");
  evals.forEach(e=>{const rows=treated().map(o=>({o,t:(DATA.cells[o.id]||{})[e]?.trans})).filter(r=>r.t);if(!rows.length)return;
    let ms=1;rows.forEach(r=>ms=Math.max(ms,r.t.gained,r.t.lost));
    // the label column is measured (13px label, ~0.58 em per glyph, plus the dot and gaps) so the longest name is never cut
    const mL=Math.max(140,Math.ceil(Math.max(...rows.map(r=>niceOrg(r.o).length))*FS*0.58)+26);
    const rh=18,mR=16,half=90,W=mL+2*half+mR,mT=32,H=mT+rows.length*rh+44,cx=mL+half,X=v=>cx+(v/ms)*half;
    const svg=natural(el("svg",{viewBox:`0 0 ${W} ${H}`,width:W,height:H}),W);
    // diverging gradient fills (pale at centre -> full colour at the end), like the ranked bars
    const defs=el("defs");
    const grad=(id,c0,c1)=>{const g2=el("linearGradient",{id,x1:"0",y1:"0",x2:"1",y2:"0"});
      g2.appendChild(el("stop",{offset:"0","stop-color":c0}));g2.appendChild(el("stop",{offset:"1","stop-color":c1}));defs.appendChild(g2);};
    grad("gGain",diverge(0.25),diverge(1));grad("gLost",diverge(-1),diverge(-0.25));
    svg.appendChild(defs);
    svg.appendChild(el("line",{x1:cx,y1:mT-6,x2:cx,y2:H-36,stroke:"var(--axis)"}));
    // header words + tick counts
    svg.appendChild(el("text",{x:cx-4,y:mT-12,"text-anchor":"end",fill:diverge(-1),"font-size":FS,"font-weight":600},[txt("← lost")]));
    svg.appendChild(el("text",{x:cx+4,y:mT-12,"text-anchor":"start",fill:diverge(1),"font-size":FS,"font-weight":600},[txt("gained →")]));
    [...new Set([Math.round(ms/2),ms])].filter(v=>v>0).forEach(v=>{svg.appendChild(el("text",{x:X(v),y:H-22,"text-anchor":"middle",fill:"var(--muted)","font-size":FS},[txt(""+v)]));
      svg.appendChild(el("text",{x:X(-v),y:H-22,"text-anchor":"middle",fill:"var(--muted)","font-size":FS},[txt(""+v)]));});
    svg.appendChild(el("text",{x:cx,y:H-22,"text-anchor":"middle",fill:"var(--muted)","font-size":FS},[txt("0")]));
    svg.appendChild(el("text",{x:cx,y:H-6,"text-anchor":"middle",fill:"var(--muted)","font-size":FS,class:"axlab"},[txt("items")]));
    const _tr=[];
    rows.forEach((r,i)=>{const y=mT+i*rh+2,h=rh-5,net=r.t.gained-r.t.lost;const gg=el("g",{cursor:"pointer"});
      gg.appendChild(el("rect",{x:cx,y,width:Math.max(0,X(r.t.gained)-cx),height:h,rx:2,fill:"url(#gGain)"}));
      gg.appendChild(el("rect",{x:X(-r.t.lost),y,width:Math.max(0,cx-X(-r.t.lost)),height:h,rx:2,fill:"url(#gLost)"}));
      gg.appendChild(el("line",{x1:X(net),y1:y-1,x2:X(net),y2:y+h+1,stroke:"var(--fg)","stroke-width":1.4}));
      gg.addEventListener("mousemove",ev=>showTT(ev,`${orgHead(r.o)} · ${ph(DATA.evals[e].display)}<br>gained ${r.t.gained} · lost ${r.t.lost} · kept ${(r.t.keptPos||0)+(r.t.keptNeg||0)}<br>net ${net>=0?"+":""}${net} · churn ${r.t.gained+r.t.lost}`));gg.addEventListener("mouseleave",hideTT);
      svg.appendChild(gg);_tr.push(gg);
      rowLabel(svg,mL-6,y+h/2+4,r.o,FS);
    });
    if(_tr.length>1)fadeGroup(_tr);
    grid.appendChild(panelBox(pretty(DATA.evals[e].display),svg));
  });
  const lg=div("legend");lg.innerHTML=`<span class="k"><span style="width:10px;height:2px;background:var(--fg);display:inline-block"></span>vertical tick = net change (gained - lost)</span>`;
  s.appendChild(grid);s.appendChild(lg);container.appendChild(s);
}
