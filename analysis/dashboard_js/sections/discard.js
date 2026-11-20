// Discard rate vs score.
function discardScatter(container){
  const s=sec("Are the scores survivorship-biased? - discard vs score","discard");
  howto(s,`<b>One point per eval</b> (averaged over organisms), coloured by property. x = <b>discard rate</b> (share of samples that couldn't be scored), y = <b>mean score</b>. Points in the top-right - high score AND high discard - are the ones to worry about: the score may rest on the outputs that survived. Bubble size = organisms contributing.`);
  // aggregate per eval across the visible organisms. Kept in JS on purpose (not in analysis/dashboard_stats.py): a mean over whichever organisms the
  // filters leave, and the chips reach 6,376 distinct visible sets (x, y and n for each of 74 evals, 13 MB) against a 1.8 MB blob without
  // transcripts. The per-cell discard and mean are exported; only their mean over treated() (select.js) is formed here.
  const rows=[];Object.keys(DATA.evals).forEach(e=>{const ds=[],ms=[];
    treated().forEach(o=>{const c=(DATA.cells[o.id]||{})[e];if(c&&c.discard!=null&&c.mean!=null){ds.push(c.discard);ms.push(c.mean);}});
    if(ds.length)rows.push({e,x:ds.reduce((s,v)=>s+v,0)/ds.length,y:ms.reduce((s,v)=>s+v,0)/ms.length,n:ds.length,axis:DATA.evals[e].axis});});
  if(!rows.length){s.appendChild(empty("No evals carry both discard and score."));container.appendChild(s);return;}
  const axcol=bucketColor;
  const W=Math.min(820,xyWidth()),H=460,mL=62,mR=W<600?20:150,mT=22,mB=60;   // drawn at the width it is shown at (rebuilt on resize), so 13px is 13px
  // a mean score can be below 0 (a signed score): the y range widens down to the lowest eval so no bubble leaves the frame
  const yLo=Math.min(0,Math.floor(Math.min(...rows.map(r=>r.y))*4)/4);
  const X=v=>mL+v*(W-mL-mR),Y=v=>H-mB-((v-yLo)/(1-yLo))*(H-mT-mB);
  const svg=natural(el("svg",{viewBox:`0 0 ${W} ${H}`,width:W,height:H}),W);
  svg.appendChild(el("rect",{x:mL,y:mT,width:W-mL-mR,height:H-mT-mB,fill:"var(--plot)",stroke:"var(--line)"}));
  // shade the top-right "worry" zone
  svg.appendChild(el("rect",{x:X(0.5),y:mT,width:X(1)-X(0.5),height:Y(0.5)-mT,fill:"color-mix(in srgb,var(--bad) 8%,transparent)",stroke:"none"}));
  for(let t=0;t<=1.0001;t+=0.25){svg.appendChild(el("line",{x1:X(t),y1:mT,x2:X(t),y2:H-mB,stroke:"var(--grid3)"}));
    svg.appendChild(el("text",{x:X(t),y:H-mB+21,"text-anchor":"middle",fill:"var(--muted)","font-size":FS},[txt((t*100).toFixed(0)+"%")]));}
  for(let t=yLo;t<=1.0001;t+=0.25){svg.appendChild(el("line",{x1:mL,y1:Y(t),x2:W-mR,y2:Y(t),stroke:"var(--grid3)"}));
    svg.appendChild(el("text",{x:mL-6,y:Y(t)+4.5,"text-anchor":"end",fill:"var(--muted)","font-size":FS},[txt(t.toFixed(2))]));}
  svg.appendChild(el("text",{x:(mL+W-mR)/2,y:H-8,"text-anchor":"middle",fill:"var(--fg)","font-size":FS,"font-weight":600},[txt("discard rate (fraction unscored) →")]));
  svg.appendChild(el("text",{x:15,y:(mT+H-mB)/2,"text-anchor":"middle",fill:"var(--fg)","font-size":FS,"font-weight":600,transform:`rotate(-90 15 ${(mT+H-mB)/2})`},[txt("mean score")]));
  const _mk=[];
  // Labels: greedy placement on the same idea as figure 1 (a label may touch no bubble and no label already placed; it tries the
  // right, left, above and below of its bubble at growing distances, with a leader line when it is not adjacent). A bubble that
  // finds no free spot stays unlabelled (its tooltip still names it). Custom: page layout, no upstream equivalent.
  const dRad=r=>4+Math.sqrt(r.n)*0.9,boxHit=(a,b)=>Math.max(0,Math.min(a.x1,b.x1)-Math.max(a.x0,b.x0))*Math.max(0,Math.min(a.y1,b.y1)-Math.max(a.y0,b.y0));
  const bubBox=rows.map(r=>({x0:X(r.x)-dRad(r),x1:X(r.x)+dRad(r),y0:Y(r.y)-dRad(r),y1:Y(r.y)+dRad(r)}));
  const dLabs=[],dPlace=new Map();
  rows.map((r,i)=>({r,i})).filter(({r})=>r.x>0.3||r.y>0.85).sort((a,b)=>a.r.y-b.r.y||a.r.x-b.r.x).forEach(({r,i})=>{
    const nm=pretty(DATA.evals[r.e].display),lw=nm.length*FS*0.62+4,lh=17,rr=dRad(r),cx0=X(r.x),cy0=Y(r.y);
    // at most ~60px from its bubble: a label that finds no free spot stays unlabelled (its tooltip still names it)
    for(let ring=0;ring<6;ring++){const d=rr+3+ring*10;let ok=null;
      for(const [dx,dy] of [[1,0],[-1,0],[0,-1],[0,1],[1,-1],[-1,-1],[1,1],[-1,1]]){
        const ax=cx0+dx*d*(dy&&dx?.72:1),ay=cy0+dy*d*(dx?(dy?.72:1):1);
        const x0=dx>0?ax:dx<0?ax-lw:ax-lw/2,y0=dy>0?ay:dy<0?ay-lh:ay-lh/2,box={x0,x1:x0+lw,y0,y1:y0+lh};
        if(box.x0<mL||box.x1>W-6||box.y0<mT-8||box.y1>H-mB+2)continue;
        if(bubBox.some((b,j)=>j!==i&&boxHit(box,b)>0)||bubBox[i]&&boxHit(box,bubBox[i])>0||dLabs.some(o=>boxHit(box,o)>0))continue;
        ok=box;break;}
      if(ok){dLabs.push(ok);dPlace.set(i,{box:ok,ring});break;}}});
  svg.setAttribute("data-unlabelled",String(rows.filter((r,i)=>(r.x>0.3||r.y>0.85)&&!dPlace.has(i)).length));   // bubbles that found no free spot (their tooltip still names them)
  rows.forEach((r,i)=>{const g=el("g",{cursor:"pointer"});const col=axcol(r.axis);const rr=dRad(r);
    g.appendChild(el("circle",{cx:X(r.x),cy:Y(r.y),r:rr,fill:col,"fill-opacity":.55,stroke:"var(--halo)","stroke-width":1}));
    const pl=dPlace.get(i);
    if(pl){const b=pl.box,nx=Math.max(b.x0,Math.min(b.x1,X(r.x))),ny=Math.max(b.y0,Math.min(b.y1,Y(r.y)));
      if(Math.hypot(nx-X(r.x),ny-Y(r.y))>rr+4)g.appendChild(el("line",{x1:X(r.x),y1:Y(r.y),x2:nx,y2:ny,stroke:"var(--muted)","stroke-width":1,opacity:.7,class:"dlead"}));
      g.appendChild(el("text",{x:(b.x0+b.x1)/2,y:b.y1-4.5,"text-anchor":"middle",class:"dlab","data-box":[b.x0,b.y0,b.x1,b.y1].map(v=>v.toFixed(1)).join(","),fill:"var(--fg)","font-size":FS,stroke:"var(--halo)","stroke-width":2.4,"paint-order":"stroke"},[txt(pretty(DATA.evals[r.e].display))]));}
    g.addEventListener("mousemove",ev=>showTT(ev,`<b>${ph(DATA.evals[r.e].display)}</b> (${ph(r.axis)})<br>discard ${(r.x*100|0)}% · mean score ${r.y.toFixed(2)} · ${r.n} organisms`));g.addEventListener("mouseleave",hideTT);
    svg.appendChild(g);_mk.push(g);});
  fadeGroup(_mk);
  // property colour legend
  const lg=div("legend");DATA.axes.forEach(a=>{const k=document.createElement("span");k.className="k";
    k.innerHTML=`<span class="dot" style="background:${axcol(a.name)}"></span>${escHtml(pretty(a.short))}`;lg.appendChild(k);});
  const w=div("figwrap center");w.appendChild(svg);s.appendChild(w);s.appendChild(lg);container.appendChild(s);
}
