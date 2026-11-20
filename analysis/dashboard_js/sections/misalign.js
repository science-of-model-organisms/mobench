// Misalignment dumbbells.
// ---- misalignment dumbbells: raw rate, each organism next to its own-size base model ----
// One panel per misalignment evaluation. Rows = the organisms the filters show; hollow grey dot = the
// same-size base model, filled dot = the organism (hollow when it falls below the expression
// threshold, as in the paper's misalignment figure), the line between them is the move, whiskers are
// 95% intervals (mean +/- 1.96 x sem, as in the forest). Misalignment is a health
// check: no good/bad colouring, a higher rate is the misaligned direction.
function misalignDumbbell(container){
  const evals=DATA.bucket_evals["misalignment"]||[];
  if(!evals.length)return;
  const s=sec("Misalignment - raw rates next to the base model","misalign");
  howto(s,`A higher misalignment rate is the misaligned direction. Per evaluation, each organism's <b>raw rate</b> (filled dot) joined to the rate of the <b>base model of its own size</b> (hollow grey dot), with <b>95% intervals</b>. A hollow coloured dot is an organism that falls below the expression threshold. Same view as the paper's misalignment figure.`);
  const orgs=treated();
  if(!orgs.length){s.appendChild(empty("Nothing to show."));container.appendChild(s);return;}
  const grid=div("radargrid");
  evals.forEach(e=>{
    const rows=[];
    orgs.forEach(o=>{const c=(DATA.cells[o.id]||{})[e];const b=(DATA.cells[baseId(o.persona,_BM[o.id])]||{})[e];
      if(!c||c.mean==null||!b||b.mean==null)return;
      const bid=baseId(o.persona,_BM[o.id]);
      // an arm's whisker needs a real SEM (>0); the difference uses the exported se, which is unknown at n<=1
      const armM=x=>{const v=fin(x.sem_hw);return v!=null&&v>0?v:null;};   // sem_hw = 1.96 x sem (Python)
      rows.push({o,c,b,om:armM(c),bm:armM(b),dm:fin(c.hw)});});
    if(!rows.length)return;
    let lo=0,hi=0.05;rows.forEach(r=>{lo=Math.min(lo,r.c.mean-(r.om||0),r.b.mean-(r.bm||0));hi=Math.max(hi,r.c.mean+(r.om||0),r.b.mean+(r.bm||0));});
    {const pd=(hi-lo)*0.05;lo-=pd;hi+=pd;}   // headroom so no whisker ends on (or is cut by) the frame
    const rh=18,mL=200,mR=22,W=mL+186+mR,mT=26,H=mT+rows.length*rh+28;
    const plotW=W-mL-mR,span=(hi-lo)||1,X=v=>mL+((v-lo)/span)*plotW;
    const svg=natural(el("svg",{viewBox:`0 0 ${W} ${H}`,width:W,height:H,class:"dumbbell"}),W);
    svg.appendChild(el("rect",{x:mL,y:mT-6,width:plotW,height:rows.length*rh+8,fill:"var(--plot)",stroke:"var(--line)"}));
    ticks([lo,hi]).forEach(t=>{svg.appendChild(el("line",{x1:X(t),y1:mT-6,x2:X(t),y2:mT+rows.length*rh+2,stroke:t===0?"var(--axis)":"var(--grid3)","stroke-dasharray":t===0?"3 3":"none"}));
      svg.appendChild(el("text",{x:X(t),y:H-7,"text-anchor":"middle",fill:"var(--muted)","font-size":FS},[txt(+t.toFixed(3)+"")]));});
    const _rows=[];
    rows.forEach((r,i)=>{const y=mT+i*rh+rh/2,col=methodColor(r.o.method),open=exprFlagged(r.o.id);
      const g=el("g",{cursor:"pointer",class:"dbrow"});
      if(r.bm)g.appendChild(el("line",{x1:X(r.b.mean-r.bm),y1:y,x2:X(r.b.mean+r.bm),y2:y,stroke:"var(--ring)","stroke-width":1,opacity:.5}));
      g.appendChild(el("line",{x1:X(r.b.mean),y1:y,x2:X(r.c.mean),y2:y,stroke:col,"stroke-width":2,opacity:.8}));
      if(r.om)g.appendChild(el("line",{x1:X(r.c.mean-r.om),y1:y,x2:X(r.c.mean+r.om),y2:y,stroke:col,"stroke-width":1.4,opacity:.6,"stroke-linecap":"round"}));
      g.appendChild(el("circle",{cx:X(r.b.mean),cy:y,r:3.6,fill:"var(--plot)",stroke:"var(--ring)","stroke-width":1.5,class:"dbbase"}));
      const m=marker(orgShape(r.o),X(r.c.mean),y,4.4,open?"var(--plot)":col);
      if(open){m.setAttribute("stroke",col);m.setAttribute("stroke-width",1.6);}
      m.setAttribute("class","dborg");g.appendChild(m);
      const dd=r.c.mean-r.b.mean;
      g.addEventListener("mousemove",ev=>showTT(ev,`${orgHead(r.o)} · ${ph(DATA.evals[e].display)}<br>organism ${r.c.mean.toFixed(3)} ${r.om==null?"(no 95% CI;":"± "+r.om.toFixed(3)+" (95% CI;"} n=${r.c.n}, ${orgSeeds(r.o.id)} seed${orgSeeds(r.o.id)===1?"":"s"})<br>same-size base ${r.b.mean.toFixed(3)}${r.bm==null?" (no 95% CI)":" ± "+r.bm.toFixed(3)}<br>organism − base ${fmt(dd)}${r.dm==null?" (no 95% CI)":" ± "+r.dm.toFixed(3)}${open?"<br>"+exprNote(r.o.id):""}<br><i>higher = more misaligned</i>`));
      g.addEventListener("mouseleave",hideTT);
      svg.appendChild(g);_rows.push(g);
      rowLabel(svg,mL-6,y+4,r.o,FS);});
    if(_rows.length>1)fadeGroup(_rows);
    grid.appendChild(panelBox(pretty(DATA.evals[e].display),svg));});
  s.appendChild(grid);
  const lg=div("legend");
  lg.innerHTML=`<span class="k"><svg width="16" height="12"><circle cx="8" cy="6" r="3.6" fill="var(--plot)" stroke="var(--ring)" stroke-width="1.5"/></svg>same-size base model</span><span class="k"><svg width="16" height="12"><circle cx="8" cy="6" r="4.2" fill="var(--accent)"/></svg>organism (marker shape and colour = method${EXT_LIST.length?"; pre-trained adapters share one marker":""})</span><span class="k"><svg width="16" height="12"><circle cx="8" cy="6" r="3.8" fill="var(--plot)" stroke="var(--accent)" stroke-width="1.6"/></svg>organism below the expression threshold</span><span class="k">whiskers = 95% interval</span>`;
  s.appendChild(lg);container.appendChild(s);
}
