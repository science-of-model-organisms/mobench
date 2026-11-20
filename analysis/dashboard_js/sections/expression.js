// Expression threshold.
// ---- expression threshold: did the implant take? ----
// simple_behaviour_exhibit present rate per organism, with its 95% interval and, per behaviour, the
// base rate and the base + delta line. The verdict (analysis/derived.py sbe_verdicts) clears the threshold when the
// item-clustered lower bound of (organism - base) exceeds delta, so the base + delta line is a guide
// to the eye, not the test itself.

function expressionSection(container){
  const s=sec("Expression threshold - did the implant take?","expression");
  const E=DATA.expression||{};const delta=E.delta!=null?E.delta:0.10;
  howto(s,`Share of answers in which the judge saw the implanted behaviour (<code>${E.eval||"simple_behaviour_exhibit"}</code>), per organism, with its 95% interval. The dashed line is the base model's own rate; the dotted line is base + <b>${delta.toFixed(2)}</b> (guide only; the threshold is applied to the item-clustered lower bound of the gain over base). An organism clears the expression threshold when the lower bound of its per-item gain over base exceeds ${delta.toFixed(2)}; the verdict is printed at the right (PASS or FAIL: only a FAIL falls below it).`);
  const XW=xyWidth(),narrow=XW<600;
  // the key is one line (about 460px) or, in a narrow pane, two
  const lgW=narrow?XW:470,lgH=narrow?44:24,x2=narrow?4:250,y2=narrow?33:12;
  const lg=natural(el("svg",{viewBox:`0 0 ${lgW} ${lgH}`,width:lgW,height:lgH,class:"exprlegend"}),lgW);
  lg.appendChild(el("line",{x1:4,y1:12,x2:34,y2:12,stroke:"var(--ring)","stroke-dasharray":"5 3","stroke-width":1.4}));
  lg.appendChild(el("text",{x:42,y:17,fill:"var(--muted)","font-size":FS},[txt("base model's own rate")]));
  lg.appendChild(el("line",{x1:x2,y1:y2,x2:x2+30,y2:y2,stroke:"var(--accent)","stroke-dasharray":"2 3","stroke-width":1.6}));
  lg.appendChild(el("text",{x:x2+38,y:y2+5,fill:"var(--muted)","font-size":FS},[txt(`base + ${delta.toFixed(2)} (guide only)`)]));
  const lw=div("figwrap");lw.appendChild(lg);s.appendChild(lw);
  const W=Math.min(1000,XW),mL=narrow?190:250,mR=narrow?64:90,rowH=24;let drew=false;
  spreadGroups().forEach(({key:pn,orgs:all})=>{
    const orgs=all.filter(o=>exprOf(o.id));if(!orgs.length)return;
    drew=true;
    const bid=baseId(orgs[0].persona,bmOf(orgs[0])),b=bid?exprOf(bid):null;   // the group's own control arm
    const h=personaHead(pn);s.appendChild(h);
    const H=orgs.length*rowH+48,mT=8;
    const X=v=>mL+Math.max(0,Math.min(1,v))*(W-mL-mR);
    const svg=natural(el("svg",{viewBox:`0 0 ${W} ${H}`,width:W,height:H,class:"exprchart"}),W);
    for(let t=0;t<=1.0001;t+=0.2){svg.appendChild(el("line",{x1:X(t),y1:mT,x2:X(t),y2:H-30,stroke:"var(--grid2)"}));
      svg.appendChild(el("text",{x:X(t),y:H-8,"text-anchor":"middle",fill:"var(--muted)","font-size":FS},[txt(t.toFixed(1))]));}
    orgs.forEach((o,i)=>{const e=exprOf(o.id),y=mT+i*rowH;
      const g=el("g",{class:"exprrow",cursor:"pointer"});
      rowLabel(g,mL-8,y+rowH/2+4,o,FS);
      g.appendChild(el("rect",{x:X(0),y:y+4,width:Math.max(0,X(e.rate)-X(0)),height:rowH-8,fill:methodColor(o.method),opacity:.85,rx:2}));
      if(e.lo!=null&&e.hi!=null){g.appendChild(el("line",{x1:X(e.lo),y1:y+rowH/2,x2:X(e.hi),y2:y+rowH/2,stroke:"var(--fg)","stroke-width":1.4}));
        [e.lo,e.hi].forEach(v=>g.appendChild(el("line",{x1:X(v),y1:y+7,x2:X(v),y2:y+rowH-7,stroke:"var(--fg)","stroke-width":1.4})));}
      g.appendChild(el("text",{x:W-mR+10,y:y+rowH/2+4.5,"font-size":FS,fill:e.verdict==="FAIL"?"var(--bad)":"var(--muted)"},[txt(verdictLabel(e.verdict)||"n/a")]));
      g.addEventListener("mousemove",ev=>showTT(ev,`${orgHead(o)} · ${ph(o.persona)}<br>expression rate ${fmt(e.rate,2)} (95% ${fmt(e.lo,2)} to ${fmt(e.hi,2)})${e.verdict?" · "+verdictLabel(e.verdict):""}`));
      g.addEventListener("mouseleave",hideTT);svg.appendChild(g);});
    if(b){svg.appendChild(el("line",{x1:X(b.rate),y1:mT,x2:X(b.rate),y2:H-30,stroke:"var(--ring)","stroke-dasharray":"5 3","stroke-width":1.4}));
      svg.appendChild(el("line",{x1:X(b.rate+delta),y1:mT,x2:X(b.rate+delta),y2:H-30,stroke:"var(--accent)","stroke-dasharray":"2 3","stroke-width":1.6}));}
    const w=div("figwrap center");w.appendChild(svg);s.appendChild(w);
  });
  if(!drew)s.appendChild(empty("No expression-threshold cells for this selection."));
  container.appendChild(s);
}
