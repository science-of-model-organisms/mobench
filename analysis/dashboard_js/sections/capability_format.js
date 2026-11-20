// The capability drill's format-sensitivity figure.
// ---- capability_format: free-form vs multiple-choice (a format artifact, not the persona) ----
// The mc and free arms are stored as derived cells (`capability_format:{mc,free}`); it is NOT
// in the capability composite, so it never shows in the property radar. When the capability property
// is drilled we render it beside that radar: the same items asked with options shown vs withheld,
// to make the point that the mc-minus-free gap is uniform across organisms. Filter-aware: reads
// the group's organisms plus their own control arm.
function capabilityFormatFig(group){
  const MC="capability_format:mc",FREE="capability_format:free";
  const bid=group.length?baseId(group[0].persona,bmOf(group[0])):null,base=bid==null?undefined:ORG_BY_ID[bid];   // the group's own control arm
  const orgs=[...group,...(base?[base]:[])].sort(ctrlLast);
  const rows=orgs.map(o=>{const c=DATA.cells[o.id]||{};const mc=c[MC],fr=c[FREE];
    return (mc&&mc.mean!=null&&fr&&fr.mean!=null)?{o,mc:mc.mean,mcSem:mc.sem_hw,mcSeeds:mc.n_seeds,fr:fr.mean,frSem:fr.sem_hw,frSeeds:fr.n_seeds}:null;}).filter(Boolean);
  if(!rows.length)return null;
  const label=o=>o.is_base?"base":niceOrg(o);
  const wrap=document.createElement("div");
  const mcVals=rows.map(r=>r.mc),frVals=rows.map(r=>r.fr),gap=rows.map(r=>r.mc-r.fr);
  const lo=a=>Math.min(...a).toFixed(2),hi=a=>Math.max(...a).toFixed(2);
  const how=document.createElement("div");how.className="howto";
  how.innerHTML=`<b>Format sensitivity of capability_format.</b> The same mmlu/wmdp items asked twice - options <b>shown</b> (multiple-choice) and <b>withheld</b> (free-form exact match). Shown ${lo(mcVals)}-${hi(mcVals)}, withheld ${lo(frVals)}-${hi(frVals)}: the gap is ${lo(gap)}-${hi(gap)} for every organism, so capability_format measures the <b>answer format, not the persona</b>.`;
  wrap.appendChild(how);
  const n=rows.length,W=Math.max(560,150+n*88),H=440,mL=92,mR=210,mT=28,mB=112;
  const X=i=>mL+(i+0.5)/n*(W-mL-mR);
  const Y=v=>H-mB-Math.max(0,Math.min(1,v))*(H-mT-mB);   // fixed 0..1 exact-match rate axis
  // High-contrast dashboard-native pair (blue vs red, from the palette/diverge scale) - differ
  // in both hue and lightness so the two arms are clearly separable; not the picture's blue/orange.
  const C_MC="#4c72b0",C_FREE="#c44e52";
  const svg=natural(el("svg",{viewBox:`0 0 ${W} ${H}`,width:W,height:H}),W);
  svg.appendChild(el("rect",{x:mL,y:mT,width:W-mL-mR,height:H-mT-mB,fill:"var(--plot)",stroke:"var(--line)"}));
  for(let t=0;t<=1.0001;t+=0.25){svg.appendChild(el("line",{x1:mL,y1:Y(t),x2:W-mR,y2:Y(t),stroke:"var(--grid)","stroke-width":1}));
    svg.appendChild(el("text",{x:mL-8,y:Y(t)+4.5,"text-anchor":"end",fill:"var(--muted)","font-size":FS},[txt(t.toFixed(2))]));}
  const yc=(mT+H-mB)/2;
  svg.appendChild(el("text",{x:22,y:yc,"text-anchor":"middle",fill:"var(--fg)","font-size":FS,"font-weight":600,transform:`rotate(-90 22 ${yc})`},[txt("exact-match rate")]));
  const arm=(cx,v,hw,col,nSeeds)=>{const g=el("g");
    if(hw!=null&&isFinite(hw)){
      g.appendChild(el("line",{x1:cx,y1:Y(v+hw),x2:cx,y2:Y(v-hw),stroke:col,"stroke-width":1.6}));
      g.appendChild(el("line",{x1:cx-3,y1:Y(v+hw),x2:cx+3,y2:Y(v+hw),stroke:col,"stroke-width":1.6}));
      g.appendChild(el("line",{x1:cx-3,y1:Y(v-hw),x2:cx+3,y2:Y(v-hw),stroke:col,"stroke-width":1.6}));}
    g.appendChild(el("circle",{cx,cy:Y(v),r:5.5,fill:col,stroke:"var(--halo)","stroke-width":1}));return g;};
  // One group per organism, tagged data-method so the drill's implant-method chips hide its
  // column (applyDrillVis), and collected for fadeGroup so hovering one dims the rest.
  const groups=[];
  rows.forEach((r,i)=>{const cx=X(i);
    const gg=el("g",{cursor:"pointer"});if(!r.o.is_base)gg.setAttribute("data-method",serOf(r.o));
    gg.appendChild(arm(cx,r.mc,r.mcSem,C_MC,r.mcSeeds));
    gg.appendChild(arm(cx,r.fr,r.frSem,C_FREE,r.frSeeds));
    gg.appendChild(el("text",{x:cx,y:H-mB+14,"text-anchor":"end",fill:"var(--fg)","font-size":FS,transform:`rotate(-40 ${cx} ${H-mB+14})`},[txt(label(r.o))]));
    const hit=el("rect",{x:cx-11,y:mT,width:22,height:H-mT-mB,fill:"transparent"});
    hit.addEventListener("mousemove",e=>showTT(e,`<b>${escHtml(label(r.o))}</b><br>mc ${r.mc.toFixed(3)} · free ${r.fr.toFixed(3)}<br>gap ${(r.mc-r.fr).toFixed(3)}`));
    hit.addEventListener("mouseleave",hideTT);gg.appendChild(hit);
    svg.appendChild(gg);groups.push(gg);});
  fadeGroup(groups);
  const lx=W-mR+24;let ly=mT+12;
  [["options shown (mc)",C_MC],["options withheld (free)",C_FREE]].forEach(([t,c])=>{
    svg.appendChild(el("circle",{cx:lx+6,cy:ly,r:5.5,fill:c}));
    svg.appendChild(el("text",{x:lx+18,y:ly+4,fill:"var(--fg)","font-size":FS},[txt(t)]));ly+=24;});
  const fw=document.createElement("div");fw.className="figwrap";fw.appendChild(svg);wrap.appendChild(fw);
  return wrap;
}
