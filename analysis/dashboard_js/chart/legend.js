// Legends and keys shared by several figures.
function legendInPlot(svg,x,y){
  let yy=y;
  spreadGroups().forEach(g=>{
    svg.appendChild(el("circle",{cx:x+6,cy:yy,r:5,fill:spreadColor(g.key)}));
    const lt=el("text",{x:x+16,y:yy+4,fill:"var(--fg)","font-size":FS},[txt(spreadLabel(g.key))]);
    if(state.mode==="model"){lt.setAttribute("data-model",g.key);lt.setAttribute("cursor","pointer");}
    svg.appendChild(lt);yy+=20;});
  yy+=6;
  visibleMethods().filter(m=>!isExtMethod(m)).forEach(m=>{
    svg.appendChild(marker(methodShape(m),x+6,yy,5,"var(--ring)"));
    const lt=el("text",{x:x+16,y:yy+4,fill:"var(--fg)","font-size":FS},[txt(pretty(m))]);
    const c=methodCite(m);
    if(c){const a=el("a",{href:c.url,target:"_blank",rel:"noopener","class":"mcite"});
      a.appendChild(el("text",{x:x+16,y:yy+20,fill:"var(--muted)","font-size":12},[txt("("+c.short+")")]));
      svg.appendChild(a);}
    svg.appendChild(lt);yy+=c?37:20;});
  if(visibleMethods().some(isExtMethod)){svg.appendChild(marker("hexagon",x+6,yy,5,"var(--ring)"));   // one entry for every pre-trained adapter
    svg.appendChild(el("text",{x:x+16,y:yy+4,fill:"var(--fg)","font-size":FS},[txt("Pre-trained adapters")]));yy+=20;}
  controlLegendItems().forEach(name=>{svg.appendChild(marker("minus",x+6,yy,5,"var(--ring)"));
    svg.appendChild(el("text",{x:x+16,y:yy+4,fill:"var(--fg)","font-size":FS},[txt(name)]));yy+=20;});
}
function dualLegend(){
  const l=document.createElement("div");l.className="legend";
  spreadGroups().forEach(g=>{const k=document.createElement("span");k.className="k";
    k.innerHTML=`<span class="dot" style="background:${spreadColor(g.key)}"></span>${escHtml(spreadLabel(g.key))}`;modelTrigger(k,g.key);l.appendChild(k);});
  const sep=document.createElement("span");sep.textContent="│";sep.style.color="var(--line)";l.appendChild(sep);
  visibleMethods().filter(m=>!isExtMethod(m)).forEach(m=>{const k=document.createElement("span");k.className="k";
    const sv=el("svg",{width:16,height:16,viewBox:"0 0 16 16"});sv.appendChild(marker(methodShape(m),8,8,5,"var(--muted)"));
    k.appendChild(sv);k.appendChild(txt(pretty(m)));
    const c=methodCite(m);
    if(c){k.appendChild(txt(" "));const a=document.createElement("a");a.href=c.url;a.target="_blank";a.rel="noopener";a.className="mcite";
      a.textContent="("+c.short+")";k.appendChild(a);}
    l.appendChild(k);});
  if(visibleMethods().some(isExtMethod)){const k=document.createElement("span");k.className="k";   // one entry for every pre-trained adapter
    const sv=el("svg",{width:16,height:16,viewBox:"0 0 16 16"});sv.appendChild(marker("hexagon",8,8,5,"var(--muted)"));
    k.appendChild(sv);k.appendChild(txt("Pre-trained adapters"));l.appendChild(k);}
  controlLegendItems().forEach(name=>{const k=document.createElement("span");k.className="k";
    const sv=el("svg",{width:16,height:16,viewBox:"0 0 16 16"});sv.appendChild(marker("minus",8,8,5,"var(--muted)"));
    k.appendChild(sv);k.appendChild(txt(name));l.appendChild(k);});
  return l;
}

function methodColorLegend(){
  const l=document.createElement("div");l.className="legend";
  DATA.meta.methods.forEach(m=>{const k=document.createElement("span");k.className="k";k.dataset.method=m;
    k.innerHTML=`<span class="dot" style="background:${methodColor(m)}"></span>${escHtml(pretty(m))}`;l.appendChild(k);});
  return l;
}
// One entry per organism (each dose/variant its own colour); tagged data-method so the
// method filter hides its entry too.
// `orgs`: the organisms of the figure it sits under (a drill-down lists its own behaviour's organisms, not every visible one)
function orgColorLegend(orgs){
  const l=document.createElement("div");l.className="legend";
  (orgs||treated()).forEach(o=>{const k=document.createElement("span");k.className="k";k.dataset.method=serOf(o);
    k.innerHTML=`<span class="dot" style="background:${orgColorKey(o)}"></span>${escHtml(niceOrg(o))}`;l.appendChild(k);});
  return l;
}

function spreadLegend(){const l=div("legend");spreadGroups().forEach(g=>{
  const k=document.createElement("span");k.className="k";k.innerHTML=`<span class="dot" style="background:${spreadColor(g.key)}"></span>${escHtml(spreadLabel(g.key))}`;modelTrigger(k,g.key);l.appendChild(k);});return l;}
// In "compare models" mode the spread dimension IS the base model, so its names (group headings, legend entries) open that model's drawer.
function modelTrigger(n,key){if(state.mode==="model"&&BASE_MODELS.includes(key)){n.dataset.model=key;n.title="Open the "+bmName(key)+" drill-down";}return n;}
function personaHead(pn){const h=div("persona-h");h.textContent=spreadHead(pn);return modelTrigger(h,pn);}

// Colour key mapping each of the property's evals to its bar colour (paired-bars companion to radarKey).
function evalColorLegend(spokes){
  const l=div("legend");
  spokes.forEach(sp=>{const k=document.createElement("span");k.className="k";
    k.innerHTML=`<span class="dot" style="background:${evalColor(sp.key)}"></span>${escHtml(pretty(sp.short))}`;l.appendChild(k);});
  const k=document.createElement("span");k.className="k";k.innerHTML="further from centre = larger change (&Delta; vs base, score units; + = better)";l.appendChild(k);
  const h=document.createElement("span");h.className="k";h.innerHTML="hover a row to highlight its implant method";l.appendChild(h);
  return l;
}

function methodColorLegendAll(){const l=div("legend");visibleMethods().forEach(m=>{
  const k=document.createElement("span");k.className="k";k.innerHTML=`<span class="dot" style="background:${methodColor(m)}"></span>${escHtml(pretty(m))}`;l.appendChild(k);});return l;}

function radarKey(kind,opts){
  const l=document.createElement("div");l.className="legend";
  const dashKey=`<span class="k flagkey"><svg width="30" height="12"><line x1="2" y1="6" x2="28" y2="6" stroke="var(--ring)" stroke-width="2" stroke-dasharray="7 4"/></svg>${opts&&opts.model?"dashed polygon = at least one of its organisms falls below the expression threshold":"dashed outline = falls below the expression threshold (verdict not PASS)"}</span>`;
  if(kind==="paper"&&opts&&opts.axis){   // a single property (a drill-down): only the rule that applies to its spokes
    const a=opts.axis,share=axShare(a),health=HEALTH_AXES.has(a);
    l.innerHTML=(share?"":`<span class="k">${DELTA_DEF}</span>`)+`<span class="k">${axisFormula(a)}</span>`+
      (share?"":health?`<span class="k">further out = better; the outer ring (10) is <b style="color:var(--fg)">base</b></span><span class="k">${MATCHED_NOTE}</span>`
        :`<span class="k">further out = better; a value below base is drawn inside the dashed base ring (at 0)</span><span class="k">${MATCHED_NOTE}</span><span class="k">${RADAR_NOTE}</span>`)+dashKey;
    return l;
  }
  if(kind==="paper"){
    l.innerHTML=
      `<span class="k">${DELTA_DEF}</span>`+
      `<span class="k"><span>further out = better. Behaviour (generality, introspection, activations): ${FORMULA.behaviour}; a value below base is drawn inside the dashed base ring (at 0)</span></span>`+
      `<span class="k"><svg width="24" height="12"><circle cx="12" cy="6" r="5" fill="none" stroke="var(--ring)" stroke-width="1.6"/></svg><span>Health (coherence, capability): ${FORMULA.health}, so the outer ring (10) is <b style="color:var(--fg)">base</b></span></span>`+
      `<span class="k"><span>Robustness: ${FORMULA.share}</span></span>`+`<span class="k">${MATCHED_NOTE}</span><span class="k">${RADAR_NOTE}</span>`+dashKey;
    return l;
  }
  l.innerHTML=
    `<span class="k">further out from centre = better (&Delta; vs base in score units, 0 = base; misalignment: outward = less misbehaviour)</span>`+dashKey;
  return l;
}
