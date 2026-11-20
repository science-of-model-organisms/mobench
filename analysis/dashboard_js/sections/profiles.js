// Profiles: per property (with the inline drill), per organism, per model.
// Only these sections depend on the selected property, so a property click repaints just them.
function selectBucket(b){const prev=state.bucket;state.bucket=(prev===b?null:b);updateBucketSelection(prev);rebuildFigs(["heatmap"]);}
function setBucketBanner(bnr){
  bnr.innerHTML=state.bucket
    ?`<b>Selected property: ${ph(state.bucket)}</b> - drill-down opened inline below the property; the effect matrix is also expanded. <span class="clearsel">clear ✕</span>`
    :`Tip: <b>click any property title</b> (marked "click to drill ›") and its drill-down slides open right underneath.`;
  if(state.bucket)bnr.querySelector(".clearsel").onclick=()=>selectBucket(state.bucket);
}
// In-place update of the property-radar section (no re-render, so the scroll position is kept):
// slide the inline drill open/closed under the selected panel and update panel highlights.
function updateBucketSelection(prev){
  const sec=document.querySelector('section[data-fig="radars"]');
  if(!sec){render();return;}   // section not present (e.g. filtered) - fall back
  // collapse & remove any open inline drills (animated)
  sec.querySelectorAll(".drillinline").forEach(n=>{const hgt=n.scrollHeight;
    n.style.maxHeight=hgt+"px";n.style.overflow="hidden";
    requestAnimationFrame(()=>{n.style.transition="max-height .28s ease,opacity .28s ease";n.style.maxHeight="0";n.style.opacity="0";});
    setTimeout(()=>n.remove(),300);});
  // panel highlight + title state
  sec.querySelectorAll(".rpanel[data-bucket]").forEach(p=>{const on=p.dataset.bucket===state.bucket;
    p.classList.toggle("selpanel",on);
    const t=p.querySelector(".rt"),ax=DATA.axes.find(a=>a.name===p.dataset.bucket);
    if(t){t.classList.toggle("sel",on);
      t.innerHTML=`${escHtml(cap(pretty(ax?ax.short:p.dataset.bucket)))} <span class="drillhint">${on?"✓ selected":"click to drill ›"}</span>`;}
  });
  // insert fresh inline drills after the newly-selected panel in each persona grid
  if(state.bucket)sec.querySelectorAll(".radargrid[data-group]").forEach(grid=>{
    const sel=grid.querySelector(`.rpanel[data-bucket="${state.bucket}"]`);
    if(sel)sel.after(drillInline(grid.dataset.group));});
  const bnr=sec.querySelector(".selbanner");if(bnr)setBucketBanner(bnr);
}

// A spoke click in the per-organism view jumps to the per-property view with that property's drill open
// (the property view's own panel click toggles; this one only ever opens).
function spokeToBucket(key){state.radarView="bucket";state.bucket=key;rebuildFigs(["radars","heatmap"]);}

// Row 1: one radar per organism, spokes = properties
function radarByOrganism(container){
  const s=document.createElement("section");s.dataset.fig="byOrg";s.innerHTML=`<h2>Profile per organism <span class="pill">${PAPER_SCALE?"score":"&Delta; vs base"}</span></h2>`;
  howto(s,PAPER_SCALE
    ?`Each mini-radar is one <b>implant × behaviour</b> combination; the spokes are the <b>properties</b> on the radar (misalignment and expression are off it by design; see their panels). ${DELTA_DEF}. <b>Behaviour spokes</b>: ${FORMULA.behaviour}, so the base is a dashed ring at 0 and values below the base model are drawn inside it; <b>health spokes</b>: ${FORMULA.health}, so the base is the <b>outer ring</b>; <b>robustness</b>: ${FORMULA.share} (the base model has none: no behaviour to keep). ${RADAR_NOTE} A <b>dashed outline</b> marks an organism that falls below the expression threshold. Hover for the values and for &Delta; with its 95% interval. <b>Click a property label</b> to highlight it everywhere and drill into its individual evals below.`
    :`Each mini-radar is one <b>implant × behaviour</b> combination; the spokes are the <b>properties</b> on the radar (misalignment and expression are off it by design; see their panels). Distance from the grey centre = how far that property moved from the base model, as &Delta; in score units (outward = better). A <b>dashed outline</b> marks an organism that falls below the expression threshold. <b>Click a property label</b> to highlight it everywhere and drill into its individual evals below.`);
  const spokes=radarAxes().map(a=>({key:a.name,short:a.short}));
  spreadGroups().forEach(({key:pn,orgs})=>{
    const h=personaHead(pn);s.appendChild(h);
    const grid=document.createElement("div");grid.className="radargrid";
    orgs.forEach(o=>{
      const vals=spokes.map(sp=>PAPER_SCALE?radarOf(o.id,sp.key):axisEff(o.id,sp.key));
      const flagged=exprFlagged(o.id);
      const lines=spokes.map(sp=>{const v=PAPER_SCALE?radarOf(o.id,sp.key):null,r=axisRead(o.id,sp.key);
        const dl=r==null?"":(r.share?"share kept "+fmt(r.d,2):(PAPER_SCALE?"page Δ ":"Δ ")+fmt(r.d,2)+(r.lo==null?"":" ["+fmt(r.lo,2)+", "+fmt(r.hi,2)+"]")+(r.back?" (+ = more misbehaviour)":""));
        const md=PAPER_SCALE&&v!=null&&!(r&&r.share)?" · matched "+matchedDelta(sp.key,v):"";   // the Δ that drives the radius
        return `${ph(sp.short)}: ${PAPER_SCALE?(v==null?"n/a":"score "+v.toFixed(1)):""}${md}${dl==""?"":(PAPER_SCALE?" · ":"")+dl}`;}).join("<br>");
      const svg=drawRadar(spokes,[{values:vals,color:spreadColor(pn),dashed:flagged,
          tip:`${orgHead(o)}<br>${lines}${flagged?"<br><i>"+exprNote(o.id)+"</i>":""}`}],
        {R:104,selLabel:state.bucket,onSpoke:spokeToBucket,paper:PAPER_SCALE});
      // an organism outside the paper's set (e.g. a retired variant) has no radar value: say so
      if(PAPER_SCALE&&vals.every(v=>v==null))svg.appendChild(el("text",{x:svg.viewBox.baseVal.width/2,y:svg.viewBox.baseVal.height/2+4,"text-anchor":"middle",fill:"var(--muted)","font-size":FS,class:"noreading"},[txt("no radar value")]));
      const pn1=panel(null,niceOrg(o),false,false,svg),rt1=pn1.querySelector(".rt");
      rt1.dataset.org=o.id;rt1.title="Open this organism's drill-down";grid.appendChild(pn1);
    });
    s.appendChild(grid);
  });
  s.appendChild(radarKey(PAPER_SCALE?"paper":"delta"));
  container.appendChild(s);
}

// Row 2: one radar per (property × behaviour), spokes = organisms (implant methods)
function radarByBucket(container){
  const s=document.createElement("section");s.dataset.fig="byBucket";s.innerHTML=`<h2>Implant spread per property <span class="pill">${PAPER_SCALE?"score":"&Delta; vs base"}</span></h2>`;
  howto(s,`Each mini-radar is one <b>property × behaviour</b>; the spokes are the <b>implant methods</b> (a method's doses and variants averaged). ${PAPER_SCALE?`${DELTA_DEF}. Each method gets a score on the property, by the formula printed under the panel title: <b>behaviour</b> properties (generality, introspection, activations) ${FORMULA.behaviour}, so the base is a dashed ring at 0 and a value below it is drawn inside it; <b>health</b> properties (coherence, capability) ${FORMULA.health}, so the base is the outer ring; <b>robustness</b> ${FORMULA.share}. ${RADAR_NOTE} Hover for the score and the signed &Delta; (or share kept) with its 95% interval.`:`This build has no paper-scale radar values, so the panels are drawn as <b>&Delta; vs base</b> in score units (robustness is the share kept).`} A spiky panel means the method choice matters there; a round one means they behave alike. <b>Click a panel title to drill into that property's individual evals</b> (below and in the effect matrix).`);
  const bnr=document.createElement("div");bnr.className="selbanner";
  setBucketBanner(bnr);
  s.appendChild(bnr);
  spreadGroups().forEach(({key:pn,orgs})=>{
    // spokes = implant METHODS (doses/variants of a method averaged), not individual organisms
    const methods=serList(orgs);
    const spokes=methods.map(m=>({key:m,short:serShort(m)}));
    const h=personaHead(pn);s.appendChild(h);
    const grid=document.createElement("div");grid.className="radargrid";grid.dataset.group=pn;
    radarAxes().forEach(a=>{
      // a share axis (robustness) is plotted as the share kept on 0..1, not a Δ: axisEff drops it, so read the oriented value directly
      const share=axShare(a.name);
      // every number below is computed in Python (analysis/dashboard_stats.py, derived.series) for the series, one implant method on one
      // behaviour x base model; serStats (select.js) finds it. d = mean oriented Δ (share kept for robustness), dlo/dhi its 95% CI,
      // n the organisms behind it, s the mean 0-10 score and md/mdc the matched Δ.
      const sts=methods.map(m=>groupStats(pn,m,a.name));
      const dvals=sts.map(st=>st?st.d??null:null);
      const scores=sts.map(st=>st?st.s??null:null);
      const vals=PAPER_SCALE?scores:dvals;
      // tooltip: each method's mean with its 95% CI (the mean of n independent organisms has se = sqrt(sum se^2)/n)
      const tl=methods.map((m,i)=>{if(dvals[i]==null)return `${escHtml(serLabel(m))}: n/a`;const st=sts[i],n=st.n;
        const d=axBack(a.name)?-1:1,v=dvals[i];
        const ci=st.dlo==null?"":" · 95% CI ["+fmt(d>0?st.dlo:-st.dhi)+", "+fmt(d>0?st.dhi:-st.dlo)+"]";
        const md=PAPER_SCALE&&scores[i]!=null&&!share?" · matched "+matchedDeltaMean(st)+" · page ":" · ";
        return `${escHtml(serLabel(m))}: ${PAPER_SCALE?"score "+(scores[i]==null?"n/a":scores[i].toFixed(1))+md:""}${share?"share kept "+v.toFixed(2):"Δ "+fmt(d*v)}${ci}${n>1?" (mean of "+n+")":""}`;});
      const svg=drawRadar(spokes,[{values:vals,color:spreadColor(pn),tip:`<b>${ph(a.short)}</b> · ${escHtml(spreadLabel(pn))}<br>${tl.join("<br>")}`}],{R:104,share,paper:PAPER_SCALE,noNote:true});
      const pb=panel(a.name,axisLabel(a.name,a.short)+(axShare(a.name)?" (share kept)":""),true,state.bucket===a.name,svg);
      // the formula that applies to this property, under its title
      const fm=div("rsub");fm.textContent=PAPER_SCALE?axisFormula(a.name):"Δ vs base, in score units (no paper-scale radar values in this build)";pb.insertBefore(fm,svg);
      grid.appendChild(pb);
      // inline, sliding drill-down right after the selected property panel
      if(state.bucket===a.name)grid.appendChild(drillInline(pn));
    });
    s.appendChild(grid);
  });
  s.appendChild(radarKey(PAPER_SCALE?"paper":"delta"));
  container.appendChild(s);
}
// Row 3: one radar per BASE MODEL for one behaviour, one polygon per implant method (doses/variants of a
// method averaged), same spokes and same scale as the per-organism view. Δ uses each organism's own base arm.
function radarByModel(container){
  const s=document.createElement("section");s.dataset.fig="byModel";s.innerHTML=`<h2>Implant methods per base model <span class="pill">${PAPER_SCALE?"score":"&Delta; vs base"}</span></h2>`;
  howto(s,`Each radar is one <b>base model</b> for the chosen behaviour; each polygon is one <b>implant method</b> (doses and variants averaged), coloured as in the method key. The spokes and the scale are the per-organism view's${PAPER_SCALE?" (the score against each organism's own base model; "+DELTA_DEF+")":""}. A polygon is <b>dashed</b> if any organism of that method falls below the expression threshold. A method that was not run on a model is simply absent from its radar.`);
  // only what the filters leave: the selected implant methods, and (behaviour view) the selected behaviours or (model view) the chosen
  // behaviour on the toggled base models. The base-model radio of the behaviour view is NOT applied: this view compares base models.
  const {all:treatedAll,pers,pn}=modelViewPick();
  if(state.mode!=="model"&&pers.length>1)s.appendChild(radioGroup("Behaviour",pers,()=>state.modelPersona,v=>{state.modelPersona=v;rebuildFigs(["radars"]);}));
  const spokes=radarAxes().map(a=>({key:a.name,short:a.short}));
  const models=BASE_MODELS.length?BASE_MODELS:[...new Set(treatedAll.map(o=>o.base_model))];
  const grid=document.createElement("div");grid.className="radargrid";grid.dataset.persona=pn;
  const present=new Set();
  models.forEach(bm=>{
    const orgs=modelPool(pn,bm);if(!orgs.length)return;
    const methods=serList(orgs);
    const polys=methods.map(m=>{present.add(m);const mo=seriesOrgs(orgs,m);
      const values=spokes.map(sp=>{const st=serStatsOf(pn,bm,m,sp.key);return st?(PAPER_SCALE?st.ps:st.pd)??null:null;});   // Python: the series mean (dashboard_stats.py)
      const failing=mo.filter(o=>exprFlagged(o.id));
      return {values,color:serColor(m),fill:.06,width:2.4,method:m===CTRL?undefined:m,dashed:failing.length>0,
        tip:`<b>${escHtml(serLabel(m))}</b> · ${bmName(bm)}${failing.length?"<br><i>"+(failing.length===mo.length?"falls below the expression threshold":failing.length+" of "+mo.length+" organisms fall below the expression threshold")+"</i>":""}`};});
    const pm=panel(null,bmName(bm),false,false,drawRadar(spokes,polys,{R:104,paper:PAPER_SCALE})),rtm=pm.querySelector(".rt");
    rtm.dataset.model=bm;rtm.title="Open the "+bmName(bm)+" drill-down";grid.appendChild(pm);
  });
  if(!grid.children.length)grid.appendChild(empty(treatedAll.length?"No organisms for this behaviour.":"Nothing to show: no organism is selected (see the behaviour, base-model and implant-method filters)."));
  s.appendChild(grid);
  const l=document.createElement("div");l.className="legend";
  [...DATA.meta.methods,CTRL].filter(m=>present.has(m)).forEach(m=>{const k=document.createElement("span");k.className="k";if(m!==CTRL)k.dataset.method=m;
    k.innerHTML=`<span class="dot" style="background:${serColor(m)}"></span>${escHtml(serLabel(m))}`;l.appendChild(k);});
  s.appendChild(l);s.appendChild(radarKey(PAPER_SCALE?"paper":"delta",{model:true}));
  container.appendChild(s);
}
// ONE radar panel with a toggle between the three views (per property is the default view, on top by
// default; per property). Each view is still built by its own function; this lifts its body in.
function radars(container){
  const s=document.createElement("section");s.dataset.fig="radars";
  s.innerHTML=`<h2>Profiles <span class="pill">${PAPER_SCALE?"score":"&Delta; vs base"}</span></h2>`;
  const tg=div("viewtoggle");tg.setAttribute("role","group");tg.setAttribute("aria-label","Radar view");
  [["bucket","Per property"],["organism","Per organism"],["model","Per model"]].forEach(([k,l])=>{
    const b=document.createElement("button");b.type="button";b.className="chip"+(state.radarView===k?" on":"");
    b.dataset.view=k;b.setAttribute("aria-pressed",String(state.radarView===k));b.textContent=l;
    b.onclick=()=>{state.radarView=k;rebuildFigs(["radars"]);};tg.appendChild(b);});
  s.appendChild(tg);
  if(state.radarView==="model"){const n=div("viewnote");
    n.textContent=PAPER_SCALE
      ?"Each organism is read against its own base model (score vs its own base), so cross-model differences mix implant strength with base-model differences."
      :"Each organism's Δ is read against its own same-size base model, so cross-model differences mix implant strength with base-model differences.";s.appendChild(n);}
  if(state.radarView!=="bucket"){const n=div("viewhint");
    n.textContent=state.radarView==="model"?"Click a base model's title to open its drill-down: every organism on it, side by side, then the methods × properties grid."
      :"Click an organism's title to open its drill-down: how it compares with the other methods on its model, every one of its evals, and its profile against base.";s.appendChild(n);}
  const tmp=document.createElement("div");
  ({bucket:radarByBucket,model:radarByModel}[state.radarView]||radarByOrganism)(tmp);
  const inner=tmp.querySelector("section");
  [...inner.children].filter(c=>c.tagName!=="H2").forEach(c=>s.appendChild(c));
  container.appendChild(s);
}
// Full-width inline drill for one persona + the selected property (individual evals, per organism).
function drillInline(pn){   // pn: a spread-group key (behaviour, or base model in model mode)
  const wrap=div("drillinline");const box=div("drillbox");
  const bkt=state.bucket,evals=DATA.bucket_evals[bkt]||[];
  const ax=DATA.axes.find(a=>a.name===bkt),paperDrill=PAPER_SCALE&&!!ax&&ax.radar!==false&&evals.length>=3;
  box.innerHTML=`<div class="drilltitle">▾ Drill-down: <b>${escHtml(pretty(bkt))}</b> · ${escHtml(spreadLabel(pn))} <span class="pill">individual evals, ${paperDrill?"score":"&Delta; vs base"}</span></div>`;
  if(!evals.length){box.appendChild(empty("This property has no composite evals to drill into."));wrap.appendChild(box);return wrap;}
  // method filter sits RIGHT ABOVE the drill radar (only shows/hides polygons - no re-render)
  const drillOrgs=groupOrgs(pn);   // the control gets its own local chip, and only when the global control chip shows it
  box.appendChild(chipGroup("Show implant methods",[...DATA.meta.methods.filter(m=>!isExtMethod(m)),...(drillOrgs.some(isControl)?[CTRL]:[])],state.drillMethods,
    {label:m=>m===CTRL?CONTROL_NAMES.prompting+" (control)":pretty(m),cls:"drillmethods",onchange:toggleDrillVis,scoped:true}));
  if(EXT_LIST.length)box.appendChild(chipGroup("Pre-trained adapters",EXT_LIST,state.drillMethods,{label:pretty,cls:"drillmethods",onchange:toggleDrillVis,scoped:true}));
  const spokes=evals.map(e=>({key:e,short:DATA.evals[e]?.display||e}));
  const orgs=groupOrgs(pn);
  const shareDrill=axShare(bkt);   // share kept (robustness) off the 0-10 radar: plotted 0..1, not as a Δ
  const g=div("radargrid");
  if(spokes.length<3){
    // <3 evals: a radar is degenerate (2 spokes = a line), so show paired diverging bars instead.
    g.appendChild(panel(null,"individual evals (Δ vs base)",false,false,drawPairedBars(spokes,orgs)));
    box.appendChild(g);box.appendChild(evalColorLegend(spokes));
  }else{
    const polys=orgs.map(o=>({values:spokes.map(sp=>paperDrill?radarEvalOf(o.id,sp.key):shareDrill?fin((evalRead(o.id,sp.key)||{}).o):evalEff(o.id,sp.key)),color:orgColorKey(o),fill:.04,width:2,method:serOf(o),
      dashed:exprFlagged(o.id),tip:`${orgHead(o)}${exprFlagged(o.id)?"<br><i>"+exprNote(o.id)+"</i>":""}`}));
    const dp=panel(null,paperDrill?"individual evals (score, each eval on its own)":"individual evals",false,false,drawRadar(spokes,polys,{R:Math.max(80,Math.min(150,Math.floor((xyWidth()-60)/2)-130)),paper:paperDrill,share:shareDrill&&!paperDrill,noNote:paperDrill}));
    if(paperDrill){const fm=div("rsub");fm.textContent=axisFormula(bkt);dp.insertBefore(fm,dp.querySelector("svg"));}   // the rule of THIS property, under the title
    g.appendChild(dp);
    box.appendChild(g);box.appendChild(orgColorLegend(orgs));box.appendChild(radarKey(paperDrill?"paper":"delta",paperDrill?{axis:bkt}:undefined));
  }
  // capability_format is a capability diagnostic excluded from the composite, so it is absent
  // from the radar above; surface its mc-vs-free split right here in the capability drill-down.
  if(bkt==="capability"){const fig=capabilityFormatFig(orgs);if(fig)box.appendChild(fig);}
  applyDrillVis(box);
  wrap.appendChild(box);return wrap;
}

// Show/hide drill series + legend items by data-method - pure toggle, no re-render.
function applyDrillVis(root){[...DATA.meta.methods,CTRL].forEach(m=>{const show=state.drillMethods.has(m);
  root.querySelectorAll(`[data-method="${m}"]`).forEach(n=>{n.style.display=show?"":"none";});});}
function toggleDrillVis(){
  document.querySelectorAll(".drillinline").forEach(applyDrillVis);
  // keep every drill method-filter (one per persona) in sync with the shared state
  document.querySelectorAll(".drillmethods .chip").forEach(c=>c.classList.toggle("on",state.drillMethods.has(c.dataset.item)));
}
