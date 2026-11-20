// The drill-down drawer: one organism or one base model.
// ---- drill-down drawer: one organism, or one base model, in a right-hand drawer ----
// Opened by a click on anything carrying data-org / data-model (a scatter point, a row label, a radar title in the Per organism /
// Per model views, an "i" on a base-model chip, a model name in a legend or heading). It reads only DATA, through the page's own
// helpers (gridHeat, drawRadar, evalRead, axisRead ...), so every number is the one the figures show; a missing value reads "no data".
// The organism drawer leads with its model's methods × properties grid (its own row outlined, the other rows clickable), then the
// forest of all its evals (collapsed), then its profile against base, the robustness / misalignment tiles and links. The model drawer
// leads with a radar per organism on that model, then the grid and the untreated base's raw scores.
const DR={cur:null,stack:[],opener:null,down:null};

const dh=(tag,cls,html)=>{const d=document.createElement(tag);if(cls)d.className=cls;if(html!=null)d.innerHTML=html;return d;};
const dNoData=m=>dh("div","dr-nodata",m||"no data");
const modelFamily=bm=>/gemma/i.test(bm)?"gemma":"qwen";
function datasetLinks(persona,bm){   // the public HF datasets that carry this behaviour on this model family
  const ds=(DATA.meta.datasets||[]).filter(d=>/^https:\/\/huggingface\.co\/datasets\//.test(d.url||""));
  return ds.filter(d=>d.name.toLowerCase().includes(modelFamily(bm))&&(!persona||d.name.toLowerCase().includes(persona.toLowerCase())));
}
// The room a drawer figure has (CSS px): the drawer's own width less its padding, so a figure drawn at this width shows at 1:1 (13px text).
// The drawer is 45vw, at least 500px, at most 860px, and the whole window under 700px.
function drW(){const d=document.getElementById("dr"),vw=window.innerWidth||1000;
  const w=(d&&d.clientWidth)||(vw<=700?vw:Math.max(500,Math.min(860,vw*0.45)));return Math.max(300,Math.floor(w-80));}
function verdictBadge(id){
  const e=((DATA.expression||{}).orgs||{})[id],v=e?e.verdict:null;
  if(!e||!v)return `<span class="badge" title="no expression verdict for this organism">expression: no data</span>`;
  const pass=v==="PASS"||v==="PASS_INDIST";   // only FAIL is below the threshold (the in-distribution-only pass counts as a pass)
  return `<span class="badge ${pass?"ok":"no"}" title="verdict ${escHtml(verdictLabel(v))} · rate ${e.rate!=null?e.rate.toFixed(2):"n/a"}">expression: ${escHtml(verdictLabel(v))}</span>`;
}
// Forest of ALL of an organism's evals, one panel per property (the forest section's idiom: dot, 95% CI whisker, dashed zero).
function evalForest(o){
  const wrap=dh("div"),bid=baseId(o.persona,bmOf(o)),col=methodColor(o.method);
  DATA.axes.forEach(a=>{
    const evs=DATA.bucket_evals[a.name]||[],share=axShare(a.name);
    wrap.appendChild(dh("div","dr-prop",`${escHtml(cap(axisLabel(a.name,a.short)))}<span class="g">${evs.length} evals${share?" · share kept (0–1)":axBack(a.name)?" · Δ, + = more misbehaviour":" · Δ vs base"}</span>`));
    const rows=evs.map(e=>({e,c:(DATA.cells[o.id]||{})[e],r:evalRead(o.id,e)}));
    if(!rows.some(x=>x.r)){wrap.appendChild(dNoData());return;}
    const rh=20,mT=8,mB=30,W=Math.min(460,drW()),mR=58,nch=W<400?20:28,mL=Math.min(W<400?150:210,Math.max(110,Math.ceil(7*Math.max(...rows.map(x=>clip(DATA.evals[x.e]?.display||x.e,nch).length))+16)));
    const plotW=W-mL-mR,Hh=mT+rows.length*rh+mB;
    let dm=share?1:0.05;if(!share)rows.forEach(x=>{if(x.r)dm=Math.max(dm,Math.abs(x.r.d),x.r.lo==null?0:Math.abs(x.r.lo),x.r.hi==null?0:Math.abs(x.r.hi));});
    const lo0=share?0:-dm,X=d=>mL+((Math.max(lo0,Math.min(dm,d))-lo0)/(dm-lo0))*plotW;
    const svg=natural(el("svg",{viewBox:`0 0 ${W} ${Hh}`,width:W,height:Hh}),W);
    svg.appendChild(el("rect",{x:mL,y:mT-2,width:plotW,height:rows.length*rh+4,fill:"var(--plot)",stroke:"var(--line)"}));
    if(!share)svg.appendChild(el("line",{x1:X(0),y1:mT-2,x2:X(0),y2:mT+rows.length*rh+2,stroke:"var(--axis)","stroke-dasharray":"3 3"}));
    rows.forEach((x,i)=>{const y=mT+i*rh+rh/2,r=x.r,label=clip(DATA.evals[x.e]?.display||x.e,nch);
      svg.appendChild(el("text",{x:mL-6,y:y+4.5,"text-anchor":"end",fill:"var(--fg)","font-size":FS},[txt(label)]));
      if(!r){svg.appendChild(el("text",{x:mL+8,y:y+4.5,fill:"var(--faint)","font-size":FS,"font-style":"italic"},[txt("no data")]));return;}
      const g=el("g",{cursor:"default"}),nv=x.c&&x.c.votes===false;
      if(r.lo!=null)g.appendChild(el("line",{x1:X(r.lo),y1:y,x2:X(r.hi),y2:y,stroke:col,"stroke-width":1.8,opacity:nv?.3:.6,"stroke-linecap":"round"}));
      g.appendChild(el("circle",{cx:X(r.d),cy:y,r:3.8,fill:(r.sig||share)?col:"var(--plot)",stroke:col,"stroke-width":1.5,opacity:nv?.45:1}));
      svg.appendChild(el("text",{x:W-mR+8,y:y+4.5,fill:"var(--muted)","font-size":FS,"font-family":"var(--mono)"},[txt((share?r.d.toFixed(2):fmt(r.d,2))+(r.attr?"\u2020":""))]));
      const b=(DATA.cells[bid]||{})[x.e];
      g.appendChild(el("rect",{x:mL,y:y-rh/2,width:plotW,height:rh,fill:"transparent"}));
      g.addEventListener("mousemove",ev=>showTT(ev,`<b>${escHtml(pretty(DATA.evals[x.e]?.display||x.e))}</b><br>${readTip(r,x.c&&x.c.mean,b&&b.mean)}${nv?"<br><i>cell barred from the composite (not voting)</i>":""}`));
      g.addEventListener("mouseleave",hideTT);svg.appendChild(g);});
    (share?[0,0.5,1]:[-dm,0,dm]).forEach(d=>svg.appendChild(el("text",{x:X(d),y:Hh-9,"text-anchor":"middle",fill:"var(--muted)","font-size":FS},[txt(d===0?"0":share?String(d):fmt(d,2))])));
    const p=dh("div","dr-fp");p.appendChild(svg);wrap.appendChild(p);
  });
  wrap.appendChild(dh("div","dr-note","Filled dot = the 95% CI of Δ excludes 0 (hollow = not distinguishable from base); whisker = 95% CI; faded = cell not voting in the composite. Δ is organism minus the same-size base, on each eval's own 0–1 scale. † = differential attrition: the organism and its base left very different shares of rows unscored (gap above the bound in the tooltip), so that Δ may be biased (report only)."));
  return wrap;
}
const radarSpokes=()=>DATA.axes.filter(a=>a.radar!==false).map(a=>({key:a.name,short:a.short}));
const radarVals=(id,spokes)=>spokes.map(s=>PAPER_SCALE?radarOf(id,s.key):axisEff(id,s.key));
const radarTip=o=>`${orgHead(o)}${exprFlagged(o.id)?"<br><i>"+exprNote(o.id)+"</i>":""}`;
function orgRadar(o){
  const spokes=radarSpokes(),vals=radarVals(o.id,spokes);
  if(vals.every(v=>v==null))return dNoData("no data (no radar value for this organism)");
  const bid=baseId(o.persona,bmOf(o)),polys=[];
  if(PAPER_SCALE&&bid&&RADAR.orgs[bid]){const bv=radarVals(bid,spokes);if(bv.some(v=>v!=null))polys.push({values:bv,color:"var(--muted)",fill:.02,width:1.6,tip:"<b>base model</b> (same size)"});}
  polys.push({values:vals,color:methodColor(o.method),fill:.16,width:2.8,dashed:exprFlagged(o.id),tip:radarTip(o)});
  const w=dh("div"),r=dh("div","dr-radar");r.appendChild(drawRadar(spokes,polys,{R:104,paper:PAPER_SCALE,maxW:Math.min(410,drW())}));w.appendChild(r);
  w.appendChild(dh("div","dr-note",`${PAPER_SCALE?`Behaviour spokes: ${FORMULA.behaviour} (dashed ring at 0 = base); health spokes: ${FORMULA.health} (outer ring = base). ${RADAR_NOTE} `:"Δ vs base; centre ring = base. "}<span style="color:var(--muted)">Grey = same-size base model</span>, coloured = this organism${exprFlagged(o.id)?" (dashed: below the expression threshold)":""}.`));
  return w;
}
function orgStats(o){
  const w=dh("div"),st=dh("div","dr-stats");
  const rob=AXMETA.robustness?axisRead(o.id,"robustness"):null,mis=AXMETA.misalignment?axisRead(o.id,"misalignment"):null;
  const tile=(l,v,d)=>st.appendChild(dh("div","dr-stat",`<div class="l">${l}</div><div class="v">${v}</div><div class="d">${d}</div>`));
  tile("Robustness kept",rob?rob.o.toFixed(2):"no data",rob?(rob.olo==null?"share kept (no CI)":`share kept · 95% CI [${rob.olo.toFixed(2)}, ${rob.ohi.toFixed(2)}]`):"not measured for this organism");
  tile("Misalignment Δ",mis?fmt(mis.d,2):"no data",mis?`organism − base, + = more misbehaviour${mis.lo==null?"":" · 95% CI ["+fmt(mis.lo,2)+", "+fmt(mis.hi,2)+"]"}`:"not measured for this organism");
  w.appendChild(st);
  const evs=DATA.bucket_evals.misalignment||[],bid=baseId(o.persona,bmOf(o));
  if(evs.length){const rows=evs.map(e=>{const c=(DATA.cells[o.id]||{})[e],b=(DATA.cells[bid]||{})[e],r=evalRead(o.id,e);
      const f=x=>x!=null&&x.mean!=null?x.mean.toFixed(3):"no data";
      return `<tr><td>${escHtml(pretty(DATA.evals[e]?.display||e))}</td><td>${f(c)}</td><td>${f(b)}</td><td>${r?fmt(r.d,3)+(r.lo==null?"":` [${fmt(r.lo,2)}, ${fmt(r.hi,2)}]`):"no data"}</td></tr>`;}).join("");
    w.appendChild(dh("div","dr-tblw",`<table class="dr-tbl"><thead><tr><th>Misalignment evaluation (raw rate)</th><th>organism</th><th>same-size base</th><th>Δ (+ = more) [95% CI]</th></tr></thead><tbody>${rows}</tbody></table>`));}
  return w;
}
function datasetLinkNodes(pairs){   // pairs: [persona, base model]; HF dataset links, or "no data"
  const seen=new Set(),out=[];
  pairs.forEach(([p,bm])=>datasetLinks(p,bm).forEach(d=>{if(seen.has(d.url))return;seen.add(d.url);
    const a=dh("a",null,escHtml(d.label)+" (HF dataset) ↗");a.href=d.url;a.target="_blank";a.rel="noopener noreferrer";out.push(a);}));
  return out.length?out:[dh("span","na","HF dataset: no data")];
}
function orgLinks(o){
  const l=dh("div","dr-links"),a=dh("a",null,"Transcripts for this organism ↗");a.tabIndex=0;a.onclick=()=>gotoTranscripts(o);
  a.onkeydown=e=>{if(e.key==="Enter")gotoTranscripts(o);};l.appendChild(a);
  datasetLinkNodes([[o.persona,bmOf(o)]]).forEach(n=>l.appendChild(n));return l;
}
// "Transcripts for this organism": select it in the transcripts picker and scroll there. The picker lists only what the current
// filters show, so an organism that is filtered out gets a note instead.
function gotoTranscripts(o){
  closeDrawer();
  const s=document.querySelector('section[data-fig="transcripts"]');if(!s){state.txOrg=o.id;return;}
  setTab("transcripts");   // the section lives in the Transcripts tab: switch before scrolling to it
  // the tab has its own base-model picker: go to the organism's base model first (this tab's own choice; the Results filters are untouched)
  if(BASE_MODELS.includes(bmOf(o))&&state.txBase!==bmOf(o)&&s.querySelector(".txbar")){
    const bchip=[...s.querySelectorAll(".txbar .chip")].find(c=>c.dataset.item===bmOf(o));if(bchip)bchip.click();}
  else if(BASE_MODELS.includes(bmOf(o)))state.txBase=bmOf(o);
  // the negative-prompt control is listed only while the Results control chip is on: arriving by this link lists it for this visit
  // (set AFTER the base-model switch above, which clears it: a base change ends a visit)
  state.txWant=isControl(o)?o.id:null;
  if(window._txDraw&&s.querySelector(".txbar"))window._txDraw();   // redraw: the picker now lists the control (txWant) and this base model
  const live=document.querySelector('section[data-fig="transcripts"]')||s;
  live.scrollIntoView({behavior:"auto",block:"start"});
  setTimeout(()=>{const sec=document.querySelector('section[data-fig="transcripts"]')||live;const lab=txOrgLabel(o);
    const chip=[...sec.querySelectorAll(".txbar .chip")].find(c=>c.textContent===lab);
    // the picker only redraws on a CHANGE (radioGroup's guard), so the organism is chosen THROUGH the chip: setting state.txOrg first
    // made the click a no-op and left the previous organism's transcripts on screen
    if(chip){chip.click();return;}
    if(!sec.querySelector(".txbar")){state.txOrg=o.id;return;}   // not drawn yet: its first draw reads state.txOrg
    drToast(`This organism has no transcripts in the picker (the negative-prompt control is listed only while the Results control chip is on).`);},700);
}
function drToast(m){const t=dh("div","dr-toast",escHtml(m));document.body.appendChild(t);setTimeout(()=>t.remove(),5000);}
// The methods × properties Δ grid of a set of organisms (the page's own gridHeat); `highlight` outlines one row.
function methodGrid(orgs,highlight){
  const w=gridHeat(orderOrgs(orgs),DATA.axes.map(a=>({key:a.name,short:a.short})),(o,k)=>axCell(o,k,k==="misalignment"?MIS_TIP:null),false,{values:true,highlight,fitW:drW()+20});
  w.classList.remove("center");const box=dh("div","dr-heat");box.appendChild(w);return box;
}
function baseRawTable(bm,personas){
  const AB={generality:"gen",robustness:"rob",coherence:"coh",capability:"cap",activations:"act",introspection:"intro",misalignment:"mis"};
  const head=DATA.axes.map(a=>`<th title="${escHtml(axisLabel(a.name,a.short))}">${escHtml(AB[a.name]||axisLabel(a.name,a.short).slice(0,5))}</th>`).join("")+`<th title="expression rate (share of answers showing the behaviour)">expr</th>`;
  const rows=personas.map(p=>{const bid=baseId(p,bm);if(!bid)return `<tr><td>${escHtml(pretty(p))}</td><td colspan="${DATA.axes.length+1}">no data</td></tr>`;
    const cells=DATA.axes.map(a=>{const v=axisMean(bid,a.name);return `<td>${v==null?"no data":v.toFixed(2)}</td>`;}).join("");
    const ex=exprOf(bid);return `<tr><td>${escHtml(pretty(p))}</td>${cells}<td>${ex&&ex.rate!=null?ex.rate.toFixed(2):"no data"}</td></tr>`;}).join("");
  const w=dh("div");w.appendChild(dh("div","dr-tblw",`<table class="dr-tbl tight"><thead><tr><th>Base</th>${head}</tr></thead><tbody>${rows}</tbody></table>`));
  w.appendChild(dh("div","dr-note","Abbreviations: gen = generality, rob = robustness, coh = coherence, cap = capability, act = activations, intro = introspection, mis = misalignment, expr = expression rate. The untreated base model's own score per property (mean over the property's composite evals, 0–1; misalignment = raw rate, higher = more misaligned). “no data” = the base arm has no score for that property."));
  return w;
}
function radarSmallMultiples(orgs){
  const spokes=radarSpokes(),g=dh("div","dr-radars");
  orderOrgs(orgs).forEach(o=>{
    const vals=radarVals(o.id,spokes),title=niceOrg(o);
    const svg=vals.every(v=>v==null)?dNoData("no data")
      :drawRadar(spokes,[{values:vals,color:methodColor(o.method),fill:.16,width:2.6,dashed:exprFlagged(o.id),tip:radarTip(o)}],{R:104,paper:PAPER_SCALE,maxW:drW()});
    const p=panel(null,title,false,false,svg),rt=p.querySelector(".rt");rt.dataset.org=o.id;rt.title="Open "+title+" drill-down";g.appendChild(p);});
  return g;
}
function drSection(title,note,node){const s=dh("div","dr-sec");{const h3=document.createElement("h3");h3.textContent=cap(title);s.appendChild(h3);};if(note)s.appendChild(dh("div","dr-note",note));if(node)s.appendChild(node);return s;}

function renderOrganism(o){
  const f=dh("div"),mid=bmOf(o);
  f.appendChild(dh("div",null,`<h2 class="dr-title">${escHtml(niceOrg(o))}</h2><div class="dr-sub">${escHtml(o.id)}</div>
    <div class="dr-meta"><span><span class="k">Model</span><b data-model="${escHtml(mid)}" title="Open the model drill-down">${escHtml(bmName(mid))}</b> <span class="dr-sub">${escHtml(mid)}</span></span>
    <span><span class="k">Behaviour</span><b>${escHtml(pretty(o.persona))}</b></span>${verdictBadge(o.id)}<span><span class="k">Seeds</span>${orgSeeds(o.id)}</span></div>`));
  const siblings=modelOrgs(mid,o.persona);
  f.appendChild(drSection(`Against the other methods on ${bmName(mid)}`,`Δ vs the same-size base, ${ph(o.persona)} organisms on ${escHtml(bmName(mid))}; this organism's row is outlined. Click a row label to open that organism.`,methodGrid(siblings,o.id)));
  const det=dh("details","dr-det");det.innerHTML=`<summary>All evals, forest of Δ vs base with 95% CI (click to expand)</summary>`;
  const inner=dh("div");inner.appendChild(evalForest(o));det.appendChild(inner);f.appendChild(det);
  f.appendChild(drSection("Profile vs base",null,orgRadar(o)));
  f.appendChild(drSection("Robustness kept and misalignment",null,orgStats(o)));
  f.appendChild(drSection("Links",null,orgLinks(o)));
  return f;
}
function renderModel(bm){
  const f=dh("div"),orgs=modelOrgs(bm),pers=DATA.meta.personas.filter(p=>orgs.some(o=>o.persona===p));
  f.appendChild(dh("div",null,`<h2 class="dr-title">${escHtml(bmName(bm))}</h2><div class="dr-sub">${escHtml(bm)}</div>
    <div class="dr-meta"><span><span class="k">Organisms</span><b>${orgs.length}</b></span><span><span class="k">Behaviours</span>${pers.map(p=>escHtml(pretty(p))+" ("+modelOrgs(bm,p).length+")").join(", ")||"no data"}</span></div>`));
  if(!orgs.length){f.appendChild(dNoData());return f;}
  const rad=dh("div");pers.forEach(p=>{rad.appendChild(dh("div","dr-prop",escHtml(pretty(p))));rad.appendChild(radarSmallMultiples(modelOrgs(bm,p)));});
  f.appendChild(drSection("Profile of every organism",PAPER_SCALE?`${DELTA_DEF}. ${FORMULA.behaviour} (dashed ring at 0 = base); ${FORMULA.health} (outer ring = base); robustness: ${FORMULA.share}. ${RADAR_NOTE} Dashed = below the expression threshold. Click a title to open that organism.`:"Δ vs base. Click a title to open that organism.",rad));
  const grid=dh("div");
  pers.forEach(p=>{grid.appendChild(dh("div","dr-prop",escHtml(pretty(p))));grid.appendChild(methodGrid(modelOrgs(bm,p)));});
  grid.appendChild(dh("div","dr-prop","Reference: the untreated base model"));grid.appendChild(baseRawTable(bm,pers));
  f.appendChild(drSection("Methods × properties, Δ vs this model's base","Each cell is the organism minus the same-size base (blue = better, dot = 95% CI excludes 0). Click a row label to open that organism.",grid));
  const l=dh("div","dr-links");datasetLinkNodes(pers.map(p=>[p,bm])).forEach(n=>l.appendChild(n));
  f.appendChild(drSection("Links",null,l));
  return f;
}
function renderDrawer(){
  if(!DR.cur)return;hideTT();
  const body=document.getElementById("dr-body"),isO=DR.cur.kind==="org";
  document.getElementById("dr-kind").textContent=isO?"Organism":"Base model";
  document.getElementById("dr-back").style.display=DR.stack.length?"":"none";
  body.innerHTML="";
  const o=isO?ORG_BY_ID[DR.cur.id]:null;
  body.appendChild(isO?(o?renderOrganism(o):dNoData("no data for this organism")):renderModel(DR.cur.id));
  body.scrollTop=0;
}
function openDrawer(kind,id,push){
  if(kind==="org"&&!ORG_BY_ID[id])return;
  const dr=document.getElementById("dr"),wasOpen=dr.classList.contains("open");
  if(wasOpen&&push&&DR.cur&&!(DR.cur.kind===kind&&DR.cur.id===id))DR.stack.push(DR.cur);
  if(!wasOpen){DR.stack=[];DR.opener=document.activeElement;}
  DR.cur={kind,id};renderDrawer();
  dr.classList.add("open");document.getElementById("dr-scrim").classList.add("open");
  dr.setAttribute("aria-label",(kind==="org"?"Organism":"Base model")+" drill-down");
  document.getElementById("dr-close").focus({preventScroll:true});
}
// Closing leaves the page exactly as it was: the drawer is an overlay, so scroll and every filter are untouched.
function closeDrawer(){
  const dr=document.getElementById("dr");if(!dr.classList.contains("open"))return;
  dr.classList.remove("open");document.getElementById("dr-scrim").classList.remove("open");DR.cur=null;DR.stack=[];hideTT();
  const f=DR.opener;DR.opener=null;if(f&&f.focus&&document.contains(f)){try{f.focus({preventScroll:true});}catch(_){}}
}
document.getElementById("dr-close").onclick=closeDrawer;
document.getElementById("dr-scrim").addEventListener("click",closeDrawer);
document.getElementById("dr-back").onclick=()=>{const p=DR.stack.pop();if(p){DR.cur=p;renderDrawer();}};
document.addEventListener("keydown",e=>{if(e.key==="Escape"&&document.getElementById("dr").classList.contains("open")&&!document.querySelector(".ovl")){e.preventDefault();closeDrawer();}});
document.addEventListener("pointerdown",e=>{DR.down={x:e.clientX,y:e.clientY};},true);
// One capture-phase click handler for every trigger (so it also wins over a chip's own toggle and the figure-enlarge click).
document.addEventListener("click",e=>{
  const t=e.target;if(!t||!t.closest||t.closest(".dr-top,.ovl"))return;
  let hit=t.closest("[data-org],[data-model]");
  if(!hit&&t.closest("svg.zsvg")){   // pointer capture on a zoomable svg retargets the click to the svg: look under the cursor (a drag is not a click)
    if(DR.down&&Math.hypot(e.clientX-DR.down.x,e.clientY-DR.down.y)>6)return;
    hit=(document.elementsFromPoint(e.clientX,e.clientY).map(n=>n.closest&&n.closest("g[data-org]")).find(Boolean))||null;}
  if(!hit)return;
  e.preventDefault();e.stopPropagation();
  if(hit.dataset.org)openDrawer("org",hit.dataset.org,true);else openDrawer("model",hit.dataset.model,true);
},true);
