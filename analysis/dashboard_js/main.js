// Section order, partial rebuilds, the table of contents, render() and start-up. Runs last.
// Figure order (the paper's: expression test, profiles, then the property x property scatter) + which function builds each - lets us rebuild ONLY the sections an
// interaction touches (property selection / drill-method chips), instead of the whole page.
const FIG_ORDER=["expression","radars","xy","rankedBars","forest",
  "heatmap","misalign","corr","discard","dist","trans","caveats"];
function rebuildFigs(names){
  const main=document.getElementById("main");
  names.forEach(name=>{
    const build=FIG_BUILDERS[name];if(!build)return;
    const old=main.querySelector(`section[data-fig="${name}"]`);
    const tmp=document.createElement("div");build(tmp);const neu=tmp.querySelector("section");
    if(neu)neu.dataset.fig=name;
    if(old){old.remove();}
    if(neu){
      let anchor=null;const idx=FIG_ORDER.indexOf(name);
      for(let k=idx+1;k<FIG_ORDER.length&&!anchor;k++)anchor=main.querySelector(`section[data-fig="${FIG_ORDER[k]}"]`);
      main.insertBefore(neu,anchor);
    }
  });
  decorateSections();
}

function decorateSections(){
  const main=document.getElementById("main"),toc=document.getElementById("toc");
  toc.innerHTML='<div class="toc-h">On this page</div>';
  const tx=document.getElementById("txmain");
  // the Transcripts tab's sections get an eyebrow and an info badge but no ToC entry (it is its own tab)
  [...tx.querySelectorAll("section")].forEach(s=>{const h2=s.querySelector("h2");if(!h2)return;
    if(!s.querySelector(".eyebrow")){const eb=div("eyebrow");eb.textContent="Transcripts";s.insertBefore(eb,s.firstChild);}
    const fig=s.dataset.fig;
    if(fig&&EXPLAIN[fig]&&!s.querySelector(".infobadge")){
      const b=infoBadge(EXPLAIN[fig],"About: "+h2.textContent);s.appendChild(b);}});
  const secs=[...main.querySelectorAll("section")];
  const linkArr=[];
  secs.forEach((s,i)=>{const h2=s.querySelector("h2");if(!h2)return;
    const id="sec"+i;s.id=id;
    if(!s.querySelector(".eyebrow")){const eb=div("eyebrow");eb.textContent=String(i+1).padStart(2,"0");s.insertBefore(eb,s.firstChild);}
    // info-sign explainer (top-right), detail on hover
    const fig=s.dataset.fig;
    if(fig&&EXPLAIN[fig]&&!s.querySelector(".infobadge")){
      const b=infoBadge(EXPLAIN[fig],"About: "+h2.textContent);s.appendChild(b);}
    const a=document.createElement("a");a.href="#"+id;a.textContent=h2.textContent;a.dataset.i=linkArr.length;
    toc.appendChild(a);linkArr.push(a);});
  // scroll-spy with a "progress rail": everything above the current section reads as passed.
  function setActive(idx){linkArr.forEach((a,k)=>{a.classList.toggle("active",k===idx);a.classList.toggle("passed",k<idx);});}
  if(window._io)window._io.disconnect();
  window._io=new IntersectionObserver(ents=>{
    ents.forEach(en=>{if(en.isIntersecting){const idx=[...secs].indexOf(en.target);if(idx>=0)setActive(idx);}});
  },{rootMargin:"-12% 0px -80% 0px",threshold:0});
  secs.forEach(s=>window._io.observe(s));
  setActive(0);
  // Proportional rail (LessWrong-style): each entry's height tracks its section's height,
  // so the rail reads like a minimap - long sections get more space, short ones less.
  sizeToc(secs,linkArr);
  window._sizeToc=()=>sizeToc(secs,linkArr);
  if(!window._tocResize){window._tocResize=true;
    let t,xw=xyWidth();addEventListener("resize",()=>{clearTimeout(t);t=setTimeout(()=>{window._sizeToc&&window._sizeToc();
      const w=xyWidth();if(Math.abs(w-xw)>=24){xw=w;const present=WIDTH_FIGS.filter(f=>document.querySelector(`section[data-fig="${f}"]`));if(present.length)rebuildFigs(present);}},150);});
    addEventListener("load",()=>window._sizeToc&&window._sizeToc());}
}
function sizeToc(secs,links){
  const hs=secs.map(s=>s.getBoundingClientRect().height||1);
  const mx=Math.max(1,...hs);
  links.forEach((a,i)=>{a.style.minHeight=Math.round(22+(hs[i]/mx)*70)+"px";});
}

// Builders for the sections that get targeted (partial) rebuilds.
const FIG_BUILDERS={expression:expressionSection,radars:radars,xy:xyChart,heatmap:heatmap,discard:discardScatter,corr:corrMatrix};
// Drawn at the width they are shown at (so their text is the size on screen): rebuilt when the pane width changes.
const WIDTH_FIGS=["xy","expression","discard","corr"];

// ---- render ----
function render(){
  renderFilters();
  const main=document.getElementById("main");main.innerHTML="";
  document.getElementById("txmain").innerHTML="";   // the Transcripts pane is rebuilt too: else every filter change appends another copy
  main.appendChild(paperTiles());
  expressionSection(main);
  radars(main);
  xyChart(main);
  rankedBars(main);
  forest(main);
  heatmap(main);
  misalignDumbbell(main);
  corrMatrix(main);
  discardScatter(main);
  distributions(main);
  transitionBars(main);
  transcripts(document.getElementById("txmain"));
  diagnosticsSection(main);
  caveatsSection(main);
  decorateSections();
}
document.getElementById("subline").innerHTML =
  `${DATA.meta.n_treated} organisms · ${DATA.meta.n_evaluations!=null?DATA.meta.n_evaluations+" evaluations":DATA.meta.n_evals+" evals"} · ${DATA.meta.personas.length} behaviour${DATA.meta.personas.length===1?"":"s"} · ${BASE_MODELS.length||1} base model${BASE_MODELS.length>1?"s":""} · ${DATA.meta.methods.length} implant methods`
  + (DATA.meta.has_base?"":" · NO BASE ARM");
document.getElementById("footer").textContent =
  "Δ = organism − the same-size base of its own behaviour, in the score's 0–1 units (+ = better; misalignment: + = more misbehaviour; robustness: share kept). 95% CI = Δ ± 1.96·SE; a dot = the CI excludes 0. Aggregation: analysis/aggregate.py.";
// --bar-h = the height of the pinned filter bar (0 when it is not pinned: narrow or short panes). Section scroll-margin and the ToC
// rail read it, so an anchor jump or ToC click never leaves a heading under the bar. Custom (no CSS-only way to read an element's height).
function syncBarHeight(){const f=document.getElementById("filters");if(!f)return;
  const pinned=getComputedStyle(f).position==="sticky";
  document.documentElement.style.setProperty("--bar-h",(pinned?Math.ceil(f.getBoundingClientRect().height):0)+"px");}
if(typeof ResizeObserver!=="undefined")new ResizeObserver(syncBarHeight).observe(document.getElementById("filters"));
window.addEventListener("resize",syncBarHeight);
// ---- tabs: Results | Transcripts ----
// Both panes stay in the DOM (state, pickers and the lazy transcript load are kept); the inactive one is `hidden`. The URL hash
// carries the choice: "#transcripts" opens the Transcripts tab, anything else ("", "#results", "#secN") the Results tab.
// SVG figures are sized from the window (xyWidth), not from their container, so a figure drawn while its tab was hidden is the right size;
// what does need the visible layout (the ToC rail heights, the lazy transcript IntersectionObserver) is redone on show.
function setTab(name,fromHash){
  const tx=name==="transcripts";
  document.body.dataset.tab=tx?"transcripts":"results";
  document.getElementById("main").hidden=tx;document.getElementById("txmain").hidden=!tx;
  document.querySelectorAll("#tabs .tab").forEach(b=>{const on=b.dataset.tab===(tx?"transcripts":"results");
    b.classList.toggle("on",on);b.setAttribute("aria-selected",on?"true":"false");b.tabIndex=on?0:-1;});
  if(!fromHash){const h=tx?"#transcripts":"#results";if(location.hash!==h){try{history.replaceState(null,"",h);}catch(e){location.hash=h;}}}
  if(window._sizeToc&&!tx)window._sizeToc();
  syncBarHeight();   // the pinned filter bar is hidden on the Transcripts tab: --bar-h goes to 0, so no heading sits under a bar that is not there
  if(!fromHash){const t=document.getElementById("tabs");
    if(t&&t.getBoundingClientRect().top<0)window.scrollTo(0,t.offsetTop);}
}
function tabFromHash(){return location.hash==="#transcripts"?"transcripts":"results";}
document.querySelectorAll("#tabs .tab").forEach(b=>{b.addEventListener("click",()=>setTab(b.dataset.tab));
  b.addEventListener("keydown",e=>{let o=null;
    if(e.key==="ArrowRight"||e.key==="ArrowLeft")o=b.dataset.tab==="results"?"transcripts":"results";
    else if(e.key==="Home")o="results";else if(e.key==="End")o="transcripts";
    if(o){e.preventDefault();setTab(o);document.getElementById("tab-"+o).focus();}});});
// Where the page lands after a hash change (also back / forward): the target section for "#secN", else the top of the tab (the leftover
// offset of the tab just left is meaningless in the one just opened). Instant: the page's smooth scroll would still be moving.
try{history.scrollRestoration="manual";}catch(e){}   // back / forward must land where landOnHash says, not on the offset the browser remembers
function landOnHash(){
  const t=tabFromHash(),m=location.hash.match(/^#(sec\d+)$/),e=t==="results"&&m&&document.getElementById(m[1]);
  if(e)e.scrollIntoView({behavior:"instant",block:"start"});
  else{const bar=document.getElementById("tabs");window.scrollTo({top:bar?bar.offsetTop:0,behavior:"instant"});}
}
window.addEventListener("hashchange",()=>{const t=tabFromHash();
  if(t!==document.body.dataset.tab)setTab(t,true);
  landOnHash();});
initState();renderTop();render();syncBarHeight();setTab(tabFromHash(),true);
// A reload keeps the scroll position (history.scrollRestoration is "manual" so back / forward land via landOnHash): the position is saved
// when the page is left, with the hash it was saved under, and restored here only when the page opens at that same hash. A visit to another
// hash lands where that hash says.
const SCROLL_KEY="mo_dashboard_scroll";
function saveScroll(){try{sessionStorage.setItem(SCROLL_KEY,JSON.stringify({p:location.pathname,h:location.hash,y:Math.round(window.scrollY)}));}catch(e){}}
window.addEventListener("pagehide",saveScroll);window.addEventListener("beforeunload",saveScroll);
(function restoreScroll(){let s=null;try{s=JSON.parse(sessionStorage.getItem(SCROLL_KEY)||"null");}catch(e){}
  if(!s||s.p!==location.pathname||s.h!==location.hash||!(s.y>0))return;
  const go=()=>window.scrollTo({top:s.y,behavior:"instant"});go();requestAnimationFrame(go);window.addEventListener("load",go,{once:true});})();
window.addEventListener("mousemove",moveTT);
// A theme change (the OS preference or <html data-theme>) repaints: the diverging scale and the
// method colours are computed in JS, so CSS alone cannot restyle them.
function _themeChanged(){renderTop();render();if(typeof DR!=="undefined"&&DR.cur)renderDrawer();}
try{new MutationObserver(_themeChanged).observe(document.documentElement,{attributes:true,attributeFilter:["data-theme"]});}catch(e){}
try{const mq=window.matchMedia("(prefers-color-scheme: dark)");
  if(mq.addEventListener)mq.addEventListener("change",_themeChanged);else if(mq.addListener)mq.addListener(_themeChanged);}catch(e){}
