// The filter bar and its chip / radio groups.
// ---- filters ----

function renderFilters(){   // rebuilt every render, so chips always reflect current state
  const f=document.getElementById("filters");f.style.display="";f.innerHTML="";
  // The per-base-model TAB: one radio chip per entry in DATA.meta.base_models (7B/14B/32B),
  // exactly one active, so each figure shows a single base model at a time. If this tab is
  // missing or shows only "?", DATA.meta.base_models was not populated — build_dashboard_data
  // was called WITHOUT `rows` (base_model is derived from rows, not the Matrix). Fix the
  // build call, not this template.
  const MODES=["Compare behaviours on a model","Compare models on a behaviour"];
  if(BASE_MODELS.length>1)f.appendChild(radioGroup("Compare",MODES,()=>MODES[state.mode==="model"?1:0],v=>setMode(v===MODES[1]?"model":"behaviour")));
  // the negative-prompt control is the last chip (CTRL in the same set: All turns it on, None off; select.js decides what it shows)
  // Pre-trained adapters (external, mobench user-config) are not implant methods: their own row, same state.methods set, All/None scoped to it.
  const methodChips=()=>{const g=chipGroup("Implant method",[...[...DATA.meta.methods].filter(m=>!isExtMethod(m)).sort((a,b)=>morder(a)-morder(b)),...(hasControlArm()?[CTRL]:[])],state.methods,
    {label:m=>m===CTRL?"Negative prompt (control)":pretty(m),cls:"methodchips",scoped:true});
    if(!EXT_LIST.length)return g;
    const w=document.createElement("div");w.className="fgroups";w.style.display="contents";w.appendChild(g);
    w.appendChild(chipGroup("Pre-trained adapters",EXT_LIST,state.methods,{label:pretty,cls:"methodchips",scoped:true}));return w;};
  if(state.mode==="model"){   // pick ONE behaviour; toggle any of the base models that carry it
    f.appendChild(radioGroup("Behaviour",DATA.meta.personas,()=>state.behaviour,v=>{state.behaviour=v;render();},pretty));
    const carry=modelsCarrying(state.behaviour);
    f.appendChild(modelInfo(chipGroup("Base model",carry,state.models,{label:bmName,cls:"modelchips"})));
    f.appendChild(methodChips());return;}
  if(BASE_MODELS.length)f.appendChild(modelInfo(radioGroup("Base model",BASE_MODELS,()=>state.baseModel,v=>{state.baseModel=v;render();},bmName)));
  // The behaviour (persona) filter is global; the implant-method filter lives in the drill-down.
  if(DATA.meta.personas.length>1)
    f.appendChild(chipGroup("Behaviour (persona)",DATA.meta.personas,state.personas,{pretty:true}));
  f.appendChild(methodChips());
}
// An "i" on each base-model chip opens that model's drill-down drawer. It is drawn by CSS (an empty span), so a chip's text stays its name.
function modelInfo(group){group.querySelectorAll(".chip").forEach(c=>{const id=c.dataset.item;if(!id)return;
  const i=document.createElement("span");i.className="minfo";i.dataset.model=id;i.title="Open the "+bmName(id)+" drill-down";
  i.setAttribute("role","button");i.setAttribute("aria-label","Open the "+bmName(id)+" drill-down");c.appendChild(i);});return group;}
// single-select chips (radio semantics): exactly one active.
function radioGroup(title,items,get,pick,label){
  const g=document.createElement("div");g.className="fgroup";
  const head=document.createElement("div");head.className="fhead";
  const l=document.createElement("label");l.textContent=title;head.appendChild(l);g.appendChild(head);
  const w=document.createElement("div");w.className="chips";
  items.forEach(it=>{const c=document.createElement("span");c.className="chip"+(get()===it?" on":"");
    c.textContent=label?label(it):it;c.dataset.item=it;c.onclick=()=>{if(get()!==it)pick(it);};w.appendChild(c);});
  g.appendChild(w);return g;
}
function chipGroup(title,items,set,opts={}){
  const change=opts.onchange||render;   // let a caller repaint only its own view
  const g=document.createElement("div");g.className="fgroup"+(opts.cls?" "+opts.cls:"");
  const head=document.createElement("div");head.className="fhead";
  const l=document.createElement("label");l.textContent=title;head.appendChild(l);
  const all=document.createElement("button");all.type="button";all.className="mini";all.textContent="All";
  all.onclick=()=>{items.forEach(i=>set.add(i));change();};
  const none=document.createElement("button");none.type="button";none.className="mini";none.textContent="None";
  none.onclick=()=>{if(opts.scoped)items.forEach(i=>set.delete(i));else set.clear();change();};   // scoped: the set is shared with another row (state.methods)
  head.appendChild(all);head.appendChild(none);g.appendChild(head);
  const w=document.createElement("div");w.className="chips";
  items.forEach(it=>{const c=document.createElement("span");c.dataset.item=it;
    c.className="chip"+(set.has(it)?" on":"");           // reflect state on every rebuild
    c.textContent=opts.label?opts.label(it):opts.pretty?pretty(it):it;
    // toggle the chip's own class directly so it works whether or not `change` repaints.
    c.onclick=()=>{const on=!set.has(it);on?set.add(it):set.delete(it);c.classList.toggle("on",on);change();};w.appendChild(c);});
  g.appendChild(w);return g;
}
