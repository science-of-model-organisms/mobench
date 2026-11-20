// The selection layer: which organisms are visible, in what order, and how they group into spread groups and series.
// It is the ONLY place that reads the filters (state.mode / methods / personas / baseModel / behaviour / models) to decide which
// organisms a view shows. Figures, legends, the transcript picker, the tiles and the drawer get their organisms from these functions
// and never filter DATA.organisms themselves, so a rule (the control arm last, the control its own series, the filters applied) is
// written once and every view follows it.
//   orgVisible(o,opts)    the filter predicate (opts.allModels: ignore the behaviour view's one-base-model radio)
//   pool(opts)            the visible treated organisms, in data order
//   treated(opts)         the same in the page's one ordering (orderOrgs: the negative-prompt control last)
//   visibleBases()        the visible base arms
//   implants()            treated() without the control arm (it is not an implant)
//   spreadGroups()        treated() split by the spread dimension (behaviour; base model in "compare models" mode)
//   groupOrgs(key)        one spread group
//   modelViewPick() / modelPool(pn,bm)   Per model: the behaviour shown, and its organisms on one base model
//   serOf / serList / seriesOrgs   the series a Profiles spoke or polygon averages over
//   visibleMethods()      the implant methods the method filter shows (legends)
//   baseId(persona,bm)    the base arm a Δ is read against
//   modelOrgs(bm,pn) / modelsCarrying(pn) / hidingAny()   unfiltered lookups (the drawer, the filter chips, Diagnostics)
// The filters: implant method plus the control chip (both views; a base arm has none), then behaviour + base model (behaviour view) or one behaviour +
// the toggled base models (model view). `opts.allModels` drops only the behaviour view's base-model radio (Per model compares models).
// The negative-prompt control is its own chip in the implant-method row (CTRL, a key in state.methods, off until switched on): it shows
// whenever that chip is on, whatever the Prompting chip says, and Prompting alone shows only the positive prompting organisms.
const orgVisible=(o,opts)=>(o.is_base||(isControl(o)?state.methods.has(CTRL):state.methods.has(o.method)))&&(state.mode==="model"?o.persona===state.behaviour&&state.models.has(bmOf(o))
  :state.personas.has(o.persona)&&(!!(opts&&opts.allModels===true)||!state.baseModel||bmOf(o)===state.baseModel));

// A meaningful method ordering (light → heavy) so rows/spokes group sensibly, not alphabetically.
const METHOD_ORDER=["prompting","few_shot_icl","oct_demos","sft_behaviour","dpo_behaviour",
  "distillation_seqkd","steering","oct_behaviour"];
const morder=m=>{const i=METHOD_ORDER.indexOf(m);return i<0?99:i;};

// ONE ordering for every list of organisms on the page (heatmap rows, radar panels, forest / bars / expression / misalignment rows,
// legends, the transcript picker, the drill-down grids): the negative-prompt CONTROL arm always comes LAST, then group, method, name.
// A list that is sorted by a value (forest, ranked bars) puts ctrlLast first in its own comparator; nothing else sorts organisms.
const ctrlLast=(a,b)=>Number(isControl(a))-Number(isControl(b));

// A SERIES is what a Profiles spoke or polygon averages over: one implant method, but the negative-prompt control is its OWN series (it
// shares method "prompting" with the real Prompting organism, and averaging the two blended a treatment with its control and gave the
// treatment the control's expression verdict). The control is always last.
const CTRL="__control";
const serOf=o=>isControl(o)?CTRL:o.method;
const serList=orgs=>{const ks=DATA.meta.methods.filter(m=>orgs.some(o=>!isControl(o)&&o.method===m));if(orgs.some(isControl))ks.push(CTRL);return ks;};
const seriesOrgs=(orgs,k)=>orgs.filter(o=>serOf(o)===k);
// The numbers of one series, computed in Python (analysis/dashboard_stats.py, DATA.derived.series). A series is one implant method on ONE base
// model for ONE behaviour, and every filter keeps or drops a whole series, so its numbers never depend on a filter: this only looks them up,
// by the behaviour and base model the series sits on ("*": all base models, when the page has none to pick).
const serStatsOf=(persona,bm,ser,ax)=>(((DATA.derived.series||{})[persona+"|"+(bm||"*")]||{})[ser]||{})[ax]||null;
// ... for a spread group (behaviour mode: the group is a behaviour, on the picked base model; model mode: the group is a base model).
const groupStats=(spreadKey,ser,ax)=>state.mode==="model"?serStatsOf(state.behaviour,spreadKey,ser,ax):serStatsOf(spreadKey,state.baseModel,ser,ax);

const orgCmp=(a,b)=>ctrlLast(a,b)||(state.mode==="model"?BASE_MODELS.indexOf(bmOf(a))-BASE_MODELS.indexOf(bmOf(b)):a.persona.localeCompare(b.persona))||morder(a.method)-morder(b.method)||a.short.localeCompare(b.short);
const orderOrgs=a=>[...a].sort(orgCmp);
const pool=opts=>DATA.organisms.filter(o=>!o.is_base&&orgVisible(o,opts));
const treated=opts=>orderOrgs(pool(opts));
const visibleBases=()=>DATA.organisms.filter(o=>o.is_base&&orgVisible(o));
// The Transcripts tab's own selection (its filter bar is hidden; it has a Base model row of its own). The base model is the tab's pick
// (state.txBase) until the reader makes one, then the Results bar's. The tab lists every organism of that base model whatever the
// Results method / behaviour filters say, plus the negative-prompt control only while the Results control chip is on. `has(o)` says
// whether the organism has transcripts. Nothing here writes to the Results filters.
const txBaseModel=()=>BASE_MODELS.includes(state.txBase)?state.txBase:(state.baseModel||BASE_MODELS[0]||null);
// state.txWant: the control the drill-down link asked for, listed even while the Results control chip is off (until another organism is picked).
const txPool=(base,has)=>DATA.organisms.filter(o=>has(o)&&(!BASE_MODELS.length||bmOf(o)===base)&&(o.is_base||!isControl(o)||state.methods.has(CTRL)||o.id===state.txWant));
// The organism to show after a base-model change: the one last picked on that base, else the same method · behaviour there, else the first.
function txPickOrg(orgs,base){
  const mem=(state.txOrgByBase||{})[base],prev=ORG_BY_ID[state.txOrg];
  const same=prev&&orgs.find(o=>o.is_base===prev.is_base&&o.persona===prev.persona&&isControl(o)===isControl(prev)&&(o.is_base||o.method===prev.method));
  return orgs.find(o=>o.id===mem)||same||orgs[0];}
const implants=()=>treated().filter(o=>!isControl(o));
// Per model: one behaviour on one base model, from the pool that ignores the behaviour view's base-model radio (data order, as the
// series means are summed in it).
const modelPool=(pn,bm)=>pool({allModels:true}).filter(o=>o.persona===pn&&bmOf(o)===bm);
// Per model shows ONE behaviour: the picked one in "compare models" mode, else the view's own pick (state.modelPersona, kept to one of
// the behaviours visible on some base model). all = that pool, pers = those behaviours, pn = the one shown.
function modelViewPick(){const all=pool({allModels:true}),pers=DATA.meta.personas.filter(p=>all.some(o=>o.persona===p));
  if(!pers.includes(state.modelPersona))state.modelPersona=pers[0]||null;
  return {all,pers,pn:state.mode==="model"?state.behaviour:state.modelPersona};}

const spreadKey=o=>state.mode==="model"?bmOf(o):o.persona;
const spreadKeys=()=>state.mode==="model"?BASE_MODELS:DATA.meta.personas;

// The visible organisms grouped by the spread dimension (empty groups dropped: loyalty has no Gemma panel).
function spreadGroups(){const ts=treated();
  return spreadKeys().map(k=>({key:k,orgs:ts.filter(o=>spreadKey(o)===k)})).filter(g=>g.orgs.length);}
const groupOrgs=k=>treated().filter(o=>spreadKey(o)===k);

// The implant methods the method filter leaves, in data order (the legends' method entries).
const visibleMethods=()=>DATA.meta.methods.filter(m=>state.methods.has(m));
const controlLegendItems=()=>Object.keys(CONTROL_NAMES).filter(m=>state.methods.has(CTRL)&&DATA.organisms.some(o=>o.method===m&&isControl(o)))
  .map(m=>CONTROL_NAMES[m]+" (control)");
// True while any filter hides an organism (Diagnostics withholds its run-wide totals then). The control chip is off by default, so it
// alone does not count: the totals are the run's, control included, until another filter narrows the selection.
const hasControlArm=()=>DATA.organisms.some(isControl);   // the filter bar shows the control chip only when the data has the arm
const hidingAny=()=>DATA.organisms.some(o=>!isControl(o)&&!orgVisible(o));

// The control arm for this persona, resolved to the SELECTED base model so a 32B organism's
// Δ is never taken against the 14B base (a persona can carry a base per base model).
function baseId(persona,bm){const cands=DATA.organisms.filter(o=>o.is_base&&o.persona===persona);
  const want=bm||state.baseModel;   // bm: the per-model radar view passes each organism's own base model
  // an explicit bm never falls back to another family's base: no same-model base -> null
  const b=cands.find(o=>!want||o.base_model===want)||(bm?null:cands[0]);return b?b.id:null;}

// Unfiltered by design: the drawer shows every organism on a model (optionally of one behaviour), whatever the filters hide, and the
// compare-models filter offers every base model that carries the behaviour.
const modelOrgs=(bm,pn)=>DATA.organisms.filter(o=>!o.is_base&&bmOf(o)===bm&&(pn==null||o.persona===pn));
const modelsCarrying=pn=>BASE_MODELS.filter(b=>DATA.organisms.some(o=>!o.is_base&&o.persona===pn&&bmOf(o)===b));

function setMode(m){state.mode=m;
  if(m==="model"&&!DATA.meta.personas.includes(state.behaviour))state.behaviour=DATA.meta.personas[0]||null;
  render();}
function initState(){   // once: everything starts selected
  DATA.meta.methods.forEach(m=>{state.methods.add(m);state.drillMethods.add(m);});
  state.drillMethods.add(CTRL);   // the drill-down's own chip for the control (shown there only while the global control chip is on)
  DATA.meta.personas.forEach(p=>state.personas.add(p));
  state.baseModel=BASE_MODELS[0]||null;   // behaviour mode: exactly one base model at a time
  BASE_MODELS.forEach(b=>state.models.add(b));   // model mode: any subset
}
