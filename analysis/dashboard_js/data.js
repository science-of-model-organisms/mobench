// Lookups and readings over DATA: organisms, base models, the control arm, Δ readings with their CI, radar values, expression verdicts.
const _ORGSEEDS={};DATA.organisms.forEach(o=>_ORGSEEDS[o.id]=o.n_seeds||1);
function orgSeeds(id){return _ORGSEEDS[id]||1;}

const _P={};DATA.organisms.forEach(o=>_P[o.id]=o.persona);
const _BM={};DATA.organisms.forEach(o=>_BM[o.id]=o.base_model);
const BASE_MODELS=(DATA.meta.base_models||[]).filter(x=>x&&x!=="?");

const ORG_BY_ID={};DATA.organisms.forEach(o=>ORG_BY_ID[o.id]=o);

// A base model's full name from its id, for legends and chips (colour carries the model elsewhere):
// "Qwen/Qwen2.5-14B-Instruct" -> "Qwen2.5-14B", "google/gemma-4-31B-it" -> "Gemma-4-31B". Last path part, no chat-tuning
// suffix, first letter capitalised. Full names never collide (Qwen2.5-32B vs Qwen3-32B); two ids that still
// would (same slug under two orgs) keep their org prefix.
const _BMNAME={};
function bmName(bm){const id=String(bm||"");if(id in _BMNAME)return _BMNAME[id];
  const base=x=>{const t=String(x).split("/").pop().replace(/-(instruct|it|chat)$/i,"");return t.charAt(0).toUpperCase()+t.slice(1);};
  const n=base(id),clash=BASE_MODELS.some(m=>m!==id&&base(m)===n);
  return _BMNAME[id]=clash?id.replace(/-(instruct|it|chat)$/i,""):n;}
// An organism's OWN base model, whatever the page is showing.
function bmOf(o){return o.base_model!==undefined?o.base_model:(_BM[o.org]||_BM[o.id]);}

// Organism names. A point or row is named by the paper's short method name only (OCT, SFT, DPO, Steering,
// Distillation, Prompting, Few-shot; the negative-prompt arm is the control); the variant, base model and
// ids live in the hover (orgHead) and colour carries the base model or behaviour.
const CONTROL_NAMES={prompting:"Negative prompt"};   // method -> the control arm's name
const CONTROL_VARIANTS=["negative"];                  // a [variant] that marks the control arm
function variantOf(o){return (/\[(.+?)\]/.exec(o.id||o.org||"")||[])[1]||"";}
// Every (organism, eval) pair with no score cell, grouped by eval: organisms x evals minus the filled cells, the same arithmetic as
// meta.n_cells_possible - meta.n_cells (analysis/dashboard.py: len(organisms) * len(eval_names) possible). Unfiltered on purpose (run-wide).
function missingCells(){const out=[];
  Object.keys(DATA.evals).forEach(e=>{const orgs=DATA.organisms.filter(o=>!(DATA.cells[o.id]||{})[e]);if(orgs.length)out.push({e,orgs});});
  return out;}
function isControl(o){return !!CONTROL_NAMES[o.method]&&CONTROL_VARIANTS.includes(variantOf(o));}
function niceOrg(o){return isControl(o)?CONTROL_NAMES[o.method]+" (control)":pretty(o.method||"");}

function kOf(id){const m=/k=(\d+)/.exec(id);return m?+m[1]:null;}
function acell(o,ax){return (DATA.axis_cells[o]||{})[ax]||null;}
function ad(o,ax){const c=acell(o,ax);return c&&c.delta!=null?c.delta:null;}   // oriented Δ (+ = better); parcoords / bump
// A cell barred by an eval_meta.yaml gate (Matrix.gate_failure) carries votes:false; every
// average over DATA.bucket_evals must skip it, as Python's axis_delta does. A blob
// from before the flag existed has no key, which reads as voting.
function votes(c){return !!c&&c.votes!==false;}
// An organism's mean over a property's voting evals, and its base arm's mean over the SAME evals (so a gated cell cannot make the two
// ends average different sets). Both are computed in Python (analysis/dashboard_stats.py) and read back here.
function axisMean(orgId,ax){const v=((DATA.derived.axis_mean||{})[orgId]||{})[ax];return v==null?null:v;}
function axisBaseMean(orgId,ax){const v=((DATA.derived.axis_mean_base||{})[orgId]||{})[ax];return v==null?null:v;}

// ---- Δ vs base: organism minus the same-size base on the same items, in the score's own 0-1 units ----
// Python exports per cell/axis `delta` (oriented: + = better) and `se` (combined SE of the difference;
// share-kept cells: the organism's own SE), and with them the 95% CI = delta +/- 1.96*se (`ci_lo`, `ci_hi`, clipped to [0, 1] for a
// share kept) and a significance flag (`sig`: the CI excludes 0), from analysis/dashboard_stats.py. Nothing is computed here. A reading r = {o: oriented Δ, d: DISPLAYED Δ, lo, hi
// (displayed), sig, share}.
// SIGN CONVENTION (one place): a backwards axis (misalignment) is DISPLAYED as organism minus base, so
// + = MORE misbehaviour; colours always use the oriented value (blue = better organism). Every other
// axis is displayed oriented (+ = better). Introspection etc. are signed Δ like the rest.
const AXMETA={};DATA.axes.forEach(a=>AXMETA[a.name]=a);
function axBack(ax){return !!(AXMETA[ax]&&AXMETA[ax].runs_backwards);}
// SHARE AXES (robustness): the value is the share kept (read against 0), NOT a difference from base.
// This is the ONE place that decides it, so changing robustness to a Δ is a one-line edit here
// (return false) plus the Python flag `self_referenced`.
function axShare(ax){return !!(AXMETA[ax]&&AXMETA[ax].self_referenced);}
function evalAxis(e){return (DATA.evals[e]||{}).axis;}
function fin(x){return x!=null&&isFinite(x)?x:null;}
function mkRead(back,share,c){
  const delta=fin(c.delta);if(delta==null)return null;
  const olo=fin(c.ci_lo),ohi=fin(c.ci_hi),sg=back?-1:1;   // the interval is on the oriented delta; a backwards axis is DISPLAYED flipped
  const a=olo==null?null:sg*olo,b=ohi==null?null:sg*ohi;
  return {o:delta,d:sg*delta,se:fin(c.se),attr:c.attrition_flag===true?{o:fin(c.discard),b:fin(c.base_discard)}:null,olo,ohi,lo:a==null?null:Math.min(a,b),hi:a==null?null:Math.max(a,b),share,back,sig:!!c.sig};}
function axisRead(org,ax){const c=acell(org,ax);if(!c)return null;
  const share=axShare(ax);
  return mkRead(axBack(ax),share,c);}
function evalRead(org,e){const c=(DATA.cells[org]||{})[e];if(!c)return null;
  return mkRead(axBack(evalAxis(e)),!!c.self_ref,c);}
function axisEff(org,ax){const r=axisRead(org,ax);return r&&!r.share?r.o:null;}   // oriented Δ for the Δ radars; a share axis is not a Δ spoke
function evalEff(org,e){const r=evalRead(org,e);return r&&!r.share?r.o:null;}
function fmtCI(r){return r.lo==null?"":" · 95% CI ["+fmt(r.lo)+", "+fmt(r.hi)+"]";}
// Tooltip body for one reading: organism and base values, Δ, CI, significance.
function readTip(r,orgRaw,baseRaw,what){
  const f=x=>x==null?"n/a":x.toFixed(3);
  if(r==null)return "unmeasurable";
  if(r.share)return `share kept ${f(orgRaw)} (not a difference from base)`+(r.lo==null?"":` · 95% CI [${r.lo.toFixed(3)}, ${r.hi.toFixed(3)}]`);
  return `organism ${f(orgRaw)} · base ${f(baseRaw)}<br>Δ ${fmt(r.d)}${r.lo==null?" · no 95% CI (a standard error is unknown)":fmtCI(r)}${r.sig?" · significant (CI excludes 0)":""}`+(r.back?"<br><i>"+SIGN_NOTE_BACK+"</i>":"")+attrNote(r);}
// Report-only (#254): the two arms discarded very different shares of their rows, so this Δ may be biased. Never gates or recolours anything.
function attrNote(r){if(!r||!r.attr)return "";const p=x=>x==null?"n/a":Math.round(x*100)+"%";
  return `<br><i>&dagger; differential attrition: ${p(r.attr.o)} of this organism's rows unscored vs ${p(r.attr.b)} of its base's (gap &gt; ${((DATA.meta||{}).attrition_bound)??"the bound"}), so this Δ may be biased</i>`;}
const SIGN_NOTE_BACK="+ = more misbehaviour than base";

// ---- the paper's radar scale: values computed server-side by analysis/dashboard.py radar_block ----
// (analysis/radar_scale.py radar_value, via paper_figures.fig1_scores), never recomputed here.
const RADAR=(DATA.radar&&DATA.radar.orgs)?DATA.radar:null;
const PAPER_SCALE=!!RADAR;   // internal name: true when the build carries the 0-10 radar block
// What a reader sees on every 0-10 radar: Delta once, and the formula of each kind of spoke (analysis/radar_scale.py radar_value).
const DELTA_DEF="Δ = organism − same-size base, in the score's 0–1 units";
const FORMULA={behaviour:"score = 10 × matched-item Δ (base = 0)",health:"score = 10 × (1 + Δ), capped at 10 (base = 10)",share:"score = 10 × share kept"};
// Radar values use matched items (the items scored for BOTH the organism and its base), so the Δ that drives a radius is the matched-item Δ,
// which can differ slightly from the page's cell-mean Δ. MATCHED_NOTE says so in the key; matchedDelta() reads it back from a score.
const MATCHED_NOTE="Matched items: a radar value counts only the items scored for both the organism and its base model, so its Δ can differ slightly from the page's cell-mean Δ (heatmap, forest, bars).";
function matchedDelta(axis,score){if(score==null||!isFinite(score))return null;
  if(axShare(axis))return "share kept "+(score/10).toFixed(2);
  if(HEALTH_AXES.has(axis))return score>=10-1e-9?"Δ ≥ 0 (score capped at 10)":"Δ "+fmt(score/10-1,2);
  return "Δ "+fmt(score/10,2);}
// A series' tooltip Δ: the MEAN of its members' read-back Δs (not the Δ of the mean score, which is wrong once a member sits at the cap: a
// health score of 10 only says Δ >= 0, a behaviour score of 10 only Δ >= 1). With any member capped it is a lower bound, labelled so.
// The mean (`md`) and the capped flag (`mdc`) come from Python (dashboard_stats.py); this only formats them.
function matchedDeltaMean(st){return st.md==null?null:st.mdc?"Δ ≥ "+fmt(st.md,2)+" (some capped)":"Δ "+fmt(st.md,2);}
const HEALTH_AXES=new Set(["coherence","capability"]);
function axisFormula(name){return axShare(name)?FORMULA.share:HEALTH_AXES.has(name)?FORMULA.health:FORMULA.behaviour;}
// The radar's properties (misalignment is not one of them: it has its own section), in the data's order.
// Unlike the paper's radars, a value below base is drawn (the centre is the lowest value needed), not clipped to the centre.
const RADAR_NOTE="Unlike the paper's radars, values below base are drawn (not clipped to the centre).";
const radarAxes=()=>DATA.axes.filter(a=>a.radar!==false);
function radarOf(id,ax){const o=RADAR&&RADAR.orgs[id];const v=o?o[ax]:null;return v==null||!isFinite(v)?null:v;}
function radarEvalOf(id,e){const o=RADAR&&RADAR.evals&&RADAR.evals[id];const v=o?o[e]:null;return v==null||!isFinite(v)?null:v;}

// The expression threshold verdict (analysis/derived.py sbe_verdicts). Anything but PASS / PASS_INDIST, a
// missing verdict included, is drawn dashed, like the paper's radars.
function exprVerdict(id){const e=((DATA.expression||{}).orgs||{})[id];return e?e.verdict:null;}
function exprFlagged(id){const v=exprVerdict(id);return v!=="PASS"&&v!=="PASS_INDIST";}
// What a reader sees for a verdict: PASS or FAIL (the in-distribution-only pass counts as a pass), anything else as it is.
function verdictLabel(v){return v==="PASS_INDIST"?"PASS":v;}
function exprNote(id){const v=exprVerdict(id);
  return v==="FAIL"?"falls below the expression threshold":"no verdict clearing the expression threshold ("+(verdictLabel(v)||"none")+")";}

function exprOf(o){
  const E=DATA.expression||{};const e=(E.orgs||{})[o];
  if(e&&e.rate!=null)return e;
  const c=(DATA.cells[o]||{})[E.eval||"simple_behaviour_exhibit"];
  if(!c||c.mean==null)return null;
  const hw=c.sem_hw!=null?c.sem_hw:0;   // 1.96 x sem, from Python
  return {rate:c.mean,lo:c.mean-hw,hi:c.mean+hw,verdict:null};
}
