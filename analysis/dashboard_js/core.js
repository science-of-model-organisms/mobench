// The data blob, the page state, and DOM / number / label helpers every other file uses.
const DATA = JSON.parse(document.getElementById("data").textContent);
// There is no CI multiplier here: every 95% interval (a flat 1.96 on the cell's `sem` / `se`, as Cell.ci95 in aggregate.py) is computed in
// Python (analysis/dashboard_stats.py) and arrives in the blob as `hw`, `sem_hw`, `ci_lo`, `ci_hi`. The significance dots use the SAME
// interval as the forest whisker (a dot = the CI excludes 0).

const state = { inclFail:false, mode:"behaviour", behaviour:null, models:new Set(), methods:new Set(), personas:new Set(), drillMethods:new Set(), bucket:null, radarView:"bucket", modelPersona:null, baseModel:null, txOrg:null, txBucket:null };
const NS="http://www.w3.org/2000/svg";
// Figure text is 13 CSS px (the paper's figure size). A figure is drawn at the width it is shown at (its viewBox width = its CSS width),
// so a font-size attribute is the size on screen; `natural` stops it being scaled UP past that (it only ever shrinks in a narrow box).
const FS=13;
// Width of a text in CSS px at `size` in the page font: measured with a canvas where there is one (a browser), else 0.6 em per
// character (jsdom has no canvas). Used to give a label the room it needs (radar side padding).
const _TW={};
function textW(s,size){const k=size+"|"+s;if(k in _TW)return _TW[k];let w=String(s).length*size*0.6;
  try{if(typeof navigator!=="undefined"&&!/jsdom/i.test(navigator.userAgent||"")){
    const c=textW.c||(textW.c=document.createElement("canvas").getContext("2d"));
    if(c){c.font=size+"px "+(getComputedStyle(document.body).fontFamily||"sans-serif");w=c.measureText(String(s)).width;}}}catch(e){}
  return _TW[k]=w;}
function natural(svg,W){svg.style.maxWidth=W+"px";return svg;}

function el(t,a={},k=[]){const n=document.createElementNS(NS,t);for(const x in a)n.setAttribute(x,a[x]);for(const c of k)n.appendChild(c);return n;}
// Numbers read with a true minus sign (U+2212), and a negative that rounds to zero is "0.00", never "-0.00": applied to every SVG text and
// tooltip, and by fmt(). A hyphen counts as a minus only before a digit and after a space, bracket, comma or the start ("t18-20" is safe).
const tidyNum=s=>String(s).replace(/(^|[\s(\[,>])-(?=[.\d])/g,"$1\u2212").replace(/\u2212(0\.0+)(?!\d)/g,"$1");
function txt(s){return document.createTextNode(tidyNum(s));}
function fmt(x,d=3){if(x==null)return "-";const s=x.toFixed(d);return /^-0\.?0*$/.test(s)?s.slice(1):(x>=0?"+":"\u2212")+s.replace("-","");}

// Human-readable labels: underscores → spaces, a few known acronyms tidied. Applied to
// every eval / method / axis / organism label the reader sees (raw ids stay in the data).
const _ACR={icl:"ICL",oct:"OCT",sft:"SFT",dpo:"DPO",mcq:"MCQ",mmlu:"MMLU",gpqa:"GPQA",ifeval:"IFEval",
  oeq:"open-ended",dod:"Δ-of-Δ",wmdp:"WMDP",ood:"OOD",mc:"MC"};
// The paper's method names (analysis/paper_style.py METHODS): OCT, SFT, DPO, Steering, Distillation,
// Prompting, Few-shot. Applied to every label that carries a method id, before the generic tidy-up.
const METHOD_NAMES={oct_behaviour:"OCT",oct_demos:"OCT demos",sft_behaviour:"SFT",dpo_behaviour:"DPO",
  steering:"Steering",distillation_seqkd:"Distillation",
  prompting:"Prompting",few_shot_icl:"Few-shot"};
function pretty(s){if(s==null)return "";
  const re=pretty.re||(pretty.re=new RegExp("\\b("+Object.keys(METHOD_NAMES).join("|")+")\\b","g"));
  return String(s).replace(re,m=>METHOD_NAMES[m]).replace(/_/g," ")
  .replace(/\b([a-z]+)\b/g,(w)=>_ACR[w]||w);}
// Sentence case for a TITLE (section / card / overlay): the first letter is capitalised, nothing else is touched, so acronyms and
// proper nouns stay as written. A first word with a digit in it ("gsm8k") is a benchmark name, not a sentence start: left alone.
// Applied where titles are rendered, never to the data (axis names stay lower-case in ticks, legends and tooltips).
function cap(s){return String(s==null?"":s).replace(/^([a-z])(?![^\s]*\d)/,m=>m.toUpperCase());}
// An axis (paper: "property") as the reader sees it; misalignment is labelled like every other.
function axisLabel(name,short){return pretty(short||name);}

// A label for HTML (a tooltip, a note, an innerHTML string): pretty() then escaped. Data-derived text never goes into HTML unescaped.
const ph=s=>escHtml(pretty(s));
function escHtml(x){return String(x).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));}

// A label clipped to n characters: clips the head and keeps a " · behaviour" suffix (niceOrg), which is what tells two rows apart.
function clip(s,n){s=pretty(s);if(s.length<=n)return s;const i=s.lastIndexOf(" · ");
  const tail=i>0?s.slice(i):"";const head=tail?s.slice(0,i):s;
  return head.slice(0,Math.max(1,n-1-tail.length))+"…"+tail;}

// ---- how-to line ----
function howto(section,html){const d=document.createElement("div");d.className="howto";d.innerHTML=html;section.appendChild(d);}

// Small DOM builders: a section with its title, a div, an empty-state note, a titled panel.
function sec(title,fig){const s=document.createElement("section");if(fig)s.dataset.fig=fig;s.innerHTML=`<h2>${cap(title)}</h2>`;return s;}
function div(c){const d=document.createElement("div");d.className=c;return d;}
function empty(m){const d=document.createElement("div");d.className="empty";d.textContent=m;return d;}
function panelBox(title,svg){const d=div("rpanel");const t=div("rt");t.textContent=cap(title);d.appendChild(t);d.appendChild(svg);return d;}

// Deterministic [0,1) from a string id and an integer seed (FNV-1a + a finaliser); used for the scatter's visual jitter.
function seedUnit(id,seed){let h=(2166136261^(seed*16777619))>>>0;const t=String(id);
  for(let i=0;i<t.length;i++){h^=t.charCodeAt(i);h=Math.imul(h,16777619)>>>0;}
  h^=h>>>13;h=Math.imul(h,0x5bd1e995)>>>0;h^=h>>>15;return (h>>>0)/4294967296;}
function clearNode(n){while(n.firstChild)n.removeChild(n.firstChild);}

function mix(a,b,t){return [a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t,a[2]+(b[2]-a[2])*t];}
function nice(lo,hi){if(lo===hi){lo-=1;hi+=1;}const p=(hi-lo)*0.08;return [lo-p,hi+p];}
function ticks(r){const[lo,hi]=r;const st=niceStep((hi-lo)/5);const o=[];for(let t=Math.ceil(lo/st)*st;t<=hi;t+=st)o.push(+t.toFixed(6));return o;}
function niceStep(x){const p=Math.pow(10,Math.floor(Math.log10(x)));const f=x/p;return (f<1.5?1:f<3?2:f<7?5:10)*p;}
