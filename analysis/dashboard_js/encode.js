// Visual encodings: colour (behaviour / base model / method / organism / property / eval), marker shape, point labels, hover headers, the theme.
// ---- encodings: colour = the SPREAD dimension (behaviour, or base model in "models on a behaviour" mode), marker shape = implant method ----
// Two compare modes. The PICK dimension is one radio value; the SPREAD dimension is what every figure shows side by side and colours:
//   behaviour mode (default): pick = base model, spread = behaviour;  model mode: pick = behaviour, spread = base model.
// Popping, high-contrast on white; primary is an orange-brown (matches the earlier report).
const PCOLORS=["#dd8452","#4c72b0","#55a868","#8172b3","#937860","#da8bc3","#64b5cd","#ccb974"];

const spreadColor=k=>PCOLORS[Math.max(0,spreadKeys().indexOf(k))%PCOLORS.length];
const orgCol=o=>spreadColor(spreadKey(o));
const spreadLabel=k=>state.mode==="model"?bmName(k):pretty(k);
const spreadHead=k=>(state.mode==="model"?"Base model: ":"Behaviour: ")+spreadLabel(k);
const colourBy=()=>state.mode==="model"?"base model":"behaviour";

const serLabel=k=>k===CTRL?CONTROL_NAMES.prompting+" (control)":pretty(k);
const serShort=k=>k===CTRL?CONTROL_NAMES.prompting:pretty(k);
const serColor=k=>k===CTRL?"var(--muted)":methodColor(k);

// A saturated categorical palette for colouring by IMPLANT METHOD (drill-down + legends).
const MPALETTE=["#dd8452","#4c72b0","#55a868","#8172b3","#937860","#da8bc3","#64b5cd","#ccb974","#c44e52","#8c8c8c"];
// The paper's method palette (same hexes as analysis/paper_style.py METHODS from PR #453; keep in sync):
// oct, sft, dpo, steering, seqkd, prompting (sysprompt), few-shot (icl). Matched on the method name, first
// hit wins, on whole "_"-separated tokens ("doctor" is not "oct"); an unknown method falls back to MPALETTE. oct_demos is a darker shade (#8A6A10) of the OCT gold
// so it is told apart from oct_behaviour; it must stay ahead of "oct".
const METHOD_COLORS=[["oct_demos","#8A6A10"],["oct","#B38B1A"],["sft","#1A9576"],["dpo","#135DF6"],["steering","#0E9BD2"],
  ["seqkd","#FE4F7F"],["prompting","#9B5A34"],["icl","#93548C"]];
// On a dark page the three darkest method colours (DPO blue, prompting brown, OCT-demos gold) fall
// under 3:1 against the plot, so they are lifted; the paper's light palette is untouched.
const DARK_LIFT={"#135DF6":"#6C97FF","#9B5A34":"#CF9068","#8A6A10":"#CDA22E","#93548C":"#C189BA"};
// Pre-trained (external) adapters: methods whose organisms carry `external` (a mobench user-config adapter, not an implant the paper
// built). They are NOT implant methods: own filter row, ONE shared marker (hexagon, the one SHAPES entry no paper method uses), and a
// cool-grey palette distinct from every paper method colour (they are told apart by name on the row labels). Empty on a paper-only page.
const EXT_METHODS=new Set(DATA.organisms.filter(o=>o.external).map(o=>o.method));
const isExtMethod=m=>EXT_METHODS.has(m);
const EXT_LIST=[...EXT_METHODS].sort();
const PAPER_METHODS=DATA.meta.methods.filter(m=>!EXT_METHODS.has(m));   // shapes / palette index over these only: the paper methods keep theirs
const EXT_COLORS=["#5C6B7A","#8E9AA6","#3E4A56"],EXT_DARK=["#9DB0C2","#C3CDD6","#7F93A6"];
const methodColor=m=>{if(EXT_METHODS.has(m)){const i=EXT_LIST.indexOf(m)%EXT_COLORS.length;return (typeof isDark==="function"&&isDark()?EXT_DARK:EXT_COLORS)[i];}
  const hit=METHOD_COLORS.find(([k])=>("_"+m+"_").includes("_"+k+"_"));
  const c=hit?hit[1]:MPALETTE[PAPER_METHODS.indexOf(m)%MPALETTE.length];
  return (typeof isDark==="function"&&isDark()&&DARK_LIFT[c])||c;};

// A distinct colour per ORGANISM (every dose/variant included) - for the drill-down.
const ORG_PALETTE=["#dd8452","#4c72b0","#55a868","#8172b3","#937860","#da8bc3","#64b5cd","#ccb974",
  "#c44e52","#8c8c8c","#c98a5a","#6a8ec7","#79b47a","#a48bc0","#a68a72","#4fa39a","#d1a05a","#9aa0a6"];
const _ORGCOL={};

// stable colour per organism "short" (method[variant]@scale), so each dose gets its own colour
[...new Set(DATA.organisms.filter(o=>!o.is_base).map(o=>o.short))].forEach((k,i)=>_ORGCOL[k]=ORG_PALETTE[i%ORG_PALETTE.length]);
function orgColorKey(o){return _ORGCOL[o.short]||"#888";}

const SHAPES=["circle","square","triangle","diamond","plus","cross","star","hexagon"];
const methodShape=m=>EXT_METHODS.has(m)?"hexagon":SHAPES[PAPER_METHODS.indexOf(m)%SHAPES.length];
// The negative-prompt control gets its own marker, a thick minus ("the negative of prompting"'s plus), wherever a
// marker encodes the method; legends add a matching entry (controlLegendItems).
const orgShape=o=>isControl(o)?"minus":methodShape(o.method);

// Figure-1 legend only: "(Maiya et al., 2025)" after a method name, from meta.method_citations (analysis/method_citations.yaml).
// Only https URLs become links; a method with no (or an unsafe) citation gets nothing - no empty brackets.
const methodCite=m=>{const c=((DATA.meta.method_citations||{})[m])||null;
  return c&&c.short&&/^https:\/\//.test(c.url||"")?c:null;};

// ONE fixed colour per property, used everywhere a property is colour-coded (all distinct).
const _BUCKETCOL={};DATA.axes.forEach((a,i)=>_BUCKETCOL[a.name]=MPALETTE[i%MPALETTE.length]);
function bucketColor(n){return _BUCKETCOL[n]||"#8c8c8c";}

const EVAL_PALETTE=["#dd8452","#4c72b0","#55a868","#8172b3","#937860","#da8bc3","#64b5cd","#ccb974","#c44e52","#8c8c8c"];
// ONE fixed colour per individual eval, stable across the whole page (used by the paired-bars
// drill). Ordered over every property's evals so two evals in one property never collide.
const _EVALCOL={};[...new Set(DATA.axes.flatMap(a=>DATA.bucket_evals[a.name]||[]))].forEach((e,i)=>_EVALCOL[e]=EVAL_PALETTE[i%EVAL_PALETTE.length]);
function evalColor(e){return _EVALCOL[e]||"#8c8c8c";}

// ---- colour helpers ----
// The reader's theme: <html data-theme> wins, else the OS preference (a page without matchMedia is light).
function isDark(){const t=document.documentElement.getAttribute("data-theme");
  if(t==="dark")return true;if(t==="light")return false;
  try{return !!(window.matchMedia&&window.matchMedia("(prefers-color-scheme: dark)").matches);}catch(e){return false;}}
// Diverging scale, blue = better, red = worse. The dark scale starts from a dark neutral (a pale
// near-zero cell would glare on a dark page) and lifts the ends, so every cell keeps a visible edge.
function diverge(t){t=Math.max(-1,Math.min(1,t));
  const dk=isDark(),g=dk?[62,61,56]:[200,197,186],good=dk?[92,146,226]:[76,114,176],bad=dk?[212,104,96]:[196,106,98];
  const c=t>=0?mix(g,good,t):mix(g,bad,-t);return `rgb(${c[0]|0},${c[1]|0},${c[2]|0})`;}
// Text on a cell is picked by contrast against the cell's own fill (white or near-black), whichever
// theme: the cell colour is fixed, so the ink must not follow the page's text colour.
function inkOn(fill){const m=/rgb\((\d+),(\d+),(\d+)\)/.exec(fill||"");if(!m)return "var(--fg)";
  const L=[1,2,3].map(i=>{const v=+m[i]/255;return v<=0.03928?v/12.92:Math.pow((v+0.055)/1.055,2.4);});
  const lum=0.2126*L[0]+0.7152*L[1]+0.0722*L[2];
  return (1.05/(lum+0.05))>=((lum+0.05)/0.05)?"#ffffff":"#000000";}
// WCAG contrast of two [r,g,b] colours, and a "#rrggbb" token read from the page's CSS (fallback when it cannot be read).
function rgbOf(c){const m=/^#([0-9a-f]{6})$/i.exec(String(c||"").trim());if(!m)return null;const n=parseInt(m[1],16);return [n>>16&255,n>>8&255,n&255];}
function relLum(c){const L=c.map(v=>{v/=255;return v<=0.03928?v/12.92:Math.pow((v+0.055)/1.055,2.4);});return 0.2126*L[0]+0.7152*L[1]+0.0722*L[2];}
function contrast(a,b){const x=relLum(a),y=relLum(b);return (Math.max(x,y)+0.05)/(Math.min(x,y)+0.05);}
function cssRgb(name,fallback){try{return rgbOf(getComputedStyle(document.documentElement).getPropertyValue(name))||fallback;}catch(e){return fallback;}}
// Figure 1's point labels are written in their marker's colour, so two stacked labels with the same text (the two "Neg. prompt"
// controls) still say whose they are. A marker colour that reads under 4.5:1 (WCAG AA for text) on the plot or on the label's halo is
// mixed toward black (light page) or white (dark page) in 5% steps until it passes: same hue, readable in both themes.
function labelInk(col){const c=rgbOf(col);if(!c)return "var(--fg)";
  const dk=isDark(),to=dk?[255,255,255]:[0,0,0];
  const bgs=[cssRgb("--plot",dk?[25,24,22]:[255,255,255]),cssRgb("--halo",dk?[30,29,26]:[255,255,255])];
  for(let k=0;k<=20;k++){const m=mix(c,to,k/20).map(Math.round);
    if(bgs.every(b=>contrast(m,b)>=4.5))return "#"+m.map(v=>v.toString(16).padStart(2,"0")).join("");}
  return dk?"#ffffff":"#000000";}

// The Figure 1 ON-POINT label only: two control points stack at ~1000 px wide, so it is abbreviated there. Every other view (legend,
// tooltips, Profiles, lists) keeps niceOrg's full "Negative prompt (control)".
const CONTROL_POINT_NAMES={prompting:"Neg. prompt"};
function pointLabel(o){return isControl(o)&&CONTROL_POINT_NAMES[o.method]?CONTROL_POINT_NAMES[o.method]:niceOrg(o);}

// The hover header for an organism: short name, then its id, variant, base model and behaviour.
function orgHead(o){const id=o.id||o.org||"",v=variantOf(o);
  return `<b>${escHtml(niceOrg(o))}</b><br><span class="ttd">${escHtml(id)}</span><br>variant: ${escHtml(v||"none")} &middot; base model: ${escHtml(bmName(bmOf(o)))} &middot; behaviour: ${escHtml(pretty(o.persona||_P[id]))}`;}
