// The shared tooltip and the hover-fade of a group of marks.
const tt=document.getElementById("tt");
// A box can be PINNED (a click, a tap or Enter on a badge): it stops following the pointer and takes pointer events, so a tall one can be
// scrolled and read; Esc or a click outside closes it. While pinned, hovering a mark does not replace it.
tt.setAttribute("role","tooltip");
const ttPinned=()=>tt.classList.contains("pinned");
function showTT(e,h){if(ttPinned())return;tt.innerHTML=tidyNum(h);tt.style.opacity=1;moveTT(e);}
// The ONE place every hover box (figure marks, the (i) badges, the drawer) is placed: beside the pointer, flipped to its other side when it
// would cross the viewport edge, then clamped inside it, and capped to the viewport (with its own scroll) when taller or wider than it.
// position:fixed, so no overflow ancestor clips it; z-index 90 keeps it over the sticky header.
const TT_MARGIN=8,TT_GAP=14;
function moveTT(e){
  if(ttPinned())return;
  const vw=document.documentElement.clientWidth||window.innerWidth,vh=window.innerHeight;
  tt.style.maxWidth=Math.min(320,vw-2*TT_MARGIN)+"px";tt.style.maxHeight=(vh-2*TT_MARGIN)+"px";tt.style.overflowY="auto";
  const w=tt.offsetWidth,h=tt.offsetHeight;
  let x=e.clientX+TT_GAP,y=e.clientY+TT_GAP;
  if(x+w>vw-TT_MARGIN)x=e.clientX-TT_GAP-w;
  if(y+h>vh-TT_MARGIN)y=e.clientY-TT_GAP-h;
  x=Math.max(TT_MARGIN,Math.min(x,vw-TT_MARGIN-w));y=Math.max(TT_MARGIN,Math.min(y,vh-TT_MARGIN-h));
  tt.style.left=x+"px";tt.style.top=y+"px";}
function hideTT(){if(ttPinned())return;tt.style.opacity=0;}
// While pinned the box holds a button, so it is a (non-modal) dialog named by its badge; unpinned it is a plain tooltip again.
function unpinTT(){tt.classList.remove("pinned");tt.style.opacity=0;tt.scrollTop=0;
  tt.setAttribute("role","tooltip");tt.removeAttribute("aria-modal");tt.removeAttribute("aria-label");
  document.querySelectorAll(".infobadge[aria-expanded=true]").forEach(b=>{b.setAttribute("aria-expanded","false");b.removeAttribute("aria-controls");});}
function pinTT(label){tt.classList.add("pinned");tt.setAttribute("role","dialog");tt.setAttribute("aria-modal","false");tt.setAttribute("aria-label",label);}
document.addEventListener("keydown",e=>{if(e.key==="Escape"&&ttPinned())unpinTT();});
tt.addEventListener("click",e=>{if(ttPinned()&&e.target.closest&&e.target.closest(".ttx"))unpinTT();});   // the x: a phone-sized box can cover the whole screen, so there is nothing "outside" to tap
document.addEventListener("click",e=>{if(ttPinned()&&!tt.contains(e.target)&&!(e.target.closest&&e.target.closest(".infobadge")))unpinTT();});
// The (i) badge of a figure: hover shows the box, focus shows it too, a click / tap / Enter / Space pins it, Esc closes. Reachable by keyboard
// (tabindex, role=button, aria-describedby the box) and by touch (the tap is the click).
function infoBadge(h,label){
  const b=document.createElement("span");b.className="infobadge";b.textContent="i";b.title="";
  b.tabIndex=0;b.setAttribute("role","button");b.setAttribute("aria-label",label||"About this figure");
  const name=label||"About this figure";
  b.setAttribute("aria-expanded","false");
  const at=()=>{const r=b.getBoundingClientRect();return {clientX:r.left+r.width/2,clientY:r.bottom};};
  // aria-describedby points at the box only while the box shows this badge's text (an empty #tt would describe nothing)
  const describe=on=>{if(on)b.setAttribute("aria-describedby","tt");else b.removeAttribute("aria-describedby");};
  const show=e=>{showTT(e,h);if(!ttPinned())describe(true);},hide=()=>{hideTT();if(!ttPinned())describe(false);};
  b.addEventListener("mousemove",show);b.addEventListener("mouseleave",hide);
  b.addEventListener("focus",()=>show(at()));b.addEventListener("blur",hide);
  const toggle=()=>{if(ttPinned()){unpinTT();return;}tt.innerHTML='<button type="button" class="ttx" aria-label="Close">×</button>'+tidyNum(h);tt.style.opacity=1;moveTT(at());pinTT(name);
    describe(false);b.setAttribute("aria-expanded","true");b.setAttribute("aria-controls","tt");};
  b.addEventListener("click",e=>{e.stopPropagation();toggle();});
  b.addEventListener("keydown",e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();toggle();}else if(e.key==="Escape"&&ttPinned()){unpinTT();}});
  return b;}

// Universal hover-fade: hovering one mark in a group dims all the others.
function fadeGroup(nodes){nodes=nodes.filter(Boolean);nodes.forEach(n=>{
  n.addEventListener("mouseenter",()=>nodes.forEach(m=>{if(m!==n)m.classList.add("dim");else m.classList.add("hot");}));
  n.addEventListener("mouseleave",()=>nodes.forEach(m=>{m.classList.remove("dim");m.classList.remove("hot");}));});}
