// Semantic zoom (ctrl/cmd+wheel, pinch, drag) and its button bar.
// ---- zoom: wheel / pinch to zoom, drag to pan, buttons + Reset, for charts that repaint from a view window ----
// Semantic zoom: the axes are re-ranged and the chart redrawn, so markers, labels and ticks keep their size and
// stay readable (a viewBox zoom would only scale them up). `view` is {x:[lo,hi],y?:[lo,hi]} in data units, `base` the
// full range (a zoom never leaves it), `area` the plot rectangle in svg user units. `repaint()` redraws the svg.
let _zid=0;

function makeZoom(svg,W,area,view,base,repaint,axes){
  axes=axes||"xy";const MAXZ=40,ptrs=new Map();let pinch=null;
  const toFrac=e=>{const r=svg.getBoundingClientRect(),k=r.width>0?r.width/W:1;
    return {fx:(((e.clientX-r.left)/k)-area.x)/area.w,fy:(((e.clientY-r.top)/k)-area.y)/area.h};};
  const zoomed=()=>["x","y"].some(a=>axes.includes(a)&&(view[a][1]-view[a][0])<(base[a][1]-base[a][0])*0.999);
  const sync=()=>{const z=zoomed();svg.style.touchAction=z?"none":"pan-y";svg.classList.toggle("zoomed",z);};
  function zoomAt(fx,fy,f){
    ["x","y"].forEach(a=>{if(!axes.includes(a))return;const v=view[a],b=base[a],span=v[1]-v[0],bs=b[1]-b[0];
      const ns=Math.max(bs/MAXZ,Math.min(bs,span/f)),fr=Math.max(0,Math.min(1,a==="x"?fx:1-fy)),anchor=v[0]+fr*span;
      let lo=anchor-fr*ns;lo=Math.max(b[0],Math.min(lo,b[1]-ns));view[a]=[lo,lo+ns];});
    sync();repaint();}
  function panBy(dfx,dfy){
    ["x","y"].forEach(a=>{if(!axes.includes(a))return;const v=view[a],b=base[a],span=v[1]-v[0];
      const d=(a==="x"?-dfx:dfy)*span;const lo=Math.max(b[0],Math.min(v[0]+d,b[1]-span));view[a]=[lo,lo+span];});
    sync();repaint();}
  const inArea=f=>f.fx>=0&&f.fx<=1&&f.fy>=0&&f.fy<=1;
  // A plain wheel / trackpad scroll must scroll the PAGE. Zoom only on ctrl/cmd+wheel (a trackpad pinch arrives as a ctrlKey
  // wheel event, so pinch still zooms) and via the buttons. A plain scroll over the chart shows a brief hint instead.
  let hintEl=null,hintT=null;
  const hint=()=>{const host=svg.parentNode;if(!host||!host.appendChild)return;
    if(!hintEl||hintEl.parentNode!==host){hintEl=document.createElement("div");hintEl.className="zwheelhint";
      hintEl.textContent="Pinch or \u2318/Ctrl+scroll to zoom";host.classList.add("zwrapx");host.appendChild(hintEl);}
    hintEl.classList.add("on");clearTimeout(hintT);hintT=setTimeout(()=>hintEl&&hintEl.classList.remove("on"),1400);};
  svg.addEventListener("wheel",e=>{const f=toFrac(e);if(!inArea(f))return;
    if(!(e.ctrlKey||e.metaKey)){hint();return;}
    e.preventDefault();
    zoomAt(f.fx,f.fy,Math.exp(-Math.max(-120,Math.min(120,e.deltaY))*0.004));},{passive:false});
  svg.addEventListener("pointerdown",e=>{const f=toFrac(e);if(!inArea(f))return;
    ptrs.set(e.pointerId,{x:e.clientX,y:e.clientY});
    if(ptrs.size===2){const [p,q]=[...ptrs.values()];pinch=Math.hypot(p.x-q.x,p.y-q.y)||1;}
    if(!zoomed()&&ptrs.size<2)return;   // drag-to-pan only applies while zoomed in (a pinch may start from the full view)
    try{svg.setPointerCapture(e.pointerId);}catch(_){}
    svg.classList.add("dragging");});
  svg.addEventListener("pointermove",e=>{const p=ptrs.get(e.pointerId);if(!p)return;
    const r=svg.getBoundingClientRect(),k=r.width>0?r.width/W:1;
    if(ptrs.size===2&&pinch){ptrs.set(e.pointerId,{x:e.clientX,y:e.clientY});
      const [a,b]=[...ptrs.values()],d=Math.hypot(a.x-b.x,a.y-b.y)||1,
        f=toFrac({clientX:(a.x+b.x)/2,clientY:(a.y+b.y)/2});zoomAt(f.fx,f.fy,d/pinch);pinch=d;return;}
    const dx=(e.clientX-p.x)/k,dy=(e.clientY-p.y)/k;ptrs.set(e.pointerId,{x:e.clientX,y:e.clientY});
    if(zoomed())panBy(dx/area.w,dy/area.h);});
  const up=e=>{ptrs.delete(e.pointerId);pinch=null;if(!ptrs.size)svg.classList.remove("dragging");};
  svg.addEventListener("pointerup",up);svg.addEventListener("pointercancel",up);
  svg.classList.add("zsvg");svg.style.touchAction="pan-y";
  return {zoomed,
    by(f){zoomAt(0.5,0.5,f);},
    reset(){["x","y"].forEach(a=>{if(base[a])view[a]=[...base[a]];});sync();repaint();}};
}
// The In / Out / Reset buttons for a zoomable chart; `host` must be position:relative (.zwrap).
function zoomBar(host,zooms){
  const bar=div("zbar");
  [["+","Zoom in",z=>z.by(1.6)],["−","Zoom out",z=>z.by(1/1.6)],["Reset zoom","Reset zoom",z=>z.reset()]].forEach(([t,lab,fn])=>{
    const b=document.createElement("button");b.type="button";b.className="zbtn";b.textContent=t;b.title=lab;b.setAttribute("aria-label",lab);
    if(t==="Reset zoom")b.dataset.zoomReset="1";
    b.onclick=()=>zooms().forEach(fn);bar.appendChild(b);});
  host.appendChild(bar);}
