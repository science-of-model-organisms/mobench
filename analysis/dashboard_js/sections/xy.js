// Figure 1: two properties against each other.
// Coherence vs generality (the paper's trade-off), interactive: a dropdown per axis (x and y), any property of DATA.axes
// (capability, generality, misalignment, ...). Values are sign-normalised so up/right = better.
const XY={x:"generality",y:"coherence",jitter:true,fit:false};

function xyChart(container){
  const names=DATA.axes.map(a=>a.name);
  if(!names.includes(XY.x))XY.x=names[0];
  if(!names.includes(XY.y))XY.y=names[Math.min(1,names.length-1)];
  const axo=n=>DATA.axes.find(a=>a.name===n)||{};
  const plane={behaviour:XY.x,cost:XY.y,behaviour_backwards:!!axo(XY.x).runs_backwards,
    cost_backwards:!!axo(XY.y).runs_backwards,
    points:treated().map(o=>({org:o.id,id:o.id,persona:o.persona,base_model:o.base_model,method:o.method,scale:o.scale,short:o.short}))};
  const tmp=document.createElement("div");
  scatter(tmp,plane,`${escHtml(pretty(XY.y))} vs ${escHtml(pretty(XY.x))}`,
    `Each point is one implant × behaviour${state.mode==="model"?" × base model":""}, named by its method: <b>marker shape = method${EXT_LIST.length?" (pre-trained adapters share one marker)":""}, colour = ${colourBy()}</b> (see the legend). Pick any two properties with the dropdowns. Both axes are sign-normalised: up and right are better. The base model sits at the crosshair; the top-right is an organism that did well on both. <b>Hover</b> a point for its full details (organism id, variant, base model, values with 95% intervals). <b>Pinch or &#8984;/Ctrl+scroll to zoom, drag to pan once zoomed</b> (or use + / &minus; / Reset zoom); plain scrolling still scrolls the page, and zooming spreads points apart rather than enlarging them. Every point is labelled; where labels would collide they move out with a leader line. <b>Jitter is visual only</b>: it nudges points that sit on (nearly) the same spot apart by a small fixed offset so they can be told apart. Hover and the tooltip always show the true values; untick it to see the exact positions.`,"xy");
  const sect=tmp.querySelector("section");
  const ctl=div("xyctl");
  [["x","X axis"],["y","Y axis"]].forEach(([k,lab])=>{
    const l=document.createElement("label");l.appendChild(txt(lab));
    const sel=document.createElement("select");sel.id="xy-"+k;sel.setAttribute("aria-label",lab);
    DATA.axes.forEach(a=>{const o=document.createElement("option");o.value=a.name;o.textContent=axisLabel(a.name);
      if(a.name===XY[k])o.selected=true;sel.appendChild(o);});
    sel.onchange=()=>{XY[k]=sel.value;rebuildFigs(["xy"]);};
    l.appendChild(sel);ctl.appendChild(l);});
  const jl=document.createElement("label");const jc=document.createElement("input");jc.type="checkbox";jc.id="xy-jitter";jc.checked=XY.jitter!==false;
  jc.setAttribute("aria-label","Jitter overlapping points (visual only)");
  jc.onchange=()=>{XY.jitter=jc.checked;if(sect._repaint)sect._repaint();};
  jl.appendChild(jc);jl.appendChild(txt("Jitter overlapping points (visual only)"));ctl.appendChild(jl);
  const fl=document.createElement("label");const fc=document.createElement("input");fc.type="checkbox";fc.id="xy-fit";fc.checked=!!XY.fit;
  fc.setAttribute("aria-label","Fit line (r)");
  fc.onchange=()=>{XY.fit=fc.checked;if(sect._repaint)sect._repaint();};
  fl.appendChild(fc);fl.appendChild(txt("Fit line (r)"));ctl.appendChild(fl);
  const how=sect.querySelector(".howto");
  if(how)how.after(ctl);else sect.querySelector("h2").after(ctl);
  container.appendChild(sect);
}
