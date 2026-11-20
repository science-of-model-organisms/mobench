// Click a radar or the effect matrix to enlarge it in an overlay.
// ---- enlarge: click a radar or the heatmap to open it bigger in an overlay (close button, Esc, or click outside) ----
let _ovl=null;
function closeOverlay(){hideTT();if(!_ovl)return;document.removeEventListener("keydown",_ovl.key,true);_ovl.root.remove();
  const f=_ovl.prev;_ovl=null;if(f&&f.focus){try{f.focus();}catch(_){}}}
function openOverlay(svg,title){
  closeOverlay();
  const root=div("ovl");root.setAttribute("role","dialog");root.setAttribute("aria-modal","true");root.setAttribute("aria-label",title||"Enlarged figure");
  const box=div("ovlbox"),head=div("ovlhead"),t=div("ovlt");t.textContent=title||"";
  const x=document.createElement("button");x.type="button";x.className="ovlclose";x.textContent="\u2715 Close";x.setAttribute("aria-label","Close enlarged figure");
  head.appendChild(t);head.appendChild(x);
  const body=div("ovlbody"),clone=svg.cloneNode(true);
  const vb=(svg.getAttribute("viewBox")||"").split(/\s+/).map(Number),nw=vb[2]||+svg.getAttribute("width")||600;
  const avail=Math.max(260,(window.innerWidth||1000)-72),target0=Math.max(Math.min(nw*2.4,avail,1500),Math.min(nw,avail));
  // a radar is roughly square, so it is also fitted to the pane HEIGHT (box padding 18+18, header ~52, body padding 24) instead of
  // being cropped and scrolled; the wide, scrollable effect matrix keeps its width-based size
  const nh=vb[3]||+svg.getAttribute("height")||nw,availH=Math.max(200,(window.innerHeight||800)-36-52-24-6);
  const target=svg.closest(".heatscroll")?target0:Math.max(240,Math.min(target0,availH*nw/nh));
  clone.setAttribute("width",Math.round(target));clone.removeAttribute("height");clone.style.height="auto";clone.style.maxWidth="none";clone.style.cursor="default";
  clone.removeAttribute("class");body.appendChild(clone);
  // cloneNode drops listeners, so the overlay forwards hover to the original figure's own handlers (same tooltip text)
  const srcEls=[svg,...svg.querySelectorAll("*")],cloneEls=[clone,...clone.querySelectorAll("*")];
  clone.addEventListener("mousemove",e=>{const i=cloneEls.indexOf(e.target);if(i<0)return;
    srcEls[i].dispatchEvent(new MouseEvent("mousemove",{clientX:e.clientX,clientY:e.clientY,bubbles:true,cancelable:true}));});
  clone.addEventListener("mouseout",()=>hideTT());box.appendChild(head);box.appendChild(body);root.appendChild(box);
  root.addEventListener("click",e=>{if(e.target===root)closeOverlay();});x.onclick=closeOverlay;
  const key=e=>{if(e.key==="Escape"){e.preventDefault();e.stopPropagation();closeOverlay();}};
  document.addEventListener("keydown",key,true);
  _ovl={root,key,prev:document.activeElement};document.body.appendChild(root);x.focus();
  body.style.maxHeight=Math.max(200,(window.innerHeight||800)-130)+"px";
}
document.getElementById("main").addEventListener("click",e=>{
  const svg=e.target.closest&&e.target.closest("svg");if(!svg||e.target.closest("[data-noenlarge]"))return;
  const sec=svg.closest("section[data-fig]"),fig=sec&&sec.dataset.fig;
  if(fig!=="radars"&&fig!=="heatmap")return;
  const host=svg.closest(".rpanel, .heatscroll");if(!host||svg.closest(".legend"))return;
  const rt=host.querySelector(".rt"),h2=sec.querySelector("h2");
  openOverlay(svg,((rt?rt.textContent.replace(/click to drill.*$/,"").replace(/\u2713 selected/,""):"")||"").trim()||(h2?h2.textContent:""));});
