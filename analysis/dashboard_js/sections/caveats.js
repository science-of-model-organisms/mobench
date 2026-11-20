// Run caveats.
// Run caveats supplied by the page builder (DATA.caveats); absent for a plain render.
function caveatsSection(container){
  const cv=DATA.caveats||[];if(!cv.length)return;
  const s=sec("Before quoting a number","caveats");const ul=document.createElement("ul");ul.className="insights";
  cv.forEach(t=>{const li=document.createElement("li");li.textContent=t;ul.appendChild(li);});
  s.appendChild(ul);container.appendChild(s);
}
