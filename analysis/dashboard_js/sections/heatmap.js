// Effect matrix.
// ---- heatmap (properties, with drill to evals) ----
function heatmap(container){
  const s=document.createElement("section");s.dataset.fig="heatmap";s.innerHTML=`<h2>Effect matrix <span class="pill">&Delta; vs base</span></h2>`;
  howto(s,`Every <b>organism</b> (rows) against every <b>property</b> (columns). Each cell is <b>&Delta; vs base</b>: the organism minus the same-size base, in the score's own 0&ndash;1 units (higher = better, except <b>misalignment</b>, where + = more misbehaviour). Colour: <span style="color:${diverge(1)}">blue = better</span>, <span style="color:${diverge(-1)}">red = worse</span>, pale = near zero, <b>symmetric about 0 with one scale per column</b> (&plusmn; that column's largest |&Delta;|, printed under it). <b>Robustness</b> is the share kept, not a &Delta; (its scale is 0&ndash;1). A small dot = the 95% interval of &Delta; <b>excludes 0</b>. <b>Click a column</b> to expand a property into its individual evals.`);
  const orgs=treated();const cols=DATA.axes;
  if(!orgs.length){s.innerHTML+='<div class="empty">Nothing to show for this selection.</div>';container.appendChild(s);return;}
  const cellOf=(o,k)=>axCell(o,k,k==="misalignment"?MIS_TIP+" Click the column for each evaluation's raw rate beside its base model's.":null);
  s.appendChild(gridHeat(orgs,cols.map(a=>({key:a.name,short:a.short})),cellOf,true,{values:true}));
  const note=document.createElement("div");note.className="selbar";
  note.innerHTML=`Cell number = &Delta; vs the same-size base (score units; misalignment: + = more misbehaviour). Colour scale: symmetric about 0, one scale per column (&plusmn; printed under each column; robustness 0&ndash;1 share kept); bar on the right: blue = better, red = worse. Dot = 95% CI of &Delta; excludes 0. Hollow cell = ran but unmeasurable; empty = not run. Corner triangle (eval drill-down) = differential attrition: the organism and its base left very different shares of rows unscored (gap above the bound in the tooltip), so that &Delta; may be biased (report only).`;
  s.appendChild(note);
  if(state.bucket){
    const evals=DATA.bucket_evals[state.bucket]||[];
    const sb=document.createElement("div");sb.className="selbar";sb.innerHTML=`Expanded property: <b>${escHtml(pretty(state.bucket))}</b> - its individual evals (click the column again to close)`;
    s.appendChild(sb);
    const mis=state.bucket==="misalignment";
    const evCell=(o,k)=>{const c=DATA.cells[o.id]?.[k];if(!c)return null;
      const rd=evalRead(o.id,k),b0=(DATA.cells[baseId(o.persona,_BM[o.id])]||{})[k],bm0=b0&&b0.mean!=null?b0.mean:null;
      const r={d:rd?rd.d:null,o:rd?rd.o:null,share:rd?rd.share:false,sig:rd?rd.sig:false,attr:!!(rd&&rd.attr),n:c.n,body:rd?readTip(rd,c.mean,bm0):null};
      if(mis){const b=b0,bm=bm0;
        // "organism / base": both raw rates. Misalignment: higher = more misaligned.
        r.txt=(c.mean!=null?c.mean.toFixed(2):"-")+" / "+(bm!=null?bm.toFixed(2):"-");
        r.tip=`Organism ${c.mean!=null?c.mean.toFixed(3):"n/a"} / same-size base model ${bm!=null?bm.toFixed(3):"n/a"}. `+MIS_TIP;}
      return r;};
    if(mis){const lg=document.createElement("div");lg.className="selbar";
      lg.innerHTML=`Cell text = <b>organism / same-size base model</b>, both raw rates (e.g. <b>0.43 / 0.11</b>). Hover a cell for Δ and its interval. Colour: blue = better (less misbehaviour), red = worse; the tooltip's Δ is organism − base, so + = more misbehaviour.`;s.appendChild(lg);}
    s.appendChild(gridHeat(orgs,evals.map(e=>({key:e,short:DATA.evals[e]?.display||e})),evCell,false,{values:true,cw:mis?78:0}));
  }
  container.appendChild(s);
}
