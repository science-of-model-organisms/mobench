// Diagnostics tables.
// ---- Diagnostics: the tables only the retired static scorecard showed ----
// DATA.diagnostics is built by analysis/diagnostics.py (same helpers, same strings as the
// scorecard's tables). Collapsed by default; each row names the organism it is about
// (row_orgs) so the Base model / behaviour filters apply. Absent in an older cached blob.
function diagEsc(t){return String(t).replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));}
function diagRich(t){return diagEsc(t).replace(/\*\*(.+?)\*\*/g,"<b>$1</b>").replace(/`([^`]+)`/g,"<code>$1</code>");}
function diagnosticsSection(container){
  const tabs=DATA.diagnostics||[],miss=missingCells();if(!tabs.length&&!miss.length)return;
  const s=sec("Diagnostics","diagnostics");
  const det=document.createElement("details");det.className="diag";
  det.innerHTML=`<summary>${tabs.length} diagnostic tables: persistence, share kept, refusal agreement, two-judge panel, response coherence, axis voters, seeds, coverage, cost${miss.length?"; the missing cells":""}</summary>`;
  const byId=ORG_BY_ID,hiding=hidingAny();
  tabs.forEach(t=>{
    const d=div("dt");d.dataset.key=t.key;
    d.innerHTML=`<h3>${diagEsc(t.caption)}</h3><p class="dnote">${diagRich(t.note)}</p>`;
    if(t.run_wide&&hiding){
      // A run-wide table sums over EVERY organism and cannot be filtered per row, so while a
      // base-model or behaviour filter hides organisms it is withheld rather than shown unlabelled.
      d.appendChild(empty("Totals over all organisms in the run, so not shown while a base model or behaviour filter is hiding some. Select every behaviour (and a run with a single base model) to see it."));
    } else if(t.header.length){
      const keep=t.rows.map((r,i)=>{const o=t.row_orgs[i];
        if(o==="*")return !hiding;               // an ALL total is wrong once the view is filtered
        return !o||!byId[o]||orgVisible(byId[o]);});
      // a row about the negative-prompt control arm goes last (stable: every other row keeps the table's own order)
      const isCtl=i=>{const o=t.row_orgs[i];return o&&o!=="*"&&byId[o]&&isControl(byId[o]);};
      const rows=t.rows.map((r,i)=>({r,i})).filter(x=>keep[x.i]).sort((x,y)=>Number(isCtl(x.i))-Number(isCtl(y.i))).map(x=>x.r);
      const w=div("dscroll");const tb=document.createElement("table");
      const th=document.createElement("thead"),tr=document.createElement("tr");
      t.header.forEach(h=>{const c=document.createElement("th");c.textContent=h;tr.appendChild(c);});
      th.appendChild(tr);tb.appendChild(th);
      const bd=document.createElement("tbody");
      rows.forEach(r=>{const x=document.createElement("tr");
        r.forEach(v=>{const c=document.createElement("td");c.textContent=v;x.appendChild(c);});bd.appendChild(x);});
      tb.appendChild(bd);w.appendChild(tb);
      if(!rows.length)d.appendChild(empty("No rows (none in this tree, or none for the selected base model and behaviours)."));
      else d.appendChild(w);
    }
    det.appendChild(d);});
  if(miss.length){   // the cells behind the Coverage line in Details: every organism x eval pair with no score
    const n=miss.reduce((t,m)=>t+m.orgs.length,0);
    const d=div("dt");d.dataset.key="missing_cells";
    d.innerHTML=`<h3>Missing cells (${n})</h3><p class="dnote">Organism &times; eval pairs with no score cell: the ${n} behind the Coverage line in Details, listed by eval. Run-wide: the filters above do not apply.</p>`;
    const w=div("dscroll"),tb=document.createElement("table");
    tb.innerHTML="<thead><tr><th>Eval</th><th>Missing</th><th>Organisms (method · behaviour · base model)</th></tr></thead>";
    const bd=document.createElement("tbody");
    miss.forEach(({e,orgs})=>{const x=document.createElement("tr");
      [pretty(DATA.evals[e]?.display||e),String(orgs.length),orderOrgs(orgs).map(o=>(o.is_base?"base":niceOrg(o))+" · "+pretty(o.persona)+" · "+bmName(bmOf(o))).join("; ")]
        .forEach(v=>{const c=document.createElement("td");c.textContent=v;x.appendChild(c);});bd.appendChild(x);});
    tb.appendChild(bd);w.appendChild(tb);d.appendChild(w);det.appendChild(d);
  }
  s.appendChild(det);container.appendChild(s);
}
