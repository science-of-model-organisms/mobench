// The page top: key and Details.
// The top of the page: a three-item key and one collapsed Details note (the per-figure "i" help holds the method).
function renderTop(){
  const M=DATA.meta,nfmt=n=>n==null?"-":n.toLocaleString();
  // Share of unscored samples, and how much of it is structural (mu_decisiveness ties / fermi non-numeric,
  // unscorable by design) rather than discards on judged evals (refusals + judge-couldn't-parse). Computed in Python (derived.unscored).
  const pct=M.n_samples?DATA.derived.unscored.pct.toFixed(0):0;
  const pctTxt=pct+"%";
  const structPct=Math.round(DATA.derived.unscored.struct_pct);
  // One line per results dataset, labelled in words and linked to its Hugging Face page. The list comes from the build
  // (meta.datasets, analysis/dashboard.py dataset_links); an older blob without it falls back to the plain provenance text.
  const dsets=(M.datasets||[]).filter(d=>/^https:\/\/huggingface\.co\/datasets\//.test(d.url||""));
  const dsLines=dsets.map(d=>`<li class="dataset">${escHtml(d.label)} &mdash; <a href="${escHtml(d.url)}" target="_blank" rel="noopener noreferrer">${escHtml(d.name)}</a></li>`).join("");
  const src=dsets.length?`Data (Hugging Face datasets):<ul>${dsLines}</ul>${M.run_date?`Eval run: <b>${escHtml(M.run_date)}</b>.`:""}`
    :[M.hf_dataset?`Data: HF dataset <b>${escHtml(M.hf_dataset)}</b>.`:"",M.run_date?`Eval run: <b>${escHtml(M.run_date)}</b>.`:""].filter(Boolean).join(" ");
  document.getElementById("intro").innerHTML=`
  <div class="key">
    <span class="k"><span class="d">&Delta;</span> = organism &minus; the same-size base model</span>
    <span class="k"><span class="swatch" style="background:${diverge(1)}"></span><b>blue</b> = better<span class="swatch" style="background:${diverge(-1)};margin-left:8px"></span><b>red</b> = worse</span>
    <span class="k"><span class="dot" style="background:var(--fg);width:7px;height:7px"></span>dot = significant (the 95% CI of &Delta; excludes 0, |&Delta;| &gt; 1.96 SE)</span>
  </div>
  <details class="introdetails" open><summary>Details</summary><ul>
    ${M.n_cells!=null&&M.n_cells_possible!=null?`<li class="coverage">Coverage: <b>${nfmt(M.n_cells)}</b> of ${nfmt(M.n_cells_possible)} organism &times; eval cells filled${M.n_cells<M.n_cells_possible?` (${nfmt(M.n_cells_possible-M.n_cells)} missing)`:""} &middot; ${nfmt(M.n_samples)} samples.</li>`:""}
    <li class="caveats">How to read a &Delta;: on this page it is the difference of the two arms' cell means, whereas the radar values use only items scorable in both arms, so the two can differ slightly. With fewer than two items in either arm there is no interval and no dot, so a missing dot can mean &ldquo;not computable&rdquo; as well as &ldquo;not significant&rdquo;.</li>
    <li>Unscored samples: <b>${pctTxt}</b>${structPct>0?` (~${structPct}% of them are mu-decisiveness ties and fermi non-numeric answers, unscorable by design, not parse errors)`:""}.</li>
    <li>Drill down: in Profiles, <b>Per property</b> (the default) opens a property's individual evals when you click its title, and so does a column of the effect matrix; <b>Per organism</b> and <b>Per model</b>, a scatter point, a row label, a base-model <b>i</b> chip or a model name in a legend open a side drawer for that organism or base model. Hover the <b>i</b> on a figure for its method.</li>
    ${src?`<li>${src}</li>`:""}
    <li>${paperMapHtml()}</li>
  </ul></details>`;
}
// "How this maps to the paper": the radar rules, the expression threshold and the evaluation count (inside Details).
function paperMapHtml(){
  const M=DATA.meta,n=M.n_evaluations,R=RADAR,delta=((DATA.expression||{}).delta??0.10).toFixed(2);
  const radars=R?`<li><b>Profile radars show a score per property</b> (Profiles: all three views, and the drill-down radars). ${DELTA_DEF}. <b>Behaviour spokes</b> (generality, introspection, activations): ${FORMULA.behaviour}, capped at 10; the base model is a dashed ring at 0 and a value below it is drawn inside the ring (the centre is the lowest value needed). <b>Health spokes</b> (coherence, capability): ${FORMULA.health}, clipped to [0, 10], so the <b>outer ring</b> is the base. <b>Robustness</b>: ${FORMULA.share}; the base model has no robustness reading (no behaviour to keep). ${MATCHED_NOTE} ${RADAR_NOTE} The heatmap, bars and forest show &Delta; vs base in score units, with 95% intervals.</li>`
    :`<li><b>This build has no paper-scale radar values</b>, so the radars are drawn as &Delta; vs base in score units.</li>`;
  return `<span class="papermap"><b>How this maps to the paper.</b><ul>${radars}
    <li><b>Expression threshold.</b> An organism clears it (<code>PASS</code>) when the lower bound of its gain over base exceeds ${delta}, on questions it did not see or on the ones it did; otherwise it is a <code>FAIL</code>, the only verdict below the threshold. An organism below it is drawn with a <b>dashed outline</b> on the radars and is not eligible for the Highest generality and Most robust tiles.</li>
    <li><b>${n!=null?n+" evaluations":"Evaluations"}</b>, the paper's set${n!=null?`; one evaluation can report several score cells (${M.n_evals} here)`:""}. Misalignment is a property like the others but is not a radar spoke.</li></ul></span>`;
}
