// Transcripts, decompressed on first view.
// ---- transcripts: whole answers, organism beside base, decompressed on first view ----
// The payload is base64(gzip(JSON)) with one shared string table (analysis/dashboard.py
// pack_transcripts). It is decoded only when the Transcripts tab is first opened (the section is hidden until then, so the observer waits), and only
// the chosen organism x property is drawn, so the page opens as fast as before.
let TX;  // undefined = not loaded yet, null = none in this build
async function loadTx(){
  if(TX!==undefined)return TX;
  const b64=(document.getElementById("transcripts").textContent||"").trim();
  if(!b64||typeof DecompressionStream==="undefined"){TX=null;return TX;}
  const bin=atob(b64),bytes=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)bytes[i]=bin.charCodeAt(i);
  const txt=await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"))).text();
  const d=JSON.parse(txt),T=d.texts,un=x=>x==null?null:T[x];
  const side=s=>s?{...s,p:un(s.p),sys:un(s.sys),a:un(s.a),r:un(s.r)}:null;
  for(const o in d.orgs)for(const b in d.orgs[o])d.orgs[o][b]=d.orgs[o][b].map(e=>({...side(e),b:side(e.b)}));
  TX=d;return TX;
}
// organism pill: method · behaviour (the base model is the picker above it)
function txOrgLabel(o){return o.is_base?"base · "+o.persona:niceOrg(o)+" · "+o.persona;}
function txPre(t){const p=document.createElement("pre");p.className="txtext";p.textContent=t;return p;}
function txScore(sd){const bits=[];if(sd.s!=null)bits.push("score "+(+sd.s).toFixed(2));else bits.push("unscored");
  Object.entries(sd.lab||{}).forEach(([k,v])=>bits.push(pretty(k)+": "+v));return bits.join(" · ");}
function txSide(sd,who,cls){const d=div("txside"+(cls?" "+cls:""));const h=div("who");
  h.innerHTML=`<b></b><span></span>`;h.querySelector("b").textContent=who;h.querySelector("span").textContent=txScore(sd);
  d.appendChild(h);d.appendChild(txPre(sd.a));
  if(sd.r){const r=document.createElement("details");r.innerHTML="<summary>Judge rationale</summary>";r.appendChild(txPre(sd.r));d.appendChild(r);}
  return d;}
function transcripts(container){
  const s=sec("Transcripts","transcripts");
  howto(s,`The answers behind the numbers. For each organism and property, up to <b class="txcap">a capped number of</b> whole answers (never cut), each beside the <b>base model's answer to the same item</b>. Picks are seeded and alternate a high-score row (score ≥ 0.5; on a judged behaviour eval, the judge found it present) and a low-score one. WMDP, GPQA and capability-format items are never shown. Contact details in transcripts (emails, phone and account numbers) are masked; the people and organisations in eval scenarios are fictional.`);
  const body=div("");s.appendChild(body);container.appendChild(s);
  const draw=window._txDraw=()=>{body.innerHTML="";
    if(TX===null){body.appendChild(empty("No transcripts in this build (rendered with --transcripts 0, or the browser cannot decompress)."));return;}
    s.querySelector(".txcap").textContent=TX.cap;
    // The Transcripts tab has its own base-model picker (the global filter bar is hidden here). It starts from the base model picked in
    // the Results bar and follows it until the reader picks one here; after that this choice is the tab's own and never writes to the
    // Results filters. The tab lists every organism of that base model (the Results method / behaviour filters do not apply here), the
    // negative-prompt control only while the Results control chip is on (as before).
    const base=txBaseModel();
    // the drawer link's one-visit listing of a control applies to the base model it asked for only
    const want=state.txWant&&ORG_BY_ID[state.txWant];if(want&&bmOf(want)!==base)state.txWant=null;
    const mine=txPool(base,o=>!!TX.orgs[o.id]);
    const orgs=[...orderOrgs(mine.filter(o=>!o.is_base)),...mine.filter(o=>o.is_base)].sort(ctrlLast);   // stable: the control ends the picker
    const bar0=div("txbar");
    if(BASE_MODELS.length>1)bar0.appendChild(radioGroup("Base model",BASE_MODELS,()=>base,v=>{state.txBase=v;state.txWant=null;draw();},bmName));
    if(BASE_MODELS.length>1)body.appendChild(bar0);
    if(!orgs.length){body.appendChild(empty("No transcripts for "+(base?bmName(base):"this base model")+"."));return;}
    if(!orgs.some(o=>o.id===state.txOrg))state.txOrg=txPickOrg(orgs,base).id;
    (state.txOrgByBase=state.txOrgByBase||{})[base]=state.txOrg;
    const have=TX.orgs[state.txOrg]||{};
    const order=DATA.axes.map(a=>a.name).filter(b=>have[b]).concat(Object.keys(have).filter(b=>!DATA.axes.some(a=>a.name===b)).sort());
    if(!order.includes(state.txBucket))state.txBucket=order[0];
    const bar=div("txbar");
    const lab=txOrgLabel;
    bar.appendChild(radioGroup("Organism",orgs.map(lab),
      ()=>{const o=orgs.find(x=>x.id===state.txOrg);return o?lab(o):null;},
      v=>{const o=orgs.find(x=>lab(x)===v);if(o){state.txOrg=o.id;if(state.txWant&&o.id!==state.txWant)state.txWant=null;draw();}}));
    bar.appendChild(radioGroup("Property",order.map(pretty),()=>pretty(state.txBucket),
      v=>{state.txBucket=order.find(b=>pretty(b)===v);draw();}));
    body.appendChild(bar);
    // EVAL row: the property's exact evals (those the picked organism has transcripts for), multi-select with All / None like the Results
    // filters. It starts at All and goes back to All whenever the organism or the property changes (the set of evals is a different one).
    const entries=have[state.txBucket]||[];
    const evs=[...new Set(entries.map(e=>e.e))];
    const evKey=state.txOrg+"|"+state.txBucket;
    if(state.txEvalKey!==evKey||!(state.txEvals instanceof Set)){state.txEvalKey=evKey;state.txEvals=new Set(evs);}
    const evName=e=>pretty((DATA.evals[e]||{}).display||e);
    const evBar=div("txbar");evBar.appendChild(chipGroup("Eval",evs,state.txEvals,{label:evName,onchange:draw,cls:"txevals"}));body.appendChild(evBar);
    const org=ORG_BY_ID[state.txOrg];
    const shown=entries.filter(e=>state.txEvals.has(e.e));
    if(!shown.length)body.appendChild(empty("No evals selected: pick at least one above."));
    shown.forEach((e,i)=>{const c=div("txcard");const h=div("txhead");
      const ev=(DATA.evals[e.e]||{}).display||e.e;
      h.innerHTML=`<b></b><span></span>`;h.querySelector("b").textContent=`${i+1}. ${pretty(ev)}`;
      h.querySelector("span").textContent=[e.cond?"condition "+e.cond:"",e.sid].filter(Boolean).join(" · ");c.appendChild(h);
      if(e.sys){const d=document.createElement("details");d.className="txprompt";d.innerHTML="<summary>System prompt</summary>";d.appendChild(txPre(e.sys));c.appendChild(d);}
      const pd=document.createElement("details");pd.className="txprompt";pd.open=true;
      pd.innerHTML=`<summary>Prompt${e.psrc==="bank"?" (from the item bank, by item id)":""}</summary>`;
      pd.appendChild(e.p!=null?txPre(e.p):empty("Prompt text not stored with this result."));c.appendChild(pd);
      const cols=div("txcols");
      cols.appendChild(txSide(e,org&&org.is_base?"base":(org?niceOrg(org):e.sid),""));
      if(e.b)cols.appendChild(txSide(e.b,"base",'base'));
      else if(!(org&&org.is_base))cols.appendChild(empty("Base has no answer to this item."));
      c.appendChild(cols);body.appendChild(c);});
    if(window._sizeToc)window._sizeToc();
  };
  if(TX!==undefined){draw();return;}
  body.appendChild(empty("Loading transcripts…"));
  if(window._txIo)window._txIo.disconnect();   // a rebuild replaces this section: do not keep observing (and retaining) the detached one
  const io=window._txIo=new IntersectionObserver(ents=>{if(ents.some(en=>en.isIntersecting)){io.disconnect();
    loadTx().then(draw,err=>{TX=null;draw();console.error(err);});}},{rootMargin:"600px 0px"});
  io.observe(s);
}
