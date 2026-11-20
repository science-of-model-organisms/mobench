// The paper highlight tiles.
// ---- paper tiles: the highlights, recomputed from whatever is selected ----
// Axis names are the paper's (core/registry.py VALID_AXES). Organisms below the expression threshold are not eligible for
// "highest generality" or "most robust" (the paper's convention); the negative-prompt control is not an implant, so it is
// never a tile.
const TILE_AXES={generality:"generality",robustness:"robustness",coherence:"coherence"};
function paperTilesData(){
  const pool=implants();
  const E=(DATA.expression||{}).orgs||{},hasThr=Object.values(E).some(e=>e&&e.verdict);
  const failN=hasThr?pool.filter(o=>exprVerdict(o.id)==="FAIL").length:0;
  const eligible=pool.filter(o=>!hasThr||state.inclFail||exprVerdict(o.id)!=="FAIL");  // only FAIL is below the threshold; the in-distribution-only pass and a missing verdict stay eligible
  const pick=(axis,orgs,sign)=>{let b=null;orgs.forEach(o=>{const r=axisRead(o.id,axis);if(!r)return;
    if(b===null||(sign>0?r.o>b.r.o:r.o<b.r.o))b={o,r};});return b;};
  return {hasThr,pool,failN,excluded:pool.length-eligible.length,
    gen:pick(TILE_AXES.generality,eligible,1),rob:pick(TILE_AXES.robustness,eligible,1),coh:pick(TILE_AXES.coherence,pool,-1)};
}
function paperTiles(){
  const d=paperTilesData(),wrap=div("");
  const grid=div("tiles");grid.id="tiles";
  const tile=(key,label,hit,valueOf,unit,none)=>{
    const t=div("tile");t.dataset.tile=key;
    const lab=div("tl");lab.textContent=label;t.appendChild(lab);
    const v=div("tv"),n=div("tn"),dt=div("td");v.textContent=hit?valueOf(hit.r):"-";t.appendChild(v);
    if(hit){n.textContent=niceOrg(hit.o);
      dt.textContent=bmName(bmOf(hit.o))+" \u00b7 "+pretty(hit.o.persona)+" \u00b7 "+unit(hit.r);
      t.title=(hit.o.id)+(hit.r.olo==null?"":" \u00b7 95% CI ["+fmt(hit.r.olo)+", "+fmt(hit.r.ohi)+"]");}
    else{n.textContent=none||"no organism in this selection";}
    t.appendChild(n);t.appendChild(dt);
    // the two tiles that skip organisms below the expression threshold say so, in one muted line (the coherence tile is unaffected)
    if(key!=="coherence"&&d.hasThr&&d.failN>0){const ex=div("tx");ex.textContent=state.inclFail?"incl. organisms that failed the expression test":"excl. "+d.failN+" that failed the expression test";t.appendChild(ex);}
    grid.appendChild(t);};
  const delta=r=>"\u0394 vs the same-size base"+(r.lo==null?"":", 95% CI ["+fmt(r.lo,2)+", "+fmt(r.hi,2)+"]");
  tile("generality","Highest generality",d.gen,r=>fmt(r.d,2),delta);
  tile("robustness","Most robust",d.rob,r=>r.o.toFixed(2),()=>"share kept");
  const worst=d.coh&&d.coh.r.o<0?d.coh:null;
  tile("coherence","Largest coherence degradation",worst,r=>fmt(r.d,2),delta,d.coh?"no organism is below its base":null);
  wrap.appendChild(grid);
  const note=div("tilenote");
  note.textContent=!d.pool.length?"":d.hasThr
    ?(state.inclFail&&d.failN?d.failN+" of "+d.pool.length+" organisms shown fall below the expression threshold; they are included in Highest generality and Most robust.":d.excluded?d.excluded+" of "+d.pool.length+" organisms shown fall below the expression threshold and are not eligible for Highest generality or Most robust.":"Every organism shown clears the expression threshold.")
    :"This data carries no expression-threshold verdicts, so no organism is excluded from Highest generality or Most robust.";
  if(d.hasThr&&d.failN>0){const b=document.createElement("button");b.type="button";b.className="linkbtn";b.id="tile-incl";
    b.textContent=state.inclFail?"exclude them again":"include them";
    b.onclick=()=>{state.inclFail=!state.inclFail;render();};
    note.appendChild(txt(" "));note.appendChild(b);}
  wrap.appendChild(note);return wrap;
}
