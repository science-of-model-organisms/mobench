// Eval correlation matrix.
function corrMatrix(container){
  const s=sec("Are we measuring many things or few? - eval correlation","corr");
  howto(s,`Correlation of <b>Δ vs base</b> across organisms, for every pair of composite evals. <span style="color:${diverge(1)}">blue</span> = move together, <span style="color:${diverge(-1)}">red</span> = move oppositely. Blocks of blue mean several evals are really one measurement.`);
  const evals=Object.keys(DATA.evals).filter(e=>DATA.evals[e].in_composite&&DATA.evals[e].axis!=="unknown");
  // Kept in JS on purpose (not in analysis/dashboard_stats.py): r depends on WHICH organisms the filters leave, and the chips (methods x behaviours x
  // base models, both views) reach 6,376 distinct visible sets in the real data, i.e. 3,576,936 r values (63 MB) to precompute, against a 1.8 MB blob
  // without transcripts. Only the per-cell inputs (cell.delta) are exported; the pairwise r is formed here over treated() (select.js).
  const orgs=treated();const vec={};evals.forEach(e=>vec[e]=orgs.map(o=>{const c=(DATA.cells[o.id]||{})[e];return votes(c)&&c.delta!=null?c.delta:null;}));
  function corr(a,b){const xs=[],ys=[];for(let i=0;i<a.length;i++)if(a[i]!=null&&b[i]!=null){xs.push(a[i]);ys.push(b[i]);}
    if(xs.length<3)return null;const mx=xs.reduce((s,v)=>s+v,0)/xs.length,my=ys.reduce((s,v)=>s+v,0)/ys.length;
    let sxy=0,sx=0,sy=0;for(let i=0;i<xs.length;i++){const dx=xs[i]-mx,dy=ys[i]-my;sxy+=dx*dy;sx+=dx*dx;sy+=dy*dy;}
    return sx>0&&sy>0?sxy/Math.sqrt(sx*sy):null;}
  const n=evals.length;if(n<2){s.appendChild(empty("Not enough composite evals."));container.appendChild(s);return;}
  // Drawn at the width it is shown at (rebuilt on resize): the cell size follows the pane, so the matrix and its colour scale always fit.
  // Labels stay 13px; where the cells are too small for every label (a row needs 14px of pitch, a 65-degree column label 14 / sin 65), only
  // every k-th is printed (hover any cell for the names), and r is printed only in a cell of 24px or more.
  const XW=xyWidth(),wide=XW>=600,mL=wide?216:124,cbGut=wide?130:0,ANG=65*Math.PI/180,rPad=wide?0:Math.ceil(20*7.4*Math.cos(ANG));   // narrow: no colour-bar gutter, but the last rotated label needs room
  const cell=Math.max(4,Math.min(27,Math.floor((XW-mL-cbGut-rPad-6)/n)));
  const rowK=Math.max(1,Math.ceil(14/cell)),colK=Math.max(1,Math.ceil(14/(cell*Math.sin(ANG))));
  const rowCh=Math.floor((mL-14)/7.4),colCh=20,mT=Math.ceil(Math.min(colCh,22)*7.4*Math.sin(ANG))+18,W=mL+n*cell+cbGut+rPad+6,H=mT+n*cell+8+(wide?0:0);
  const svg=natural(el("svg",{viewBox:`0 0 ${W} ${H}`,width:W,height:H}),W);
  evals.forEach((e,i)=>{const c=mL+i*cell+cell/2,full=pretty(DATA.evals[e].display);
    if(i%colK===0){const t=el("text",{x:c,y:mT-7,transform:`rotate(-65 ${c} ${mT-7})`,"text-anchor":"start",fill:"var(--fg)","font-size":FS},[txt(clip(DATA.evals[e].display,colCh))]);
      t.appendChild(el("title",{},[txt(full)]));svg.appendChild(t);}
    if(i%rowK===0){const t=el("text",{x:mL-8,y:mT+i*cell+cell/2+4,"text-anchor":"end",fill:"var(--fg)","font-size":FS},[txt(clip(DATA.evals[e].display,rowCh))]);
      t.appendChild(el("title",{},[txt(full)]));svg.appendChild(t);}});
  const CR=[],vals=[],rTxt=r=>{const v=r.toFixed(1);return v==='1.0'?'1':v==='-1.0'?'−1':v.replace('-0.','−.').replace('0.','.');};
  evals.forEach((e,i)=>evals.forEach((f,j)=>{const r=i===j?(corr(vec[e],vec[e])==null?null:1):corr(vec[e],vec[f]);   // the diagonal is r=1 only when r is defined at all (>=3 organisms, nonzero variance)
    const x=mL+j*cell,y=mT+i*cell,rect=el("rect",{x:x+1,y:y+1,width:cell-1,height:cell-1,fill:r==null?"var(--grid3)":diverge(r),stroke:"var(--halo)","stroke-width":.8});
    if(r!=null){rect.addEventListener("mousemove",ev=>showTT(ev,`${ph(DATA.evals[e].display)} × ${ph(DATA.evals[f].display)}<br>r = ${r.toFixed(2)}`));rect.addEventListener("mouseleave",hideTT);
      // r on top of its cell (drawn after every rect, in the next layer) in the ink that reads on that fill; a cell too small for 13px text keeps colour + hover only
      if(cell>=24)vals.push(el("text",{x:x+cell/2,y:y+cell/2+4.5,"text-anchor":"middle",fill:inkOn(diverge(r)),"font-size":FS,"pointer-events":"none"},[txt(rTxt(r))]));}
    svg.appendChild(rect);CR.push({r:rect,i,j});}));
  vals.forEach(t=>svg.appendChild(t));
  CR.forEach(o=>o.r.addEventListener("mouseenter",()=>CR.forEach(m=>m.r.classList.toggle("dim",m.i!==o.i&&m.j!==o.j))));
  svg.addEventListener("mouseleave",()=>CR.forEach(m=>m.r.classList.remove("dim")));
  // vertical colour bar on the RIGHT: +1 (together) top → −1 (opposite) bottom
  if(wide)vRamp(svg,mL+n*cell+22,mT,mT+n*cell,16,-1,1,[["+1 together",1],["0",0],["−1 opposite",-1]]);
  const w=div("figwrap center");w.appendChild(svg);s.appendChild(w);
  if(!wide)s.appendChild(rampLegend(-1,1,[["−1 opposite",0],["0",0.5],["+1 together",1]]));   // a narrow pane puts the colour scale under the matrix
  container.appendChild(s);
}
