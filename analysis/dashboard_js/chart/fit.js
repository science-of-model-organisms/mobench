// Figure 1's optional line of best fit, one per colour group, with Pearson r.
// Matches the paper's generality-vs-coherence figure (analysis/paper_figures.py generality_coherence_delta): an unweighted
// least-squares line of degree 1 (np.polyfit(xs, ys, 1)) over the implant methods' points, drawn from the smallest to the largest x, with
// r the Pearson correlation of the same points (a real minus sign). The paper fits one panel's methods (one point per implant method: the
// seven in analysis/paper_style.py METHODS; the negative-prompt control and the base model are not in it); here the group is the colour
// group of the figure (a behaviour, or a base model in "compare models" mode) over the implant points the filters leave. The control is
// never fitted, even with its chip on. Fits use the TRUE values, never the jittered positions. Custom: no Inspect equivalent (page layout).
// Kept in JS on purpose (not in analysis/dashboard_stats.py): the fit is over the points the filters and the two axis dropdowns leave, i.e. one
// line per (x property, y property, behaviour, base model, subset of the 7 implant-method chips): ~35,000 fits (~3.5 MB) to precompute, against
// a 1.8 MB blob without transcripts. The points' x and y are the exported axis_cells deltas.
function olsFit(xs,ys){const n=xs.length;if(n<2)return null;   // the paper fits any panel with 2 or more methods (`len(q) >= 2`)
  const mx=xs.reduce((s,v)=>s+v,0)/n,my=ys.reduce((s,v)=>s+v,0)/n;let sxx=0,sxy=0,syy=0;
  for(let i=0;i<n;i++){const dx=xs[i]-mx,dy=ys[i]-my;sxx+=dx*dx;sxy+=dx*dy;syy+=dy*dy;}
  if(!(sxx>0))return null;
  const slope=sxy/sxx,r=syy>0?sxy/Math.sqrt(sxx*syy):NaN;
  return {n,slope,icpt:my-slope*mx,r,x0:Math.min(...xs),x1:Math.max(...xs)};}
// pts: {x,y,...organism fields} (true values). Returns {fits:[{key,...olsFit}], skipped:[key of each group with fewer than 2 points]}.
function fitGroups(pts){const groups=new Map();
  pts.filter(p=>!isControl(p)).forEach(p=>{const k=spreadKey(p);if(!groups.has(k))groups.set(k,[]);groups.get(k).push(p);});
  const fits=[],skipped=[];
  spreadKeys().forEach(k=>{const g=groups.get(k);if(!g)return;
    const f=olsFit(g.map(p=>p.x),g.map(p=>p.y));if(f&&isFinite(f.r))fits.push({key:k,...f});else skipped.push(k);});
  return {fits,skipped};}
const hitBox=(a,b)=>Math.max(0,Math.min(a.x1,b.x1)-Math.max(a.x0,b.x0))*Math.max(0,Math.min(a.y1,b.y1)-Math.max(a.y0,b.y0));
const fitR=r=>"r = "+r.toFixed(2).replace("-","−");
