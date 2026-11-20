// The frontier scatter (figure 1) and its width.
// ---- scatter (frontier) ----
function scatter(container,plane,title,how,fig){
  const s=document.createElement("section");if(fig)s.dataset.fig=fig;s.innerHTML=`<h2>${cap(title)}</h2>`;howto(s,how);
  // Position by Δ vs base (oriented: up / right = better). Reading carries the CI for the tooltip.
  const pts=plane.points.map(p=>({...p,
    rx:axisRead(p.org,plane.behaviour),ry:axisRead(p.org,plane.cost)})).map(p=>({...p,x:p.rx?p.rx.o:null,y:p.ry?p.ry.o:null})).filter(p=>p.x!=null&&p.y!=null);
  if(!pts.length){s.innerHTML+='<div class="empty">No measured points for this selection (needs a base arm and both axes).</div>';container.appendChild(s);return;}
  // The viewBox is drawn at (about) the width it is shown at, and the svg is never scaled UP (max-width = W below), so one
  // viewBox unit is at most one CSS px and the sizes below are true on-screen px. Narrow panes: no right gutter, legend under.
  const XW=xyWidth(),narrow=XW<820,W=XW,H=narrow?Math.round(Math.max(430,Math.min(640,W*1.3))):Math.round(Math.max(560,Math.min(640,W*0.64)));
  const mL=narrow?68:112,mR=narrow?14:216,mT=34,mB=narrow?96:100;
  // data-driven ranges (0 always included); the green best-quadrant may be a sliver - that's fine
  const xs=[0,...pts.map(p=>p.x)],ys=[0,...pts.map(p=>p.y)];
  const base={x:nice(Math.min(...xs),Math.max(...xs)),y:nice(Math.min(...ys),Math.max(...ys))};
  const view={x:[...base.x],y:[...base.y]};
  // height:auto lets the viewBox set the height, so the plot fills its box (a fixed height attribute left a blank band)
  const svg=el("svg",{viewBox:`0 0 ${W} ${H}`,width:W,height:H});svg.style.width="100%";svg.style.height="auto";svg.style.maxWidth=W+"px";   // never upscaled beyond 1:1
  const cid="zc"+(++_zid);
  const tiny=W<420,R0=tiny?5.5:6.25,FS0=tiny?11.5:FS;   // marker radius / label font size at zoom 1, in CSS px (= viewBox units: the svg is shown at 1:1)
  const fitNote=document.createElement("div");fitNote.className="pkey fitnote";fitNote.style.display="none";
  const fitSet=fitGroups(pts);   // fitted once, on the true values: the figure only re-draws it
  const keyEl=document.createElement("div");keyEl.className="pkey";keyEl.style.display="none";   // numbered-callout key (see the label placer)
  const paint=()=>{
    hideTT();clearNode(svg);
    const xr=view.x,yr=view.y;
    // Zoom spreads points apart; it must not inflate them: markers grow at most 1.15x, labels at most 1.1x.
    const zl=Math.min((base.x[1]-base.x[0])/(xr[1]-xr[0]),(base.y[1]-base.y[0])/(yr[1]-yr[0])),sc=Math.min(1.15,Math.sqrt(Math.max(1,zl)));
    const r=R0*sc,fs=FS0*Math.min(1.1,1+0.5*(sc-1));
    const X=v=>mL+(v-xr[0])/(xr[1]-xr[0])*(W-mL-mR);
    const Y=v=>H-mB-(v-yr[0])/(yr[1]-yr[0])*(H-mT-mB);
    svg.appendChild(el("clipPath",{id:cid},[el("rect",{x:mL,y:mT,width:W-mL-mR,height:H-mT-mB})]));
    const plot=el("g",{"clip-path":`url(#${cid})`});   // marks that may leave the view are clipped to the plot frame
    // plot frame
    svg.appendChild(el("rect",{x:mL,y:mT,width:W-mL-mR,height:H-mT-mB,fill:"var(--plot)",stroke:"var(--line)"}));
    // faint quadrant tints: top-right = best (green), bottom-left = worst (red), others amber
    const xm=Math.max(mL,Math.min(W-mR,X(0))),ym=Math.max(mT,Math.min(H-mB,Y(0)));
    const quad=(x,y,w,h,c)=>{if(w>0&&h>0)svg.appendChild(el("rect",{x,y,width:w,height:h,fill:c,stroke:"none"}));};
    quad(xm,mT,W-mR-xm,ym-mT,"color-mix(in srgb,#3fa06a 22%,transparent)");            // top-right best (green)
    quad(mL,ym,xm-mL,H-mB-ym,"color-mix(in srgb,#c85b52 14%,transparent)");            // bottom-left worst (red)
    quad(mL,mT,xm-mL,ym-mT,"color-mix(in srgb,#d0912f 9%,transparent)");               // top-left (amber)
    quad(xm,ym,W-mR-xm,H-mB-ym,"color-mix(in srgb,#d0912f 9%,transparent)");           // bottom-right (amber)
    // gridlines (seaborn whitegrid)
    const tdig=r=>{const st=niceStep((r[1]-r[0])/5);return st>=0.02?2:st>=0.002?3:4;};   // enough decimals that zoomed ticks stay distinct
    for(const t of ticks(xr)){svg.appendChild(el("line",{x1:X(t),y1:mT,x2:X(t),y2:H-mB,stroke:"var(--grid)","stroke-width":1}));
      svg.appendChild(el("text",{x:X(t),y:H-mB+18,"text-anchor":"middle",fill:"var(--muted)","font-size":FS},[txt(fmt(t,tdig(xr)))]));}
    for(const t of ticks(yr)){svg.appendChild(el("line",{x1:mL,y1:Y(t),x2:W-mR,y2:Y(t),stroke:"var(--grid)","stroke-width":1}));
      svg.appendChild(el("text",{x:mL-8,y:Y(t)+4.5,"text-anchor":"end",fill:"var(--muted)","font-size":FS},[txt(fmt(t,tdig(yr)))]));}
    // zero crosshair (base)
    plot.appendChild(el("line",{x1:X(0),y1:mT,x2:X(0),y2:H-mB,stroke:"var(--axis)","stroke-dasharray":"4 3","stroke-width":1}));
    plot.appendChild(el("line",{x1:mL,y1:Y(0),x2:W-mR,y2:Y(0),stroke:"var(--axis)","stroke-dasharray":"4 3","stroke-width":1}));
    // axis titles with the evals they average over
    const bw=plane.behaviour_backwards;
    // Uppercase only the axis NAME (not the whole string). Both axes are oriented: + = better.
    const axSuffix=a=>axShare(a)?"share kept":axBack(a)?"Δ vs base, sign flipped (+ = less misbehaviour)":"Δ vs base";
    const behLabel=`${pretty(plane.behaviour).toUpperCase()} - ${axSuffix(plane.behaviour)}`;
    const xTfs=Math.min(FS,Math.max(11,(W-mL-mR)/(behLabel.length*0.62)));
    svg.appendChild(el("text",{x:(mL+W-mR)/2,y:H-42,"text-anchor":"middle",fill:"var(--fg)","font-size":xTfs.toFixed(1),"font-weight":600},
      [txt(behLabel)]));
    // direction words at the x-axis ends (kept, per request)
    svg.appendChild(el("text",{x:mL+4,y:H-26,"text-anchor":"start",fill:"var(--muted)","font-size":FS},[txt(bw?"← more "+plane.behaviour:"← less "+plane.behaviour)]));
    svg.appendChild(el("text",{x:W-mR-4,y:H-26,"text-anchor":"end",fill:"var(--muted)","font-size":FS},[txt(bw?"less "+plane.behaviour+" →":"more "+plane.behaviour+" →")]));
    const yc=(mT+H-mB)/2;
    const yTitle=`${pretty(plane.cost).toUpperCase()} - ${axSuffix(plane.cost)}`,yTfs=Math.min(FS,Math.max(11,(H-mT-mB)/(yTitle.length*0.6)));
    svg.appendChild(el("text",{x:14,y:yc,"text-anchor":"middle",fill:"var(--fg)","font-size":yTfs.toFixed(1),"font-weight":600,transform:`rotate(-90 14 ${yc})`},
      [txt(yTitle)]));
    // direction words at the y-axis ends: rotated words + separate UP/DOWN arrows (not rotated,
    // so the arrows actually point up and down).
    // Wide layout only: the end-words sit in their own column (x=36) left of the tick labels (which end at mL-8 and are at most
    // ~46px wide), and are anchored AWAY from their arrow (so a long word never runs into it). Narrow panes have no room for
    // the column beside the ticks, so the words are dropped there (the axis title already says which way is better).
    if(!narrow){
      svg.appendChild(el("text",{x:36,y:mT+30,"text-anchor":"end",fill:"var(--muted)","font-size":FS,transform:`rotate(-90 36 ${mT+30})`},[txt("better on "+pretty(plane.cost))]));
      svg.appendChild(el("text",{x:36,y:mT+16,"text-anchor":"middle",fill:"var(--muted)","font-size":15,"font-weight":700},[txt("↑")]));
      svg.appendChild(el("text",{x:36,y:H-mB-30,"text-anchor":"start",fill:"var(--muted)","font-size":FS,transform:`rotate(-90 36 ${H-mB-30})`},[txt("worse on "+pretty(plane.cost))]));
      svg.appendChild(el("text",{x:36,y:H-mB-12,"text-anchor":"middle",fill:"var(--muted)","font-size":15,"font-weight":700},[txt("↓")]));
    }
    // quadrant labels - serif italic + white halo so they read on the tints and look deliberate.
    const xN=pretty(plane.behaviour),yN=pretty(plane.cost);
    const AMBER="var(--amber)";
    const corner=(x,y,anch,col,t)=>svg.appendChild(el("text",{x,y,"text-anchor":anch,fill:col,
      "font-family":"var(--serif)","font-style":"italic","font-weight":600,"font-size":narrow?11:13.5,opacity:0.5,
      stroke:"var(--halo)","stroke-width":3.2,"paint-order":"stroke","stroke-linejoin":"round"},[txt(t)]));
    // drawn before (so behind) the points and labels, faint, and also treated as obstacles by the label placer
    const caps=[];const capBox=(x,y,anch,t)=>{const w=t.length*(narrow?11:13.5)*0.5;caps.push({x0:anch==="end"?x-w:x,x1:anch==="end"?x:x+w,y0:y-14,y1:y+4});};
    const cap=(x,y,anch,col,t)=>{if(t.length*(narrow?11:13.5)*0.5>(W-mL-mR)/2-12)return;   // a caption too wide for half the plot is dropped (it would collide with its neighbour)
      corner(x,y,anch,col,t);capBox(x,y,anch,t);};
    cap(W-mR-10,mT+20,"end","var(--good)",`better on both`);
    cap(mL+10,H-mB-12,"start","var(--bad)",`worse on both`);
    cap(mL+10,mT+20,"start",AMBER,`better ${yN}, worse ${xN}`);
    cap(W-mR-10,H-mB-12,"end",AMBER,`better ${xN}, worse ${yN}`);
    // base as grey diamond; label placed AWAY from the plot edge & haloed so it never conflicts.
    plot.appendChild(marker("diamond",X(0),Y(0),Math.min(r,R0*1.4)*0.75,"var(--basemark)"));
    // (its "untreated base" label is placed below, once the points are known, and painted on top of them)
    svg.appendChild(plot);
    // points + labels (grouped so hover fades the rest). EVERY point in view is labelled: a greedy placer tries 8 positions
    // around the marker (then wider rings, with a leader line), avoiding markers, placed labels, captions and the plot edge.
    // Custom (not Inspect): this is page layout, no upstream equivalent.
    const pgroups=[],inPlot=(x,y)=>x>=mL&&x<=W-mR&&y>=mT&&y<=H-mB;
    const vis=pts.filter(p=>inPlot(X(p.x),Y(p.y)));   // points panned out of view are not drawn at all
    const pos=vis.map(p=>({p,x:X(p.x),y:Y(p.y)}));
    pos.forEach(q=>{q.tx=q.x;q.ty=q.y;});   // the TRUE pixel position (what jitter is measured against)
    const ox=X(0),oy=Y(0);                   // the base crosshair, in pixels
    const F={mL,mR,mT,mB,W,H};
    jitterPoints(pos,r,F,ox,oy,XY.jitter!==false);
    // optional fit lines (one per colour group) on the true values: drawn here (behind the points); each r label is placed after the point labels
    // (below), at the free spot nearest the line's right end
    const fitOn=XY.fit===true,fitDraw=[];
    if(fitOn){
      fitSet.fits.forEach(f=>{
        const ends=[[f.x1,f.slope*f.x1+f.icpt,1],[f.x0,f.slope*f.x0+f.icpt,-1]].map(([vx,vy,side])=>({x:X(vx),y:Y(vy),side}));
        fitDraw.push({f,ends,pick:null,txtR:fitR(f.r)});
        const col=spreadColor(f.key);
        plot.appendChild(el("line",{class:"fitline",x1:ends[1].x,y1:ends[1].y,x2:ends[0].x,y2:ends[0].y,stroke:col,"stroke-width":2,"stroke-dasharray":"7 4",opacity:0.9,"pointer-events":"none"}));
      });
    }
    fitNote.style.display=fitOn?"":"none";
    if(fitOn)fitNote.textContent="Fit lines: least squares over each colour group's implant points, on the true values (the negative-prompt control is not in a fit; r = Pearson)"+(fitSet.skipped.length?"; no line for "+fitSet.skipped.map(k=>spreadLabel(k)).join(", ")+" (fewer than 2 points).":".");
    const baseIn=ox>=mL&&ox<=W-mR&&oy>=mT&&oy<=H-mB;   // zoomed/panned away from the crosshair: no base label (it would float)
    const {bbest,numbered,bfs}=placePointLabels(pos,F,{r,fs,ox,oy,baseIn,caps});
    // r labels: candidates at the line's right end first (beside it, above, below), then its left end, then along the line; the first one that
    // touches no marker, point label, caption, base label or earlier r label wins, else the one with the least overlap. Only inside the plot.
    {const used=[],lbl=pos.map(q=>q.lab&&q.lab.box).filter(Boolean),mkb=pos.map(q=>({x0:q.x-r-2,x1:q.x+r+2,y0:q.y-r-2,y1:q.y+r+2}));
      const others=[...caps,...lbl,...mkb,{x0:ox-8,x1:ox+8,y0:oy-8,y1:oy+8},...(baseIn&&bbest?[bbest.box]:[])];
      fitDraw.forEach(d=>{const lw=d.txtR.length*FS*0.6+4,lh=FS+3,[e0,e1]=d.ends;
        const mk=(x,y)=>({x0:x,x1:x+lw,y0:y,y1:y+lh});
        const cands=[];
        [e0,e1].forEach(e=>{const sx=e.side>0?e.x+6:e.x-6-lw;cands.push(mk(sx,e.y-lh/2),mk(sx,e.y-lh-4),mk(sx,e.y+4),mk(e.x-lw/2,e.y-lh-6),mk(e.x-lw/2,e.y+6));});
        [0.9,0.75,0.65,0.55,0.45,0.35].forEach(t=>{const x=e1.x+(e0.x-e1.x)*t,y=e1.y+(e0.y-e1.y)*t;cands.push(mk(x-lw/2,y-lh-5),mk(x-lw/2,y+5));});
        let best=null,bs=Infinity;
        for(const c of cands){if(!(c.x0>=mL+2&&c.x1<=W-mR-2&&c.y0>=mT+2&&c.y1<=H-mB-2))continue;
          let sc=0;[...others,...used].forEach(o=>{sc+=hitBox(c,o);});
          for(let k=0;k<=12;k++){const px=e1.x+(e0.x-e1.x)*k/12,py=e1.y+(e0.y-e1.y)*k/12;if(px>c.x0&&px<c.x1&&py>c.y0&&py<c.y1)sc+=6;}   // its own line counts too
          if(sc<bs){bs=sc;best=c;if(sc===0)break;}}
        d.pick=best;if(best)used.push(best);});}
    keyEl.textContent="";
    numbered.sort((u,v)=>+u.tag-+v.tag).forEach(n=>{const k=document.createElement("span");k.className="pk";
      k.appendChild(txt(n.tag+" = "+n.q.full));keyEl.appendChild(k);});
    keyEl.style.display=numbered.length?"":"none";
    for(const q of pos){
      const p=q.p,col=orgCol(p),ink=labelInk(col),g=el("g",{cursor:"pointer","data-org":p.id,"data-cx":q.x.toFixed(2),"data-cy":q.y.toFixed(2),"data-r":r.toFixed(2),
        "data-lbox":[q.lab.box.x0,q.lab.box.y0,q.lab.box.x1,q.lab.box.y1].map(v=>v.toFixed(2)).join(",")});
      const bx0=q.lab.box,tx=(bx0.x0+bx0.x1)/2,ty=(bx0.y0+bx0.y1)/2;
      const nx=Math.max(bx0.x0,Math.min(bx0.x1,q.x)),ny=Math.max(bx0.y0,Math.min(bx0.y1,q.y));
      if(Math.hypot(nx-q.x,ny-q.y)>r+4)   // label is not adjacent: a leader line ties it to its marker
        g.appendChild(el("line",{class:"plead",x1:q.x,y1:q.y,x2:nx,y2:ny,stroke:ink,"stroke-width":1,opacity:0.6}));
      g.appendChild(marker(orgShape(p),q.x,q.y,r,col));
      g.appendChild(el("text",{class:"plab",x:tx,y:ty+fs*0.34,"text-anchor":"middle",fill:ink,"font-size":fs.toFixed(2),
        stroke:"var(--halo)","stroke-width":(fs*0.24).toFixed(2),"paint-order":"stroke","stroke-linejoin":"round"},[txt(q.name)]));
      const one=(ax,r)=>`${ph(ax)} ${axShare(ax)?"share kept ":"Δ "}${fmt(r.o)}${r.olo==null?"":" · 95% CI ["+fmt(r.olo)+", "+fmt(r.ohi)+"]"}${r.sig?" · significant":""}`;
      g.addEventListener("mousemove",e=>showTT(e,`${orgHead(p)}<br>${one(plane.behaviour,p.rx)}<br>${one(plane.cost,p.ry)}<br><i>both oriented: + = better</i>`));
      g.addEventListener("mouseleave",hideTT);plot.appendChild(g);pgroups.push(g);
    }
    fadeGroup(pgroups);
    fitDraw.forEach(d=>{if(!d.pick)return;const col=spreadColor(d.f.key);
      svg.appendChild(el("text",{class:"fitr",x:d.pick.x0,y:d.pick.y0+FS,fill:labelInk(col),"font-size":FS,"font-weight":700,"data-key":d.f.key,
        stroke:"var(--halo)","stroke-width":3,"paint-order":"stroke","pointer-events":"none"},[txt(d.txtR)]));});
    const bb=bbest.box;plot.setAttribute("data-basebox",[bb.x0,bb.y0,bb.x1,bb.y1].map(v=>v.toFixed(2)).join(","));
    const bcx=(bb.x0+bb.x1)/2,bcy=(bb.y0+bb.y1)/2,bnx=Math.max(bb.x0,Math.min(bb.x1,ox)),bny=Math.max(bb.y0,Math.min(bb.y1,oy));
    const blay=el("g",{class:"baselab","pointer-events":"none"});
    if(baseIn&&Math.hypot(bnx-ox,bny-oy)>r+3)blay.appendChild(el("line",{x1:ox,y1:oy,x2:bnx,y2:bny,stroke:"var(--muted)","stroke-width":1,opacity:0.75}));
    if(baseIn)blay.appendChild(el("text",{x:bcx,y:bcy+bfs*0.34,"text-anchor":"middle",fill:"var(--muted)","font-size":bfs.toFixed(2),"font-style":"italic",
      stroke:"var(--halo)","stroke-width":(bfs*0.28).toFixed(2),"paint-order":"stroke","stroke-linejoin":"round"},[txt("untreated base")]));
    plot.appendChild(blay);
    // in-plot legend (right gutter): behaviour / base-model colours + method shapes
    if(!narrow)legendInPlot(svg,W-mR+28,mT+8);
  };
  paint();
  s._repaint=paint;
  const w=document.createElement("div");w.className="figwrap center";w.appendChild(svg);
  const z=makeZoom(svg,W,{x:mL,y:mT,w:W-mL-mR,h:H-mT-mB},view,base,paint);
  const zr=div("zrow");zoomBar(zr,()=>[z]);s.appendChild(zr);s.appendChild(w);s.appendChild(fitNote);s.appendChild(keyEl);if(narrow)s.appendChild(dualLegend());
  container.appendChild(s);
}

// Width (CSS px) the scatter is drawn at: the viewport minus the page gutters, the ToC rail (wide screens) and the card padding,
// capped at 1000 so a big monitor never gets a bigger chart (the svg is never upscaled).
function xyWidth(){const vw=typeof window!=="undefined"&&window.innerWidth>0?window.innerWidth:1000;
  return Math.round(Math.max(300,Math.min(1000,vw-(vw<=600?58:114)-(vw>1000?244:0))));}
