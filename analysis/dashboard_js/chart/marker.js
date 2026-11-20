// Marker shapes and the organism row label.
function marker(shape,cx,cy,r,fill){
  const a={fill:fill,stroke:"var(--bg)","stroke-width":1.3};
  if(shape==="circle")return el("circle",{cx,cy,r,...a});
  if(shape==="square")return el("rect",{x:cx-r,y:cy-r,width:2*r,height:2*r,...a});
  if(shape==="triangle")return el("path",{d:`M${cx} ${cy-r} L${cx+r} ${cy+r} L${cx-r} ${cy+r} Z`,...a});
  if(shape==="diamond")return el("path",{d:`M${cx} ${cy-r} L${cx+r} ${cy} L${cx} ${cy+r} L${cx-r} ${cy} Z`,...a});
  if(shape==="minus"){const w=r*0.5;return el("rect",{x:cx-r,y:cy-w,width:2*r,height:2*w,...a});}
  if(shape==="plus"){const w=r*0.5;return el("path",{d:`M${cx-w} ${cy-r} H${cx+w} V${cy-w} H${cx+r} V${cy+w} H${cx+w} V${cy+r} H${cx-w} V${cy+w} H${cx-r} V${cy-w} H${cx-w} Z`,...a});}
  if(shape==="cross"){const w=r*0.42;const s=r*0.72;return el("path",{d:`M${cx-s} ${cy-s+w} L${cx-s+w} ${cy-s} L${cx} ${cy-w} L${cx+s-w} ${cy-s} L${cx+s} ${cy-s+w} L${cx+w} ${cy} L${cx+s} ${cy+s-w} L${cx+s-w} ${cy+s} L${cx} ${cy+w} L${cx-s+w} ${cy+s} L${cx-s} ${cy+s-w} L${cx-w} ${cy} Z`,...a});}
  if(shape==="star"){let d="";for(let i=0;i<10;i++){const ang=-Math.PI/2+i*Math.PI/5;const rr=i%2?r*0.45:r;d+=(i?"L":"M")+(cx+Math.cos(ang)*rr).toFixed(1)+" "+(cy+Math.sin(ang)*rr).toFixed(1)+" ";}return el("path",{d:d+"Z",...a});}
  // hexagon
  let d="";for(let i=0;i<6;i++){const ang=-Math.PI/2+i*Math.PI/3;d+=(i?"L":"M")+(cx+Math.cos(ang)*r).toFixed(1)+" "+(cy+Math.sin(ang)*r).toFixed(1)+" ";}
  return el("path",{d:d+"Z",...a});
}

// A row label for a list figure: a dot in the organism's colour (base model or behaviour), then its short name.
// The label is a drill-down trigger (data-org: a click opens that organism's drawer, see openDrawer).
function rowLabel(parent,x,y,o,size,opts){const r=size||11,hl=opts&&opts.highlight;
  parent.appendChild(el("circle",{cx:x-r*0.35,cy:y-r*0.32,r:Math.max(2.6,r*0.33),fill:orgCol(o)}));
  parent.appendChild(el("text",{x:x-r*1.1,y,"text-anchor":"end",fill:hl?"var(--accent)":"var(--fg)","font-size":r,"font-weight":hl?700:400,
    "data-org":o.id||o.org,"data-noenlarge":"1",cursor:"pointer"},[txt(opts&&opts.maxCh?clip(niceOrg(o),opts.maxCh):niceOrg(o)),...(opts&&opts.maxCh?[el("title",{},[txt(niceOrg(o))])]:[])]));}
