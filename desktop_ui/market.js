/* Veltra Marketplace tab: smartness vs. speed on this PC. All figures come from GET /api/catalog; pure SVG, no libraries.
   Scores are the model makers' published numbers; a missing number stays "geen cijfer". */
(()=>{
'use strict';
var V=window.V;
if(!V)return;
var $=V.$,esc=V.esc,fmt=V.fmt;
var MET=[['overall_iq','Totaal'],['code','Code'],['reasoning','Redeneren'],['knowledge','Kennis']];
var BARS=[['code','Code'],['reasoning','Redeneren'],['knowledge','Kennis'],['tools_json','Tools/JSON']];
var COL={green:'#34D399',blue:'#5B8CFF',orange:'#F59E0B',red:'#6B6B78',gray:'#6B6B78'};
var M={metric:'overall_iq',filters:{fit:false,tools:false,commercial:false},sort:'',q:'',compare:false,picked:[],sel:null,dl:null,dlTimer:0,busy:false};

/* ---- view + navigation ------------------------------------------------------------------------------------------- */
var view=document.createElement('section');
view.className='view';view.id='view-market';view.setAttribute('aria-label','Marketplace');
view.innerHTML='<header class="vhead"><button class="icon-btn menu-btn" type="button" data-menu aria-label="Menu openen"></button><div class="vtitle"><h1>Marketplace</h1><p class="sub" id="mkSub">Slimheid tegenover snelheid op deze pc.</p></div><div class="vhead-r"><button class="btn btn-ghost btn-sm" type="button" id="mkRefresh">Vernieuwen</button></div></header>'+
'<div class="scroll"><div class="page mk-page"><div class="card pad mk-chart-card"><div class="card-h"><div><h2>Slimheid vs. snelheid op jouw pc</h2><p class="muted" id="mkBasis"></p></div><div class="seg" id="mkMetric" role="radiogroup" aria-label="Y-as">'+
MET.map((m)=>'<button type="button" role="radio" class="seg-btn" data-met="'+m[0]+'">'+m[1]+'</button>').join('')+'</div></div><div id="mkChart"></div><div class="mk-legend"><span><i style="background:#34D399"></i>GPU READY</span><span><i style="background:#5B8CFF"></i>CPU READY</span><span><i style="background:#F59E0B"></i>HYBRID</span><span><i style="background:#6B6B78"></i>TE GROOT</span><span>Grootte van de bol = geheugengebruik</span><span><i class="ring"></i>Open rand = Veltra kan het nog niet draaien</span></div><p class="muted small" id="mkMissing"></p></div>'+
'<div class="mk-tools"><div class="mk-pills" id="mkPills"></div><input type="search" id="mkSearch" placeholder="Zoek model…" aria-label="Zoek model"><label class="mk-cmp"><button type="button" class="switch" id="mkCompare" role="switch" aria-checked="false"><span></span></button>Vergelijk (selecteer 2–3 modellen)</label><button type="button" class="btn btn-primary btn-sm" id="mkCompareGo" hidden>Vergelijken</button></div>'+
'<div class="mk-grid" id="mkGrid"></div><p class="muted small">Scores zijn de door de makers gepubliceerde cijfers van de originele modellen, niet van de gecomprimeerde .vlt-bestanden. De makers meten verschillend (wel of geen “thinking”, andere benchmarkversies), dus kleine verschillen zeggen weinig. Veltra IQ is het gemiddelde van de beschikbare kolommen en wordt alleen getoond bij minstens drie.</p></div></div>';
document.querySelector('.main').appendChild(view);
var nav=document.querySelector('.nav'),btn=document.createElement('button');
btn.type='button';btn.className='nav-btn';btn.setAttribute('data-view','market');btn.title='Marketplace';btn.setAttribute('aria-label','Marketplace');
btn.innerHTML=V.ic('<path d="M3 9l1.5-5h15L21 9"/><path d="M4 9v11h16V9"/><path d="M3 9a3 3 0 0 0 6 0 3 3 0 0 0 6 0 3 3 0 0 0 6 0"/><path d="M9 20v-6h6v6"/>',19)+'<span>Markt</span>';
nav.insertBefore(btn,nav.children[1]);
var menu=view.querySelector('.menu-btn');menu.innerHTML=V.ic(V.P.menu,18);

/* ---- data -------------------------------------------------------------------------------------------------------- */
function models(){return((V.st.catalog&&V.st.catalog.models)||[]).filter((m)=>m.scores);}
function speed(m){return m.measured_tok_s!=null?{v:m.measured_tok_s,m:true}:m.est_tok_s!=null?{v:m.est_tok_s,m:false}:null;}
function fits(m){return m.loaded||m.color==='green'||m.color==='blue'||m.color==='orange';}
function color(m){return m.loaded?(m.runs_on==='cuda'?COL.green:COL.blue):COL[m.color]||COL.gray;}
function speedText(m){var s=speed(m);if(!s)return 'snelheid nog niet geschat';return(s.m?'':'~')+fmt(s.v,s.v<100?1:0)+' tok/s '+(s.m?'(Gemeten ✓)':'(Geschat)');}
function sizeText(m){return(m.size_estimated?'~':'')+fmt(m.size_gib,1)+' GB';}
function visible(){
var q=M.q.toLowerCase();
var list=models().filter((m)=>(!M.filters.fit||(fits(m)&&m.supported))&&(!M.filters.tools||(m.best_for||[]).indexOf('Code')>=0||m.scores.tools_json!=null)&&(!M.filters.commercial||m.commercial)&&(!q||(m.name+' '+(m.best_for||[]).join(' ')+' '+m.license).toLowerCase().indexOf(q)>=0));
if(M.sort==='fast')list.sort((a,b)=>((speed(b)||{v:-1}).v)-((speed(a)||{v:-1}).v));
else if(M.sort==='smart')list.sort((a,b)=>(b.scores.overall_iq==null?-1:b.scores.overall_iq)-(a.scores.overall_iq==null?-1:a.scores.overall_iq));
else list.sort((a,b)=>(b.supported?1:0)-(a.supported?1:0));
return list;
}

/* ---- Pareto chart ------------------------------------------------------------------------------------------------ */
function renderChart(list){
var box=$('#mkChart'),pts=[],missing=[];
list.forEach((m)=>{var y=m.scores[M.metric],s=speed(m);if(y==null||!s)missing.push(m.name);else pts.push({m:m,x:s.v,y:y,s:s});});
$('#mkMissing').textContent=missing.length?'Niet in de grafiek (geen cijfer op deze as): '+missing.join(', ')+'.':'';
if(!pts.length){box.innerHTML='<p class="muted">Geen modellen met een cijfer voor deze as.</p>';return;}
var W=900,H=360,L=46,R=18,T=14,B=40;
var xmin=Math.log(Math.max(0.2,Math.min.apply(null,pts.map((p)=>p.x))*0.7)),xmax=Math.log(Math.max.apply(null,pts.map((p)=>p.x))*1.4);
var ys=pts.map((p)=>p.y),ymin=Math.max(0,Math.floor((Math.min.apply(null,ys)-8)/10)*10),ymax=Math.min(100,Math.ceil((Math.max.apply(null,ys)+8)/10)*10);
var X=(v)=>L+(Math.log(v)-xmin)/(xmax-xmin)*(W-L-R),Y=(v)=>T+(1-(v-ymin)/(ymax-ymin))*(H-T-B);
var h='<svg viewBox="0 0 '+W+' '+H+'" role="img" aria-label="Spreidingsgrafiek van slimheid tegenover snelheid">';
for(var y=ymin;y<=ymax;y+=10)h+='<line class="mk-grid-l" x1="'+L+'" x2="'+(W-R)+'" y1="'+Y(y)+'" y2="'+Y(y)+'"/><text class="mk-ax" x="'+(L-8)+'" y="'+(Y(y)+4)+'" text-anchor="end">'+y+'</text>';
[0.5,1,2,5,10,20,50,100,200,500].forEach((v)=>{if(Math.log(v)>=xmin&&Math.log(v)<=xmax)h+='<line class="mk-grid-l" y1="'+T+'" y2="'+(H-B)+'" x1="'+X(v)+'" x2="'+X(v)+'"/><text class="mk-ax" y="'+(H-B+16)+'" x="'+X(v)+'" text-anchor="middle">'+String(v).replace('.',',')+'</text>';});
h+='<text class="mk-ax" x="'+((L+W-R)/2)+'" y="'+(H-6)+'" text-anchor="middle">snelheid op deze pc (tok/s, logaritmisch) →</text>';
/* best choices: the models this PC can run for which nothing else is both faster and smarter */
var own=pts.filter((p)=>p.m.supported&&fits(p.m));
var front=own.filter((a)=>!own.some((b)=>b!==a&&b.x>=a.x&&b.y>=a.y&&(b.x>a.x||b.y>a.y))).sort((a,b)=>a.x-b.x);
if(front.length>1)h+='<polyline class="mk-front" points="'+front.map((p)=>X(p.x)+','+Y(p.y)).join(' ')+'"/><text class="mk-front-l" x="'+(W-R)+'" y="'+(T+10)+'" text-anchor="end">- - -  Beste keuzes voor jouw pc</text>';
var maxRam=Math.max.apply(null,pts.map((p)=>p.m.ram_needed_gib||1));
pts.slice().sort((a,b)=>(b.m.ram_needed_gib||0)-(a.m.ram_needed_gib||0)).forEach((p)=>{
var r=7+17*Math.sqrt(Math.min(1,(p.m.ram_needed_gib||1)/Math.min(maxRam,40))),c=color(p.m);
h+='<circle class="mk-dot'+(M.sel===p.m.id?' sel':'')+'" tabindex="0" role="button" data-dot="'+esc(p.m.id)+'" cx="'+X(p.x)+'" cy="'+Y(p.y)+'" r="'+r.toFixed(1)+'" fill="'+c+'" fill-opacity="'+(p.m.supported?0.55:0.08)+'" stroke="'+c+'" stroke-width="1.6"'+(p.m.supported?'':' stroke-dasharray="4 3"')+' aria-label="'+esc(p.m.name)+'"/>'+
'<text class="mk-name" x="'+X(p.x)+'" y="'+(Y(p.y)+r+13)+'" text-anchor="middle">'+esc(p.m.name.replace(/-Instruct.*$/,''))+'</text>';
});
box.innerHTML=h+'</svg><div class="mk-tip" id="mkTip" hidden></div>';
}
function tipFor(m){
var parts=['<b>'+esc(m.name)+'</b>',esc(speedText(m)),'Geheugen: ~'+fmt(m.ram_needed_gib,1)+' GB'];
var best=null;BARS.forEach((b)=>{var v=m.scores[b[0]];if(v!=null&&(!best||v>best.v))best={v:v,n:b[1]};});
parts.push(best?'Beste score: '+best.n+' '+fmt(best.v,1):'Geen gepubliceerde scores');
if(!m.supported)parts.push('<i>Veltra kan dit nog niet draaien</i>');
return parts.join('<br>');
}

/* ---- cards ------------------------------------------------------------------------------------------------------- */
function bar(m,b){
var v=m.scores[b[0]],bench=m.scores[b[0]+'_bench'];
return '<div class="mk-bar" title="'+esc(v!=null?bench+' · '+m.scores.mode:'geen cijfer op de modelkaart')+'"><span>'+b[1]+'</span><div class="bar"><i style="width:'+(v!=null?v:0)+'%"></i></div><b>'+(v!=null?fmt(v,0):'—')+'</b></div>';
}
function action(m){
var d=M.dl;
if(d&&d.state==='running'&&d.id===m.id){
var label=d.phase==='verify'?'Controleren (SHA-256)':d.phase==='convert'?'Omzetten naar .vlt':'Downloaden';
return '<div class="mk-dl"><div class="bar'+(d.percent==null?' indet':'')+'"><i style="width:'+(d.percent||0)+'%"></i></div><div class="mk-dl-r"><span>'+label+(d.percent!=null?' '+fmt(d.percent,0)+'%':'')+(d.phase==='download'&&d.mb_per_s?' · '+fmt(d.mb_per_s,0)+' MB/s':'')+(d.sha256_ok?' · SHA-256 ✓':'')+'</span><button type="button" class="btn btn-sm btn-ghost" data-mk-cancel>Stoppen</button></div></div>';
}
if(m.loaded)return '<span class="badge b-green"><span class="dot"></span>Actief ✓</span>';
if(!m.supported)return '<button type="button" class="btn btn-sm" disabled title="De engine ondersteunt deze architectuur of tokenizer nog niet">Nog niet ondersteund</button>';
if(m.local)return '<button type="button" class="btn btn-primary btn-sm" data-mk-load="'+esc(m.file)+'">Laden</button>';
if(m.installable){
var err=d&&d.state==='error'&&d.id===m.id?'<p class="err-text">'+esc(d.error||'Downloaden mislukt')+'</p>':'';
return '<button type="button" class="btn btn-primary btn-sm" data-mk-get="'+esc(m.id)+'"'+(d&&d.state==='running'?' disabled':'')+' title="'+(m.prebuilt?'Kant-en-klaar .vlt van '+esc(String(m.download_url).split('/')[2])+', gecontroleerd met SHA-256':'Officiële bestanden van Hugging Face; Veltra zet ze op deze pc om')+'">Downloaden'+(m.download_gib!=null?' ('+fmt(m.download_gib,1)+' GB)':'')+'</button>'+err;
}
return '<span class="muted small">Niet te downloaden: zet '+esc(m.file)+' in de modellenmap.</span>';
}
function cardHTML(m){
var sp=speed(m),picked=M.picked.indexOf(m.id)>=0,iq=m.scores.overall_iq;
return '<article class="card mk-card'+(M.sel===m.id?' sel':'')+(m.supported?'':' dim')+'" data-card="'+esc(m.id)+'">'+
'<div class="mk-top"><h3>'+esc(m.name)+'</h3>'+(iq!=null?'<span class="mk-iq" title="Gemiddelde van '+m.scores.iq_columns+' gepubliceerde kolommen">IQ '+fmt(iq,0)+'<small>'+m.scores.iq_columns+'/4</small></span>':'<span class="mk-iq none" title="Te weinig gepubliceerde scores">geen IQ</span>')+'</div>'+
'<div class="mk-chips"><span class="mk-pill">'+(m.is_moe?'MoE '+fmt(m.active_params_b,1)+'B actief':'Dense '+fmt(m.total_params_b,1)+'B')+'</span><span class="mk-pill'+(m.commercial?'':' warn')+'">'+esc(m.license)+'</span><span class="mk-pill">'+esc(m.languages)+'</span>'+
V.badge(V.statusColor(m.status,m.color),m.loaded?'ACTIEF':m.status||'onbekend','sm')+'</div>'+
'<div class="mk-speed"><span class="'+(sp&&sp.m?'ok':'')+'">'+esc(speedText(m))+'</span><span>'+sizeText(m)+(m.supported?'':' (geschat)')+'</span></div>'+
'<div class="mk-bars">'+BARS.map((b)=>bar(m,b)).join('')+'</div>'+
'<p class="muted small mk-note">'+esc(m.notes||m.reason||'')+' <a href="'+esc(m.card_url)+'" target="_blank" rel="noopener noreferrer">Modelkaart</a></p>'+
'<div class="mk-act">'+action(m)+'<button type="button" class="btn btn-sm btn-ghost" data-mk-cmp="'+esc(m.id)+'" aria-pressed="'+picked+'">'+(picked?'✓ Gekozen':'Vergelijken')+'</button></div></article>';
}
function renderPills(){
var p=[['fit','Past op mijn pc',M.filters.fit],['fast','Snelste',M.sort==='fast'],['smart','Slimste',M.sort==='smart'],['tools','Code & Tools',M.filters.tools],['commercial','Commercieel (Apache 2.0)',M.filters.commercial]];
$('#mkPills').innerHTML=p.map((x)=>'<button type="button" class="pill-btn'+(x[2]?' on':'')+'" aria-pressed="'+x[2]+'" data-pill="'+x[0]+'">'+x[1]+'</button>').join('')+
'<button type="button" class="pill-btn" disabled title="De engine kan dit via de API (search_n); dit filter is nog niet gekoppeld">Deep Search 8× (filter volgt)</button>';
}
function render(){
if(V.st.view!=='market')return;
var cat=V.st.catalog;
renderPills();
V.$$('[data-met]').forEach((b)=>{var on=b.getAttribute('data-met')===M.metric;b.classList.toggle('on',on);b.setAttribute('aria-checked',on);});
if(!cat){$('#mkGrid').innerHTML='<p class="muted">Modellen laden…</p>';return;}
$('#mkBasis').textContent=cat.bandwidth_gb_s>0?'Geschatte snelheden komen uit de gemeten geheugensnelheid van deze pc ('+fmt(cat.bandwidth_gb_s,0)+' GB/s)'+(cat.calibrated_from?', bijgesteld met '+cat.calibrated_from+' eigen benchmark'+(cat.calibrated_from>1?'s':''):' — zonder eigen benchmark is dit grof en op laptops vaak te hoog')+'. Berekend voor de processor; deze build gebruikt geen videokaart.':'De snelheid wordt geschat zodra Veltra even niets doet.';
var list=visible();
renderChart(list);
var html=list.length?list.map(cardHTML).join(''):'<p class="muted">Geen model past bij deze filters.</p>';
var grid=$('#mkGrid');if(grid._h!==html){grid._h=html;grid.innerHTML=html;}
$('#mkCompareGo').hidden=M.picked.length<2;
}
function refresh(){return V.loadCatalog().then(()=>{render();});}

/* ---- download / load --------------------------------------------------------------------------------------------- */
function pollDl(){
clearTimeout(M.dlTimer);
V.api('/api/models/download/status',{timeout:8000}).then((d)=>{
var was=M.dl&&M.dl.state;M.dl=d;
if(d.state==='running'){M.dlTimer=setTimeout(pollDl,900);render();return;}
if(was==='running'){
if(d.state==='done'){V.toast('Model klaar. Je kunt het nu laden.','green');refresh();return;}
if(d.state==='error'&&d.error!=='cancelled')V.toast('Downloaden mislukt: '+d.error,'red');
if(d.error==='cancelled')V.toast('Gestopt. Wat al binnen is blijft bewaard; later hervat de download daar.','gray');
}
render();
},()=>{M.dlTimer=setTimeout(pollDl,3000);});
}
function download(id){
V.api('/api/models/download',{method:'POST',body:{id:id},timeout:20000}).then((d)=>{M.dl=Object.assign({state:'running',id:id},d);render();pollDl();},(e)=>{V.toast(V.errText(e),'orange');});
}

/* ---- compare ----------------------------------------------------------------------------------------------------- */
var CC=['#7C89FF','#22D3EE','#F59E0B'];
function radar(list){
var axes=[['code','Code'],['reasoning','Redeneren'],['knowledge','Kennis'],['tools_json','Tools'],['speed','Snelheid op deze pc']];
var maxSp=Math.max.apply(null,list.map((m)=>(speed(m)||{v:0}).v).concat([1]));
var cx=210,cy=180,R=130,pt=(k,f)=>{var a=-Math.PI/2+k*2*Math.PI/axes.length;return[cx+Math.cos(a)*R*f,cy+Math.sin(a)*R*f];};
var h='<svg viewBox="0 0 420 360" role="img" aria-label="Radargrafiek">';
[0.25,0.5,0.75,1].forEach((f)=>{h+='<polygon class="mk-grid-l" fill="none" points="'+axes.map((x,k)=>pt(k,f).join(',')).join(' ')+'"/>';});
axes.forEach((x,k)=>{var e=pt(k,1),l=pt(k,1.17);h+='<line class="mk-grid-l" x1="'+cx+'" y1="'+cy+'" x2="'+e[0]+'" y2="'+e[1]+'"/><text class="mk-ax" x="'+l[0]+'" y="'+l[1]+'" text-anchor="middle">'+x[1]+'</text>';});
list.forEach((m,n)=>{
var vals=axes.map((x)=>x[0]==='speed'?(speed(m)?speed(m).v/maxSp:null):(m.scores[x[0]]!=null?m.scores[x[0]]/100:null));
h+='<polygon points="'+vals.map((v,k)=>pt(k,v||0).join(',')).join(' ')+'" fill="'+CC[n]+'" fill-opacity=".14" stroke="'+CC[n]+'" stroke-width="2"/>';
vals.forEach((v,k)=>{if(v!=null){var p=pt(k,v);h+='<circle cx="'+p[0]+'" cy="'+p[1]+'" r="3" fill="'+CC[n]+'"/>';}});
});
return h+'</svg>';
}
function compare(){
var list=M.picked.map((id)=>models().filter((m)=>m.id===id)[0]).filter(Boolean);
if(list.length<2)return;
var old=$('#mkModal');if(old)old.remove();
var rows=[['Snelheid op deze pc',(m)=>speedText(m)],['Veltra IQ',(m)=>m.scores.overall_iq!=null?fmt(m.scores.overall_iq,0)+' ('+m.scores.iq_columns+'/4 kolommen)':'geen cijfer']].concat(
BARS.map((b)=>[b[1],(m)=>m.scores[b[0]]!=null?fmt(m.scores[b[0]],1)+' · '+m.scores[b[0]+'_bench']:'geen cijfer']),
[['Meetwijze van de maker',(m)=>m.scores.mode],['Bestand',(m)=>sizeText(m)],['RAM nodig',(m)=>'~'+fmt(m.ram_needed_gib,1)+' GB'],['VRAM nodig (CUDA-build)',(m)=>'~'+fmt(m.vram_needed_gib,1)+' GB'],
['Context in Veltra',(m)=>m.supported?'tot 32K (instelbaar)':'n.v.t.'],['Licentie',(m)=>m.license],['Talen',(m)=>m.languages],['Past op deze pc',(m)=>m.status||'onbekend'],['Deep Search 8×',(m)=>m.supported?'via de API (search_n)':'n.v.t.']]);
var box=document.createElement('div');box.id='mkModal';
box.innerHTML='<div class="mk-modal" role="dialog" aria-modal="true" aria-label="Modellen vergelijken"><div class="card-h"><div><h2>Vergelijken</h2><p class="muted">Ontbrekende punten in de radar zijn cijfers die de maker niet publiceert. De snelheidsas is relatief aan de snelste van deze keuze.</p></div><button type="button" class="btn btn-sm" data-x>Sluiten</button></div>'+
'<div class="mk-cmp-body"><div>'+radar(list)+'<div class="mk-legend">'+list.map((m,n)=>'<span><i style="background:'+CC[n]+'"></i>'+esc(m.name)+'</span>').join('')+'</div></div><div class="tbl-wrap"><table class="hist"><thead><tr><th></th>'+list.map((m)=>'<th>'+esc(m.name)+'</th>').join('')+'</tr></thead><tbody>'+
rows.map((r)=>'<tr><td>'+r[0]+'</td>'+list.map((m)=>'<td>'+esc(r[1](m))+'</td>').join('')+'</tr>').join('')+'</tbody></table></div></div></div>';
document.body.appendChild(box);
box.addEventListener('click',(e)=>{if(e.target===box||e.target.closest('[data-x]'))box.remove();});
box.querySelector('[data-x]').focus();
}
window.addEventListener('keydown',(e)=>{if(e.key==='Escape'&&$('#mkModal')){e.preventDefault();e.stopPropagation();$('#mkModal').remove();}},true);

/* ---- events ------------------------------------------------------------------------------------------------------ */
view.addEventListener('click',(e)=>{
var t=e.target,b;
if((b=t.closest('[data-met]'))){M.metric=b.getAttribute('data-met');render();return;}
if((b=t.closest('[data-pill]'))){var k=b.getAttribute('data-pill');if(k==='fast'||k==='smart')M.sort=M.sort===k?'':k;else M.filters[k]=!M.filters[k];render();return;}
if((b=t.closest('[data-dot]'))){select(b.getAttribute('data-dot'));return;}
if((b=t.closest('[data-mk-get]'))){download(b.getAttribute('data-mk-get'));return;}
if(t.closest('[data-mk-cancel]')){V.api('/api/models/download/cancel',{method:'POST',body:{}}).catch((er)=>{V.toast(V.errText(er),'orange');});return;}
if((b=t.closest('[data-mk-load]'))){b.disabled=true;b.textContent='Laden…';V.loadModel({file:b.getAttribute('data-mk-load')}).then(refresh);return;}
if((b=t.closest('[data-mk-cmp]'))){
var id=b.getAttribute('data-mk-cmp'),at=M.picked.indexOf(id);
if(at>=0)M.picked.splice(at,1);else if(M.picked.length<3)M.picked.push(id);else V.toast('Je kunt er hooguit drie vergelijken.','gray');
render();return;
}
if(t.closest('#mkCompare')){M.compare=!M.compare;$('#mkCompare').setAttribute('aria-checked',M.compare);view.classList.toggle('comparing',M.compare);if(!M.compare){M.picked=[];render();}return;}
if(t.closest('#mkCompareGo')){compare();return;}
if(t.closest('#mkRefresh')){refresh();return;}
if(M.compare&&(b=t.closest('[data-card]'))&&!t.closest('a,button')){b.querySelector('[data-mk-cmp]').click();}
});
view.addEventListener('keydown',(e)=>{var d=e.target.closest&&e.target.closest('[data-dot]');if(d&&(e.key==='Enter'||e.key===' ')){e.preventDefault();select(d.getAttribute('data-dot'));}});
function select(id){
M.sel=id;render();
var card=view.querySelector('[data-card="'+id+'"]');
if(card)card.scrollIntoView({behavior:'smooth',block:'center'});
}
function tip(e){
var d=e.target.closest&&e.target.closest('[data-dot]'),box=$('#mkTip');
if(!box)return;
if(!d){box.hidden=true;return;}
var m=models().filter((x)=>x.id===d.getAttribute('data-dot'))[0];if(!m)return;
box.innerHTML=tipFor(m);box.hidden=false;
var host=$('#mkChart').getBoundingClientRect(),r=d.getBoundingClientRect();
box.style.left=Math.max(4,Math.min(host.width-box.offsetWidth-4,r.left-host.left+r.width/2-box.offsetWidth/2))+'px';
box.style.top=Math.max(0,r.top-host.top-box.offsetHeight-8)+'px';
}
view.addEventListener('mouseover',tip);view.addEventListener('focusin',tip);
$('#mkSearch').addEventListener('input',function(){M.q=this.value.trim();render();});
document.addEventListener('veltra:catalog',render);
var last='';
new MutationObserver(()=>{var v=document.body.dataset.view;if(v===last)return;last=v;if(v==='market'){render();refresh();pollDl();}}).observe(document.body,{attributes:true,attributeFilter:['data-view']});
/* keep the "Actief" state current while the page is open */
var loadedFile='';
setInterval(()=>{if(V.st.view!=='market'||document.hidden)return;var f=(V.st.status&&V.st.status.loaded&&V.st.status.file)||'';if(f!==loadedFile){loadedFile=f;refresh();}},1500);
/* five tabs: two shorter labels so they fit (the full names stay in the tooltips) */
document.querySelectorAll('.nav-btn').forEach((b)=>{var s=b.querySelector('span'),v=b.getAttribute('data-view');if(s&&v==='bench')s.textContent='Bench';if(s&&v==='settings')s.textContent='Opties';});
window.VMarket={state:M,render:render,compare:compare};
})();
