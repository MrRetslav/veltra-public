/* Veltra desktop add-ons. Loaded after the main script; uses its bridge (window.V). */
(()=>{
'use strict';
var V=window.V;
if(!V)return;
var $=V.$;
var wv=window.chrome&&window.chrome.webview;
function host(msg){if(wv&&wv.postMessage)wv.postMessage(msg);}

/* ---- own title bar (frameless window) ---------------------------------------------------------------------------- */
var winMax=null;
function titlebar(){
if(!window.__VELTRA_FRAMELESS||!wv)return;
document.body.classList.add('framed');
var bar=document.createElement('div');
bar.id='titlebar';
bar.innerHTML='<span class="tb-name">Veltra</span><span class="tb-sp"></span>'+
'<button type="button" data-win="minimize" aria-label="Minimaliseren" title="Minimaliseren"><svg viewBox="0 0 10 10" width="10" height="10"><path d="M0 5.5h10" stroke="currentColor"/></svg></button>'+
'<button type="button" data-win="maximize" aria-label="Maximaliseren" title="Maximaliseren"></button>'+
'<button type="button" data-win="close" class="tb-close" aria-label="Sluiten naar systeemvak" title="Sluiten (blijft in het systeemvak)"><svg viewBox="0 0 10 10" width="10" height="10"><path d="M0 0l10 10M10 0L0 10" stroke="currentColor"/></svg></button>'+
'<i class="tb-edge tb-top" data-edge="resize_top"></i><i class="tb-edge tb-tl" data-edge="resize_topleft"></i><i class="tb-edge tb-tr" data-edge="resize_topright"></i>';
document.body.insertBefore(bar,document.body.firstChild);
winMax=bar.querySelector('[data-win="maximize"]');
setMaximized(false);
bar.addEventListener('click',(e)=>{var b=e.target.closest('[data-win]');if(b)host({type:'window',action:b.getAttribute('data-win')});});
bar.addEventListener('mousedown',(e)=>{
if(e.button!==0)return;
var edge=e.target.closest('[data-edge]');
if(edge){e.preventDefault();host({type:'window',action:edge.getAttribute('data-edge')});return;}
if(!window.__VELTRA_APPREGION&&!e.target.closest('button')&&e.detail===1)host({type:'window',action:'drag'});
});
bar.addEventListener('dblclick',(e)=>{if(!window.__VELTRA_APPREGION&&!e.target.closest('button,[data-edge]'))host({type:'window',action:'maximize'});});
}
function setMaximized(on){
document.body.classList.toggle('maximized',on);
if(!winMax)return;
winMax.innerHTML=on?'<svg viewBox="0 0 10 10" width="10" height="10" fill="none" stroke="currentColor"><path d="M2.5 2.5v-2h7v7h-2M.5 2.5h7v7h-7z"/></svg>'
:'<svg viewBox="0 0 10 10" width="10" height="10" fill="none" stroke="currentColor"><path d="M.5.5h9v9h-9z"/></svg>';
winMax.title=on?'Vorig formaat':'Maximaliseren';
winMax.setAttribute('aria-label',winMax.title);
}

/* ---- extra settings cards ----------------------------------------------------------------------------------------- */
var hotkey={enabled:false,registered:false,keys:'Alt + Spatie'};
function switchCard(id,title,text,extra){
return '<section class="card set"><div class="set-h"><div><h3 id="'+id+'L">'+title+'</h3><p>'+text+'</p>'+(extra||'')+'</div>'+
'<button type="button" class="switch" id="'+id+'" role="switch" aria-checked="false" aria-labelledby="'+id+'L"><span></span></button></div></section>';
}
function settingsCards(){
var box=$('#settings');if(!box)return;
var h=switchCard('swIdle','Geheugen vrijgeven na 15 min inactiviteit','Haalt het model uit RAM/VRAM als je Veltra een kwartier niet gebruikt, zodat games de ruimte krijgen. Bij je volgende vraag laadt het vanzelf weer.');
if(wv)h+=switchCard('swHotkey','Snelstart met Alt + Spatie','Opent een klein invoervenster boven elk programma. Esc sluit het weer.','<p class="err-text" id="hotkeyErr"></p>');
h+=switchCard('swSound','Zachte meldingstoon bij lange taken','Een korte toon als een benchmark klaar is terwijl je in een ander venster zit.');
var wrap=document.createElement('div');wrap.className='pro-cards';wrap.innerHTML=h;
box.appendChild(wrap);
bindSwitch('swIdle',()=>Number(V.LS.get('idle_unload_min',0))>0,(on)=>{V.LS.set('idle_unload_min',on?15:0);V.uisFlush();});
bindSwitch('swSound',()=>V.LS.get('sound',false)===true,(on)=>{V.LS.set('sound',on);if(on)ping();});
if(wv)bindSwitch('swHotkey',()=>hotkey.enabled,(on)=>{hotkey.enabled=on;host({type:'hotkey',enabled:on});});
}
function bindSwitch(id,get,set){
var b=$('#'+id);if(!b)return;
b.setAttribute('aria-checked',get()?'true':'false');
b.addEventListener('click',()=>{var on=b.getAttribute('aria-checked')!=='true';b.setAttribute('aria-checked',on?'true':'false');set(on);});
}
function showHotkey(){
var b=$('#swHotkey');if(!b)return;
b.setAttribute('aria-checked',hotkey.enabled?'true':'false');
$('#hotkeyErr').textContent=!hotkey.enabled?'':!hotkey.registered?'Alt + Spatie en Ctrl + Alt + Spatie zijn allebei al in gebruik door andere programma’s.':
hotkey.keys!=='Alt + Spatie'?'Alt + Spatie is al in gebruik door een ander programma. Veltra gebruikt daarom '+hotkey.keys+'.':'';
}

/* ---- messages from the window ------------------------------------------------------------------------------------- */
if(wv&&wv.addEventListener){
wv.addEventListener('message',(ev)=>{
var d=ev.data;
if(typeof d==='string'){try{d=JSON.parse(d);}catch(e){return;}}
if(!d||!d.type)return;
if(d.type==='window_state')setMaximized(!!d.maximized);
else if(d.type==='hotkey_state'){hotkey.enabled=!!d.enabled;hotkey.registered=!!d.registered;if(d.keys)hotkey.keys=String(d.keys);showHotkey();}
else if(d.type==='copied'&&copyWait){clearTimeout(copyWait.t);if(d.ok)copyWait.ok();else copyWait.fail(new Error('Windows kon het klembord niet openen'));copyWait=null;}
else if(d.type==='open_chat'&&d.id){V.loadChats().then(()=>V.openChat(String(d.id)));}
});
}

/* ---- zero-overhead freeze: decorative work stops while the engine is busy (the app already tracks that as is-busy) ---- */
var frozen=false;
function syncFreeze(){frozen=document.body.classList.contains('is-busy');document.body.classList.toggle('veltra-generating',frozen);}
new MutationObserver(syncFreeze).observe(document.body,{attributes:true,attributeFilter:['class']});
syncFreeze();

/* ---- pointer spotlight along card edges; one passive listener, at most one style write per frame ---- */
var spotEl=null,spotX=0,spotY=0,spotRaf=0;
function spotPaint(){
spotRaf=0;
if(!spotEl||frozen)return;
var r=spotEl.getBoundingClientRect();
spotEl.style.setProperty('--mx',(spotX-r.left)+'px');
spotEl.style.setProperty('--my',(spotY-r.top)+'px');
}
document.addEventListener('pointermove',(e)=>{
if(frozen||!e.target.closest)return;
var el=e.target.closest('.card,.chip,.tile');
if(!el)return;
spotEl=el;spotX=e.clientX;spotY=e.clientY;
if(!spotRaf)spotRaf=requestAnimationFrame(spotPaint);
},{passive:true});

/* ---- shake the field next to a fresh error message (invalid cluster address, wrong path, ...) ---- */
function shake(el){if(!el)return;el.classList.remove('shake');void el.offsetWidth;el.classList.add('shake');}
var lastErr=new WeakMap();
new MutationObserver((list)=>{
for(var i=0;i<list.length;i++){
var t=list[i].target.nodeType===1?list[i].target:list[i].target.parentElement;
var box=t&&t.closest&&t.closest('.err-text');
if(!box)continue;
var txt=box.textContent.trim();
if(txt&&lastErr.get(box)!==txt){var prev=box.previousElementSibling;shake(prev&&prev.matches('form,.row-flex,input')?prev:box);}
lastErr.set(box,txt);
}
}).observe(document.querySelector('.main'),{subtree:true,childList:true,characterData:true});

/* ---- memory: one segmented bar per memory kind, from the engine's own figures --------------------------------- */
var GIB=1073741824;
function seg(cls,label,gib,total){return gib>0.005?'<i class="ms '+cls+'" style="width:'+Math.max(0.6,100*gib/total).toFixed(2)+'%" title="'+label+': '+V.fmt(gib,2)+' GB"></i>':'';}
function legend(cls,label,gib){return '<span><i class="'+cls+'"></i>'+label+' <b>'+V.fmt(gib,1)+' GB</b></span>';}
function memRow(title,total,w,kv,free,note){
var other=Math.max(0,total-free-w-kv);
return '<div class="mem-row"><div class="bar-label"><span>'+V.esc(title)+'</span><span>'+V.fmt(total,1)+' GB</span></div><div class="mem-bar">'+
seg('m-w','Modelgewichten',w,total)+seg('m-kv','KV-cache (context)',kv,total)+seg('m-o','Windows en overig',other,total)+seg('m-f','Vrij',free,total)+'</div><div class="mem-leg">'+
legend('m-w','Modelgewichten',w)+legend('m-kv','KV-cache',kv)+legend('m-o','Windows/overig',other)+legend('m-f','Vrij',free)+'</div>'+(note?'<p class="muted small">'+note+'</p>':'')+'</div>';
}
function renderMemory(){
var grid=$('#hwCards'),hw=V.st.hw,s=V.st.status||{};
if(!grid||!hw||!hw.ram)return;
var card=$('#memCard');
if(!card){card=document.createElement('div');card.id='memCard';card.className='card pad';grid.parentNode.insertBefore(card,grid);}
var w=(s.model_bytes||0)/GIB,kv=(s.kv_bytes||0)/GIB,ram=hw.ram;
var h='<div class="card-h"><div><h2>Geheugen</h2><p class="muted">'+(s.loaded?'Geladen: '+V.esc(V.baseName(s.file||s.model))+', '+V.fmt(s.max_context,0)+' tokens context.':'Geen model in het geheugen.')+'</p></div></div>'+
memRow('Werkgeheugen (RAM)',ram.total_gib,w,kv,ram.free_gib,s.loaded?'Modelgewichten = grootte van het modelbestand; Windows kan een deel daarvan als vrijgeefbare schijfcache tellen.':'');
(hw.gpus||[]).forEach((g)=>{if(g.vram_total_gib>=1)h+=memRow(g.name,g.vram_total_gib,0,0,g.vram_free_gib,hw.backends&&hw.backends.cuda_built?'':'Deze build rekent op de processor: het model gebruikt geen videogeheugen.');});
if(card._h!==h){card._h=h;card.innerHTML=h;}
}
/* ---- the GPU drawing follows the measured load (NVIDIA only); unknown load stays idle and says so --------------- */
function renderGpuLoad(){
var gpus=(V.st.hw&&V.st.hw.gpus)||[];
document.querySelectorAll('[data-gpu]').forEach((el,n)=>{
var g=gpus.filter((x,k)=>String(x.index!=null?x.index:k)===el.getAttribute('data-gpu'))[0]||gpus[n];
var load=g&&g.load_pct!=null?g.load_pct:null,lvl=load==null||load<15?'idle':load<80?'mid':'high';
if(el.getAttribute('data-load')!==lvl)el.setAttribute('data-load',lvl);
var dur=load==null?'7s':(Math.round((7-6*load/100)*2)/2)+'s';   // half-second steps: no animation restart on every sample
if(el.style.getPropertyValue('--fan')!==dur)el.style.setProperty('--fan',dur);
var t=load==null?'GPU-belasting onbekend (alleen NVIDIA-kaarten melden die)':'GPU-belasting: '+load+'%';
var art=el.querySelector('.hw-art');if(art&&art.title!==t)art.title=t;
});
}
/* ---- cluster picture: this PC and the nodes, measured ping, and the layer split as what it is: a plan ----------- */
function renderTopology(){
var anchor=$('#computeCard');if(!anchor)return;
var box=$('#topo'),c=V.st.compute||{},mode=V.st.sel&&V.st.sel.mode;
var nodes=(c.nodes||[]);
if(mode!=='cluster'||!nodes.length){if(box)box.remove();return;}
if(!box){box=document.createElement('div');box.id='topo';box.className='card pad';anchor.parentNode.insertBefore(box,anchor.nextSibling);}
var plan={};((c.layer_plan&&c.layer_plan.parts)||[]).forEach((p)=>{plan[p.device]=p;});
function part(key){var p=plan[key];return p?'lagen '+p.first_layer+'–'+p.last_layer+' ('+p.n_layers+')':'';}
var h='<div class="card-h"><div><h2>Cluster</h2><p class="muted">'+(c.effective==='cluster'?'Elk verzoek gaat in zijn geheel naar één bereikbare node (om de beurt).':'Geen node bereikbaar: verzoeken draaien op deze PC.')+'</p></div></div>'+
'<div class="topo"><div class="topo-me"><b>Deze PC</b><span>'+part('local')+'</span></div><div class="topo-links">'+
nodes.map((n)=>{var on=n.status==='online';return '<div class="topo-row"><svg viewBox="0 0 120 12" preserveAspectRatio="none" aria-hidden="true"><line x1="0" y1="6" x2="120" y2="6" class="'+(on?'flow-on':'flow-off')+'"/></svg><span class="topo-ms">'+(on&&n.latency_ms!=null?V.fmt(n.latency_ms,0)+' ms':'geen verbinding')+'</span><div class="topo-node'+(on?' on':'')+'"><b>'+V.esc(n.node)+'</b><span>'+V.esc(on?(n.model||'online'):(n.error||'offline'))+'</span><span>'+part(n.node)+'</span></div></div>';}).join('')+'</div></div>'+
(c.layer_plan&&c.layer_plan.parts&&c.layer_plan.parts.length?'<p class="muted small">De laagverdeling is een plan: een model over meerdere machines splitsen zit nog niet in de engine.</p>':'');
if(box._h!==h){box._h=h;box.innerHTML=h;}
}
/* ---- a running remote test can always be stopped, whatever page is open ---------------------------------------- */
function renderTestBar(){
var on=!!V.st.rtBusy,bar=$('#testBar'),orig=$('#rtCancel');
if(!on){if(bar)bar.remove();return;}
if(!bar){
bar=document.createElement('div');bar.id='testBar';bar.setAttribute('role','status');
bar.innerHTML='<span><b>Remote GPU-test loopt op deze PC</b></span><button type="button" class="btn btn-danger btn-sm">Test annuleren</button>';
bar.querySelector('button').addEventListener('click',()=>{var o=$('#rtCancel');if(o&&!o.disabled)o.click();});
document.body.appendChild(bar);
}
var b=bar.querySelector('button');
if(orig){b.disabled=orig.disabled;if(b.textContent!==orig.textContent)b.textContent=orig.textContent;}
}
setInterval(()=>{
if(document.hidden)return;
renderTestBar();
if(V.st.view==='hardware'){renderMemory();renderTopology();renderGpuLoad();}
},900);

/* ---- command palette (Ctrl + K) -------------------------------------------------------------------------------- */
var pal=null,palItems=[],palShown=[],palSel=0;
function paletteCommands(){
var list=[];
function add(group,label,run,hint,off){list.push({group:group,label:label,run:run,hint:hint||'',off:!!off});}
add('Actie','Nieuwe chat',()=>V.newChatAction());
[['chats','Chats'],['hardware','Hardware & Cluster'],['bench','PC Benchmarks'],['market','Marketplace'],['settings','Instellingen']].forEach((v)=>{
if(document.getElementById('view-'+v[0]))add('Ga naar',v[1],()=>V.setView(v[0]));
});
var cat=V.st.catalog||{},seen={};
(cat.models||[]).concat(cat.local||[]).forEach((m)=>{
if(!m.file||seen[m.file]||m.local===false)return;seen[m.file]=1;
if(!m.loaded)add('Model','Wissel naar '+(m.name||m.file),()=>{V.setView('settings');V.loadModel({file:m.file});},m.status||'');
});
(V.st.chats||[]).forEach((c)=>{add('Chat',c.title||'Naamloze chat',()=>V.openChat(String(c.id)));});
add('Actie','Benchmark deze PC starten',()=>{V.setView('bench');setTimeout(()=>{var b=$('#benchBtn');if(b&&!b.disabled)b.click();},150);});
add('Actie','VRAM vrijgeven (model uit het geheugen)',()=>{V.api('/api/model/unload',{method:'POST',body:{},timeout:15000}).then((r)=>{V.toast(r.unloaded?'Model uit het geheugen gehaald.':'Er was geen model geladen.','green');V.pollStatus();},(e)=>{V.toast(e.status===409?'Veltra is bezig; probeer het na het antwoord.':V.errText(e),'orange');});});
add('Deep Search','Deep Search 4× / 8× aan- of uitzetten',null,'knop nog niet gekoppeld (API: search_n)',true);
return list;
}
/* fuzzy: every typed letter in order; consecutive and word-start letters score higher. -1 = no match */
function score(q,text){
var t=text.toLowerCase(),p=0,s=0,last=-2;
for(var i=0;i<q.length;i++){
var ch=q[i];if(ch===' ')continue;
var k=t.indexOf(ch,p);if(k<0)return -1;
s+=k===last+1?6:1;if(k===0||t[k-1]===' ')s+=4;
last=k;p=k+1;
}
return s-t.length*0.01;
}
function paletteRender(){
var q=pal.querySelector('input').value.trim().toLowerCase();
palShown=palItems.map((c,n)=>({c:c,s:q?score(q,c.group+' '+c.label):-n*0.001})).filter((x)=>!q||x.s>=0).sort((a,b)=>b.s-a.s).slice(0,12).map((x)=>x.c);
palSel=Math.max(0,Math.min(palSel,palShown.length-1));
pal.querySelector('.pal-list').innerHTML=palShown.length?palShown.map((c,k)=>'<div class="pal-item'+(k===palSel?' sel':'')+(c.off?' off':'')+'" role="option" aria-selected="'+(k===palSel)+'" data-k="'+k+'"><span class="pal-g">'+V.esc(c.group)+'</span><span class="pal-l">'+V.esc(c.label)+'</span><span class="pal-h">'+V.esc(c.hint)+'</span></div>').join(''):'<div class="pal-empty">Niets gevonden</div>';
var sel=pal.querySelector('.pal-item.sel');if(sel)sel.scrollIntoView({block:'nearest'});
}
function paletteClose(){if(pal){pal.remove();pal=null;}}
function paletteRun(k){var c=palShown[k];if(!c||c.off)return;paletteClose();c.run();}
function paletteOpen(){
if(pal){paletteClose();return;}
pal=document.createElement('div');pal.id='palette';
pal.innerHTML='<div class="pal-box" role="dialog" aria-label="Commando’s"><input type="text" placeholder="Zoek een model, chat of actie…" aria-label="Zoeken" autocomplete="off" spellcheck="false"><div class="pal-list" role="listbox"></div><div class="pal-foot">↑ ↓ kiezen · Enter uitvoeren · Esc sluiten</div></div>';
document.body.appendChild(pal);
palItems=paletteCommands();palSel=0;paletteRender();
var inp=pal.querySelector('input');inp.focus();
inp.addEventListener('input',()=>{palSel=0;paletteRender();});
pal.addEventListener('mousedown',(e)=>{if(e.target===pal){paletteClose();return;}var it=e.target.closest('[data-k]');if(it){e.preventDefault();paletteRun(+it.getAttribute('data-k'));}});
if(!V.st.catalog)V.loadCatalog().then(()=>{if(pal){palItems=paletteCommands();paletteRender();}});
}
window.addEventListener('keydown',(e)=>{
if((e.ctrlKey||e.metaKey)&&!e.altKey&&e.key.toLowerCase()==='k'){e.preventDefault();e.stopPropagation();paletteOpen();return;}
if(!pal)return;
if(e.key==='Escape'){e.preventDefault();e.stopPropagation();paletteClose();}
else if(e.key==='ArrowDown'||e.key==='ArrowUp'){e.preventDefault();palSel=(palSel+(e.key==='ArrowDown'?1:-1)+palShown.length)%Math.max(1,palShown.length);paletteRender();}
else if(e.key==='Enter'){e.preventDefault();e.stopPropagation();paletteRun(palSel);}
},true);

/* ---- result card: drawn locally on a canvas, put on the clipboard as a PNG --------------------------------------- */
var CARD_W=1200,CARD_H=630;
function drawCard(run,hw){
var cv=document.createElement('canvas');cv.width=CARD_W;cv.height=CARD_H;
var g=cv.getContext('2d'),F='"Segoe UI Variable Display","Segoe UI",system-ui,sans-serif';
g.fillStyle='#09090B';g.fillRect(0,0,CARD_W,CARD_H);
var glow=g.createRadialGradient(180,0,20,180,0,640);glow.addColorStop(0,'rgba(67,81,228,.45)');glow.addColorStop(1,'rgba(67,81,228,0)');g.fillStyle=glow;g.fillRect(0,0,CARD_W,CARD_H);
function box(x,y,w,h,r){g.beginPath();g.roundRect(x,y,w,h,r);}
function text(s,x,y,size,color,weight,max){g.font=(weight||400)+' '+size+'px '+F;g.fillStyle=color;var t=String(s);if(max)while(t.length>4&&g.measureText(t).width>max)t=t.slice(0,-2)+'…';g.fillText(t,x,y);return g.measureText(t).width;}
var grad=g.createLinearGradient(56,48,120,112);grad.addColorStop(0,'#5866F0');grad.addColorStop(1,'#3843C4');
box(56,48,64,64,18);g.fillStyle=grad;g.fill();
g.strokeStyle='#fff';g.lineWidth=5.4;g.lineCap='round';g.lineJoin='round';g.beginPath();g.moveTo(74,69);g.lineTo(88,93);g.lineTo(102,69);g.stroke();
text('Veltra',140,90,38,'#F2F2F5',700);
text(new Date().toLocaleDateString('nl-NL',{day:'numeric',month:'long',year:'numeric'}),140,114,17,'#9898A6');
var cpu=run.cpu||(hw&&hw.cpu&&hw.cpu.brand)||'onbekende processor';
var gpu=((run.gpus&&run.gpus[0])||(hw&&hw.gpus&&hw.gpus[0])||{});gpu=typeof gpu==='string'?gpu:gpu.name||'';
var onGpu=/cuda|gpu/i.test(run.device||'');
text(onGpu?gpu:cpu,56,182,30,'#F2F2F5',600,1088);
text((onGpu?'Berekend op de GPU':'Berekend op de processor ('+(run.device_label||'CPU')+')')+(!onGpu&&gpu?' · aanwezig: '+gpu:'')+' · model '+V.baseName(run.model||'onbekend')+(run.threads?' · '+run.threads+' threads':''),56,214,18,'#B4B4C0',400,1088);
var tiles=[['Decode',run.decode_tok_s!=null?V.fmt(run.decode_tok_s,1):'—','tok/s'],['Tijd tot eerste token',run.ttft_s!=null?V.fmt(run.ttft_s,2):'—','s'],['Prefill',run.prefill_tok_s!=null?V.fmt(run.prefill_tok_s,0):'—','tok/s'],['Piek-RAM',run.peak_rss_mb!=null?V.fmt(run.peak_rss_mb/1024,1):'—','GB']];
if(run.vram_peak_mb!=null)tiles.push(['Piek-VRAM',V.fmt(run.vram_peak_mb/1024,1),'GB']);
var gap=16,w=(1088-gap*(tiles.length-1))/tiles.length;
tiles.forEach((t,k)=>{var x=56+k*(w+gap);box(x,258,w,190,22);g.fillStyle='#121216';g.fill();g.strokeStyle='rgba(255,255,255,.09)';g.lineWidth=1;g.stroke();
text(t[0],x+22,296,17,'#9898A6',600,w-40);var used=text(t[1],x+22,386,k===0?72:58,k===0?'#AAB4FF':'#F2F2F5',700,w-70);text(t[2],x+28+used,386,20,'#B4B4C0');});
box(56,478,1088,96,20);g.fillStyle='rgba(255,255,255,.035)';g.fill();
text('Referentie, niet deze pc: Ryzen 9 3900 met Qwen3-30B-A3B',80,516,17,'#9898A6',600);
text('Veltra 21,3 tok/s  ·  llama.cpp 18,1 tok/s (decode, lege context)',80,550,24,'#F2F2F5',600);
text('Gemeten met de ingebouwde Veltra-benchmark op deze pc.',56,608,15,'#6F6F7C');
return cv;
}
var copyWait=null;
function b64(bytes){var s='';for(var i=0;i<bytes.length;i+=32768)s+=String.fromCharCode.apply(null,bytes.subarray(i,i+32768));return btoa(s);}
function nativeCopy(cv,blob){
if(!wv)return Promise.reject(new Error('geen klembordtoegang'));
return blob.arrayBuffer().then((buf)=>{
var px=cv.getContext('2d').getImageData(0,0,cv.width,cv.height).data,bgra=new Uint8Array(px.length);
for(var y=0;y<cv.height;y++){var from=y*cv.width*4,to=(cv.height-1-y)*cv.width*4;for(var x=0;x<cv.width*4;x+=4){bgra[to+x]=px[from+x+2];bgra[to+x+1]=px[from+x+1];bgra[to+x+2]=px[from+x];bgra[to+x+3]=255;}}
return new Promise((ok,fail)=>{copyWait={ok:ok,fail:fail,t:setTimeout(()=>{copyWait=null;fail(new Error('geen antwoord van het venster'));},8000)};
host({type:'copy_png',width:cv.width,height:cv.height,png:b64(new Uint8Array(buf)),bgra:b64(bgra)});});
});
}
function copyCard(btn){
btn.disabled=true;
V.api('/api/benchmarks/local').then((d)=>{
var runs=(d&&d.data)||[],run=runs[runs.length-1];
if(!run)throw new Error('Draai eerst een benchmark.');
var cv=drawCard(run,V.st.hw);
return new Promise((r)=>cv.toBlob(r,'image/png')).then((blob)=>{
window.VPro.lastCard=cv;
var web=navigator.clipboard&&window.ClipboardItem?navigator.clipboard.write([new ClipboardItem({'image/png':blob})]):Promise.reject(new Error('geen klembordtoegang'));
return web.catch(()=>nativeCopy(cv,blob));
});
}).then(()=>{btn.textContent='✓ Gekopieerd';V.toast('Resultaatkaart staat als afbeelding op je klembord.','green');},(e)=>{V.toast('Kopiëren mislukt: '+((e&&e.message)||e),'orange');})
.then(()=>{btn.disabled=false;setTimeout(()=>{btn.textContent='Kopieer resultaatkaart';},1800);});
}
(function(){
var hero=$('#benchBtn');if(!hero)return;
var b=document.createElement('button');b.type='button';b.className='btn';b.id='cardBtn';b.textContent='Kopieer resultaatkaart';
var wrap=document.createElement('div');wrap.className='bench-actions';hero.parentNode.insertBefore(wrap,hero);wrap.appendChild(hero);wrap.appendChild(b);
b.addEventListener('click',()=>copyCard(b));
})();

/* ---- optional soft ping when a benchmark finishes while you are somewhere else (off by default) ------------------ */
var audio=null;
function audioReady(){if(!audio){try{audio=new AudioContext();}catch(e){}}if(audio&&audio.state==='suspended')audio.resume();return audio;}
function ping(){
var a=audioReady();if(!a)return false;
[[880,0],[1318.5,0.13]].forEach((n)=>{var o=a.createOscillator(),gn=a.createGain(),t=a.currentTime+n[1];o.type='sine';o.frequency.value=n[0];gn.gain.setValueAtTime(0,t);gn.gain.linearRampToValueAtTime(0.09,t+0.02);gn.gain.exponentialRampToValueAtTime(0.0001,t+0.5);o.connect(gn);gn.connect(a.destination);o.start(t);o.stop(t+0.55);});
return true;
}
function soundOn(){return V.LS.get('sound',false)===true;}
document.addEventListener('pointerdown',()=>{if(soundOn())audioReady();},{passive:true,capture:true});
var wasBench=false;
setInterval(()=>{
var now=!!(V.st.bench&&V.st.bench.running);
if(wasBench&&!now&&soundOn()&&(document.hidden||!document.hasFocus())){ping();window.VPro.pings=(window.VPro.pings||0)+1;}
wasBench=now;
},500);

titlebar();
settingsCards();
host({type:'hello'});
window.VPro={host:host,bindSwitch:bindSwitch,shake:shake,switchCard:switchCard};
})();
