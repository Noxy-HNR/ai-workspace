/* Relay Station. All conversation content is escaped before it enters the DOM.
   Secret settings are masked server-side and are never retained in localStorage. */
"use strict";
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const state = { users: [], selected: null, chat: null, thread: 0, chats: [], messages: [], context: null,
  settings: {}, settingsLoaded: false, settingsDirty: false, activeTab: "overview", csrf: null,
  chatRequest: 0, peopleRequest: 0, eventsRequest: 0, jobsRequest: 0, drafts: new Map(), personDrafts: new Map(), events: [], presets: [],
  sessionPromise: null, refreshing: false, hasMore: false, before: null, currentUser: null, lastStatus: null };
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const uname = (u = {}) => [u.first_name,u.last_name].filter(Boolean).join(" ") || (u.username ? "@" + u.username : "Person " + (u.user_id || u.id || ""));
const initials = (u) => uname(u).replace(/^@/, "").split(/\s+/).slice(0,2).map(v => v[0]).join("").toUpperCase();
const number = (n) => Number(n || 0).toLocaleString();
const tstr = (s) => s ? new Date(s).toLocaleString([], {month:"short",day:"numeric",hour:"numeric",minute:"2-digit"}) : "Never";
const reltime = (s) => { if (!s) return "Never"; const minutes = Math.max(0,Math.floor((Date.now()-new Date(s))/60000)); return minutes<1?"Just now":minutes<60?minutes+"m ago":minutes<1440?Math.floor(minutes/60)+"h ago":Math.floor(minutes/1440)+"d ago"; };
const scopeKey = (chat, thread = 0) => `${chat}:${thread || 0}`;
const draftKey = () => `${state.selected}:${scopeKey(state.chat,state.thread)}`;
const scopeBody = () => ({chat_id:state.chat,thread_id:state.thread || 0});
const scopeQuery = (chat = state.chat, thread = state.thread) => new URLSearchParams({chat_id:chat,thread_id:thread || 0}).toString();
const empty = (title, detail, symbol="◇") => `<div class="empty"><div class="empty-icon" aria-hidden="true">${symbol}</div><h3>${esc(title)}</h3><p>${esc(detail)}</p></div>`;
const person = (u) => `<div class="person"><span class="avatar" aria-hidden="true">${esc(initials(u))}</span><div><b>${esc(uname(u))}</b><small>${esc(u.username ? "@"+u.username : "Telegram ID "+(u.id || u.user_id || ""))}</small></div></div>`;
const card = (label,value,detail,symbol="◈",cls="") => `<div class="card"><div class="card-head"><span class="k">${esc(label)}</span><span class="card-symbol" aria-hidden="true">${symbol}</span></div><div class="v ${cls}">${esc(value)}</div><div class="s" title="${esc(detail)}">${esc(detail)}</div></div>`;

let noticeTimer;
function notice(message, error=false) {
  clearTimeout(noticeTimer); $("#notice").hidden=false; $("#notice").className=error?"error":""; $("#noticeText").textContent=message;
  if (!error) noticeTimer=setTimeout(() => { $("#notice").hidden=true; },6500);
}
$("#noticeDismiss").onclick=() => { $("#notice").hidden=true; };
window.addEventListener("unhandledrejection", e => { e.preventDefault(); notice(e.reason?.message || "The request could not be completed.",true); });
async function api(path, opts={}) {
  if (!state.csrf) {
    if (!state.sessionPromise) state.sessionPromise=fetch("/api/session",{signal:AbortSignal.timeout(10000)}).then(async r => { if (!r.ok) throw new Error("Unable to open a console session."); return r.json(); }).then(j => { state.csrf=j.csrf; }).finally(() => { state.sessionPromise=null; });
    await state.sessionPromise;
  }
  const response=await fetch(path,{...opts,signal:opts.signal || AbortSignal.timeout(35000),headers:{"Content-Type":"application/json","X-Relay-CSRF":state.csrf,...opts.headers}});
  const result=await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(result.error || `Request failed (${response.status}).`);
  return result;
}
const post = (path, body={}) => api(path,{method:"POST",body:JSON.stringify(body)});
async function busy(button, action) {
  if (button.disabled) return;
  button.disabled=true; button.classList.add("busy"); button.setAttribute("aria-busy","true");
  try { return await action(); } catch(e) { notice(e.name==="TimeoutError"?"The request timed out. Refresh to check its final state before repeating it.":e.message,true); }
  finally { button.disabled=false; button.classList.remove("busy"); button.removeAttribute("aria-busy"); if (["setSave","setDiscard"].includes(button.id) && state.settingsLoaded) updateSettingsDirty(); }
}
const debounce = (fn,ms=240) => { let timer; return (...args) => { clearTimeout(timer); timer=setTimeout(() => fn(...args),ms); }; };
function setDot(selector,cls) { $(selector).className="dot "+cls; }
function bindOpen(container) { container.querySelectorAll("[data-open]").forEach(b => { b.onclick=() => openPerson(Number(b.dataset.open),b.dataset.chat?Number(b.dataset.chat):null,Number(b.dataset.thread || 0)); }); }

/* Navigation and preferences. Only the color theme is persisted on disk. */
const titles={overview:"Overview",chat:"Conversations",users:"People",queue:"Reply queue",activity:"Activity",stats:"Insights",settings:"Settings"};
function goto(tab, updateHash=true) {
  if (!(tab in titles)) tab="overview";
  state.activeTab=tab;
  $$("#nav button").forEach(b => { const active=b.dataset.tab===tab; b.classList.toggle("active",active); if(active)b.setAttribute("aria-current","page");else b.removeAttribute("aria-current"); });
  $$(".tab").forEach(el => el.classList.toggle("active",el.id==="tab-"+tab));
  $("#pageTitle").textContent=titles[tab]; document.title=`${titles[tab]} · Relay Station`;
  if (updateHash && location.hash!=="#"+tab) history.replaceState(null,"","#"+tab);
  refreshPage();
}
function refreshPage() { return ({overview:refreshStatus,chat:loadChat,users:loadUsers,queue:loadJobs,activity:loadEvents,stats:loadStats,settings:() => loadSettings(!state.settingsDirty)}[state.activeTab])(); }
$$("#nav button").forEach(b => { b.onclick=() => goto(b.dataset.tab); });
window.addEventListener("hashchange",() => goto(location.hash.slice(1),false));
$(".brand").onclick=e => {e.preventDefault();goto("overview");};
$("#globalRefresh").onclick=() => busy($("#globalRefresh"),refreshPage);
$("#overviewChats").onclick=() => goto("chat"); $("#overviewQueue").onclick=() => goto("queue"); $("#overviewPeople").onclick=() => goto("users"); $("#btnGoSettings").onclick=() => goto("settings");
try { document.documentElement.dataset.theme=localStorage.getItem("relay-theme") || "dark"; } catch (_) { /* Theme is optional. */ }
$("#themeToggle").onclick=() => {const theme=document.documentElement.dataset.theme==="light"?"dark":"light";document.documentElement.dataset.theme=theme;try{localStorage.setItem("relay-theme",theme);}catch(_){} $("#themeToggle").setAttribute("aria-label",`Switch to ${theme==="light"?"dark":"light"} theme`);};

/* Status uses cached probes from the server; auto refresh does not send a test message. */
async function refreshStatus() {
  if (state.refreshing || document.hidden) return;
  state.refreshing=true;
  try {
    const [s,queued]=await Promise.all([api("/api/status"),state.activeTab==="overview"?api("/api/jobs?status=queued&limit=4"):Promise.resolve(state.overviewQueued || {jobs:[],counts:{}})]); state.lastStatus=s;if(state.activeTab==="overview")state.overviewQueued=queued;
    const tg=s.telegram || {}, ll=s.llm || {}, p=s.poller || {}, counts=s.counts || {};
    const jobCounts=s.jobs || queued.counts || {},busyCount=Number(jobCounts.processing || 0)+Number(jobCounts.delivering || 0) || (p.busy_users || []).length, pending=Number(jobCounts.queued || 0);
    setDot("#dotTg",tg.connected===true?"ok":tg.connected===false?"bad":"warn"); setDot("#dotLlm",ll.connected===true?"ok":ll.connected===false?"bad":"warn"); setDot("#dotQ",busyCount || pending?"warn":"ok");
    $("#rsTg").textContent=tg.connected?("@"+(tg.bot_username || "connected")):tg.configured?"Offline":"Unconfigured";
    $("#rsLlm").textContent=ll.connected?"Connected":ll.connected===false?"Offline":"Checking";
    $("#rsQ").textContent=busyCount?busyCount+" active":pending?pending+" queued":"All clear";
    $("#brandOrb").className="brand-orb"+(!tg.configured || ll.connected===false?" bad":busyCount?" warn":"");
    $("#navUserCount").textContent=counts.users || ""; $("#navQueueCount").textContent=busyCount+pending || "";
    const banners=[];
    if (s.preview) banners.push(`<div class="preview"><b>Demo workspace</b> · Sample conversations, simulated Telegram and model calls. Try the controls freely.</div>`);
    if (!tg.configured) banners.push(`<div class="err"><b>Connect your Telegram bot.</b> Add your BotFather token in Settings to start receiving messages.</div>`);
    if (!s.admin_claimed) banners.push(`<div class="warn"><b>Choose an administrator.</b> Set your Telegram ID in Settings, or claim the bot with its first private message.</div>`);
    if (p.last_error && !s.preview) banners.push(`<div class="warn"><b>Poller needs attention.</b> ${esc(p.last_error)}</div>`);
    if (ll.connected===false) banners.push(`<div class="warn"><b>Model endpoint unavailable.</b> ${esc(ll.error || "Check that your local model server is running.")} <span class="mono">${esc(ll.base_url)}</span></div>`);
    $("#setupBanner").innerHTML=banners.join("");
    $("#statusCards").innerHTML=card("Telegram",tg.connected?"Connected":tg.configured?"Offline":"Set up",tg.bot_username?"@"+tg.bot_username:tg.error || "Waiting for bot token","↗",tg.connected?"healthy":"")+card("Model endpoint",ll.connected?"Ready":ll.connected===false?"Offline":"Checking",(ll.models || [])[0] || ll.model || ll.base_url || "No endpoint","◈",ll.connected?"healthy":ll.connected===false?"failed":"")+card("People",number(counts.users),number(counts.messages_today)+" messages today","♧")+card("Message polling",p.running?"Running":"Paused",p.last_poll?"Last poll "+reltime(p.last_poll):"Waiting for first poll","≋",p.running?"healthy":"");
    const items=[];
    (p.busy_users || []).forEach(id => items.push(`<div class="queue-item"><div class="queue-heading"><span class="badge running">Generating</span><b>Person ${esc(id)}</b></div></div>`));
    (queued.jobs || []).forEach(j => items.push(`<div class="queue-item"><div><div class="queue-heading"><span class="badge queued">${p.running?"Queued":"Queued · polling paused"}</span><b>${esc(uname(j))}</b></div><div class="queue-text">${esc(j.text.slice(0,110))}${j.text.length>110?"…":""}</div></div><div class="btnrow"><button class="quiet" data-open="${j.user_id}" data-chat="${j.chat_id}" data-thread="${j.thread_id || 0}">Open →</button></div></div>`));
    $("#queueBox").innerHTML=items.length?items.join("")+(pending>4?`<div class="dim" style="font-size:11px">${number(pending-4)} more queued replies · open the reply queue to see all.</div>`:""):empty("Room to breathe","No replies are waiting. New messages will appear here.","✓");bindOpen($("#queueBox"));
    $("#btnPollToggle").textContent=p.running?"Pause polling":"Resume polling";
    $("#syncStatus").innerHTML='<span class="dot ok"></span> Console connected';
    const refreshed="Synced "+new Date().toLocaleTimeString([], {hour:"numeric",minute:"2-digit"}); $("#lastRefreshed").textContent=refreshed;
    await loadPeople();
    const users=state.users.slice(0,5);
    $("#recentUsers").innerHTML=users.length?`<table aria-label="Recent people"><tbody>${users.map(u => `<tr><td>${person(u)}</td><td class="recent-id mono dim">${u.id}</td><td>${u.role==="admin"?'<span class="badge admin">Admin</span>':u.blocked?'<span class="badge blocked">Blocked</span>':'<span class="badge ok">Active</span>'}</td><td class="mono dim" title="${esc(tstr(u.last_seen))}">${esc(reltime(u.last_seen))}</td><td><button class="quiet" data-open="${u.id}" aria-label="Open ${esc(uname(u))}'s conversation">Open →</button></td></tr>`).join("")}</tbody></table>`:empty("Your first conversation starts here","Message your bot on Telegram to bring this workspace to life.","♧");
    bindOpen($("#recentUsers"));
  } catch(e) { $("#syncStatus").innerHTML='<span class="dot bad"></span> Connection lost'; notice("Console connection failed: "+e.message,true); }
  finally {state.refreshing=false;}
}
$("#btnPollToggle").onclick=() => busy($("#btnPollToggle"),async () => {const {poller}=await api("/api/poller");await post(poller.running?"/api/poller/stop":"/api/poller/start");notice(poller.running?"Incoming message polling paused.":"Incoming message polling resumed.");await refreshStatus();});
async function testInto(target,kind,button) {return busy(button,async () => {$(target).textContent="Checking the saved connection…";try{const {result}=await post("/api/test/"+kind);const good=result.connected===true || result.ok===true;$(target).innerHTML=`<div class="test-result ${good?"":"failed"}"><b>${good?"Connection healthy":"Connection needs attention"}</b><div>${esc(result.error || result.bot_username && "@"+result.bot_username || result.models?.join(", ") || "Check complete.")}</div><details><summary>Connection details</summary><pre>${esc(JSON.stringify(result,null,2))}</pre></details></div>`;await refreshStatus();}catch(e){$(target).innerHTML=`<div class="test-result failed"><b>Check failed</b><div>${esc(e.message)}</div></div>`;throw e;}});}
$("#btnTestTg").onclick=() => testInto("#quickOut","telegram",$("#btnTestTg")); $("#btnTestLlm").onclick=() => testInto("#quickOut","llm",$("#btnTestLlm"));
$("#setTestTg").onclick=() => testInto("#setOut","telegram",$("#setTestTg")); $("#setTestLlm").onclick=() => testInto("#setOut","llm",$("#setTestLlm"));

/* The unfiltered person roster is separate from the People page's filters. */
let peoplePromise;
async function loadPeople() {
  if (!peoplePromise) peoplePromise=api("/api/users").then(({users}) => {state.users=users;renderPeopleSelects();}).finally(() => {peoplePromise=null;});
  return peoplePromise;
}
function renderPeopleSelects() {
  const selected=state.selected, queueSelected=$("#queueUser").value;
  if(document.activeElement!==$("#chatUser")) {$("#chatUser").innerHTML=state.users.map(u => `<option value="${u.id}">${esc(uname(u))}${u.blocked?" · Blocked":""}</option>`).join("");if(selected)$("#chatUser").value=selected;}
  if(document.activeElement!==$("#queueUser")) {$("#queueUser").innerHTML='<option value="">All people</option>'+state.users.map(u => `<option value="${u.id}">${esc(uname(u))}</option>`).join(""); $("#queueUser").value=queueSelected;}
}
async function loadUsers() {
  const key=++state.peopleRequest;
  const query=new URLSearchParams({search:$("#userSearch").value,role:$("#userRole").value,blocked:$("#userBlocked").value});
  const {users}=await api("/api/users?"+query); if(key!==state.peopleRequest)return;
  $("#usersCount").textContent=number(users.length)+" people";
  $("#userTable tbody").innerHTML=users.length?users.map(u => `<tr class="${u.id===state.selected?"sel":""}"><td>${person(u)}</td><td class="mono dim">${u.id}</td><td>${u.role==="admin"?'<span class="badge admin">Administrator</span>':'<span class="dim">User</span>'}</td><td><span class="badge ${u.blocked?"blocked":"ok"}">${u.blocked?"Blocked":"Active"}</span></td><td class="mono">${number(u.stats?.messages)}</td><td class="event-time" title="${esc(tstr(u.last_seen))}">${esc(reltime(u.last_seen))}</td><td><button data-open="${u.id}" class="quiet" aria-label="Open ${esc(uname(u))}'s conversation">Open →</button></td></tr>`).join(""):`<tr><td colspan="7">${empty("No people match","Try another name, ID, or filter.","⌕")}</td></tr>`;
  bindOpen($("#userTable"));
}
$("#userRefresh").onclick=() => busy($("#userRefresh"),loadUsers); $("#userSearch").oninput=debounce(loadUsers); $("#userRole").onchange=loadUsers; $("#userBlocked").onchange=loadUsers;
function openPerson(uid,chat=null,thread=0) {state.selected=uid;state.chat=chat;state.thread=thread;$("#chatSearch").value="";state.messages=[];goto("chat");}
$("#chatUser").onchange=e => openPerson(Number(e.target.value));
$("#chatScope").onchange=e => {const [chat,thread]=e.target.value.split(":").map(Number);state.chat=chat;state.thread=thread;state.messages=[];$("#chatSearch").value="";loadChat();};

/* Conversation requests are keyed before any await. Late responses cannot switch
   a chat back, replace another chat's prompt, or send to the wrong recipient. */
function activeChat() {return state.chats.find(c => Number(c.chat_id)===state.chat && Number(c.thread_id || 0)===state.thread);}
function chatName(chat=activeChat()) {return chat?.label || (chat?.chat_type==="private"?"Private conversation":chat?.chat_type==="supergroup"?"Group conversation":chat?.chat_type?chat.chat_type[0].toUpperCase()+chat.chat_type.slice(1):"Conversation")+(chat?.thread_id?" · Topic "+chat.thread_id:"");}
async function loadChat(options={}) {
  const key=++state.chatRequest;
  await loadPeople(); if(key!==state.chatRequest)return;
  if(!state.selected && state.users.length)state.selected=state.users[0].id;
  if(!state.selected){$("#chatLog").innerHTML=empty("No conversations yet","Message your bot on Telegram to get started.","▤");$("#chatHeader").innerHTML="";$("#ctxPanel").innerHTML=empty("No context yet","Choose a person to inspect their model context.");setChatControls(false);return;}
  const uid=state.selected;
  const {chats}=await api(`/api/users/${uid}/chats`);if(key!==state.chatRequest || uid!==state.selected)return;
  state.chats=chats;
  const ordered=[...chats].sort((a,b) => Number(b.bookmarked || 0)-Number(a.bookmarked || 0));
  if(!chats.some(c => Number(c.chat_id)===state.chat && Number(c.thread_id || 0)===state.thread)){const chosen=chats.find(c => c.chat_type==="private") || chats[0];state.chat=chosen?Number(chosen.chat_id):uid;state.thread=chosen?Number(chosen.thread_id || 0):0;}
  $("#chatScope").innerHTML=ordered.map(c => `<option value="${scopeKey(c.chat_id,c.thread_id)}">${c.bookmarked?"★ ":""}${esc(chatName(c))} · ${c.chat_id} (${number(c.messages)})</option>`).join(""); $("#chatScope").value=scopeKey(state.chat,state.thread);$("#chatUser").value=uid;
  const chat=state.chat, thread=state.thread,query=scopeQuery(chat,thread),search=$("#chatSearch").value.trim();
  const [md,cx]=await Promise.all([api(`/api/users/${uid}/messages?${query}&limit=150${options.older && state.before?"&before="+state.before:""}&search=${encodeURIComponent(search)}`),api(`/api/users/${uid}/context?${query}`)]);
  if(key!==state.chatRequest || uid!==state.selected || chat!==state.chat || thread!==state.thread)return;
  const dataKey=`${uid}:${scopeKey(chat,thread)}:${search}`,sameData=dataKey===state.chatDataKey,previousIds=new Set(state.messages.map(m => m.id));
  if(options.older && sameData) {state.messages=[...md.messages.filter(m => !previousIds.has(m.id)),...state.messages];state.hasMore=!!md.has_more;state.before=md.next_before;}
  else if(options.keepScroll && sameData) {const merged=new Map(state.messages.map(m => [m.id,m]));md.messages.forEach(m => merged.set(m.id,m));state.messages=[...merged.values()].sort((a,b) => a.id-b.id);}
  else {state.messages=md.messages;state.hasMore=!!md.has_more;state.before=md.next_before;}
  state.chatDataKey=dataKey;state.currentUser=md.user;state.context=cx;
  setChatControls(chats.length>0);renderChat(options);restoreDrafts(md.user);renderContext();
}
function setChatControls(available) { ["chatScope","chatBookmark","chatRename","chatExport","chatExportMd","adminSend","saveChatPrompt","chatPromptReset","applyPromptPreset","saveNote","savePromptPreset"].forEach(id => {$("#"+id).disabled=!available;}); }
function renderChat(options={}) {
  const u=state.currentUser,chat=activeChat(),inContext=new Set(state.context?.context?.included_ids || []);
  $("#chatHeader").innerHTML=`<div class="conversation-header">${person(u)}<div class="scope-summary"><b>${esc(chatName())}</b>${number(chat?.messages || 0)} messages · ${esc(tstr(chat?.last_message || u.last_seen))} ${u.role==="admin"?'<span class="badge admin">Admin</span>':""} ${u.blocked?'<span class="badge blocked">Blocked</span>':""}</div></div>`;
  $("#chatBookmark").textContent=chat?.bookmarked?"★":"☆";$("#chatBookmark").setAttribute("aria-pressed",chat?.bookmarked?"true":"false");$("#chatBookmark").setAttribute("aria-label",chat?.bookmarked?"Remove conversation bookmark":"Bookmark this conversation");
  $("#adminSendTarget").textContent=`Deliver to ${chatName()} (${state.chat}${state.thread?", topic "+state.thread:""}).`;
  const log=$("#chatLog"),oldTop=log.scrollTop,oldHeight=log.scrollHeight,nearBottom=oldHeight-oldTop-log.clientHeight<65;
  const messages=state.messages.filter(m => !$("#ctxOnly").checked || inContext.has(m.id));
  let day="";
  log.innerHTML=messages.map(m => {const current=new Date(m.created_at).toLocaleDateString([], {weekday:"short",month:"short",day:"numeric"});const separator=current!==day?`<div class="message-day">${esc(current)}</div>`:"";day=current;const included=inContext.has(m.id),cls=m.kind==="error"?"err":m.kind==="system"?"sys":m.direction;return separator+`<article class="msg ${cls}${included?"":" archived"}"><div class="meta"><b>${m.direction==="in"?"Person":"Bot"}</b><span>${esc(new Date(m.created_at).toLocaleTimeString([], {hour:"numeric",minute:"2-digit"}))}</span>${m.kind!=="text"?`<span class="tag">${esc(m.kind)}</span>`:""}${m.status!=="done"?`<span class="tag">${esc(m.status)}</span>`:""}<span class="tag ${included?"inctx":"arch"}">${included?"In context":"History"}</span><button class="message-copy" data-copy="${m.id}" aria-label="Copy message" title="Copy message">⧉</button></div><div class="body">${formatMessage(m.text,$("#chatSearch").value.trim())}</div></article>`;}).join("") || empty($("#chatSearch").value?"No messages match":$("#ctxOnly").checked?"No active messages":"A fresh page",$("#chatSearch").value?"Try a shorter phrase or load a different conversation.":$("#ctxOnly").checked?"Turn off In context to view stored history.":"Messages in this chat will appear here.","▤");
  log.querySelectorAll("[data-copy]").forEach(b => {b.onclick=() => busy(b,async () => {const m=state.messages.find(x => x.id===Number(b.dataset.copy));await copyText(m?.text || "");notice("Message copied.");});});
  if(options.older)log.scrollTop=oldTop+(log.scrollHeight-oldHeight);else if(options.keepScroll && !nearBottom)log.scrollTop=oldTop;else log.scrollTop=log.scrollHeight;
  $("#chatOlder").hidden=!state.hasMore;$("#chatMessageCount").textContent=number(messages.length)+($("#chatSearch").value?" matching":" loaded")+(state.hasMore?" · earlier messages available":"");
  $("#userActions").innerHTML=`<button data-a="${u.blocked?"unblock":"block"}" ${u.blocked?'class="good"':""}>${u.blocked?"Unblock person":"Block person"}</button><button data-a="${u.role==="admin"?"revoke-admin":"grant-admin"}">${u.role==="admin"?"Revoke admin":"Grant admin"}</button><button data-a="reset-context">Reset chat context</button><button data-a="cancel">Cancel chat replies</button><button data-a="retry">Retry failed reply</button><button data-a="delete" class="danger">Delete person</button>`;
  $("#userActions").querySelectorAll("[data-a]").forEach(b => {b.onclick=() => userAction(b,b.dataset.a);});
}
function formatMessage(text,search="") {
  const safe=esc(text);
  if(search){const needle=esc(search).replace(/[.*+?^${}()|[\]\\]/g,"\\$&");return safe.replace(new RegExp(needle,"gi"),match => "<mark>"+match+"</mark>");}
  // Deliberately limited formatting: code and emphasis, without executable HTML.
  return safe.split(/(```[\s\S]*?```)/g).map(part => part.startsWith("```")?"<pre><code>"+part.slice(3,-3).replace(/^\w+\n/,"")+"</code></pre>":part.replace(/`([^`\n]+)`/g,"<code>$1</code>").replace(/\*\*([^*\n]+)\*\*/g,"<strong>$1</strong>")).join("");
}
async function copyText(text) {if(navigator.clipboard?.writeText)return navigator.clipboard.writeText(text);const el=document.createElement("textarea");el.value=text;el.style.position="fixed";el.style.opacity="0";document.body.append(el);el.select();const copied=document.execCommand("copy");el.remove();if(!copied)throw new Error("Your browser could not copy this message.");}
function renderContext() {
  const c=state.context.context,limits=state.context.limits || {},cap=Number(limits.max_tokens || 0),pct=cap?Math.min(100,c.est_tokens/cap*100):0;
  const trimmed=(c.content_trimmed_ids || []).length,trimNote=c.system_truncated || trimmed?`<div class="context-trim-note">${c.system_truncated?"System instructions shortened to fit the context budget. ":""}${trimmed?number(trimmed)+" message"+(trimmed===1?"":"s")+" shortened for the model. Full text remains in history.":""}</div>`:"";
  $("#ctxPanel").innerHTML=`<div class="panel-heading"><h2>Model context</h2><span class="badge">INSPECTOR</span></div><div class="context-summary"><b>~${number(c.est_tokens)}</b><span class="dim"> estimated tokens</span><div class="meter"><span style="width:${pct}%"></span></div><div class="dim">${number(cap)} token cap · ${number(c.included_ids.length)} messages included</div></div><p class="dim" style="font-size:11px">${number(c.eligible)} eligible · ${number(c.dropped_for_count)} outside history limit · ${number(c.dropped_for_tokens)} outside token cap</p>${trimNote}<details class="ctx-details"><summary>Effective system prompt</summary><div class="sys">${esc(c.system)}</div></details><details class="ctx-details"><summary>Messages sent to the model (${c.messages.length})</summary>${c.messages.map(m => `<div class="ctxmsg ${m.role=== "assistant"?"assistant":"user"}"><b>${esc(m.role)}</b><br>${esc(m.content.slice(0,1500))}${m.content.length>1500?" … (inspector preview shortened)":""}</div>`).join("")}</details>`;
  $("#chatInspector").hidden=!$("#showCtx").checked;$("#chatGrid").classList.toggle("no-inspector",!$("#showCtx").checked);
}
function restoreDrafts(u) {
  const chat=activeChat(),draft=state.drafts.get(draftKey()),pd=state.personDrafts.get(state.selected);
  $("#adminSendText").value=draft?.send || "";$("#chatPrompt").value=draft?.prompt ?? chat?.prompt ?? "";
  $("#userNote").value=pd?.notes ?? u.notes ?? "";$("#ovrCtx").value=pd?.limit ?? u.context_limit_override ?? "";$("#ovrSys").value=pd?.system ?? u.system_prompt_override ?? "";
  $("#chatPromptState").textContent=chat?.prompt?"CUSTOM":"DEFAULT";updateDraftIndicator();
}
function updateDraftIndicator() {const draft=state.drafts.get(draftKey());$("#chatDraftState").hidden=!(draft?.send || draft?.promptDirty || state.personDrafts.has(state.selected));$("#sendCount").textContent=number($("#adminSendText").value.length)+" characters";}
function rememberChatDraft() {if(!state.selected)return;const prior=state.drafts.get(draftKey()) || {};state.drafts.set(draftKey(),{...prior,send:$("#adminSendText").value,prompt:$("#chatPrompt").value,promptDirty:$("#chatPrompt").value!==(activeChat()?.prompt || "")});updateDraftIndicator();}
["adminSendText","chatPrompt"].forEach(id => {$("#"+id).oninput=rememberChatDraft;});
["userNote","ovrCtx","ovrSys"].forEach(id => {$("#"+id).oninput=() => {if(!state.selected)return;state.personDrafts.set(state.selected,{notes:$("#userNote").value,limit:$("#ovrCtx").value,system:$("#ovrSys").value});updateDraftIndicator();};});
$("#chatRefresh").onclick=() => busy($("#chatRefresh"),() => loadChat({keepScroll:true}));$("#chatOlder").onclick=() => busy($("#chatOlder"),() => loadChat({older:true}));$("#chatLatest").onclick=() => {$("#chatLog").scrollTop=$("#chatLog").scrollHeight;};
$("#chatSearch").oninput=debounce(() => loadChat()); $("#ctxOnly").onchange=() => {if(state.currentUser)renderChat();};$("#showCtx").onchange=() => {if(state.context)renderContext();};
async function userAction(button,action) {
  const uid=state.selected,scope=scopeBody();
  if(action==="delete" && !confirm("Delete this person and all their stored conversations, notes, and reply jobs? This cannot be undone."))return;
  if(action==="reset-context" && !confirm("Start fresh model context for this chat or topic? Stored history remains viewable; pending replies here are cancelled."))return;
  return busy(button,async () => {await post(`/api/users/${uid}/${action}`,scope);notice(action==="reset-context"?"Chat context reset. Stored history is still available.":action==="retry"?"Latest failed reply queued for retry.":action==="cancel"?"Pending replies in this chat cancelled.":"Person updated.");if(action==="delete"){state.personDrafts.delete(uid);for(const key of state.drafts.keys())if(key.startsWith(uid+":"))state.drafts.delete(key);if(state.selected===uid){state.selected=null;state.chat=null;}}if(state.selected===uid || action==="delete")await loadChat({keepScroll:true});await refreshStatus();});
}
$("#adminSend").onclick=() => {const uid=state.selected,scope=scopeBody(),key=draftKey(),text=$("#adminSendText").value.trim();if(!uid || !text){notice("Write a message first.");return;}return busy($("#adminSend"),async () => {await post(`/api/users/${uid}/send`,{...scope,text});const draft=state.drafts.get(key);if(draft?.send.trim()===text)state.drafts.set(key,{...draft,send:""});notice("Message delivered.");if(state.selected===uid && scope.chat_id===state.chat && scope.thread_id===state.thread)await loadChat({keepScroll:true});});};
$("#saveNote").onclick=() => {const uid=state.selected,key=uid,values={notes:$("#userNote").value,context_limit_override:$("#ovrCtx").value===""?null:Number($("#ovrCtx").value),system_prompt_override:$("#ovrSys").value};if(values.context_limit_override!==null && (!Number.isInteger(values.context_limit_override) || values.context_limit_override<1 || values.context_limit_override>500)){notice("Context limit must be a whole number from 1 to 500, or blank for the global default.",true);return;}return busy($("#saveNote"),async () => {await post(`/api/users/${uid}/profile`,values);const draft=state.personDrafts.get(key);if(draft && draft.notes===values.notes && draft.system===values.system_prompt_override && Number(draft.limit || 0)===Number(values.context_limit_override || 0))state.personDrafts.delete(key);notice("Private notes and person defaults saved.");if(state.selected===uid)await loadChat({keepScroll:true});});};
async function savePrompt(button,prompt) {const uid=state.selected,scope=scopeBody(),key=draftKey();return busy(button,async () => {await post(`/api/users/${uid}/chat-prompt`,{...scope,prompt});const draft=state.drafts.get(key);if(draft?.prompt===prompt)state.drafts.set(key,{...draft,promptDirty:false});notice(prompt?"Chat instructions saved.":"Chat now uses its default instructions.");if(state.selected===uid && scope.chat_id===state.chat && scope.thread_id===state.thread){const current=state.drafts.get(key);if(current && !current.promptDirty)state.drafts.set(key,{...current,prompt});await loadChat({keepScroll:true});}});}
$("#saveChatPrompt").onclick=() => savePrompt($("#saveChatPrompt"),$("#chatPrompt").value);
$("#chatPromptReset").onclick=() => {$("#chatPrompt").value="";rememberChatDraft();return savePrompt($("#chatPromptReset"),"");};
$("#chatBookmark").onclick=() => {const uid=state.selected,scope=scopeBody(),bookmark=!activeChat()?.bookmarked;return busy($("#chatBookmark"),async () => {await post(`/api/users/${uid}/chat-meta`,{...scope,bookmarked:bookmark});notice(bookmark?"Conversation bookmarked. It will appear first in the chat selector.":"Bookmark removed.");if(state.selected===uid)await loadChat({keepScroll:true});});};
let renameTarget;
$("#chatRename").onclick=() => {renameTarget={uid:state.selected,...scopeBody()};$("#chatLabelInput").value=activeChat()?.label || "";$("#editDialog").showModal();$("#chatLabelInput").focus();};
$("#editForm").onsubmit=e => {e.preventDefault();const target=renameTarget,label=$("#chatLabelInput").value.trim();busy($("#chatLabelSave"),async () => {await post(`/api/users/${target.uid}/chat-meta`,{chat_id:target.chat_id,thread_id:target.thread_id,label});$("#editDialog").close();notice("Conversation label saved.");if(state.selected===target.uid)await loadChat({keepScroll:true});});};
async function exportChat(button,markdown=false) {const uid=state.selected,chat=state.chat,thread=state.thread;return busy(button,async () => {const data=await api(`/api/users/${uid}/export?${scopeQuery(chat,thread)}`);const body=markdown?`# Conversation with ${uname(data.user)}\n\nChat: ${chat}${thread?" · Topic "+thread:""}\nExported: ${data.exported_at || new Date().toISOString()}\n${data.truncated?"\nThis export contains the most recent "+data.exported+" of "+data.total+" messages.\n":""}\n`+data.messages.map(m => `## ${m.direction==="in"?"Person":"Bot"} · ${tstr(m.created_at)}\n\n${m.text}\n`).join("\n"):JSON.stringify(data,null,2);const url=URL.createObjectURL(new Blob([body],{type:markdown?"text/markdown;charset=utf-8":"application/json"}));const link=document.createElement("a");link.href=url;link.download=`conversation-${uid}-${chat}${thread?"-topic-"+thread:""}.${markdown?"md":"json"}`;link.click();setTimeout(() => URL.revokeObjectURL(url),1000);notice(data.truncated?`Exported the latest ${number(data.exported)} of ${number(data.total)} messages. Older history remains stored.`:"Conversation exported.");});}
$("#chatExport").onclick=() => exportChat($("#chatExport"));$("#chatExportMd").onclick=() => exportChat($("#chatExportMd"),true);

/* Prompt starters are editable before applying. Saved starters come from the DB. */
const starters=[{id:"builtin-tutor",name:"Patient tutor",prompt:"You are a patient tutor. Explain one idea at a time with a short example. Ask a focused question when it helps check understanding. Keep explanations clear and conversational."},{id:"builtin-concise",name:"Concise assistant",prompt:"Be direct, practical, and concise. Lead with the answer. Use a short list only when it makes the response easier to follow."},{id:"builtin-code",name:"Coding partner",prompt:"You are a thoughtful coding partner. Ask for missing requirements when necessary, explain the reason for a change, and provide runnable examples. Be honest about assumptions and untested code."},{id:"builtin-creative",name:"Creative collaborator",prompt:"You are a creative collaborator. Offer concrete, varied ideas and help develop the promising ones. Match the person's tone and keep suggestions specific."}];
async function loadPresets() {const selected=$("#promptPreset").value;try{const {presets}=await api("/api/presets");state.presets=[...presets,...starters];}catch(_){state.presets=starters;}$("#promptPreset").innerHTML='<option value="">Choose a starter…</option>'+state.presets.map(p => `<option value="${esc(p.id)}">${esc(p.name)}</option>`).join("");$("#promptPreset").value=selected;updateStarterActions();}
$("#applyPromptPreset").onclick=() => {const preset=state.presets.find(p => String(p.id)===$("#promptPreset").value);if(!preset){notice("Choose a prompt starter first.");return;}$("#chatPrompt").value=preset.prompt;rememberChatDraft();notice("Starter added to the editor. Save instructions to apply it.");};
let starterTarget=null;
function updateStarterActions() {const selected=$("#promptPreset").value;$("#editPromptPreset").hidden=!selected || selected.startsWith("builtin-");}
$("#promptPreset").onchange=updateStarterActions;
$("#savePromptPreset").onclick=() => {const prompt=$("#chatPrompt").value.trim();if(!prompt){notice("Add some chat instructions first, then save them as a starter.");$("#chatPrompt").focus();return;}starterTarget=null;$("#starterTitle").textContent="Save prompt starter";$("#starterName").value="";$("#starterPrompt").value=prompt;$("#starterDialog").showModal();$("#starterName").focus();};
$("#editPromptPreset").onclick=() => {const preset=state.presets.find(p => String(p.id)===$("#promptPreset").value);if(!preset || String(preset.id).startsWith("builtin-"))return;starterTarget=preset.id;$("#starterTitle").textContent="Edit prompt starter";$("#starterName").value=preset.name;$("#starterPrompt").value=preset.prompt;$("#starterDialog").showModal();$("#starterName").focus();};
$("#starterForm").onsubmit=e => {e.preventDefault();const name=$("#starterName").value.trim(),prompt=$("#starterPrompt").value.trim(),id=starterTarget;if(!name || !prompt)return;busy($("#starterSave"),async () => {const result=await post("/api/presets",{...(id!==null?{id}:{}),name,prompt});$("#starterDialog").close();await loadPresets();if(result.preset?.id)$("#promptPreset").value=String(result.preset.id);updateStarterActions();notice("Prompt starter saved. Choose Use to add it to the chat editor.");});};

/* Durable queue visibility. Cancellation/retry applies to a chat, not one row. */
const jobLabel={queued:"Queued",processing:"Generating",delivering:"Delivering",done:"Delivered",failed:"Failed",interrupted:"Interrupted",cancelled:"Cancelled"};
async function loadJobs() {
  const key=++state.jobsRequest;await loadPeople();const query=new URLSearchParams({status:$("#queueStatus").value,limit:100});if($("#queueUser").value)query.set("user_id",$("#queueUser").value);
  const {jobs,counts}=await api("/api/jobs?"+query);if(key!==state.jobsRequest)return;
  const active=Number(counts.processing || 0)+Number(counts.delivering || 0),failed=Number(counts.failed || 0)+Number(counts.interrupted || 0);
  $("#queueCards").innerHTML=card("Waiting",number(counts.queued),"Persisted and ready to run","≋")+card("In progress",number(active),"Generating or delivering","↗")+card("Delivered",number(counts.done),"Replies completed","✓","healthy")+card("Needs attention",number(failed),"Failed or interrupted replies","!",failed?"failed":"");
  $("#queueUpdated").textContent=`${jobs.length} replies shown · refreshed ${new Date().toLocaleTimeString([], {hour:"numeric",minute:"2-digit"})}`;
  $("#jobsList").innerHTML=jobs.length?jobs.map(j => `<article class="queue-item"><div><div class="queue-heading"><span class="badge ${j.status==="failed"?"error":j.status==="processing" || j.status==="delivering"?"running":esc(j.status)}">${esc(jobLabel[j.status] || j.status)}</span><b>${esc(uname(j))}</b><span class="mono dim">#${j.id}</span></div><div class="queue-text">${esc(j.text.slice(0,260))}${j.text.length>260?"…":""}</div><div class="queue-meta"><span>${esc(tstr(j.created_at))}</span><span>Chat ${j.chat_id}${j.thread_id?" · Topic "+j.thread_id:""}</span><span>Updated ${esc(reltime(j.updated_at))}</span></div></div><div class="btnrow"><button class="quiet" data-open="${j.user_id}" data-chat="${j.chat_id}" data-thread="${j.thread_id || 0}">Conversation →</button>${["failed","interrupted"].includes(j.status)?`<button data-job-action="retry" data-user="${j.user_id}" data-chat="${j.chat_id}" data-thread="${j.thread_id || 0}" title="Retry the latest failed reply in this chat">Retry chat</button>`:["queued","processing","delivering"].includes(j.status)?`<button data-job-action="cancel" data-user="${j.user_id}" data-chat="${j.chat_id}" data-thread="${j.thread_id || 0}" title="Cancel all pending replies in this chat">Cancel chat</button>`:""}</div></article>`).join(""):empty("Nothing in this view",$("#queueStatus").value?"Try another status or person to see more replies.":"Incoming model requests will appear here, with their delivery status.","≋");
  bindOpen($("#jobsList"));$("#jobsList").querySelectorAll("[data-job-action]").forEach(b => {b.onclick=() => busy(b,async () => {await post(`/api/users/${b.dataset.user}/${b.dataset.jobAction}`,{chat_id:Number(b.dataset.chat),thread_id:Number(b.dataset.thread)});notice(b.dataset.jobAction==="retry"?"Latest failed reply in this chat queued for retry.":"Pending replies in this chat cancelled.");await loadJobs();});});
}
$("#queueRefresh").onclick=() => busy($("#queueRefresh"),loadJobs);$("#queueStatus").onchange=loadJobs;$("#queueUser").onchange=loadJobs;

/* Settings retain their DOM and edits when navigating or refreshing. */
const settingSections={telegram:["Telegram","Connect your bot and choose how it responds on Telegram."],llm:["Model connection","Choose the endpoint and model used for replies."],generation:["Replies & personality","Tune the response style and the bot's default instructions."],context:["Conversation memory","Balance useful recent context with the model's capacity."],behavior:["Access & workload","Control who can use the bot and how requests are queued."]};
let settingsPromise;
async function loadSettings(force=false) {if(state.settingsLoaded && !force)return;if(settingsPromise)return settingsPromise;const controls=$$("#settingsForm [data-k]");if(force)controls.forEach(el => {el.disabled=true;});settingsPromise=populateSettings();try{return await settingsPromise;}finally{settingsPromise=null;controls.forEach(el => {el.disabled=false;});}}
async function populateSettings() {
  const {settings}=await api("/api/settings");state.settings=settings;state.settingsLoaded=true;state.settingsDirty=false;
  const groups={};Object.values(settings).forEach(s => {(groups[s.section] ||= []).push(s);});const order=[...Object.keys(settingSections),...Object.keys(groups).filter(g => !(g in settingSections))].filter(g => groups[g]);
  $("#settingsNav").innerHTML=order.map((g,i) => `<button data-section="${esc(g)}" class="${i===0?"active":""}">${esc(settingSections[g]?.[0] || g)}</button>`).join("");
  $("#settingsForm").innerHTML=order.map(g => `<section class="setgroup" id="settings-${esc(g)}"><h2>${esc(settingSections[g]?.[0] || g)}</h2><p class="section-description">${esc(settingSections[g]?.[1] || "Additional bot configuration.")}</p>${g==="generation"?'<div class="generation-presets"><span>Reply style starter</span><button data-generation-preset="focused">Focused</button><button data-generation-preset="balanced">Balanced</button><button data-generation-preset="creative">Creative</button></div>':""}${groups[g].map(s => {let input;const id="setting-"+s.key;if(s.kind==="bool")input=`<input id="${id}" type="checkbox" data-k="${s.key}" ${["1","true"].includes(String(s.value).toLowerCase())?"checked":""}>`;else if(s.kind==="textarea")input=`<textarea id="${id}" data-k="${s.key}" rows="4">${esc(s.value)}</textarea>`;else if(s.kind==="secret")input=`<input id="${id}" data-k="${s.key}" type="password" placeholder="${s.set?"Configured · blank keeps existing secret":"Not configured"}" autocomplete="new-password">`;else if(s.kind==="choice")input=`<select id="${id}" data-k="${s.key}">${s.choices.map(c => `<option value="${esc(c)}" ${c===s.value?"selected":""}>${esc(c)}</option>`).join("")}</select>`;else input=`<input id="${id}" data-k="${s.key}" value="${esc(s.value)}" ${s.kind==="int" || s.kind==="float"?`type="number" step="${s.kind==="int"?"1":"any"}"`:""}>`;return `<div class="setrow ${s.kind==="bool"?"boolean":""}" data-search="${esc((s.label+" "+s.help+" "+s.key).toLowerCase())}"><label for="${id}">${esc(s.label)}<span class="help" id="help-${s.key}">${esc(s.help || (s.kind==="secret"?"Leave blank to keep the configured value.":""))}</span></label><div>${input.replace(" data-k",` aria-describedby="help-${s.key}" data-k`)}</div></div>`;}).join("")}</section>`).join("");
  $("#settingsNav").querySelectorAll("button").forEach(b => {b.onclick=() => {$$("#settingsNav button").forEach(x => x.classList.toggle("active",x===b));$("#settings-"+b.dataset.section).scrollIntoView({behavior:matchMedia("(prefers-reduced-motion: reduce)").matches?"instant":"smooth",block:"start"});};});
  $("#settingsForm").querySelectorAll("[data-k]").forEach(el => {el.oninput=updateSettingsDirty;el.onchange=updateSettingsDirty;});
  $("#settingsForm").querySelectorAll("[data-generation-preset]").forEach(b => {b.onclick=() => {const presets={focused:{gen_temperature:"0.25",gen_top_p:"0.85"},balanced:{gen_temperature:"0.7",gen_top_p:"0.9"},creative:{gen_temperature:"0.95",gen_top_p:"0.95"}};Object.entries(presets[b.dataset.generationPreset]).forEach(([k,v]) => {const el=$("#setting-"+k);if(el)el.value=v;});updateSettingsDirty();notice("Generation starter loaded. Save changes to apply it.");};});
  updateSettingsDirty();filterSettings();
}
function settingValue(el) {return el.type==="checkbox"?(el.checked?"true":"false"):el.value;}
function updateSettingsDirty() {let changed=0;$$("#settingsForm [data-k]").forEach(el => {const s=state.settings[el.dataset.k],original=s.kind==="bool"?(["1","true"].includes(String(s.value).toLowerCase())?"true":"false"):String(s.value ?? "");const modified=settingValue(el)!==original;el.closest(".setrow").dataset.modified=modified?"true":"false";if(modified)changed++;});state.settingsDirty=changed>0;$("#settingsDirty").hidden=!changed;$("#settingsDirty").textContent=changed+" unsaved change"+(changed===1?"":"s");$("#setSave").disabled=!changed;$("#setDiscard").disabled=!changed;}
function filterSettings() {const search=$("#settingsSearch").value.toLowerCase().trim();$$(".setgroup").forEach(group => {let shown=0;group.querySelectorAll(".setrow").forEach(row => {row.hidden=search && !row.dataset.search.includes(search);if(!row.hidden)shown++;});group.hidden=!shown;});}
$("#settingsSearch").oninput=filterSettings;
$("#setDiscard").onclick=() => {if(state.settingsDirty && !confirm("Discard all unsaved setting edits?"))return;busy($("#setDiscard"),() => loadSettings(true));};
$("#setSave").onclick=() => busy($("#setSave"),async () => {const payload={};const controls=$$("#settingsForm [data-k]");controls.forEach(el => {if(el.closest(".setrow").dataset.modified==="true")payload[el.dataset.k]=settingValue(el);el.disabled=true;});try{const result=await post("/api/settings",{settings:payload});state.settings=result.settings || state.settings;state.settingsLoaded=false;await loadSettings(true);$("#setMsg").textContent=`${result.updated.length} change${result.updated.length===1?"":"s"} saved`;notice("Bot settings saved.");await refreshStatus();}finally{controls.forEach(el => {el.disabled=false;});updateSettingsDirty();}});

/* Events and insights. */
async function loadEvents() {const key=++state.eventsRequest;const {events}=await api("/api/events?"+new URLSearchParams({level:$("#evLevel").value,limit:300}));if(key!==state.eventsRequest)return;state.events=events;renderEvents();}
function renderEvents() {const query=$("#evSearch").value.toLowerCase().trim(),events=state.events.filter(e => !query || [e.message,e.source,e.level].some(v => String(v).toLowerCase().includes(query)));$("#evCount").textContent=events.length+" events";$("#evTable tbody").innerHTML=events.length?events.map(e => `<tr><td class="event-time" title="${esc(e.created_at)}">${esc(tstr(e.created_at))}</td><td><span class="badge ${e.level==="error"?"error":e.level==="warning"?"warning":"ok"}">${esc(e.level)}</span></td><td class="mono dim">${esc(e.source)}</td><td>${esc(e.message)}</td></tr>`).join(""):`<tr><td colspan="4">${empty("No events match","Try another level or search phrase.","↗")}</td></tr>`;}
$("#evRefresh").onclick=() => busy($("#evRefresh"),loadEvents);$("#evLevel").onchange=loadEvents;$("#evSearch").oninput=renderEvents;
async function loadStats() {
  const s=await api("/api/stats"),counts=s.counts || {},jobs=s.jobs || {},finished=Number(jobs.done || 0),failures=Number(jobs.failed || 0)+Number(jobs.interrupted || 0);
  $("#statCards").innerHTML=card("People",number(counts.users),number(s.conversation_count)+" separate chats and topics","♧")+card("Stored messages",number(counts.messages),number(counts.messages_today)+" today","▤")+card("Replies delivered",number(finished),"Completed generation jobs","✓","healthy")+card("Delivery success",finished+failures?Math.round(finished/(finished+failures)*100)+"%":"—",number(failures)+" failed or interrupted","↗");
  const daily=new Map((s.daily || []).map(d => [d.day,d.n])),days=[];
  for(let i=29;i>=0;i--){const date=new Date();date.setUTCDate(date.getUTCDate()-i);const day=date.toISOString().slice(0,10);days.push({day,n:Number(daily.get(day) || 0)});}
  const max=Math.max(1,...days.map(d => d.n)),total=days.reduce((n,d) => n+d.n,0);
  $("#bars").innerHTML=days.map(d => `<div class="bar" style="height:${Math.max(1,d.n/max*100)}%" tabindex="0" aria-label="${d.day}: ${d.n} messages"><span class="bar-tip">${d.day} · ${number(d.n)}</span></div>`).join("");
  $("#bars").setAttribute("aria-label",`Messages per day over the last 30 days: ${number(total)} total. Peak ${number(max)}.`);$("#chartSummary").textContent=number(total)+" messages";$("#chartStart").textContent=new Date(days[0].day+"T12:00:00").toLocaleDateString([], {month:"short",day:"numeric"});$("#chartEnd").textContent="Today";
  const top=s.top_users || [],peak=Math.max(1,...top.map(t => t.n));
  $("#topUsers").innerHTML=top.length?top.map((t,i) => {const u=state.users.find(u => u.id===t.user_id) || {id:t.user_id};return `<div class="rank-row"><span class="rank-index">${String(i+1).padStart(2,"0")}</span>${person(u)}<div class="rank-meter"><span style="width:${t.n/peak*100}%"></span></div><span class="rank-count">${number(t.n)} msgs</span><button data-open="${t.user_id}" class="text-button" aria-label="Open ${esc(uname(u))}'s conversation">→</button></div>`;}).join(""):empty("Your story is just beginning","Activity insights will grow as people talk to your bot.","▥");bindOpen($("#topUsers"));
}

/* A lightweight keyboard palette; native dialogs provide the focus trap. */
const commands=[...Object.entries(titles).map(([tab,title]) => ({name:title,detail:"Open page",run:() => goto(tab)})),{name:"Refresh this page",detail:"Action",run:refreshPage},{name:"Search conversations",detail:"Action",run:() => {goto("chat");$("#chatSearch").focus();}},{name:"Change color theme",detail:"Appearance",run:() => $("#themeToggle").click()}];
let commandIndex=0,visibleCommands=[];
function renderCommands() {const search=$("#commandSearch").value.toLowerCase().trim();visibleCommands=commands.filter(c => c.name.toLowerCase().includes(search));commandIndex=Math.min(commandIndex,Math.max(0,visibleCommands.length-1));$("#commandResults").innerHTML=visibleCommands.map((c,i) => `<button class="${i===commandIndex?"active":""}" data-command="${i}"><span>${esc(c.name)}</span><small>${esc(c.detail)}</small></button>`).join("") || empty("No matching commands","Try a page name or refresh.","⌕");$("#commandResults").querySelectorAll("button").forEach(b => {b.onclick=() => runCommand(Number(b.dataset.command));});}
function runCommand(index) {$("#commandDialog").close();visibleCommands[index]?.run();}
$("#commandOpen").onclick=() => {$("#commandSearch").value="";commandIndex=0;renderCommands();$("#commandDialog").showModal();$("#commandSearch").focus();};
$("#commandSearch").oninput=() => {commandIndex=0;renderCommands();};
$("#commandSearch").onkeydown=e => {if(["ArrowDown","ArrowUp"].includes(e.key)){e.preventDefault();commandIndex=(commandIndex+(e.key==="ArrowDown"?1:-1)+visibleCommands.length)%Math.max(1,visibleCommands.length);renderCommands();$("#commandResults button.active")?.scrollIntoView({block:"nearest"});}if(e.key==="Enter"){e.preventDefault();runCommand(commandIndex);}};
$$("[data-close-dialog]").forEach(b => {b.onclick=() => $("#"+b.dataset.closeDialog).close();});
$$("dialog").forEach(dialog => {dialog.addEventListener("click",e => {if(e.target===dialog){const r=dialog.getBoundingClientRect();if(e.clientX<r.left || e.clientX>r.right || e.clientY<r.top || e.clientY>r.bottom)dialog.close();}});});
document.addEventListener("keydown",e => {if((e.ctrlKey || e.metaKey) && e.key.toLowerCase()==="k"){e.preventDefault();$("#commandOpen").click();}if(e.key==="/" && !e.ctrlKey && !e.metaKey && !e.target.matches("input,textarea,select") && !$("dialog[open]")){const input=state.activeTab==="chat"?$("#chatSearch"):state.activeTab==="users"?$("#userSearch"):state.activeTab==="activity"?$("#evSearch"):null;if(input){e.preventDefault();input.focus();}}});
window.addEventListener("beforeunload",e => {if(state.settingsDirty || [...state.drafts.values()].some(d => d.send || d.promptDirty) || state.personDrafts.size){e.preventDefault();e.returnValue="";}});

/* Boot and quiet refreshes. Draft editors are restored from memory per scope. */
loadPresets();refreshStatus();goto(location.hash.slice(1) || "overview",false);
setInterval(refreshStatus,6000);
setInterval(() => {if(document.hidden || $("dialog[open]") || document.activeElement?.matches("input,textarea,select"))return;if(state.activeTab==="chat")loadChat({keepScroll:true});if(state.activeTab==="queue")loadJobs();if(state.activeTab==="activity")loadEvents();},10000);
document.addEventListener("visibilitychange",() => {if(!document.hidden)refreshStatus();});
