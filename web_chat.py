"""
web_chat.py
===========
A tiny ChatGPT/Claude-style web UI for a trained latent-thought checkpoint.
Pure Python stdlib (http.server) -- no Flask/extra deps; torch is the only heavy
import. Loads ONE checkpoint and serves a single-page app that talks to it.

  python web_chat.py                        # asks for the checkpoint path, then serves
  python web_chat.py runs/scaled/model.pt   # load directly
  python web_chat.py --port 8100 <ckpt>

Then open http://127.0.0.1:8000 . The page is a chat interface with a debug
sidebar: chunk-border visualization toggle, per-chunk "thought" pills, an input-
segmentation view, a Score (perplexity) mode, and temperature / #chunks dials.
You can also load a different checkpoint from the UI.

Inference runs on CPU (matching generate.py) and is serialized by a lock, so this
is a single-user local tool, not a production server.
"""
import os, sys, json, argparse, threading, html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import chat_core

STATE = {"model": None, "adapter": None, "session": None,
         "chunker": None, "cfg": None, "meta": None, "path": None}
LOCK = threading.Lock()


def do_load(path):
    # Load the model AND a DialogueAdapter + a fresh DialogueSession, so the "Chat"
    # mode's full Stage-F serving (input lane + response seed + cross-turn memory)
    # is available. On a plain A→E checkpoint the adapter is zero-init (untrained
    # dialogue) but the Generate/Score modes are unaffected.
    model, adapter, chunker, cfg, ckpt = chat_core.load_dialogue_checkpoint(path)
    session = chat_core.new_dialogue_session(model, adapter, chunker, cfg)
    STATE.update(model=model, adapter=adapter, session=session, chunker=chunker, cfg=cfg,
                 meta=chat_core.ckpt_summary(cfg, ckpt),
                 path=os.path.abspath(os.path.expanduser(path.strip())))


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Latent-Thought Chat</title>
<style>
  :root{
    --bg:#1a1a19; --panel:#232320; --panel2:#2b2b28; --line:#3a3a36;
    --text:#ecece8; --muted:#9a9a92; --accent:#d97757; --accent2:#5b8def;
    --user:#31302c; --chunk:#3a3730; --chunkbd:#6b5a45;
  }
  *{box-sizing:border-box}
  body{margin:0;height:100vh;display:flex;font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
       background:var(--bg);color:var(--text)}
  /* sidebar */
  aside{width:290px;flex:none;background:var(--panel);border-right:1px solid var(--line);
        display:flex;flex-direction:column;padding:16px;gap:14px;overflow-y:auto}
  aside h1{font-size:15px;margin:0 0 2px;font-weight:600}
  aside .sub{color:var(--muted);font-size:12px;margin-top:-6px}
  .card{background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:12px;display:flex;flex-direction:column;gap:10px}
  .card h2{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin:0}
  label.row{display:flex;align-items:center;justify-content:space-between;gap:8px;cursor:pointer}
  input[type=text]{width:100%;background:#191917;border:1px solid var(--line);border-radius:8px;color:var(--text);padding:8px}
  input[type=range]{width:120px}
  .switch{position:relative;width:38px;height:22px;flex:none}
  .switch input{opacity:0;width:0;height:0}
  .slider{position:absolute;inset:0;background:#4a4a44;border-radius:22px;transition:.15s}
  .slider:before{content:"";position:absolute;height:16px;width:16px;left:3px;top:3px;background:#fff;border-radius:50%;transition:.15s}
  .switch input:checked+.slider{background:var(--accent)}
  .switch input:checked+.slider:before{transform:translateX(16px)}
  .seg{display:flex;background:#191917;border:1px solid var(--line);border-radius:8px;overflow:hidden}
  .seg button{flex:1;background:none;border:0;color:var(--muted);padding:7px;cursor:pointer;font:inherit}
  .seg button.on{background:var(--accent);color:#fff}
  .meta{font-size:12px;color:var(--muted);white-space:pre-wrap;word-break:break-word}
  button.btn{background:var(--panel2);border:1px solid var(--line);border-radius:8px;color:var(--text);padding:8px;cursor:pointer;font:inherit}
  button.btn:hover{border-color:var(--accent)}
  /* main */
  main{flex:1;display:flex;flex-direction:column;min-width:0}
  #thread{flex:1;overflow-y:auto;padding:26px 0}
  .wrap{max-width:760px;margin:0 auto;padding:0 22px}
  .msg{display:flex;gap:12px;margin:0 0 22px}
  .msg .who{width:26px;height:26px;border-radius:6px;flex:none;display:grid;place-items:center;font-size:12px;font-weight:700;color:#fff}
  .msg.user .who{background:var(--accent2)} .msg.bot .who{background:var(--accent)}
  .bubble{background:transparent;padding-top:2px;min-width:0}
  .msg.user .bubble{color:var(--text)}
  .txt{white-space:pre-wrap;word-wrap:break-word}
  .pills{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
  .pill{background:var(--chunk);border:1px solid var(--chunkbd);border-radius:7px;padding:3px 9px;font-size:13px}
  .pill .i{color:var(--muted);font-size:10px;margin-right:5px}
  .dbg{margin-top:8px;font-size:12px;color:var(--muted)}
  .dbg .read{margin-top:4px}
  .dbg .read .pill{background:#2a2a26;border-color:var(--line)}
  .score{display:inline-block;margin-top:6px;background:#2a2a26;border:1px solid var(--line);border-radius:7px;padding:4px 10px;font-variant-numeric:tabular-nums}
  /* composer */
  .composer{border-top:1px solid var(--line);background:var(--bg)}
  .composer .wrap{padding:14px 22px 18px;display:flex;gap:10px;align-items:flex-end}
  textarea{flex:1;resize:none;max-height:180px;background:var(--panel2);border:1px solid var(--line);border-radius:12px;
           color:var(--text);padding:12px 14px;font:inherit}
  textarea:focus{outline:none;border-color:var(--accent)}
  .send{background:var(--accent);border:0;color:#fff;border-radius:10px;width:42px;height:42px;font-size:18px;cursor:pointer;flex:none}
  .send:disabled{opacity:.5;cursor:default}
  .hint{color:var(--muted);font-size:11px;text-align:center;padding-bottom:8px}
  .empty{color:var(--muted);text-align:center;margin-top:12vh}
  .empty b{color:var(--text)}
</style></head>
<body>
<aside>
  <div><h1>Latent-Thought Chat</h1><div class="sub">chunk-level "thoughts", one Talker-decoded span each</div></div>

  <div class="card">
    <h2>Mode</h2>
    <div class="seg" id="mode">
      <button data-mode="generate" class="on">Generate</button>
      <button data-mode="dialogue">Chat</button>
      <button data-mode="score">Score</button>
    </div>
    <div class="sub" style="margin-top:2px">Chat = Stage-F two-lane serving (cross-turn memory); Clear resets it.</div>
  </div>

  <div class="card">
    <h2>Debug tools</h2>
    <label class="row">Chunk visualization
      <span class="switch"><input type="checkbox" id="viz" checked><span class="slider"></span></span></label>
    <label class="row">Show input segmentation
      <span class="switch"><input type="checkbox" id="showread" checked><span class="slider"></span></span></label>
    <label class="row">Per-message perplexity
      <span class="switch"><input type="checkbox" id="showppl"><span class="slider"></span></span></label>
  </div>

  <div class="card">
    <h2>Sampling</h2>
    <label class="row">Temperature <span id="tempv">0.90</span></label>
    <input type="range" id="temp" min="0.1" max="1.5" step="0.05" value="0.9" style="width:100%">
    <label class="row">Chunks to generate <span id="nv">3</span></label>
    <input type="range" id="nchunks" min="1" max="8" step="1" value="3" style="width:100%">
  </div>

  <div class="card">
    <h2>Checkpoint</h2>
    <div class="meta" id="meta">loading…</div>
    <input type="text" id="ckpt" placeholder="path to a .pt checkpoint"/>
    <button class="btn" id="loadbtn">Load checkpoint</button>
  </div>
  <div style="flex:1"></div>
  <button class="btn" id="clear">Clear conversation</button>
</aside>

<main>
  <div id="thread"><div class="wrap"><div class="empty" id="empty">
    <b>Type a message to begin.</b><br>Each reply is decoded chunk-by-chunk; toggle chunk visualization to see the thought boundaries.
  </div></div></div>
  <div class="composer"><div class="wrap">
    <textarea id="box" rows="1" placeholder="Message the model…"></textarea>
    <button class="send" id="send">↑</button>
  </div><div class="hint">Enter to send · Shift+Enter for newline · completion model, not a conversational assistant</div></div>
</main>

<script>
const $=s=>document.querySelector(s);
let mode="generate";
const thread=$("#thread"), box=$("#box");

function api(path,body){return fetch(path,{method:"POST",headers:{"Content-Type":"application/json"},
  body:JSON.stringify(body)}).then(r=>r.json());}

function refreshMeta(){fetch("/api/info").then(r=>r.json()).then(m=>{
  if(m.error){$("#meta").textContent="no checkpoint loaded";return;}
  $("#meta").textContent=`stage ${m.stage_reached} · step ${m.global_step}\nd_model ${m.d_model} · L=${m.max_chunk_len} · C=${m.max_chunks_per_doc}\n${m.path}`;
});}

$("#mode").addEventListener("click",e=>{const b=e.target.closest("button");if(!b)return;
  mode=b.dataset.mode;[...$("#mode").children].forEach(x=>x.classList.toggle("on",x===b));
  box.placeholder = mode==="score" ? "Text to score (perplexity)…"
                  : mode==="dialogue" ? "Message the chatbot…" : "Message the model…";});

$("#temp").oninput=e=>$("#tempv").textContent=(+e.target.value).toFixed(2);
$("#nchunks").oninput=e=>$("#nv").textContent=e.target.value;
$("#clear").onclick=()=>{thread.innerHTML='<div class="wrap"></div>';
  fetch("/api/dialogue_reset",{method:"POST"});};   // also wipe the chatbot's cross-turn memory
$("#loadbtn").onclick=()=>{const p=$("#ckpt").value.trim();if(!p)return;
  $("#meta").textContent="loading…";
  api("/api/load",{path:p}).then(m=>{if(m.error){$("#meta").textContent="ERROR: "+m.error;}else{refreshMeta();$("#ckpt").value="";}});};

function el(tag,cls,txt){const e=document.createElement(tag);if(cls)e.className=cls;if(txt!=null)e.textContent=txt;return e;}
function esc(s){return s;}

function addMsg(role, build){
  const e=$("#empty"); if(e) e.remove();
  let w=thread.querySelector(".wrap:last-child"); if(!w){w=el("div","wrap");thread.appendChild(w);}
  const m=el("div","msg "+(role==="user"?"user":"bot"));
  const who=el("div","who",role==="user"?"You":"LT"); m.appendChild(who);
  const b=el("div","bubble"); m.appendChild(b); build(b);
  w.appendChild(m); thread.scrollTop=thread.scrollHeight; return b;
}

function renderChunks(container, chunks, viz){
  if(viz){const p=el("div","pills");
    chunks.forEach((c,i)=>{const pill=el("div","pill");pill.appendChild(el("span","i","#"+(i+1)));
      pill.appendChild(document.createTextNode(c));p.appendChild(pill);});container.appendChild(p);}
  else{container.appendChild(el("div","txt",chunks.join(" ")));}
}

function send(){
  const text=box.value.trim(); if(!text) return;
  const viz=$("#viz").checked, showread=$("#showread").checked, showppl=$("#showppl").checked;
  const temp=+$("#temp").value, n=+$("#nchunks").value;
  box.value=""; box.style.height="auto";
  addMsg("user", b=>b.appendChild(el("div","txt",text)));
  const bot=addMsg("bot", b=>b.appendChild(el("div","txt","…")));
  $("#send").disabled=true;

  if(mode==="score"){
    api("/api/score",{text}).then(r=>{bot.innerHTML="";
      if(r.error){bot.appendChild(el("div","txt","error: "+r.error));}
      else{bot.appendChild(el("span","score",`avg NLL/token = ${r.nll.toFixed(3)}   ·   perplexity = ${r.ppl.toFixed(1)}`));
        if(showread){const d=el("div","dbg");d.appendChild(el("div",null,`input → ${r.read.length} chunks`));
          const rr=el("div","read");renderChunks(rr,r.read,true);d.appendChild(rr);bot.appendChild(d);}}
      $("#send").disabled=false;box.focus();});
    return;
  }
  if(mode==="dialogue"){
    api("/api/dialogue",{text,temperature:temp,n_chunks:n}).then(r=>{bot.innerHTML="";
      if(r.error){bot.appendChild(el("div","txt","error: "+r.error));}
      else{renderChunks(bot, r.reply.length?r.reply:["(empty)"], viz);
        if(showread){const d=el("div","dbg");
          d.appendChild(el("div",null,`you → ${r.read.length} chunks · reply → ${r.reply.length} thoughts`));
          const rr=el("div","read");renderChunks(rr,r.read,true);d.appendChild(rr);bot.appendChild(d);}}
      $("#send").disabled=false;box.focus();});
    return;
  }
  api("/api/generate",{text,temperature:temp,n_chunks:n,score:showppl}).then(r=>{
    bot.innerHTML="";
    if(r.error){bot.appendChild(el("div","txt","error: "+r.error));$("#send").disabled=false;return;}
    renderChunks(bot, r.gen.length?r.gen:["(empty)"], viz);
    const d=el("div","dbg");
    if(showread){d.appendChild(el("div",null,`read: ${r.read.length} chunks · generated: ${r.gen.length}`));
      const rr=el("div","read");renderChunks(rr,r.read,true);d.appendChild(rr);}
    if(showppl && r.ppl!=null){d.appendChild(el("span","score",`input perplexity = ${r.ppl.toFixed(1)}`));}
    if(showread||showppl) bot.appendChild(d);
    $("#send").disabled=false;box.focus();
  });
}

$("#send").onclick=send;
box.addEventListener("keydown",e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();send();}});
box.addEventListener("input",()=>{box.style.height="auto";box.style.height=Math.min(box.scrollHeight,180)+"px";});
refreshMeta();
</script>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _json(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            data = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif self.path == "/api/info":
            if STATE["model"] is None:
                self._json({"error": "no checkpoint"})
            else:
                self._json({**STATE["meta"], "path": STATE["path"]})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        try:
            body = self._body()
        except Exception as e:
            return self._json({"error": f"bad request: {e}"}, 400)

        if self.path == "/api/load":
            try:
                with LOCK:
                    do_load(body["path"])
                return self._json({**STATE["meta"], "path": STATE["path"]})
            except Exception as e:
                return self._json({"error": str(e)})

        if self.path == "/api/dialogue_reset":
            with LOCK:
                s = STATE.get("session")
                if s is not None:
                    s.memory.reset(); s.source_memory = None
            return self._json({"ok": True})

        if STATE["model"] is None:
            return self._json({"error": "no checkpoint loaded"})
        text = (body.get("text") or "").strip()
        if not text:
            return self._json({"error": "empty input"})

        try:
            with LOCK:
                m, ch, cfg = STATE["model"], STATE["chunker"], STATE["cfg"]
                if self.path == "/api/score":
                    nll, ppl = chat_core.score_text(m, ch, cfg, text)
                    return self._json({"nll": nll, "ppl": ppl,
                                       "read": chat_core.input_chunks(ch, text)})
                if self.path == "/api/chunks":
                    return self._json({"read": chat_core.input_chunks(ch, text)})
                if self.path == "/api/generate":
                    n = int(body.get("n_chunks", 3))
                    temp = float(body.get("temperature", 0.9))
                    gen = chat_core.generate_chunks(m, ch, cfg, text,
                                                    n_chunks=n, temperature=temp)
                    out = {"gen": gen, "read": chat_core.input_chunks(ch, text)}
                    if body.get("score"):
                        _, out["ppl"] = chat_core.score_text(m, ch, cfg, text)
                    return self._json(out)
                if self.path == "/api/dialogue":
                    n = int(body.get("n_chunks", 6))
                    temp = float(body.get("temperature", 0.9))
                    reply, read = chat_core.dialogue_reply(STATE["session"], text,
                                                           n_chunks=n, temperature=temp)
                    return self._json({"reply": reply, "read": read})
        except Exception as e:
            return self._json({"error": str(e)})
        self._json({"error": "not found"}, 404)


def ask_checkpoint(rest):
    if rest:
        return rest[0]
    while True:
        try:
            p = input("checkpoint path: ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); sys.exit(0)
        if p:
            return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?", help="path to a .pt checkpoint (else prompted)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    path = args.ckpt or ask_checkpoint([])
    print(f"[web_chat] loading {path} ...")
    do_load(path)
    print(f"[web_chat] loaded: stage={STATE['meta']['stage_reached']} "
          f"d_model={STATE['meta']['d_model']} vocab={STATE['meta']['vocab_size']}")
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"[web_chat] serving on {url}  (Ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[web_chat] bye.")


if __name__ == "__main__":
    main()
