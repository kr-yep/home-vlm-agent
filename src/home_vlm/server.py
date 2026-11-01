"""Web page: the annotated video, the list of named objects, and a chat box."""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict
from typing import Protocol

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from PIL import Image
from pydantic import BaseModel, Field

from .pipeline import SeenObject


class Source(Protocol):
    def snapshot(self) -> tuple[Image.Image | None, list[SeenObject], bytes | None]: ...
    def largest_crop(self) -> Image.Image | None: ...


class Chatter(Protocol):
    def chat(self, question: str, scene: str, images: list[Image.Image] = ()) -> str: ...


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=500)


PAGE = """<!doctype html><html lang="zh-Hant"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Home VLM Agent</title>
<style>
body{font-family:system-ui,"Noto Sans TC",sans-serif;margin:16px;background:#111;color:#eee}
main{display:grid;grid-template-columns:minmax(0,3fr) minmax(260px,1fr);gap:16px}
@media(max-width:800px){main{grid-template-columns:1fr}}
img{width:100%;border-radius:8px;background:#000}
ul{padding-left:1.2em}li{margin:.3em 0}small{color:#999}
#log p{margin:.4em 0}form{display:flex;gap:8px}input{flex:1;padding:8px;font-size:1rem}
</style>
<main><section><img src="/stream" alt="camera"></section>
<section><h3>看到的物品</h3><ul id="objects"></ul>
<h3>問問看</h3><div id="log"></div>
<form id="ask"><input id="q" placeholder="你看到什麼？" autocomplete="off"><button>送出</button></form></section></main>
<script>
const objects=document.getElementById('objects'),log=document.getElementById('log');
async function refresh(){try{const r=await fetch('/api/objects');const d=await r.json();
objects.replaceChildren(...d.objects.map(o=>{const li=document.createElement('li');
li.textContent=o.name?`${o.name}：${o.detail||''}`:'辨識中…';
const s=document.createElement('small');s.textContent=` YOLO: ${o.yolo_name} ${o.yolo_confidence}`;li.append(s);return li}))}catch(e){}}
setInterval(refresh,1000);refresh();
document.getElementById('ask').onsubmit=async e=>{e.preventDefault();const q=document.getElementById('q').value.trim();if(!q)return;
const p=document.createElement('p');p.textContent='你：'+q;log.append(p);document.getElementById('q').value='';
const a=document.createElement('p');a.textContent='…';log.append(a);
try{const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})});
const d=await r.json();a.textContent=r.ok?`助手：${d.answer}（${d.seconds}s）`:`錯誤：${d.detail||r.status}`}catch(err){a.textContent='錯誤：連不上伺服器'}};
</script></html>"""


def create_app(source: Source, chatter: Chatter) -> FastAPI:
    app = FastAPI(title="home-vlm-agent")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return PAGE

    @app.get("/api/objects")
    def objects() -> dict:
        _, seen, _ = source.snapshot()
        return {"objects": [asdict(obj) for obj in seen], "error": getattr(source, "error", None)}

    @app.get("/stream")
    async def stream() -> StreamingResponse:
        async def frames():
            last = None
            while True:
                _, _, jpeg = source.snapshot()
                if jpeg is not None and jpeg is not last:
                    last = jpeg
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                await asyncio.sleep(0.05)

        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.post("/api/chat")
    def chat(body: Question) -> dict:
        _, seen, _ = source.snapshot()
        scene = "\n".join(obj.describe() for obj in seen) or "（畫面中沒有偵測到物品）"
        crop = source.largest_crop()
        started = time.perf_counter()
        answer = chatter.chat(body.question, scene, [crop] if crop is not None else [])
        return {"answer": answer, "seconds": round(time.perf_counter() - started, 2), "scene": scene}

    return app
