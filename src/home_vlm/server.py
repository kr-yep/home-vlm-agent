"""Web page: the annotated video, the list of named objects, and a chat box."""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict
from typing import Protocol

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, StreamingResponse
from PIL import Image
from pydantic import BaseModel, Field

from .pipeline import SeenObject


class Source(Protocol):
    def snapshot(self) -> tuple[Image.Image | None, list[SeenObject], bytes | None]: ...


class Chatter(Protocol):
    def chat(self, question: str, scene: str, images: list[Image.Image] = ()) -> str: ...


class Camera(Protocol):
    def put(self, jpeg: bytes) -> bool: ...


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=500)


PAGE = """<!doctype html><html lang="zh-Hant"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Home VLM Agent</title>
<style>
body{font-family:system-ui,"Noto Sans TC",sans-serif;margin:16px;background:#111;color:#eee}
main{display:grid;grid-template-columns:minmax(0,3fr) minmax(260px,1fr);gap:16px}
@media(max-width:800px){main{grid-template-columns:1fr}}
@media(min-width:801px){main>section:first-child{position:sticky;top:16px;align-self:start}
#side{position:sticky;top:16px;align-self:start;height:calc(100vh - 32px);display:flex;flex-direction:column}#objects{flex:1;min-height:0;overflow-y:auto}}
h3{margin:16px 0 8px}#side>h3:first-child{margin-top:0}
img{width:100%;min-height:240px;border-radius:8px;background:#000}
ul{padding-left:1.2em}li{margin:.3em 0}small{color:#999}
#log{height:35vh;overflow-y:auto;margin-bottom:8px}#log p{margin:.4em 0}form{display:flex;gap:8px}input{flex:1;padding:8px;font-size:1rem}
#cam{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-top:8px}[hidden]{display:none!important}
</style>
<main><section><img id="view" alt="">
<div id="cam" hidden><button id="camBtn">開鏡頭</button><select id="camSel" hidden></select><span id="camMsg"></span></div></section>
<section id="side"><h3>問問看</h3><div id="log"></div>
<form id="ask"><input id="q" placeholder="你看到什麼？" autocomplete="off"><button>送出</button></form>
<h3>看到的物品</h3><ul id="objects"></ul></section></main>
<script>
const objects=document.getElementById('objects'),log=document.getElementById('log');
async function refresh(){try{const r=await fetch('/api/objects');const d=await r.json();
if(live&&!stream){if(!d.receiving)view.removeAttribute('src');else if(!view.getAttribute('src'))view.src='/stream'}
if(d.error){objects.textContent='影像來源有問題：'+d.error;return}
objects.replaceChildren(...d.objects.map(o=>{const li=document.createElement('li');
li.textContent=o.name?`${o.name}：${o.detail||''}`:'辨識中…';
const s=document.createElement('small');s.textContent=` YOLO: ${o.yolo_name} ${o.yolo_confidence}`;li.append(s);return li}))}catch(e){}}
document.getElementById('ask').onsubmit=async e=>{e.preventDefault();const q=document.getElementById('q').value.trim();if(!q)return;
const p=document.createElement('p');p.textContent='你：'+q;log.append(p);document.getElementById('q').value='';
const a=document.createElement('p');a.textContent='…';log.append(a);log.scrollTop=log.scrollHeight;
try{const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})});
const d=await r.json();a.textContent=r.ok?`助手：${d.answer}（${d.seconds}s）`:`錯誤：${d.detail||r.status}`}catch(err){a.textContent='錯誤：連不上伺服器'}log.scrollTop=log.scrollHeight};
const live=__LIVE__,view=document.getElementById('view'),cam=document.getElementById('cam'),camBtn=document.getElementById('camBtn'),camSel=document.getElementById('camSel'),camMsg=document.getElementById('camMsg');
const video=document.createElement('video'),canvas=document.createElement('canvas');video.muted=true;video.playsInline=true;
let stream=null;const FRAME_MS=66;
function stopCam(text){if(stream)stream.getTracks().forEach(t=>t.stop());stream=null;view.removeAttribute('src');camBtn.textContent='開鏡頭';camMsg.textContent=text||''}
async function startCam(deviceId){
if(!navigator.mediaDevices){camMsg.textContent='瀏覽器不讓這個網址開鏡頭，請改用 http://127.0.0.1:8700';return}
try{stream=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:1280},height:{ideal:720},...(deviceId?{deviceId:{exact:deviceId}}:{})}})}
catch(e){camMsg.textContent=e.name==='NotFoundError'?'找不到鏡頭':e.name==='NotAllowedError'?'瀏覽器沒有允許使用鏡頭':'開不了鏡頭：'+e.message;return}
camMsg.textContent='';camBtn.textContent='關鏡頭';video.srcObject=stream;await video.play();view.src='/stream';
stream.getVideoTracks()[0].onended=()=>stopCam('鏡頭斷線了');
const cams=(await navigator.mediaDevices.enumerateDevices()).filter(d=>d.kind==='videoinput');
const current=stream.getVideoTracks()[0].getSettings().deviceId;
camSel.replaceChildren(...cams.map(d=>{const o=document.createElement('option');o.value=d.deviceId;o.textContent=d.label||'鏡頭';o.selected=d.deviceId===current;return o}));camSel.hidden=cams.length<2;
sendFrames(stream)}
async function sendFrames(mine){while(stream===mine){const t=performance.now();
canvas.width=video.videoWidth;canvas.height=video.videoHeight;canvas.getContext('2d').drawImage(video,0,0);
const jpeg=await new Promise(done=>canvas.toBlob(done,'image/jpeg',0.85));
try{await fetch('/api/frame',{method:'POST',headers:{'Content-Type':'image/jpeg'},body:jpeg})}catch(e){stopCam('連不上伺服器');return}
await new Promise(done=>setTimeout(done,Math.max(0,FRAME_MS-(performance.now()-t))))}}
if(live){cam.hidden=false;camBtn.onclick=()=>stream?stopCam():startCam();camSel.onchange=()=>{stopCam();startCam(camSel.value)}}else view.src='/stream';
setInterval(refresh,1000);refresh();
</script></html>"""


def create_app(source: Source, chatter: Chatter, camera: Camera | None = None) -> FastAPI:
    app = FastAPI(title="home-vlm-agent")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return PAGE.replace("__LIVE__", "true" if camera else "false")

    @app.get("/api/objects")
    def objects() -> dict:
        _, seen, jpeg = source.snapshot()
        return {"objects": [asdict(obj) for obj in seen], "error": getattr(source, "error", None), "receiving": jpeg is not None}

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

    if camera:
        @app.post("/api/frame", status_code=204)
        async def frame(request: Request) -> None:
            if not await run_in_threadpool(camera.put, await request.body()):
                raise HTTPException(400, "not a JPEG image")

    @app.post("/api/chat")
    def chat(body: Question) -> dict:
        frame, seen, _ = source.snapshot()
        named = [obj.describe() for obj in seen if obj.name]
        waiting = len(seen) - len(named)
        scene = "\n".join(named + ([f"（另有 {waiting} 個物體還沒辨識）"] if waiting else [])) or "（畫面中沒有偵測到物品）"
        started = time.perf_counter()
        answer = chatter.chat(body.question, scene, [frame] if frame is not None else [])
        return {"answer": answer, "seconds": round(time.perf_counter() - started, 2), "scene": scene}

    return app
