import threading, time
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import agent

app = FastAPI(title="Deep Research Agent")

lock = threading.Lock()      # 一次只处理一个研究请求
usage = {}                   # IP -> 今天的请求次数
DAILY_LIMIT = 5

class Query(BaseModel):
    question: str

@app.post("/research")
def research(q: Query, request: Request):
    # 简单限流：每个 IP 每天最多 5 次
    ip = request.client.host
    key = f"{ip}:{time.strftime('%Y-%m-%d')}"
    if usage.get(key, 0) >= DAILY_LIMIT:
        raise HTTPException(429, "今天的次数已用完，请明天再试")
    if not q.question.strip() or len(q.question) > 200:
        raise HTTPException(400, "问题不能为空，且不能超过 200 字")

    # 一次只跑一个研究
    if not lock.acquire(blocking=False):
        raise HTTPException(503, "当前有其他研究正在进行，请稍后再试")
    try:
        usage[key] = usage.get(key, 0) + 1
        return agent.research(q.question)
    finally:
        lock.release()

@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Deep Research Agent</title>
<style>
 body{max-width:760px;margin:40px auto;font-family:sans-serif;padding:0 16px;line-height:1.6}
 textarea{width:100%;height:70px;font-size:15px}
 button{margin-top:8px;padding:8px 20px;font-size:15px}
 #out{white-space:pre-wrap;margin-top:20px}
</style></head>
<body>
<h2>Deep Research Agent</h2>
<textarea id="q" placeholder="输入你想研究的问题"></textarea><br>
<button onclick="go()">开始研究</button>
<div id="out"></div>
<script>
async function go(){
  const out = document.getElementById('out');
  out.textContent = '研究中，大约需要 30-90 秒……';
  const res = await fetch('/research', {method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({question: document.getElementById('q').value})});
  const data = await res.json();
  if(!res.ok){ out.textContent = '出错：' + data.detail; return; }
  out.textContent = '研究计划：\\n' + data.plan.map((s,i)=>(i+1)+'. '+s).join('\\n')
    + '\\n\\n' + data.answer
    + '\\n\\n参考来源：\\n' + data.sources.map((u,i)=>'['+(i+1)+'] '+u).join('\\n')
    + '\\n\\n（' + data.steps + ' 步，' + data.tokens + ' tokens）';
}
</script></body></html>"""