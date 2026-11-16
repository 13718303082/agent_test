import os, json
import numpy as np
from openai import OpenAI
import requests
from pypdf import PdfReader
import re
from dotenv import load_dotenv
load_dotenv()
from sentence_transformers import SentenceTransformer
client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"],
                base_url="https://api.deepseek.com")
embedder = SentenceTransformer("BAAI/bge-m3")


def load_pdf_and_embed(pdf_path, chunk_size=250):
    # 加载pdf、分块、向量化，全局覆盖 chunk 和 doc_vec
    global chunk, doc_vec
    if not os.path.exists(pdf_path):
        return f"错误：文件不存在 {pdf_path}"
    try:
        reader = PdfReader(pdf_path)
        text = "".join(p.extract_text() or "" for p in reader.pages)
        text = re.sub(r"\s+", " ", text).strip()
        chunk = []
        for i in range(0, len(text), chunk_size):
            chunk.append({"source": pdf_path, "text": text[i:i + chunk_size]})
        # 向量化
        doc_vec = embedder.encode([c["text"] for c in chunk], normalize_embeddings=True)
        return f"✅ 加载成功：{pdf_path}，一共 {len(chunk)} 个chunk"
    except Exception as e:
        return f"PDF读取失败：{str(e)}"




# ---------- 工具 ----------
def load_pdf(pdf_path):
    return load_pdf_and_embed(pdf_path)

def get_weather(city):
    # ① 用 Open-Meteo 自带的地理编码，把城市名转成坐标
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "zh"},
            timeout=8,
        )
        geo.raise_for_status()
        results = geo.json().get("results")
    except Exception as e:
        return f"定位服务出错：{e}"          # 网络问题，和“找不到”区分开

    if not results:
        return f"找不到城市：{city}"

    place = results[0]
    lat, lon = place["latitude"], place["longitude"]
    name = f'{place["name"]}, {place.get("country", "")}'

    # ② 查天气
    try:
        res = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon,
                    "current": "temperature_2m,weather_code", "timezone": "auto"},
            timeout=8,
        )
        res.raise_for_status()
        cur = res.json()["current"]
    except Exception as e:
        return f"天气查询失败：{e}"

    return f"{name}：{cur['temperature_2m']}°C {weather_desc(cur['weather_code'])}"


def weather_desc(code):
    if code == 0:              return "晴"
    if code in (1, 2, 3):      return "多云"
    if code in (45, 48):       return "雾"
    if 51 <= code <= 67:       return "雨"
    if 71 <= code <= 77:       return "雪"
    if 80 <= code <= 82:       return "阵雨"
    if code in (85, 86):       return "阵雪"
    if 95 <= code <= 99:       return "雷暴"
    return f"未知天气(代码{code})"

def calculate(expression):
    return str(eval(expression))   # 仅练习用

def search_notes(query,k = 3):
    q = embedder.encode([query],normalize_embeddings=True)[0]
    top = np.argsort(doc_vec@q)[::-1][:k]
    return "\n\n".join(f"[{chunk[i]['source']}] {chunk[i]['text']}" for i in top)
TOOL_MAP = {"get_weather": get_weather, "calculate": calculate,"search_notes":search_notes,"load_pdf":load_pdf}

tools = [
    {"type": "function", "function": {
        "name": "get_weather", "description": "查询城市当前天气",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string"}},
                       "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "calculate", "description": "计算数值",
        "parameters": {"type": "object",
                       "properties": {"expression": {"type": "string"}},
                       "required": ["expression"]}}},
    {"type": "function",
        "function": {
        "name": "search_notes",
        "description": "在用户的课程笔记中搜索相关内容。问题涉及课程知识时使用。",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "搜索关键词或问题"}},
            "required": ["query"]}}},
    {"type": "function",
        "function": {
        "name": "load_pdf",
        "description": "加载一个本地PDF文件，读取文本并向量化，之后可以搜索这个pdf内容。用户给pdf路径时调用",
        "parameters": {
            "type": "object",
            "properties": {"pdf_path": {"type": "string", "description": "pdf完整本地路径，如 D:\\a.pdf"}},
            "required": ["pdf_path"]
            }
        }
    }
    ]
messages = [{"role": "system", "content": "你是一个乐于助人的助手，需要时调用工具。"}]
while True:
    user_input = input("\n你：")
    if user_input == "q":
        break
    messages.append({"role": "user", "content": user_input})
    for step in range(15):
        resp = client.chat.completions.create(
            model="deepseek-flash", messages=messages,
            tools=tools
        )

        msg = resp.choices[0].message
        if msg.content:
            print("AI：",msg.content)
        if not msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content})
            break

        messages.append({"role": "assistant", "content": msg.content,"tool_calls": [tc.model_dump() for tc in msg.tool_calls]})

        for tc in msg.tool_calls:
            try:
                arg = json.loads(tc.function.arguments)
                result = TOOL_MAP[tc.function.name](**arg)
            except Exception as e:
                result = f"tool error: {e}"
            print(f"调用{tc.function.name}({tc.function.arguments})")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
