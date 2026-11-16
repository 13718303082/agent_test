import os, json
from dotenv import load_dotenv
from openai import OpenAI
from ddgs import DDGS
import trafilatura


load_dotenv()
client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"],
                base_url="https://api.deepseek.com")
MODEL = "deepseek-flash"
MAX_STEPS = 10
READ_URLS = []
#-------tools--------------
def web_search(query,max_results = 5):
    try:
        result = DDGS().text(query,max_results=max_results)
    except Exception as e:
        return f"搜索失败：{e}"
    if not result:
        return "无结果"
    return "\n\n".join(
        f"[{i+1}] {r['title']}\n链接：{r['href']}\n摘要：{r['body']}"
        for i, r in enumerate(result)
    )
def read_page(url,max_chart = 3000):
    try:
        html = trafilatura.fetch_url(url)
        text = trafilatura.extract(html) if html else None
    except Exception as e:
        return f"读取失败:{e}"
    if not text:
        return "无法提取正文"
    READ_URLS.append(url)
    n = len(READ_URLS)
    return f"[来源 {n}]\n{text[:max_chart]}"
TOOL_MAP = {"web_search":web_search,"read_page":read_page}
tools = [
{"type": "function", "function": {
        "name": "web_search",
        "description": "在互联网上搜索信息，返回标题、链接和摘要。需要最新信息或事实时使用。",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string", "description": "搜索关键词"}},
                       "required": ["query"]}}},
{"type": "function", "function": {
        "name": "read_page",
        "description": "读取网页正文。搜索摘要信息不够详细时，用它读取全文。",
        "parameters": {"type": "object",
                       "properties": {"url": {"type": "string", "description": "搜素网络链接"}},
                       "required": ["url"]}}},
]
def run_agent(messages):
    READ_URLS.clear()
    for step in range(MAX_STEPS):
        resp = client.chat.completions.create(
            model=MODEL, messages=messages,tools=tools
        )
        print("prompt_tokens:"+str(resp.usage.prompt_tokens)+"-------completion_tokens:"+str(resp.usage.completion_tokens))
        msg = resp.choices[0].message
        if msg.content:
            print("AI：", msg.content)
        if not msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content})
            print("\n参考来源")
            for i, url in enumerate(READ_URLS,1):
                print(f"[{i}{url}]")
            return
        messages.append({"role": "assistant", "content": msg.content,"tool_calls":[tc.model_dump() for tc in msg.tool_calls]})

        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
                result = TOOL_MAP[tc.function.name](**args)
            except Exception as e:
                result = f"tool error: {e}"
            print(f"  [第{step + 1}步] {tc.function.name}({tc.function.arguments})")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
    print("到达最大步数")
SYSTEM_PROMPT = """你是一个研究助手。
- 先用 web_search 搜索，再用 read_page 阅读网页全文
- 只根据读到的内容回答，不要编造
- 关键事实后用 [编号] 标注出处，编号就是 read_page 结果开头的“来源 N”
- 不要自己写参考来源列表"""
if __name__ == "__main__":
    messages = [{"role":"system","content":SYSTEM_PROMPT}]
    while True:
        q = input("\n你：")
        if q == "q":
            break
        messages.append({"role": "user", "content": q})
        run_agent(messages)