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
STATS = {"steps": 0, "tokens": 0}
#-------tools--------------
MAX_SEARCH  =  6
SEARCH_COUNT = {"n": 0}
def web_search(query,max_results = 5):
    if SEARCH_COUNT["n"] >= MAX_SEARCH:
        return f"已达到搜索上限（{MAX_SEARCH} 次），请根据已有信息写最终报告。"
    SEARCH_COUNT["n"] += 1
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
MAX_PAGES = 5
def read_page(url,focus,max_chart = 3000):
    if len(READ_URLS) >= MAX_PAGES:
        return f"已达到阅读上限（{MAX_PAGES} 页）"
    try:
        html = trafilatura.fetch_url(url)
        text = trafilatura.extract(html) if html else None
    except Exception as e:
        return f"读取失败:{e}"
    if not text:
        return "无法提取正文"
    resp = client.chat.completions.create(
        model=MODEL,messages=[{"role":"user","content":f"从下面的网页内容中，提取和「{focus}」相关的事实，"
            f"保留具体数字、日期、名称，用要点列出，不超过 300 字。"
            f"如果没有相关内容，就回答“无相关内容”。\n\n{text[:max_chart]}"}] )
    notes = resp.choices[0].message.content
    READ_URLS.append(url)
    n = len(READ_URLS)
    return f"[来源 {n}]\n{notes}"


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
                       "properties": {
                           "url": {"type": "string", "description": "搜素网络链接"},
                            "focus": {"type": "string","description": "想从这个网页里找什么信息"}},
                       "required": ["url","focus"]}}},
]
def run_agent(messages):
    READ_URLS.clear()
    SEARCH_COUNT["n"] = 0
    STATS["steps"], STATS["tokens"] = 0, 0
    for step in range(MAX_STEPS):
        resp = client.chat.completions.create(
            model=MODEL, messages=messages,tools=tools
        )
        STATS["steps"]+=1
        STATS["tokens"] += resp.usage.prompt_tokens
        print("prompt_tokens:"+str(resp.usage.prompt_tokens)+"-------completion_tokens:"+str(resp.usage.completion_tokens))
        msg = resp.choices[0].message
        if msg.content:
            print("AI：", msg.content)
        if not msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content})
            print("\n参考来源")
            for i, url in enumerate(READ_URLS,1):
                print(f"[{i}{url}]")
            return msg.content
        messages.append({"role": "assistant", "content": msg.content,"tool_calls":[tc.model_dump() for tc in msg.tool_calls]})

        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
                result = TOOL_MAP[tc.function.name](**args)
            except Exception as e:
                result = f"tool error: {e}"
            print(f"  [第{step + 1}步] {tc.function.name}({tc.function.arguments})")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
    print("到达最大步数强制生成报告")
    messages.append({"role":"user","content":"请停止搜索，根据目前已经获得的信息写最终报告。"})
    resp = client.chat.completions.create(model=MODEL, messages=messages)
    ans = resp.choices[0].message.content
    print("AI:"+ans)
    messages.append({"role": "assistant", "content": ans})
    print("\n参考来源")
    for i, url in enumerate(READ_URLS, 1):
        print(f"[{i}] {url}")
    return ans




SYSTEM_PROMPT = """你是一个研究助手。
- 先用 web_search 搜索，再用 read_page 阅读网页全文
- 只根据读到的内容回答，不要编造
- 关键事实后用 [编号] 标注出处，编号就是 read_page 结果开头的“来源 N”
- 不要自己写参考来源列表"""

def plan(question):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content":
            f"判断下面这个研究问题的复杂程度，拆成子问题：\n"
            f"- 简单问题（单一概念、单一事实）：只输出 1 个子问题\n"
            f"- 复杂问题（对比、多方面分析）：输出 2-3 个子问题\n"
            f"每个子问题都要能通过搜索回答。"
            f'用 JSON 输出，格式：{{"sub_questions": ["...", "..."]}}\n\n问题：{question}'}],
        response_format={"type": "json_object"},
    )
    subs = json.loads(resp.choices[0].message.content)["sub_questions"]
    print("sub_questions:", subs)
    for i,s in enumerate(subs,1):
        print(f"  {i}. {s}")
    return subs

if __name__ == "__main__":
    messages = [{"role":"system","content":SYSTEM_PROMPT}]
    while True:
        q = input("\n你：")
        if q == "q":
            break
        subs = plan(q)
        plan_text = "\n".join(f"{i}. {s}" for i, s in enumerate(subs, 1))
        messages.append({"role": "user", "content":
            f"{q}\n\n请按以下子问题依次研究：\n{plan_text}\n\n"
            f"要求：整个研究最多阅读 5 页。"
            f"所有子问题都有答案后，立刻写最终报告。"})
        run_agent(messages)