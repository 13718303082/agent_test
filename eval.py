import agent
import json
QUESTIONS = [
    "DeepSeek 最新的模型是什么？",
    "对比一下 LangGraph 和 CrewAI 的优缺点",
    "什么是 MCP 协议？",
    "LoRA 和全参数微调的区别是什么？",
    "vLLM 为什么推理速度快？",
]
list1 = []
list2 = []
def judge(question, answer):
    resp = agent.client.chat.completions.create(
        model=agent.MODEL,
        messages=[{"role": "user", "content":
            f"你是评审员，给下面的研究报告打分（1-10 分），考虑：是否回答了问题、内容是否具体、结构是否清晰。"
            f"注意：报告中的信息来自实时网络搜索，可能比你的知识更新。"
            f"不要用你自己的知识判断事实是否正确，不要因为你不认识某个名称、版本或日期就扣分。\n"
            f"只根据以下标准打分：\n"
            f"1. 是否完整回答了问题\n2. 内容是否具体（数字、日期、名称）\n"
            f"3. 关键事实是否有引用编号\n4. 结构是否清晰\n"
            f'只输出 JSON：{{"score": 分数, "reason": "一句话理由"}}\n\n'
            f"问题：{question}\n\n报告：{answer}"}],
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content
for i,q in enumerate(QUESTIONS,1):
    messages = [{"role": "system", "content": agent.SYSTEM_PROMPT},
                {"role": "user", "content": q}]
    ans = agent.run_agent(messages)
    score = json.loads(judge(q, ans))["score"]
    print(f"\n✅ {q} | 步数 {agent.STATS['steps']} | "
          f"输入tokens {agent.STATS['tokens']} | 读了 {len(agent.READ_URLS)} 页\n")
    list1.append({"id":i,"步数": agent.STATS['steps'],"输入tokens": agent.STATS['tokens'],"平均token":agent.STATS['tokens']/agent.STATS['steps'],"读了": str(len(agent.READ_URLS))+"页","分数":score})


for i,q in enumerate(QUESTIONS,1):
    subs = agent.plan(q)
    plan_text = "\n".join(f"{i}. {s}" for i, s in enumerate(subs, 1))

    messages = [{"role": "system", "content": agent.SYSTEM_PROMPT}]
    messages.append({"role": "user", "content":
        f"{q}\n\n请按以下子问题依次研究：\n{plan_text}\n\n"
        f"要求：整个研究最多阅读 5 页。"
        f"所有子问题都有答案后，立刻写最终报告。"})
    ans = agent.run_agent(messages)
    score = json.loads(judge(q, ans))["score"]
    print(f"\n✅ {q} | 步数 {agent.STATS['steps']} | "
          f"输入tokens {agent.STATS['tokens']} | 读了 {len(agent.READ_URLS)} 页\n")
    list2.append({"id":i,"步数": agent.STATS['steps'], "输入tokens": agent.STATS['tokens'],
                  "平均token": agent.STATS['tokens'] / agent.STATS['steps'], "读了": str(len(agent.READ_URLS)) + "页","分数":score})
print(list1)
print(list2)