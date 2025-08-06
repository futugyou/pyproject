import json
import os
import asyncio
from openai import AsyncOpenAI
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()


async def analyze_text_with_llm(text_content):
    """
    Call the LLM API to analyze the text and return structured JSON results.
    """

    system_prompt = """你是一个专业的食品安全舆情分析专家，精通自然语言处理技术。你的任务是分析用户提供的文本，从中提取食品安全事件相关信息，并进行情感分析。请严格按照要求，以JSON格式输出结果。"""

    user_prompt_template = """请分析以下文本，并提取食品安全事件三元组（主体-谓词-客体）、情感分析结果。请确保输出的JSON格式完全正确，并且内容准确无误。如果文本中没有相关信息，请返回空列表或null。

--- 待分析文本 ---

{text_content}

--- 输出要求 ---

1.  **事件三元组（event_triples）：**
    * 以列表形式返回，每个元素是一个字典。
    * 每个字典包含 `subject` (施事/主体), `predicate` (谓词), `object` (受事/客体) 和 `description` (事件描述) 四个键值对。
    * `description` 字段应结合主谓客体，用一句话概括事件。
    * 例如：`{{"subject": "朋友", "predicate": "化验出", "object": "硼砂", "description": "朋友化验出燕皮扁食中含有硼砂成分"}}`。

2.  **情感分析（sentiment_analysis）：**
    * 以字典形式返回，包含 `overall_score`, `positive_score`, `negative_score`, `sentiment_keywords` 和 `analysis_details`。
    * **`overall_score`：** 基于情感倾向，评分范围为-10到+10。负分代表负面情绪，正分代表正面情绪，0代表中性。情感强度越高，分数绝对值越大。
    * **`positive_score`：** 正面情感的分值（0-10）。
    * **`negative_score`：** 负面情感的分值（0-10）。
    * **`sentiment_keywords`：** 列表，列出所有影响情感判断的关键词（如“剧毒”、“触目惊心”、“没有”、“最好”等）。
    * **`analysis_details`：** 字符串，简要分析情感产生的原因，包括识别到的情感词、否定词、程度副词以及它们对情感倾向的影响。

**重要提示：**
* 请将所有输出数据严格封装在单个JSON对象中，键为`analysis_result`。
* 如果文本不包含任何相关事件或情感，可以返回一个表示为空的JSON对象，例如：`{{"analysis_result": null}}`。

--- 示例 ---

输入文本: \"朋友拿了米其林一星餐厅的燕皮扁食去化验 检测出了硼砂成分（剧毒）想提醒妈妈们馄饨、肠粉之类的自己做我也好信儿 也寄了我们家小馄饨去检测花了 750 块钱巨资 还好没有……不过食品安全也算是触目惊心了自己做肯定是最好的 \"

期望输出JSON：
```json
{{
  "analysis_result": {{
    "event_triples": [
      {{
        "subject": "朋友",
        "predicate": "检测出",
        "object": "硼砂成分",
        "description": "朋友在米其林一星餐厅的燕皮扁食中检测出硼砂成分（剧毒）"
      }},
      {{
        "subject": "我",
        "predicate": "检测",
        "object": "小馄饨",
        "description": "我将自家小馄饨寄去检测，结果没有问题"
      }}
    ],
    "sentiment_analysis": {{
      "overall_score": -7,
      "positive_score": 3,
      "negative_score": 7,
      "sentiment_keywords": [
        "硼砂",
        "剧毒",
        "触目惊心",
        "巨资",
        "还好没有",
        "最好"
       ],
    "analysis_details": "文本整体情感偏向负面。负面情感主要来源于对'米其林一星餐厅'的'燕皮扁食'中'剧毒'的'硼砂成分'的震惊，以及对'食品安全'感到'触目惊心'。正面情感部分来源于作者检测自家馄饨'还好没有'问题，并强调'自己做肯定是最好的'，体现了对安全食物的正面期望。"
    }}
  }}
}}
"""
    client = AsyncOpenAI(
        api_key=os.getenv("GOOGLE_API_KEY"),
        base_url=os.getenv("GOOGLE_URL"),
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": user_prompt_template.format(text_content=text_content),
        },
    ]

    try:
        async with AsyncOpenAI(
            api_key=os.getenv("GOOGLE_API_KEY"),
            base_url=os.getenv("GOOGLE_URL"),
        ) as client:
            completion = await client.chat.completions.create(
                messages=messages,
                model=os.getenv("GOOGLE_CHAT_MODEL_ID"),
                response_format={"type": "json_object"},
            )

            result_json_str = completion.choices[0].message.content
            return json.loads(result_json_str)

    except Exception as e:
        print(f"Error calling LLM API: {e}")
        return {"analysis_result": None}


async def process_jsonl_file(input_file, output_file):
    """
    Reads a JSONL file, processes each line, and saves the result to a new file.
    """
    processed_data = []

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            # Read all lines to get the total for the progress bar
            lines = f.readlines()
            total_lines = len(lines)

            # Use tqdm to display the progress bar
            for line in tqdm(lines, total=total_lines, desc="Processing JSONL"):
                data = json.loads(line)
                text = data.get("text", "")

                if text:
                    # Call LLM for analysis
                    analysis_result = await analyze_text_with_llm(text)

                    # Merge the analysis results into the original data
                    if analysis_result:
                        data.update(analysis_result)
                else:
                    data.update(
                        {
                            "analysis_result": {
                                "event_triples": [],
                                "sentiment_analysis": None,
                            }
                        }
                    )

                processed_data.append(data)

    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return

    # Write the processed data to a new JSONL file
    with open(output_file, "w", encoding="utf-8") as f:
        for item in processed_data:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + "\n")

    print(f"Processing completed! The results have been saved to '{output_file}'.")


if __name__ == "__main__":
    input_jsonl_file = "2.weibo_data_tagged.jsonl"
    output_jsonl_file = "3.weibo_data_analyzed.jsonl"
    asyncio.run(process_jsonl_file(input_jsonl_file, output_jsonl_file))
