import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import json
import asyncio
from openai import AsyncOpenAI
from tqdm import tqdm
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Dict, Any

from models.llm_response import LLMResponse

from dotenv import load_dotenv

load_dotenv()


llm_response_schema = LLMResponse.model_json_schema()
llm_response_schema_str = json.dumps(llm_response_schema, indent=2, ensure_ascii=False)


async def analyze_text_with_llm(text_content, tags_content):
    """
    Call the LLM API to analyze the text and return structured JSON results.
    """

    system_prompt = """你是一个专业的食品安全舆情分析专家，精通自然语言处理技术。你的任务是分析用户提供的文本，从中提取食品安全事件相关信息，识别广告并进行情感分析。请严格按照要求，以JSON格式输出结果。"""

    user_prompt_template = """请分析以下文本，并提取食品安全事件三元组（主体-谓词-客体）、情感分析结果以及是否为广告内容。请特别注意，我已经为您提取了文本中可能涉及的关键实体（如企业名、有害物质等），您应将这些信息作为辅助，来更精准地完成分析任务。请严格按照要求，以JSON格式输出结果。如果文本中没有相关信息，请返回空列表或null。

--- 待分析文本 ---

{text_content}

--- 辅助信息 (已识别的实体) ---

{tags_content}


--- 输出要求 ---
请严格按照以下 JSON Schema 和提供的示例格式输出，确保你的输出是有效的 JSON。

**JSON Schema:**
```json
{llm_response_schema_str}
```

--- 示例 ---

输入文本: \"朋友拿了米其林一星餐厅的燕皮扁食去化验 检测出了硼砂成分（剧毒）想提醒妈妈们馄饨、肠粉之类的自己做我也好信儿 也寄了我们家小馄饨去检测花了 750 块钱巨资 还好没有……不过食品安全也算是触目惊心了自己做肯定是最好的 \"

辅助信息 (已识别的实体):
企业名: ['米其林一星餐厅']
有害物质: ['硼砂']
食品名: ['馄饨', '肠粉']

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
    }},
    "ad_detection": {{
        "is_ad": false,
        "ad_type": null,
        "ad_keywords": null,
        "reasoning": "通过关键词和情感分析，判断这段文本没有广告的迹象。"
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
            "content": user_prompt_template.format(
                text_content=text_content,
                tags_content=tags_content,
                llm_response_schema_str=llm_response_schema_str,
            ),
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
            llm_response = LLMResponse.model_validate_json(result_json_str)
            return llm_response.model_dump()

    except ValidationError as e:
        print(f"Pydantic Validation Error for text: '{text_content}'\nError: {e}")
        return {"analysis_result": None}
    except Exception as e:
        print(f"Error calling LLM API or parsing JSON: {e}")
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
                tags = data.get("tags", {})
                tags_content_lines = []

                for k, v in tags.items():
                    tags_content_lines.append(f"{k}: {list(v.keys())}")

                tags_content = "\n".join(tags_content_lines)

                if text:
                    # Call LLM for analysis
                    analysis_result = await analyze_text_with_llm(text, tags_content)

                    # Merge the analysis results into the original data
                    if analysis_result:
                        data.update(analysis_result)
                else:
                    data.update(
                        {
                            "analysis_result": {
                                "event_triples": [],
                                "sentiment_analysis": None,
                                "ad_detection": None,
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
    output_jsonl_file = "3.1.weibo_data_analyzed_structured.jsonl"
    asyncio.run(process_jsonl_file(input_jsonl_file, output_jsonl_file))
