import json
from tokenizers import Tokenizer
from transformers import AutoTokenizer
from collections import defaultdict, Counter
from typing import List, Dict, Tuple

tokenizer = Tokenizer.from_pretrained("bert-base-chinese") 
autoTokenizer = AutoTokenizer.from_pretrained("bert-base-chinese") 

def process_text_tokenization(
    input_path: str,
):
    """
    Process JSONL files according to the new logic.
    """

    with (
        open(input_path, "r", encoding="utf-8") as fr, 
    ):
        for line in fr:
            data = json.loads(line)
            text = data.get("text", "")

            input_tokens = tokenizer.encode(text)
            print(input_tokens)

            
            auto_tokens = autoTokenizer(text)
            decoded_text = tokenizer.decode(auto_tokens['input_ids'])
            print(decoded_text) 


if __name__ == "__main__":
    input_path = "../public_opinion_monitoring/1.weibo_foodsafety.jsonl"

    process_text_tokenization(input_path)
