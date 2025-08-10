import json
import jieba
import jieba.posseg as pseg
import spacy
from collections import defaultdict, Counter
from typing import List, Dict, Tuple

# Load the spaCy model globally
spacyNlp = spacy.load("zh_core_web_sm")


class CustomSegmenter:
    """
    Responsible for initializing the custom dictionary and word segmentation engine.
    Focused on word segmentation and part-of-speech tagging.
    """

    def __init__(self, dict_path: str, engine: str = "jieba"):
        self.engine = engine
        self.dict_path = dict_path
        self.domain_dict = self.load_dict(dict_path)
        self.phrase_set = set(self.domain_dict.keys())

        if engine == "jieba":
            self.init_jieba()
        else:
            raise ValueError("engine must be 'jieba'")

    def load_dict(self, path: str) -> Dict[str, str]:
        """Load a custom dictionary from a file."""
        d = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                word = parts[0]
                tag = parts[1] if len(parts) > 1 else "UNKNOWN"
                d[word] = tag
        return d

    def init_jieba(self):
        """Initialize jieba and add a custom dictionary."""
        jieba.initialize()
        for w in self.domain_dict.keys():
            # jieba.add_word will process words and parts of speech
            jieba.add_word(w)
        # After adding the dictionary, you need to reload the jieba word segmenter
        jieba.load_userdict(self.dict_path)

    def segment(self, text: str) -> List[str]:
        """Use jieba for word segmentation."""
        return list(jieba.cut(text))

    def posseg(self, text: str) -> List[Tuple[str, str]]:
        """Use jieba for part-of-speech tagging."""
        return [(word, flag) for word, flag in pseg.cut(text)]


def spacy_process(text: str) -> Dict[str, any]:
    """
    Use spaCy's complete NLP pipeline to perform tokenization, part-of-speech tagging, and entity recognition in one go.
    """
    doc = spacyNlp(text)

    # Get tokenization and part-of-speech tags
    tokens_and_pos = [(token.text, token.pos_) for token in doc]

    # Extract entities identified by spaCy
    spacy_entities = defaultdict(list)
    _ = [spacy_entities[ent.text].append(ent.label_) for ent in doc.ents]

    return {"tokens_and_pos": tokens_and_pos, "spacy_entities": dict(spacy_entities)}


def tag_custom_units(
    words: List[str], domain_dict: Dict[str, str], count_mode: bool = False
) -> Dict[str, dict]:
    """
    Specially designed to process words in custom dictionaries and perform classification and counting.
    """
    tag_result = defaultdict(Counter if count_mode else list)
    for w in words:
        if w in domain_dict:
            category = domain_dict[w]
            if count_mode:
                tag_result[category][w] += 1
            else:
                if w not in tag_result[category]:
                    tag_result[category].append(w)
    if count_mode:
        return {k: dict(v) for k, v in tag_result.items()}
    else:
        return dict(tag_result)


def process_jsonl_optimized(
    input_path: str,
    output_path: str,
    segmenter: CustomSegmenter,
    count_mode: bool = False,
):
    """
    Process JSONL files according to the new logic.
    """
    domain_dict = segmenter.domain_dict

    with (
        open(input_path, "r", encoding="utf-8") as fr,
        open(output_path, "w", encoding="utf-8") as fw,
    ):
        for line in fr:
            data = json.loads(line)
            text = data.get("text", "")

            # 1. Use Jieba for word segmentation and part-of-speech tagging, focusing on custom lexicons
            jieba_tokens = segmenter.segment(text)
            jieba_pos_tags = segmenter.posseg(text)

            # 2. Use spaCy for general tasks and retrieve common entities
            spacy_results = spacy_process(text)

            # 3. Use Jieba's word segmentation results to specifically process custom lexicons
            custom_tags = tag_custom_units(jieba_tokens, domain_dict, count_mode)

            # 4. Combine all results into the output data
            data["jieba_tokens"] = jieba_tokens
            data["jieba_pos_tags"] = jieba_pos_tags
            data["tags"] = custom_tags
            data["spacy_pos_tags"] = spacy_results["tokens_and_pos"]
            data["spacy_entities"] = spacy_results["spacy_entities"]

            fw.write(json.dumps(data, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    dict_path = "custom_food_dict.txt"
    input_path = "1.weibo_foodsafety.jsonl"
    output_path = "2.weibo_data_tagged_optimized.jsonl"

    print("Processing using the jieba engine...")
    segmenter = CustomSegmenter(dict_path, engine="jieba")

    process_jsonl_optimized(input_path, output_path, segmenter, True)
    print(
        f"Processing is complete and the labeled data has been saved to {output_path}"
    )
