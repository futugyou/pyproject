import json
import jieba
import jieba.posseg as pseg
import spacy
from collections import defaultdict

# import hanlp
from typing import List, Dict, Tuple
from collections import defaultdict, Counter

spacyNlp = spacy.load("zh_core_web_sm")


class CustomSegmenter:
    def __init__(self, dict_path: str, engine: str = "jieba"):
        self.engine = engine
        self.dict_path = dict_path
        self.domain_dict = self.load_dict(dict_path)
        self.phrase_set = set(self.domain_dict.keys())

        if engine == "jieba":
            self.init_jieba()
        # elif engine == "hanlp":
        #     self.init_hanlp()
        else:
            raise ValueError("engine must be 'jieba' or 'hanlp'")

    def load_dict(self, path: str) -> Dict[str, str]:
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
        jieba.initialize()
        for w in self.domain_dict.keys():
            jieba.add_word(w)

    # def init_hanlp(self):
    #     # hanlp.load() now returns a pipeline.
    #     # This is a general approach to load a model for segmentation and POS tagging.
    #     self.pipeline = hanlp.load(
    #         hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE
    #     )
    #     # Add custom dictionary
    #     for w in self.domain_dict.keys():
    #         self.pipeline["cws"].add_dictionary(
    #             w, "ns"
    #         )  # The second parameter is part of speech.

    def segment(self, text: str) -> List[str]:
        if self.engine == "jieba":
            return list(jieba.cut(text))
        # elif self.engine == "hanlp":
        #     return self.pipeline(text)["tok/fine"]
        else:
            return []

    def posseg(self, text: str) -> List[Tuple[str, str]]:
        if self.engine == "jieba":
            return [(word, flag) for word, flag in pseg.cut(text)]
        # elif self.engine == "hanlp":
        #     result = self.pipeline(text)
        #     return list(zip(result["tok/fine"], result["pos/ctb"]))
        return []


def merge_semantic_units(words: List[str], phrase_set: set) -> List[str]:
    merged = []
    i = 0
    while i < len(words):
        if i + 2 < len(words):
            tri = words[i] + words[i + 1] + words[i + 2]
            if tri in phrase_set:
                merged.append(tri)
                i += 3
                continue
        if i + 1 < len(words):
            bi = words[i] + words[i + 1]
            if bi in phrase_set:
                merged.append(bi)
                i += 2
                continue
        merged.append(words[i])
        i += 1
    return merged


def tag_text_grouped(
    words: List[str], domain_dict: Dict[str, str], count_mode: bool = False
) -> Dict[str, dict]:
    """
    count_mode=False: {'有害物质': ['硼砂']}
    count_mode=True:   {'有害物质': {'硼砂': 2}}
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
    return tag_result


def process_jsonl(
    input_path: str, output_path: str, segmenter: CustomSegmenter, count_mode=False
):
    domain_dict = segmenter.domain_dict
    phrase_set = segmenter.phrase_set

    with (
        open(input_path, "r", encoding="utf-8") as fr,
        open(output_path, "w", encoding="utf-8") as fw,
    ):
        for line in fr:
            data = json.loads(line)
            text = data.get("text", "")

            tokenized_text = segmenter.segment(text)
            pos_tags = segmenter.posseg(text)

            merged_words = merge_semantic_units(tokenized_text, phrase_set)
            tags = tag_text_grouped(merged_words, domain_dict, count_mode=count_mode)
            if count_mode:
                tags = {k: dict(v) for k, v in tags.items()}

            doc = spacyNlp(text)
            entity_dict = defaultdict(list)

            entity_dict = defaultdict(list)
            _ = [entity_dict[ent.text].append(ent.label_) for ent in doc.ents]

            data["tokenized_text"] = tokenized_text
            data["pos_tags"] = pos_tags
            data["tags"] = tags
            data["entity_dict"] = dict(entity_dict)

            fw.write(json.dumps(data, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    dict_path = "custom_food_dict.txt"
    input_path = "1.weibo_foodsafety.jsonl"
    output_path = "2.weibo_data_tagged.jsonl"

    # Using the jieba engine
    print("Processing using the jieba engine...")
    segmenter = CustomSegmenter(dict_path, engine="jieba")

    # Using the hanlp engine
    # Note: Loading and running HanLP may take a long time.
    # print("\nProcessing using the hanlp engine...")
    # segmenter = CustomSegmenter(dict_path, engine="hanlp")

    process_jsonl(input_path, output_path, segmenter, True)
    print(
        f"Processing is complete and the labeled data has been saved to {output_path}"
    )
