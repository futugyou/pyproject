import json
import hanlp
from typing import List, Dict, Tuple
from collections import defaultdict, Counter

# Globally load the HanLP model and custom dictionary
# Use hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE
# This model includes multiple tasks, including word segmentation, part-of-speech tagging, and named entity recognition.
# Loading HanLP may take some time because it is a large pretrained model.
hanlp_pipeline = hanlp.load(
    hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE
)

# Custom dictionary path
DICT_PATH = "custom_food_dict.txt"


def load_hanlp_custom_dict(dict_path: str):
    """
    Load custom dictionaries and integrate them into HanLP's tokenizer and named entity recognizer.
    """
    domain_dict = {}
    custom_words = []

    with open(dict_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            word = parts[0]
            tag = parts[1] if len(parts) > 1 else "CUSTOM_TERM"
            domain_dict[word] = tag
            custom_words.append(word)

    # Integrate the custom dictionary into the HanLP tokenizer
    hanlp.utils.io.dict_util.CustomDictionary.add(custom_words)

    # Return the custom dictionary for subsequent use in the `tag_custom_units` function
    return domain_dict


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


def get_event_triples(doc):
    event_triples = []
    for token in doc["syntactic_analysis"].tokens:
        if token.pos == "VERB":
            predicate = token.text
            srl_info = token.srl
            if srl_info:
                subject = None
                obj = None
                for role in srl_info:
                    if role.label == "A0":
                        subject = role.text
                    elif role.label == "A1":
                        obj = role.text

                if subject and obj:
                    event_triples.append(
                        {"subject": subject, "predicate": predicate, "object": obj}
                    )
    return event_triples


def process_jsonl_hanlp(input_path: str, output_path: str, count_mode: bool = False):
    """
    Process JSONL files using a HanLP pipeline.
    """
    # Load a custom dictionary and make the words known to HanLP
    domain_dict = load_hanlp_custom_dict(DICT_PATH)

    with (
        open(input_path, "r", encoding="utf-8") as fr,
        open(output_path, "w", encoding="utf-8") as fw,
    ):
        for line in fr:
            data = json.loads(line)
            text = data.get("text", "")

            # Process text in one go using the HanLP pipeline
            result = hanlp_pipeline(text)

            # Extract HanLP results
            tokens = result["tok/fine"]
            pos_tags = result["pos/ctb"]
            ner_tags = result["ner/msra"]

            # Combine HanLP results
            pos_tagged_tokens = list(zip(tokens, pos_tags))

            # Extract named entities
            hanlp_entities = defaultdict(list)
            for item in ner_tags:
                entity_text = "".join(tokens[item[0] : item[1]])
                entity_label = item[2]
                hanlp_entities[entity_text].append(entity_label)

            # Use the tokenization results to specifically process custom vocabulary
            custom_tags = tag_custom_units(tokens, domain_dict, count_mode)

            # Combine all results into the output data
            data["hanlp_tokens"] = tokens
            data["hanlp_pos_tags"] = pos_tagged_tokens
            data["hanlp_entities"] = dict(hanlp_entities)
            data["custom_tags"] = custom_tags
            data["raw_event_triples"] = get_event_triples(result)

            fw.write(json.dumps(data, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    input_path = "1.weibo_foodsafety.jsonl"
    output_path = "2.weibo_data_tagged_hanlp.jsonl"

    print("Processing using the HanLP engine...")
    process_jsonl_hanlp(input_path, output_path, True)
    print(
        f"Processing is complete and the labeled data has been saved to {output_path}"
    )
