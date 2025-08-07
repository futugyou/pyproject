from sentence_transformers import SentenceTransformer
from pymilvus import (
    connections,
    CollectionSchema,
    FieldSchema,
    DataType,
    Collection,
    utility,
)

model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
vector_dim = 768
fields = [
    FieldSchema(
        name="weibo_id",
        dtype=DataType.VARCHAR,
        is_primary=True,
        auto_id=False,
        max_length=100,
    ),
    FieldSchema(name="weibo_vector", dtype=DataType.FLOAT_VECTOR, dim=vector_dim),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096),
    FieldSchema(name="attitudes_count", dtype=DataType.INT64),
    FieldSchema(name="comments_count", dtype=DataType.INT64),
]
schema = CollectionSchema(fields, "Knowledge base, storing mixed vectors")


def hybrid_vectorize(data):
    """
    Mix the text information and structured information into vector form.
    """
    text_parts = [
        data["text"],
        f"event_triples:{data['analysis_result']['event_triples']}",
        f"sentiment_keywords:{data['analysis_result']['sentiment_analysis']['sentiment_keywords']}",
        f"tags:{data['tags']}",
    ]
    combined_text = " ".join(text_parts)
    vector = model.encode(combined_text)

    return vector.tolist()


def insert_vector_to_milvusdb(raw_data_list, vectors):
    try:
        connections.connect("default", host="localhost", port="19530")
        print("Successfully connected to Milvus!")
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        exit()

    if not utility.has_collection(collection_name):
        print(f"Collection '{collection_name}' does not exist. Creating now...")
        collection = Collection(name=collection_name, schema=schema)
        print(f"Collection '{collection_name}' created successfully.")
    else:
        print(f"Collection '{collection_name}' already exists. Reusing it.")
        collection = Collection(name=collection_name)

    if not collection.has_index():
        print("Index does not exist. Creating an index...")
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128},
        }
        collection.create_index(field_name="weibo_vector", index_params=index_params)
        print("Index created successfully.")
    else:
        print("Index already exists.")

    print("Loading collection to memory...")
    collection.load()
    print("Collection loaded.")

    ids = [data["weibo_id"] for data in raw_data_list]
    texts = [data["text"] for data in raw_data_list]
    attitudes_counts = [data["attitudes_count"] for data in raw_data_list]
    comments_counts = [data["comments_count"] for data in raw_data_list]

    data_to_insert = [
        ids,
        vectors,
        texts,
        attitudes_counts,
        comments_counts,
    ]

    collection = Collection(collection_name)
    mr = collection.insert(data_to_insert)

    print(f"Successfully inserted {mr.insert_count} pieces of data into Milvus.")
    connections.disconnect(alias="default")


def search_milvusdb(search_text):
    try:
        connections.connect("default", host="localhost", port="19530")
        print("Successfully connected to Milvus!")
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        exit()

    collection = Collection(name=collection_name, schema=schema)
    collection.load()

    search_vector = model.encode(search_text).tolist()

    search_params = {"nprobe": 10}
    results = collection.search(
        data=[search_vector],
        anns_field="weibo_vector",
        param=search_params,
        limit=3,
        output_fields=["text", "attitudes_count"],
    )

    print("\nSearch results:")
    for result in results[0]:
        print(f"Distance: {result.distance}")
        print(f"Weibo content: {result.entity.get('text')}")
        print(f"Number of likes: {result.entity.get('attitudes_count')}\n")

    connections.disconnect(alias="default")


def read_jsonl_to_list(file_path):
    data_list = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        data_list.append(data)
                    except json.JSONDecodeError as e:
                        print(
                            f"Error decoding JSON on line: {line.strip()}. Error: {e}"
                        )
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return data_list


def hybrid_vectorize(file_data):
    combined_texts = [hybrid_vectorize(data) for data in raw_data_list]
    return model.encode(combined_texts).tolist()


async def process_jsonl_file(input_file):
    """
    Reads a JSONL file, processes each line.
    """

    try:
        file_data = read_jsonl_to_list(input_file)
        vectors = hybrid_vectorize(file_data)
        insert_vector_to_milvusdb(file_data, vectors)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    print(f"Processing completed!")


if __name__ == "__main__":
    input_jsonl_file = "3.1.weibo_data_analyzed_structured.jsonl"
    asyncio.run(process_jsonl_file(input_jsonl_file))
