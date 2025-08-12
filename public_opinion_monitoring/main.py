from typing import Any, Literal, Annotated
import os
import click
import asyncio

from spider.weibo_spider import search_weibo, save_data_as_jsonl
from tokenization.tokenize_handler import CustomSegmenter, process_text_tokenization

# hanlp is too big to import into venv.
# from tokenization.tokenize_hanlp import (
#     process_text_tokenization as hanlp_process_text_tokenization,
# )

from nlp.analyze import process_llm_calling
from nlp.analyze_structured import process_llm_calling as process_llm_calling_structured

from knowledge_graph.gragh_with_py2neo import (
    generate_knowledge_graph as generate_knowledge_graph_py2neo,
)
from knowledge_graph.neo_gragh import generate_knowledge_graph


@click.command()
@click.option(
    "--process",
    default="graph",
    type=click.Choice(["all", "spider", "tokenization", "nlp", "graph"]),
    help="Workflow",
)
@click.option(
    "--token_type",
    default="nomal",
    type=click.Choice(["hanlp", "nomal"]),
    help="token type",
)
@click.option(
    "--nlp_type",
    default="structured",
    type=click.Choice(["structured", "nomal"]),
    help="nlp type",
)
@click.option(
    "--graph_type",
    default="neo4j",
    type=click.Choice(["py2neo", "neo4j"]),
    help="graph type",
)
@click.option("--keyword", default="食品安全", help="keyword to search")
@click.option("--pages", default=5, help="max pages to process")
@click.option(
    "--spider_output", default="1.weibo_foodsafety.jsonl", help="save spider result"
)
@click.option(
    "--tokenize_output",
    default="2.weibo_data_tagged.jsonl",
    help="save tokenization result",
)
@click.option(
    "--llm_output",
    default="3.weibo_data_analyzed.jsonl",
    help="save llm result",
)
@click.option("--dict_path", default="custom_food_dict.txt", help="dictionary Path")
def cli(
    keyword: str,
    process: Literal["all", "spider", "tokenization", "nlp", "graph"] = "all",
    token_type: Literal["hanlp", "nomal"] = "nomal",
    nlp_type: Literal["structured", "nomal"] = "structured",
    graph_type: Literal["neo4j", "py2neo"] = "neo4j",
    pages: int = 5,
    dict_path: str = "custom_food_dict.txt",
    spider_output: str = "1.weibo_foodsafety.jsonl",
    tokenize_output: str = "2.weibo_data_tagged.jsonl",
    llm_output: str = "3.weibo_data_analyzed.jsonl",
):
    asyncio.run(
        main(
            keyword,
            process,
            token_type,
            nlp_type,
            graph_type,
            pages,
            dict_path,
            spider_output,
            tokenize_output,
            llm_output,
        )
    )


async def main(
    keyword: str,
    process: Literal["all", "spider", "tokenization", "nlp", "graph"] = "all",
    token_type: Literal["hanlp", "nomal"] = "nomal",
    nlp_type: Literal["structured", "nomal"] = "structured",
    graph_type: Literal["neo4j", "py2neo"] = "neo4j",
    pages: int = 5,
    dict_path: str = "custom_food_dict.txt",
    spider_output: str = "1.weibo_foodsafety.jsonl",
    tokenize_output: str = "2.weibo_data_tagged.jsonl",
    llm_output: str = "3.weibo_data_analyzed.jsonl",
):
    if process == "spider":
        data = search_weibo(keyword, max_pages=pages)
        save_data_as_jsonl(data, filename=spider_output)
        return

    if process == "tokenization":
        if token_type == "nomal":
            segmenter = CustomSegmenter(dict_path)
            process_text_tokenization(spider_output, tokenize_output, segmenter, True)
            return
        else:
            hanlp_process_text_tokenization(spider_output, tokenize_output, True)
            return

    if process == "nlp":
        if nlp_type == "structured":
            await process_llm_calling_structured(tokenize_output, llm_output)
            return
        else:
            await process_llm_calling(tokenize_output, llm_output)
            return

    if process == "graph":
        if graph_type == "py2neo":
            generate_knowledge_graph_py2neo(llm_output)
            return
        else:
            await generate_knowledge_graph(llm_output)
            return

    if process == "all":
        data = search_weibo(keyword, max_pages=pages)
        save_data_as_jsonl(data, filename=spider_output)

        if token_type == "nomal":
            segmenter = CustomSegmenter(dict_path)
            process_text_tokenization(spider_output, tokenize_output, segmenter, True)
        else:
            hanlp_process_text_tokenization(spider_output, tokenize_output, True)

        if nlp_type == "structured":
            await process_llm_calling_structured(tokenize_output, llm_output)
        else:
            await process_llm_calling(tokenize_output, llm_output)

        if graph_type == "py2neo":
            generate_knowledge_graph_py2neo(llm_output)
        else:
            await generate_knowledge_graph(llm_output)


if __name__ == "__main__":
    cli()
    # asyncio.run(main())
