import json

CORPUS_PATH  = "/mnt/f/Ubuntu/benchmark/beir-v1.0.0/trec-covid/corpus.jsonl"
QUERIES_PATH = "/mnt/f/Ubuntu/benchmark/beir-v1.0.0/trec-covid/queries.jsonl"
BM25_PATH    = "bm25_top100.json"

def load_corpus():
    corpus = {}
    with open(CORPUS_PATH) as f:
        for line in f:
            doc = json.loads(line)
            corpus[doc["_id"]] = doc
    print(f"Corpus 加载完成：{len(corpus)} 篇文档")
    return corpus

def load_queries():
    queries = {}
    with open(QUERIES_PATH) as f:
        for line in f:
            q = json.loads(line)
            queries[q["_id"]] = q["text"]
    print(f"Queries 加载完成：{len(queries)} 条")
    return queries

def load_bm25_results():
    with open(BM25_PATH) as f:
        results = json.load(f)
    print(f"BM25 结果加载完成：{len(results)} 条查询")
    return results