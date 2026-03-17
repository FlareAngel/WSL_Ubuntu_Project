#  BM25 需要 Elasticsearch 服务在后台运行

# from beir import util, LoggingHandler
# from beir.datasets.data_loader import GenericDataLoader
# from beir.retrieval.evaluation import EvaluateRetrieval
# from beir.retrieval import models
# from beir.retrieval.search.lexical import BM25Search

# import json, logging
# logging.basicConfig(format='%(message)s', level=logging.INFO)

# # 1. 加载数据
# data_path = "/mnt/f/Ubuntu/benchmark/beir-v1.0.0/trec-covid"
# corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

# # 2. BM25 检索 Top-100
# hostname = "localhost"
# index_name = "trec-covid"
# model = BM25Search(index=index_name, hostname=hostname, initialize=True)
# retriever = EvaluateRetrieval(model)
# results = retriever.retrieve(corpus, queries)  # dict: {qid: {doc_id: score}}

# # 3. 保存候选集供重排使用
# with open("bm25_top100.json", "w") as f:
#     json.dump(results, f)

# # 4. 记录 BM25 baseline
# ndcg, _map, recall, precision = retriever.evaluate(qrels, results, [10])
# print(f"BM25 Baseline nDCG@10: {ndcg}")



#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# 完整 retrieval.py（基于 pyserini）

import json
import os
from pyserini.search.lucene import LuceneSearcher
from pyserini.index.lucene import LuceneIndexer

# ── 路径配置 ──────────────────────────────────────────
DATA_DIR     = "/mnt/f/Ubuntu/benchmark/beir-v1.0.0/trec-covid"
CORPUS_JSONL = os.path.join(DATA_DIR, "corpus.jsonl")
QUERIES_FILE = os.path.join(DATA_DIR, "queries.jsonl")
QRELS_FILE   = os.path.join(DATA_DIR, "qrels/test.tsv")
INDEX_DIR    = "/mnt/f/Ubuntu/benchmark/beir-v1.0.0/trec-covid/pyserini_index"
OUTPUT_FILE  = "bm25_top100.json"
TOP_K        = 100

# ── Step 1：把 corpus 转成 pyserini 需要的格式 ────────
def build_pyserini_corpus(corpus_jsonl, out_dir):
    """pyserini 要求每行是 {"id": ..., "contents": ...}"""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "corpus.jsonl")
    with open(corpus_jsonl, "r") as fin, open(out_path, "w") as fout:
        for line in fin:
            doc = json.loads(line)
            pyserini_doc = {
                "id": doc["_id"],
                "contents": (doc.get("title", "") + " " + doc.get("text", "")).strip()
            }
            fout.write(json.dumps(pyserini_doc) + "\n")
    print(f"Corpus 转换完成 → {out_path}")
    return out_dir

# ── Step 2：建索引 ─────────────────────────────────────
def build_index(corpus_dir, index_dir):
    if os.path.exists(index_dir) and os.listdir(index_dir):
        print(f"索引已存在，跳过建索引：{index_dir}")
        return
    print("开始建索引...")
    os.system(
        f"python -m pyserini.index.lucene "
        f"--collection JsonCollection "
        f"--input {corpus_dir} "
        f"--index {index_dir} "
        f"--generator DefaultLuceneDocumentGenerator "
        f"--threads 4 "
        f"--storePositions --storeDocvectors --storeRaw"
    )
    print(f"索引完成 → {index_dir}")

# ── Step 3：加载 queries ───────────────────────────────
def load_queries(queries_file):
    queries = {}
    with open(queries_file, "r") as f:
        for line in f:
            q = json.loads(line)
            queries[q["_id"]] = q["text"]
    print(f"加载 {len(queries)} 条 query")
    return queries

# ── Step 4：BM25 检索 ──────────────────────────────────
def bm25_retrieve(index_dir, queries, top_k=100):
    searcher = LuceneSearcher(index_dir)
    searcher.set_bm25(k1=0.9, b=0.4)   # BEIR 推荐参数

    results = {}
    for i, (qid, query_text) in enumerate(queries.items()):
        hits = searcher.search(query_text, k=top_k)
        results[qid] = {hit.docid: hit.score for hit in hits}
        if (i + 1) % 10 == 0:
            print(f"  检索进度：{i+1}/{len(queries)}")

    print(f"检索完成，共 {len(results)} 条查询")
    return results

# ── Step 5：计算 nDCG@10 baseline ─────────────────────
def evaluate(results, qrels_file):
    qrels = {}
    with open(qrels_file, "r") as f:
        next(f)  # 跳过表头 "query-id corpus-id score"
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            qid, doc_id, rel = parts[0], parts[1], int(parts[2])  # 3列
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][doc_id] = rel

    import math
    ndcg_scores = []
    for qid, doc_scores in results.items():
        if qid not in qrels:
            continue
        ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:10]

        dcg = sum(
            (2 ** qrels[qid].get(doc_id, 0) - 1) / math.log2(i + 2)
            for i, (doc_id, _) in enumerate(ranked)
        )
        ideal = sorted(qrels[qid].values(), reverse=True)[:10]
        idcg = sum(
            (2 ** rel - 1) / math.log2(i + 2)
            for i, rel in enumerate(ideal)
        )
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0)

    ndcg10 = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0
    print(f"BM25 Baseline nDCG@10: {ndcg10:.4f}")
    return ndcg10

# ── Main ───────────────────────────────────────────────
if __name__ == "__main__":
    PYSERINI_CORPUS_DIR = os.path.join(DATA_DIR, "pyserini_corpus")

    corpus_dir = build_pyserini_corpus(CORPUS_JSONL, PYSERINI_CORPUS_DIR)
    build_index(corpus_dir, INDEX_DIR)
    queries    = load_queries(QUERIES_FILE)
    results    = bm25_retrieve(INDEX_DIR, queries, top_k=TOP_K)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f)
    print(f"\nBM25 结果已保存 → {OUTPUT_FILE}")

    evaluate(results, QRELS_FILE)