import json, math

QRELS_PATH = "/mnt/f/Ubuntu/benchmark/beir-v1.0.0/trec-covid/qrels/test.tsv"

RESULT_FILES = {
    "BM25_baseline": "bm25_top100.json",
    "Pointwise":     "results_pointwise.json",
    "Pairwise":      "results_pairwise.json",
    "Listwise":      "results_listwise.json",
    "Setwise":       "results_setwise.json",
}

def load_qrels(path):
    qrels = {}
    with open(path) as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            qid, doc_id, rel = parts[0], parts[1], int(parts[2])
            qrels.setdefault(qid, {})[doc_id] = rel
    return qrels

def ndcg_at_k(results, qrels, k=10):
    scores = []
    for qid, doc_scores in results.items():
        if qid not in qrels:
            continue
        ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        dcg  = sum((2**qrels[qid].get(d,0)-1)/math.log2(i+2) for i,(d,_) in enumerate(ranked))
        idcg = sum((2**r-1)/math.log2(i+2) for i,r in enumerate(sorted(qrels[qid].values(),reverse=True)[:k]))
        scores.append(dcg/idcg if idcg > 0 else 0)
    return sum(scores)/len(scores) if scores else 0

if __name__ == "__main__":
    import os
    qrels = load_qrels(QRELS_PATH)

    print(f"\n{'方法':<20} {'nDCG@10':>10}")
    print("-" * 32)
    for method, path in RESULT_FILES.items():
        if not os.path.exists(path):
            print(f"{method:<20} {'未生成':>10}")
            continue
        with open(path) as f:
            results = json.load(f)
        score = ndcg_at_k(results, qrels, k=10)
        print(f"{method:<20} {score:>10.4f}")