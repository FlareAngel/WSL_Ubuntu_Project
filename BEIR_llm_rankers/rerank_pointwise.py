import json
from model_loader import load_model, generate

PROMPT = """You are a relevance judge. Given a query and a passage, score the relevance on a scale of 0 to 10.
Output only a single integer and nothing else.

Query: {query}
Passage: {passage}
Score:"""

def pointwise_rerank(model, tokenizer, query, candidates, corpus, top_k=20):
    scored = []
    for doc_id, bm25_score in candidates[:top_k]:
        doc = corpus[doc_id]
        passage = (doc.get("title", "") + " " + doc.get("text", ""))[:512]
        prompt = PROMPT.format(query=query, passage=passage)
        response = generate(model, tokenizer, prompt, max_new_tokens=4)
        try:
            score = float(response.split()[0])
        except:
            score = 0.0
        scored.append((doc_id, score))
    return sorted(scored, key=lambda x: x[1], reverse=True)


if __name__ == "__main__":
    import os
    from data_loader import load_corpus, load_queries, load_bm25_results

    print("加载数据...")
    corpus   = load_corpus()
    queries  = load_queries()
    bm25     = load_bm25_results()
    model, tokenizer = load_model()

    results = {}
    for i, (qid, query_text) in enumerate(queries.items()):
        candidates = sorted(bm25[qid].items(), key=lambda x: x[1], reverse=True)
        ranked = pointwise_rerank(model, tokenizer, query_text, candidates, corpus)
        results[qid] = {doc_id: score for doc_id, score in ranked}
        print(f"Pointwise 进度：{i+1}/{len(queries)}")

    with open("results_pointwise.json", "w") as f:
        json.dump(results, f)
    print("✅ 结果已保存 → results_pointwise.json")