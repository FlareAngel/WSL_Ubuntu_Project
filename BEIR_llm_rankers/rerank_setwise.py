import json, re
from model_loader import load_model, generate

PROMPT = """You are a relevance judge. Given a query and several passages, identify the single most relevant passage.
Output only the identifier like [2] and nothing else.

Query: {query}
{passages}
Most relevant passage:"""

def setwise_rerank(model, tokenizer, query, candidates, corpus, top_k=20, window_size=4):
    docs  = [doc_id for doc_id, _ in candidates[:top_k]]
    # 初始分数 = BM25 排名倒序
    scores = {doc: top_k - i for i, doc in enumerate(docs)}
    step   = window_size // 2

    for start in range(0, len(docs) - window_size + 1, step):
        window = docs[start: start + window_size]
        passages_text = ""
        for i, doc_id in enumerate(window):
            text = (corpus[doc_id].get("title","") + " " + corpus[doc_id].get("text",""))[:256]
            passages_text += f"[{i+1}] {text}\n\n"

        prompt = PROMPT.format(query=query, passages=passages_text)
        response = generate(model, tokenizer, prompt, max_new_tokens=8)

        match = re.search(r'\[(\d+)\]', response)
        if match:
            best_idx = int(match.group(1)) - 1
            if 0 <= best_idx < len(window):
                scores[window[best_idx]] += window_size

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


if __name__ == "__main__":
    from data_loader import load_corpus, load_queries, load_bm25_results

    print("加载数据...")
    corpus  = load_corpus()
    queries = load_queries()
    bm25    = load_bm25_results()
    model, tokenizer = load_model()

    results = {}
    for i, (qid, query_text) in enumerate(queries.items()):
        candidates = sorted(bm25[qid].items(), key=lambda x: x[1], reverse=True)
        ranked = setwise_rerank(model, tokenizer, query_text, candidates, corpus)
        results[qid] = {doc_id: score for doc_id, score in ranked}
        print(f"Setwise 进度：{i+1}/{len(queries)}")

    with open("results_setwise.json", "w") as f:
        json.dump(results, f)
    print("✅ 结果已保存 → results_setwise.json")