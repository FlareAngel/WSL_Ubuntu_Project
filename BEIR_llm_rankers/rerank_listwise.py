import json, re
from model_loader import load_model, generate

PROMPT = """You are a relevance judge. Rank the following passages by relevance to the query from most to least relevant.
Output only the ranking as identifiers separated by ' > ', e.g.: [1] > [3] > [2]
Do not output anything else.

Query: {query}
{passages}
Ranking:"""

def listwise_rerank(model, tokenizer, query, candidates, corpus, top_k=20):
    docs = candidates[:top_k]
    passages_text = ""
    for i, (doc_id, _) in enumerate(docs):
        text = (corpus[doc_id].get("title","") + " " + corpus[doc_id].get("text",""))[:256]
        passages_text += f"[{i+1}] {text}\n\n"

    prompt = PROMPT.format(query=query, passages=passages_text)
    response = generate(model, tokenizer, prompt, max_new_tokens=128)

    order = re.findall(r'\[(\d+)\]', response)
    ranked, seen = [], set()
    for idx_str in order:
        idx = int(idx_str) - 1
        if 0 <= idx < len(docs) and idx not in seen:
            doc_id = docs[idx][0]
            ranked.append((doc_id, len(docs) - len(ranked)))
            seen.add(idx)

    # 未被排到的文档补到末尾
    for idx, (doc_id, _) in enumerate(docs):
        if idx not in seen:
            ranked.append((doc_id, 0))

    return ranked


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
        ranked = listwise_rerank(model, tokenizer, query_text, candidates, corpus)
        results[qid] = {doc_id: score for doc_id, score in ranked}
        print(f"Listwise 进度：{i+1}/{len(queries)}")

    with open("results_listwise.json", "w") as f:
        json.dump(results, f)
    print("✅ 结果已保存 → results_listwise.json")