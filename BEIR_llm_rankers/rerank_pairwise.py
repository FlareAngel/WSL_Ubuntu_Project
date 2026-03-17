import json
from model_loader import load_model, generate

PROMPT = """You are a relevance judge. Given a query and two passages, determine which passage is more relevant.
Output only 'A' or 'B' and nothing else.

Query: {query}
Passage A: {passage_a}
Passage B: {passage_b}
Answer:"""

def pairwise_compare(model, tokenizer, query, doc_a, doc_b, corpus):
    text_a = (corpus[doc_a].get("title","") + " " + corpus[doc_a].get("text",""))[:300]
    text_b = (corpus[doc_b].get("title","") + " " + corpus[doc_b].get("text",""))[:300]
    prompt = PROMPT.format(query=query, passage_a=text_a, passage_b=text_b)
    response = generate(model, tokenizer, prompt, max_new_tokens=4)
    return "A" if "A" in response.upper() else "B"

def pairwise_rerank(model, tokenizer, query, candidates, corpus, top_k=20):
    docs = [doc_id for doc_id, _ in candidates[:top_k]]
    wins = {doc: 0 for doc in docs}
    # Round-robin tournament
    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            winner = pairwise_compare(model, tokenizer, query, docs[i], docs[j], corpus)
            if winner == "A":
                wins[docs[i]] += 1
            else:
                wins[docs[j]] += 1
    return sorted(wins.items(), key=lambda x: x[1], reverse=True)


if __name__ == "__main__":
    import json
    from data_loader import load_corpus, load_queries, load_bm25_results

    print("加载数据...")
    corpus  = load_corpus()
    queries = load_queries()
    bm25    = load_bm25_results()
    model, tokenizer = load_model()

    results = {}
    for i, (qid, query_text) in enumerate(queries.items()):
        candidates = sorted(bm25[qid].items(), key=lambda x: x[1], reverse=True)
        ranked = pairwise_rerank(model, tokenizer, query_text, candidates, corpus)
        results[qid] = {doc_id: score for doc_id, score in ranked}
        print(f"Pairwise 进度：{i+1}/{len(queries)}")

    with open("results_pairwise.json", "w") as f:
        json.dump(results, f)
    print("✅ 结果已保存 → results_pairwise.json")