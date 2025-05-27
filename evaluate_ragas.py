from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas import evaluate
from rag_pipeline import retrieve_and_generate

def evaluate_rag():
    examples = [
        {"query": "quotes about love by Shakespeare", "ground_truths": ["Love looks not with the eyes..."]},
        {"query": "quotes on courage", "ground_truths": ["Courage is grace under pressure."]}
    ]
    evaluation_data = []
    for ex in examples:
        answer, contexts = retrieve_and_generate(ex["query"])
        evaluation_data.append({
            "query": ex["query"],
            "contexts": contexts,
            "answer": answer,
            "ground_truths": ex["ground_truths"]
        })

    results = evaluate(evaluation_data, metrics=[faithfulness, answer_relevancy, context_precision])
    print(results)

if __name__ == "__main__":
    evaluate_rag()
