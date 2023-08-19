from docs2Vector import index


def get_similar_docs(indexed, query, k=1, score=False):
    if score:
        similar_docs = indexed.similarity_search_with_score(query, k=k)
    else:
        similar_docs = index.similarity_search(query)

    return similar_docs

query = "What is Pakistan's Geographic Location?"
print(get_similar_docs(index, query, k=3))
