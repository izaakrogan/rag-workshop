# Exercise 3: RAG from Scratch

OK so in exercise 1 we used OpenAI's API and ChromaDB to build a RAG system. Now we're going to rebuild the whole thing with no embedding API and no vector database. Just sentence-transformers, numpy, and the Claude API. The point is to see every piece of the pipeline. No magic, no black boxes.

## What You'll Build

Same RAG pipeline as before, but you can see everthing:

1. **Chunking**: split documents into pieces (same as before)
2. **Embedding**: use sentence-transformers locally instead of an API
3. **Retrieval**: brute-force cosine similarity search over numpy arrays instead of a vector database
4. **Generation**: pass retrieved chunks to Claude (same as before)

## Prerequisites

- **Python 3.10 or higher** with your virtual environment active
- **sentence-transformers** installed (from exercise 2)
- An **Anthropic API key**
- The `documents/` directory (already included)

## Step 1: Install any missing dependancies

```bash
pip install sentence-transformers anthropic
```

## Step 2: Build the full pipeline

Create a file called `rag_from_scratch.py`. I've put the whole thing here so you can read through it. Have a look at each section and make sure you understand what its doing before you run it. If something doesn't make sense, come and talk to me.

```python
import os
import sys
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import anthropic

DOCUMENTS_DIR = "documents"
INDEX_FILE = "index.json"
EMBEDDINGS_FILE = "embeddings.npy"


# --- Chunking ---

def load_and_chunk(directory, chunk_size=500, overlap=50):
    """Read all text files and split them into overlapping chunks."""
    chunks = []
    for filename in sorted(os.listdir(directory)):
        if not filename.endswith((".txt", ".md")):
            continue
        filepath = os.path.join(directory, filename)
        with open(filepath, "r") as f:
            text = f.read()

        for start in range(0, len(text), chunk_size - overlap):
            chunk_text = text[start : start + chunk_size]
            if chunk_text.strip():
                chunks.append(
                    {
                        "text": chunk_text,
                        "source": filename,
                        "start": start,
                    }
                )
    return chunks


# --- Embedding ---

def embed_chunks(chunks, model):
    """Embed all chunks using a local model. Returns a numpy array."""
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings


# --- Retrieval ---

def search(query, model, embeddings, chunks, n_results=5):
    """Find the most similar chunks to a query using cosine similarity."""
    query_embedding = model.encode([query])[0]

    # Cosine similarity against all chunks
    # Since embeddings are normalised, this is just a dot product
    similarities = np.dot(embeddings, query_embedding)

    # Get the indices of the top N results
    top_indices = np.argsort(similarities)[::-1][:n_results]

    results = []
    for idx in top_indices:
        results.append(
            {
                "text": chunks[idx]["text"],
                "source": chunks[idx]["source"],
                "score": float(similarities[idx]),
            }
        )
    return results


# --- Generation ---

def generate(question, results):
    """Ask Claude to answer the question using the retrieved context."""
    client = anthropic.Anthropic()

    context = ""
    for i, r in enumerate(results):
        context += f"\n--- Chunk {i + 1} (from {r['source']}, score: {r['score']:.4f}) ---\n{r['text']}\n"

    response = client.messages.create(
        model="claude-sonnet-4-20250514",  # check https://docs.anthropic.com/en/docs/about-claude/models for latest
        max_tokens=1024,
        system="You are a helpful assistant. Answer questions based only on the provided context. If the context doesn't contain enough information to answer, say so.",
        messages=[
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            }
        ],
    )

    return response.content[0].text


# --- Index Management ---

def build_index(directory):
    """Chunk, embed, and save to disk."""
    print("Loading and chunking documents...")
    chunks = load_and_chunk(directory)
    print(f"Created {len(chunks)} chunks from {len(set(c['source'] for c in chunks))} documents")

    print("Embedding chunks (this may take a moment)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embed_chunks(chunks, model)

    # Save to disk so we don't have to re-embed every time
    np.save(EMBEDDINGS_FILE, embeddings)
    with open(INDEX_FILE, "w") as f:
        json.dump(chunks, f)

    print(f"Saved index to {INDEX_FILE} and {EMBEDDINGS_FILE}")
    return model, embeddings, chunks


def load_index():
    """Load a previously built index from disk."""
    embeddings = np.load(EMBEDDINGS_FILE)
    with open(INDEX_FILE, "r") as f:
        chunks = json.load(f)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model, embeddings, chunks


# --- Main ---

if __name__ == "__main__":
    # Build or load the index
    if os.path.exists(INDEX_FILE) and os.path.exists(EMBEDDINGS_FILE):
        print("Loading existing index...")
        model, embeddings, chunks = load_index()
    else:
        model, embeddings, chunks = build_index(DOCUMENTS_DIR)

    # Get the question
    question = " ".join(sys.argv[1:]) or "What is Dan's role in the realm?"

    print(f"\nQuestion: {question}\n")

    # Retrieve
    results = search(question, model, embeddings, chunks)

    print("Retrieved chunks:")
    for i, r in enumerate(results):
        print(f"  [{i + 1}] (score: {r['score']:.4f}) {r['source']}: {r['text'][:80]}...")
    print()

    # Generate
    answer = generate(question, results)
    print(f"Answer: {answer}")
```

## Step 3: Build the index and query

```bash
# First run builds the index (embeds all chunks)
python rag_from_scratch.py "What is Dan's role in the realm?"

# Subsequent runs load from disk
python rag_from_scratch.py "What alliance did Annie and Jess form?"
```

To rebuild the index after changing documents:

```bash
rm index.json embeddings.npy
python rag_from_scratch.py "your question here"
```

## Step 4: Compare with exercise 1

Run the same questions through both systems. Compare:

- Do they retrieve the same chunks?
- Are the similarity scores in the same ballpark?
- Does one system give better answers?

The OpenAI model (`text-embedding-3-small`) has 1536 dimensions. The local model (`all-MiniLM-L6-v2`) has 384. More dimensions doesn't automatically mean better retrieval. Try it and see.

## Quiz Questions

<details>
<summary><strong>1. Why is brute-force search fine here but not in production?</strong></summary>

Brute-force compares the query against every single chunk. It's O(n) where n is the number of chunks. With 100 chunks and 384-dimensional vectors, this takes microseconds. With 10 million chunks, it takes seconds per query, which is way too slow. Vector databases use approximate nearest neighbour (ANN) algorithms like HNSW that build a graph structure over the vectors, giving you O(log n) search. The tradeoff is you might occasionally miss the absolute nearest neighbour, but in practice "almost the nearest" is fine.

</details>

<details>
<summary><strong>2. Why do we save the index to disk instead of re-embedding every time?</strong></summary>

Embedding is the expensive step. Even with a local model, encoding hundreds of chunks takes several seconds. The vectors don't change unless the documents change, so saving them avoids redoing all that work. This is basically what vector databases do too. Embed once, query many times.

</details>

<details>
<summary><strong>3. The local model has 384 dimensions while OpenAI's has 1536. Why might the smaller model still perform well?</strong></summary>

Dimension count isn't a direct measure of quality. A well-trained 384-dimensional model can capture the semantic distinctions you need for retrieval just as well as a larger one. It's also faster to search (fewer multiplications) and uses less memory. Higher dimensions help when you need really fine-grained distinctions across a huge vocabulary of concepts, but 384 is plenty for document retrieval like what we're doing here.

</details>

<details>
<summary><strong>4. What would happen if you used a different embedding model for indexing and querying?</strong></summary>

Retrieval would break completely. Each model learns its own vector space. "Cat" might point in one direction in model A's space and a completely different direction in model B's space. Cosine similarity between vectors from different models is meaningless. This is why you always need to re-embed your entire corpus when you switch embedding models.

</details>

<details>
<summary><strong>5. In the search function, why can we use <code>np.dot(embeddings, query_embedding)</code> instead of computing cosine similarity properly?</strong></summary>

Because `all-MiniLM-L6-v2` outputs normalised vectors (unit length). When vectors are unit length, cosine similarity simplifies to just the dot product because the denominator (product of norms) is always 1.0. So we skip the norm calculations entirely, which is faster. Most embedding models normalise their outputs for exactly this reason.

</details>
