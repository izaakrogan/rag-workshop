# Exercise 2: Exploring Embeddings

In exercise 1, embeddings were a black box. You called an API and got vectors back. In this exercise we'll embed sentences locally, compute similarity ourselves, and see how retrieval actually works.

No API calls, no vector database. Just vectors and maths.

## Prerequisites

- Your venv from exercise 1 activated

Navigate to this exercise's directory:

```bash
cd 02-exploring-embeddings
```

(Or `cd ../02-exploring-embeddings` if you're still in the exercise 1 directory.)

## Step 1: Install sentence-transformers

```bash
pip install sentence-transformers
```

This pulls in PyTorch and lets you run embedding models locally. No API key needed.

## Step 2: Embed some sentences

Create a file called `explore_embeddings.py`:

```python
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "The cat sat on the mat",
    "A kitten was resting on the rug",
    "The stock market crashed yesterday",
    "Financial markets experienced a sharp decline",
    "I enjoy eating pizza on Fridays",
]

embeddings = model.encode(sentences)

print(f"Number of sentences: {len(embeddings)}")
print(f"Embedding dimension: {embeddings.shape[1]}")
print(f"First embedding (first 10 values): {embeddings[0][:10]}")
print(f"Embedding norm: {np.linalg.norm(embeddings[0]):.4f}")
```

Run it:

```bash
python explore_embeddings.py
```

Each sentence is now a vector of 384 numbers. The norm should be close to 1.0 because these are normalised vectors (unit vectors), which matters in a sec.

`all-MiniLM-L6-v2` is a small model that runs on CPU in under a second. `model.encode()` takes a list of strings and returns a numpy array of shape `(n_sentences, 384)`.

## Step 3: Compute cosine similarity

Add this to the bottom of your script:

```python
def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


print("\nPairwise cosine similarities:")
print("-" * 50)
for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        sim = cosine_similarity(embeddings[i], embeddings[j])
        print(f"{sim:.4f}  |  '{sentences[i][:40]}' vs '{sentences[j][:40]}'")
```

Run it again. You should see high similarity (~0.5-0.7) between the semantically similar pairs ("cat on the mat" / "kitten on the rug", "stock market crashed" / "financial markets declined") and low similarity (~0.0-0.2) between unrelated pairs.

Cosine similarity measures the angle between two vectors: 1.0 means they point in the same direction, 0.0 means perpendicular, -1.0 means opposite. Since our vectors are normalised, `cosine_similarity(a, b)` is equivilant to just `np.dot(a, b)` because the division by norms doesn't do anything when both norms are 1.0. I've included the full formula here because it still works if you switch to a model that doesn't normalise its outputs, but in exercise 3 we'll just use the dot product directly.

## Step 4: Simulate retrieval

Add this to the bottom of your script. Given a query, find the most similar "document":

```python
query = "What happened in the financial markets?"
query_embedding = model.encode([query])[0]

print(f"\nQuery: '{query}'")
print("Similarities to each sentence:")
print("-" * 50)

similarities = []
for i, sentence in enumerate(sentences):
    sim = cosine_similarity(query_embedding, embeddings[i])
    similarities.append((sim, sentence))
    print(f"  {sim:.4f}  |  '{sentence}'")

# Sort by similarity (highest first)
similarities.sort(reverse=True)
print(f"\nBest match: '{similarities[0][1]}'")
```

The financial query should rank the two market-related sentences highest. Thats retrieval. Five lines of numpy. Everything else in a RAG system is just scaffolding around this.

## Step 5: Have a play

Mess around with it. Some ideas:

- Add more sentences and see how they cluster
- Try queries phrased very differently from the documents. Does retrieval still work?
- What happens with ambiguous words? Try "bank" in a financial context vs a river context
- Embed a long paragraph vs a short sentence on the same topic. Does length affect similarity?

If something surprising happens, come talk to me about it.

## Quiz Questions

<details>
<summary><strong>1. Why are the embeddings 384-dimensional and not, say, 3?</strong></summary>

In 3 dimensions there just aren't enough "directions" to separate thousands of different concepts. 384 gives the model room to encode things like tone, topic, specificity all at once. You pay for it in storage and compute, but it's a small model so the cost is negligible.

</details>

<details>
<summary><strong>2. If the embeddings are already normalised, why bother writing the full cosine similarity formula?</strong></summary>

For normalised vectors, cosine similarity equals the dot product. The norms are both 1.0 so dividing by them does nothing. We wrote the full formula because it's safer: if you later switch to a model that doesn't normalise, or if you average two embeddings together, the full formula still works. The dot product shortcut is fine when you know the vectors are unit length.

</details>

<details>
<summary><strong>3. What does it mean geometrically when two embeddings have cosine similarity of 0?</strong></summary>

They're perpendicular, pointing in completely unrelated directions. Note this doesn't mean they're "opposites" (that would be -1.0), just that the model sees no relationship between them. You rarely get exactly 0 in practice; unrelated sentences usually land around 0.0 to 0.3.

</details>
