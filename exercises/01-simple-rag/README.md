# Building a RAG System

In this exercise we're going to build a Retrieval-Augmented Generation system - a pipeline that can answer questions about a set of documents. There are three stages:

1. **Ingestion** - chunk your documents, turn each chunk into a vector embedding, and store them
2. **Retrieval** - given a question, find the most relevant chunks
3. **Generation** - pass the retrieved chunks to Claude and get an answer

We're using OpenAI for embeddings, ChromaDB as a vector database, and Claude for generation. These are independant components - it's fine to use different providers for different parts of the pipeline.

## Prerequisites

- Python 3.10+
- An Anthropic API key ([console.anthropic.com](https://console.anthropic.com))
- An OpenAI API key ([platform.openai.com](https://platform.openai.com))

## Step 1: Set up your environment

Clone the repo and navigate to this exercise:

```bash
git clone https://github.com/foundersandcoders/rag-workshop.git
cd rag-workshop/exercises/01-simple-rag
```

Create a virtual environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install anthropic openai chromadb python-dotenv
```

Create a `.env` file with your/our API keys (see Discord):

```
ANTHROPIC_API_KEY=your-anthropic-api-key
OPENAI_API_KEY=your-openai-api-key
```

## Step 2: Explore the documents

The `documents/` directory contains articles about the fictional microstate of Founders and Coders. I made all of this up - Claude doesn't know any of it from its training data, so the RAG system has to actually work to answer questions about it.

```bash
ls documents/
```

You should see files like `founders_and_coders.txt`, `dan_the_archmage.txt`, `semantic_kelp.txt`, etc. Have a read through a few of them - you'll need to know what's in there to judge whether your RAG system is giving good answers.

## Step 3: Build the ingestion pipeline

Create a file called `ingest.py`. This script reads your documents, chunks them, embeds the chunks, and stores everything in ChromaDB. Read through it before you run it, and if something doesn't make sense, come and talk to me.

```python
import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

load_dotenv()

DOCUMENTS_DIR = "documents"

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

        # Simple character-based chunking with overlap
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


# --- Embedding and Storage ---

def embed_and_store(chunks):
    """Embed chunks with OpenAI and store in ChromaDB."""
    openai = OpenAI()
    db = chromadb.PersistentClient(path="./chroma_db")

    # Delete existing collection if re-running
    try:
        db.delete_collection("documents")
    except Exception:
        pass

    collection = db.create_collection(
        name="documents",
        metadata={"hnsw:space": "cosine"},
    )

    # OpenAI accepts batches of up to 2048 texts
    batch_size = 128
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c["text"] for c in batch]

        response = openai.embeddings.create(input=texts, model="text-embedding-3-small")
        embeddings = [item.embedding for item in response.data]

        collection.add(
            ids=[f"chunk_{i + j}" for j in range(len(batch))],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{"source": c["source"], "start": c["start"]} for c in batch],
        )

    print(f"Stored {len(chunks)} chunks from {len(set(c['source'] for c in chunks))} documents")
    return collection


if __name__ == "__main__":
    chunks = load_and_chunk(DOCUMENTS_DIR)
    embed_and_store(chunks)
```

Run it:

```bash
python ingest.py
```

You should see output like: `Stored 127 chunks from 12 documents`

So what's going on here? We're splitting each document into 500-character chunks with a 50-character overlap. The overlap is important - without it, sentences that fall on a boundary get cut in half and neither chunk has the complete thought. We then send each chunk to OpenAI's `text-embedding-3-small` model, which returns a 1536-dimensional vector for each one. ChromaDB stores these vectors along with the original text so we can search by similarity later. The `hnsw:space: cosine` bit tells ChromaDB to use cosine similarity for search.

## Step 4: Build the retrieval and generation pipeline

Create a file called `query.py`:

```python
import sys
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
import anthropic

load_dotenv()


def retrieve(question, n_results=5):
    """Find the most relevant chunks for a question."""
    openai = OpenAI()
    db = chromadb.PersistentClient(path="./chroma_db")
    collection = db.get_collection("documents")

    # Embed the question with the same model used for documents
    response = openai.embeddings.create(input=[question], model="text-embedding-3-small")
    query_embedding = response.data[0].embedding

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )

    return results["documents"][0], results["metadatas"][0]


def generate(question, context_chunks, metadata):
    """Ask Claude to answer the question using the retrieved context."""
    client = anthropic.Anthropic()

    # Build the context string
    context = ""
    for i, (chunk, meta) in enumerate(zip(context_chunks, metadata)):
        context += f"\n--- Chunk {i + 1} (from {meta['source']}) ---\n{chunk}\n"

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


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What is Dan's role in the realm?"

    print(f"Question: {question}\n")

    chunks, metadata = retrieve(question)

    print("Retrieved chunks:")
    for i, (chunk, meta) in enumerate(zip(chunks, metadata)):
        print(f"  [{i + 1}] {meta['source']}: {chunk[:80]}...")
    print()

    answer = generate(question, chunks, metadata)
    print(f"Answer: {answer}")
```

Run it:

```bash
python query.py "What is Dan's role in the realm?"
```

The important thing to notice here is that we embed the query with the same model we used for the documents (`text-embedding-3-small`). If you use a different model, the vectors live in different spaces and similarity search won't work at all.

We retrieve the top 5 chunks by default (`n_results=5`). More chunks means more context for Claude but also more noise. The system prompt tells Claude to only answer from the provided context - without this it might just use its own training data, which defeats the point. Each chunk is labelled with its source file so you can trace where the information came from.

## Step 5: Try it out

Ask a few questions and see how it goes:

```bash
python query.py "What is Dan's role in the realm?"
python query.py "How did Jaz and Jason first meet?"
python query.py "What alliance did Annie and Jess form?"
python query.py "What is Izaak's familiar called?"
```

You know the right answers because you've read the source documents. Check whether the retrieved chunks are actually relevant, and whether Claude's answer matches what's in them.

Now try asking something that isn't in the lore:

```bash
python query.py "What is the capital of France?"
```

The system prompt tells Claude to only answer from context - does it refuse, or does it hallucinate?

## How it works

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Your Query  │────►│   OpenAI     │────►│   ChromaDB   │
│              │     │  (embed)     │     │  (search)    │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                 │
                                          top N chunks
                                                 │
                                                 ▼
                                         ┌──────────────┐
                                         │   Claude     │
                                         │  (generate)  │
                                         └──────────────┘
```

**Ingestion** (run once): read documents from disk, split into overlapping chunks, embed each chunk with OpenAI, store embeddings + text in ChromaDB.

**Query** (run per question): embed the question with OpenAI, find the nearest chunks in ChromaDB by cosine similarity, pass the chunks + question to Claude, get an answer grounded in the context.

## Quiz questions

<details>
<summary><strong>1. What is an embedding?</strong></summary>

An embedding is a list of numbers that captures the meaning of a piece of text. Texts with similar meanings get similar numbers, which lets us find relevant documents by comparing these number lists. If a question's numbers are close to a chunk's numbers, that chunk is probably relevant.

</details>

<details>
<summary><strong>2. What is cosine similarity?</strong></summary>

It measures how similar two vectors are by looking at the angle between them, ignoring their length. If two vectors point in the same direction the similarity is 1 (identical meaning), if they're perpendicular it's 0 (unrelated), and if they point in opposite directions it's -1. We care about direction rather than magnitude because that's what captures meaning.

</details>

<details>
<summary><strong>3. Why must you use the same embedding model for documents and queries?</strong></summary>

Each model maps text into its own vector space. If you embed documents with one model and queries with another, the vectors are in completely different spaces and the numbers aren't comparable. It's like measuring one thing in kilometres and another in pounds, then trying to find the "closest" match.

</details>

<details>
<summary><strong>4. What happens if you remove the overlap between chunks?</strong></summary>

Sentences that fall on chunk boundaries get cut in half. The first half ends up in one chunk and the second half in another - neither contains the complete thought. If a question targets that sentence, retrieval might find a partial match or miss it entirely. Overlap ensures boundary sentences appear in full in at least one chunk.

</details>

<details>
<summary><strong>5. What would happen if you embedded each document as one giant chunk?</strong></summary>

The embedding would average over the entire document's meaning. If a document covers 10 different topics, the embedding sits in a vague middle ground between all of them, so a question about one specific topic gets a weak similarity score. Smaller chunks produce more focussed embeddings that match specific questions better.

</details>

<details>
<summary><strong>6. What's the tradeoff in choosing how many chunks to retrieve?</strong></summary>

More chunks means you're more likely to include the relevant information, but you also include more irrelevant text. This adds noise that can confuse Claude and uses more tokens (costing more and adding latency). Fewer chunks keeps things lean but risks missing the relevant passage. I find 3-5 chunks is a reasonable starting point.

</details>

