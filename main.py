# Standard library imports
import os
import json
import datetime
import time
import re
import argparse

# Third-party imports
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# NOTE: We removed torch / transformers heavy initialisation for summarisation to
# keep the total model footprint small (< 1 GB) and speed up CPU inference.

# --- CONFIGURATION ---
DEFAULT_COLLECTION = os.path.join("sample_data", "input", "collection1")

# These will be overridden by CLI argument
COLLECTION_DIR = DEFAULT_COLLECTION
PDF_DIR = os.path.join(COLLECTION_DIR, "pdfs")
INPUT_CONFIG_PATH = os.path.join(COLLECTION_DIR, "challenge1b_input.json")
OUTPUT_DIR = COLLECTION_DIR  # save result next to input
OUTPUT_JSON_PATH = os.path.join(OUTPUT_DIR, "challenge1b_output.json")

# Sentence-Transformers model chosen for its small size (~80 MB) and strong
# sentence-level semantic performance on CPU.
EMBEDDING_MODEL_NAME = "fine_tuned_model"
TOP_K_SECTIONS = 5

# --- STAGE 1: PDF Extraction & Chunking ---
def extract_and_chunk_pdfs(pdf_files):
    print("Stage 1: Starting PDF extraction and chunking...")
    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        file_path = os.path.join(PDF_DIR, pdf_file)
        try:
            doc = fitz.open(file_path)
            # Pylance/mypy stubs for PyMuPDF don't mark Document as iterable; use
            # an explicit index loop to satisfy the type checker.
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                page_text = page.get_text("text")  # type: ignore[attr-defined]
                if not page_text or len(page_text.strip()) < 50:
                    continue
                chunks = text_splitter.split_text(page_text)
                for chunk_text in chunks:
                    all_chunks.append({
                        "source": os.path.basename(pdf_file),
                        "page_number": page_num + 1,
                        "text": chunk_text
                    })
            doc.close()
        except Exception as e:
            print(f"  - Error processing {pdf_file}: {e}")
    print(f"  - Successfully created {len(all_chunks)} chunks from {len(pdf_files)} documents.")
    print("Stage 1: Completed.\n")
    return all_chunks

# --- STAGE 2: Embedding & Relevance Ranking ---
def rank_chunks_by_relevance(chunks, persona, job_to_be_done, model):
    print("Stage 2: Embedding and Relevance Ranking...")
    query = f"Persona: {persona}. Task: {job_to_be_done}"
    query_embedding = model.encode([query], show_progress_bar=False)
    chunk_texts = [chunk['text'] for chunk in chunks]
    chunk_embeddings = model.encode(chunk_texts, show_progress_bar=True)
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]

    for i, chunk in enumerate(chunks):
        chunk['relevance_score'] = float(similarities[i])

    ranked_chunks = sorted(chunks, key=lambda x: x['relevance_score'], reverse=True)
    for i, chunk in enumerate(ranked_chunks):
        chunk['importance_rank'] = i + 1
    print("Stage 2: Completed.\n")
    return ranked_chunks

def _select_diverse_chunks(ranked_chunks, top_k):
    """Return up to *top_k* chunks while maximising document diversity.

    The input *ranked_chunks* list is assumed to be sorted by relevance
    (highest first). The function first selects at most one chunk per
    source document to encourage coverage across different PDFs. If fewer
    than *top_k* unique-source chunks are available, the remaining slots are
    filled with the next most relevant chunks regardless of source.
    """
    selected = []
    seen_sources = set()

    # First pass: choose one chunk per source for diversity
    for chunk in ranked_chunks:
        src = chunk.get("source")
        if src in seen_sources:
            continue
        selected.append(chunk)
        seen_sources.add(src)
        if len(selected) >= top_k:
            return selected

    # Second pass: top-up with remaining high-score chunks
    for chunk in ranked_chunks:
        if len(selected) >= top_k:
            break
        if chunk not in selected:
            selected.append(chunk)

    return selected

# --- STAGE 3: Summarization & Output Formatting ---
# Lightweight semantic sentence selection instead of loading an additional large
# summariser model. We re-use the same embedding model to pick the most
# relevant sentences for each top chunk, dramatically reducing memory/time.

def _refine_text_semantic(chunk_text: str, query_embedding, model, top_n: int = 3) -> str:
    """Return the *top_n* sentences in *chunk_text* that are closest to the query.

    The function performs a naive sentence split (to avoid external NLTK/Spacy
    dependencies) and ranks each sentence by cosine similarity w.r.t the query.
    """

    # Basic sentence segmentation (keeps delimiters).
    sentences = re.split(r"(?<=[.!?])\s+", chunk_text.strip())
    if len(sentences) <= top_n:
        return " ".join(sentences)

    sentence_embeddings = model.encode(sentences, show_progress_bar=False)
    sims = cosine_similarity(query_embedding, sentence_embeddings)[0]
    # Get indices of top_n sentences.
    top_idx = sims.argsort()[::-1][:top_n]
    # Preserve original order for readability.
    top_sentences = [sentences[i] for i in sorted(top_idx)]
    return " ".join(top_sentences)


def analyze_and_format_output(ranked_chunks, input_data, query_embedding, embedding_model):
    print("Stage 3: Performing Sub-section Analysis and Formatting Output...")
    metadata_content = input_data.get("metadata", {})
    output_data = {
        "metadata": {
            "input_documents": metadata_content.get("input_documents", []),
            "persona": metadata_content.get("persona", ""),
            "job_to_be_done": metadata_content.get("job_to_be_done", ""),
            "processing_timestamp": datetime.datetime.now().isoformat()
        },
        "extracted_sections": [],
        "sub_section_analysis": []
    }

    top_chunks = _select_diverse_chunks(ranked_chunks, TOP_K_SECTIONS)
    # Reassign importance_rank after diversity filter
    for idx, ch in enumerate(top_chunks, 1):
        ch["importance_rank"] = idx

    for chunk in tqdm(top_chunks, desc="Refining Sections"):
        # Attempt to infer a simple section title: take the first line of the chunk
        # that looks like a heading (short, capitalised, or ends with ':').
        first_line = chunk["text"].split("\n", 1)[0].strip()
        if len(first_line) > 100 or len(first_line.split()) < 2:
            section_title = "N/A"
        else:
            section_title = first_line

        output_data["extracted_sections"].append({
            "document": chunk['source'],
            "page_number": chunk['page_number'],
            "section_title": section_title,
            "importance_rank": chunk['importance_rank']
        })
        refined_text = _refine_text_semantic(chunk['text'], query_embedding, embedding_model)
        output_data["sub_section_analysis"].append({
            "document": chunk['source'],
            "page_number": chunk['page_number'],
            "refined_text": refined_text
        })
    print("Stage 3: Completed.\n")
    return output_data

# --- MAIN FUNCTION ---
def main():
    parser = argparse.ArgumentParser(description="Run persona-driven PDF ranking")
    parser.add_argument("collection", nargs="?", default=DEFAULT_COLLECTION,
                        help="Path to a collection folder (contains pdfs/, challenge1b_input.json)")
    args = parser.parse_args()

    global COLLECTION_DIR, PDF_DIR, INPUT_CONFIG_PATH, OUTPUT_DIR, OUTPUT_JSON_PATH
    COLLECTION_DIR = args.collection
    PDF_DIR = os.path.join(COLLECTION_DIR, "pdfs")
    INPUT_CONFIG_PATH = os.path.join(COLLECTION_DIR, "challenge1b_input.json")
    OUTPUT_DIR = COLLECTION_DIR
    OUTPUT_JSON_PATH = os.path.join(OUTPUT_DIR, "challenge1b_output.json")

    print(f"--- Processing collection: {COLLECTION_DIR} ---\n")
    start_time = time.time()

    with open(INPUT_CONFIG_PATH, 'r') as f:
        config = json.load(f)

    meta = config.get("metadata", config)  # support root-level
    persona = meta.get("persona", "")
    job_to_be_done = meta.get("job_to_be_done", "")
    pdf_files = meta.get("input_documents")

    if pdf_files is None:
        # Fallback: use every PDF in pdfs/ folder
        pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]

    # Ensure config contains a proper metadata dict for downstream output
    if "metadata" not in config:
        config["metadata"] = {}
    config["metadata"].update({
        "persona": persona,
        "job_to_be_done": job_to_be_done,
        "input_documents": pdf_files
    })

    if not all([persona, job_to_be_done, pdf_files]):
        print("Error: 'persona', 'job_to_be_done', or 'input_documents' missing from input.json. Exiting.")
        return

    chunks = extract_and_chunk_pdfs(pdf_files)
    if not chunks:
        print("No text could be extracted from the provided PDFs. Exiting.")
        return

    print("Loading sentence-embedding model (all-MiniLM-L6-v2)…")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Model loaded successfully.\n")

    ranked_chunks = rank_chunks_by_relevance(chunks, persona, job_to_be_done, embedding_model)

    # Reuse query embedding for sentence-level refinement.
    query_text = f"Persona: {persona}. Task: {job_to_be_done}"
    query_embedding = embedding_model.encode([query_text], show_progress_bar=False)

    final_output = analyze_and_format_output(ranked_chunks, config, query_embedding, embedding_model)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(final_output, f, indent=4)

    end_time = time.time()
    print("--- Processing Complete ---")
    print(f"Total processing time: {end_time - start_time:.2f} seconds.")
    print(f"Output saved to: {OUTPUT_JSON_PATH}")
    print("---------------------------\n")

if __name__ == "__main__":
    main()
