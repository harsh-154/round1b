import os
import json
import random
from typing import List

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

"""train_model.py
===================
Fine-tune the MiniLM sentence-transformer on your labelled PDF chunks.

We assume each training collection has:
• sample_data/<collection_name>/input/input.json – contains persona, job, and list of PDFs
• sample_data/<collection_name>/output/result.json – our manually curated ground-truth

For every collection we create pairs (query, chunk_text) with a score: 1 for the
chunks that appear in extracted_sections, 0 for randomly sampled other chunks.
We then use a CosineSimilarityLoss to pull positives closer to the query
embedding and push negatives away.

Run:
  python train_model.py             # trains and saves ./fine_tuned_model
"""

# ---------- CONFIG -----------------------------------------------------------
TRAIN_ROOT = os.path.join("sample_data", "input")  # collections live here
MODEL_NAME = "all-MiniLM-L6-v2"  # Base model
OUTPUT_DIR = "fine_tuned_model"  # Saved model path
MAX_COLLECTIONS = None           # Limit to N collections for quick experiments
POSITIVE_WEIGHT = 1.0            # Weight for positive samples (kept for clarity)
NUM_NEGATIVES = 4                # Negatives sampled per positive
BATCH_SIZE = 16
NUM_EPOCHS = 3
LR = 2e-5
SEED = 42

# ---------------------------------------------------------------------------
random.seed(SEED)


def discover_collections(root: str) -> List[str]:
    """Return list of sub-directories that look like separate collections."""
    collections = []
    for entry in os.listdir(root):
        coll_path = os.path.join(root, entry)
        if not os.path.isdir(coll_path):
            continue
        # Needs both input and output subfolders
        inp = os.path.join(coll_path, "challenge1b_input.json")
        out = os.path.join(coll_path, "challenge1b_output.json")
        if os.path.isfile(inp) and os.path.isfile(out):
            collections.append(coll_path)
    return collections


def load_ground_truth(collection_path: str):
    inp_path = os.path.join(collection_path, "challenge1b_input.json")
    out_path = os.path.join(collection_path, "challenge1b_output.json")
    with open(inp_path, "r", encoding="utf-8") as f:
        inp = json.load(f)
    with open(out_path, "r", encoding="utf-8") as f:
        out = json.load(f)
    # Some JSONs embed fields directly without a 'metadata' wrapper.
    meta = inp.get("metadata", inp)  # fall back to root level
    try:
        persona = meta["persona"]
        job = meta["job_to_be_done"]
    except KeyError as e:
        raise KeyError(f"Missing expected key {e.args[0]} in {inp_path}. Ensure the JSON contains either a 'metadata' block or the keys at root level.") from e
    query = f"Persona: {persona}. Task: {job}"
    positives = {(sec["page_number"], sec["document"]): 1 for sec in out["extracted_sections"]}
    return query, positives


def create_training_examples(collection_path: str, query: str, positives: dict) -> List[InputExample]:
    """Iterate over PDF chunks saved during previous run to build examples.
    We rely on cached chunk JSONs. If unavailable, extract again on the fly.
    """
    chunks_file = os.path.join(collection_path, "chunks.json")
    if not os.path.isfile(chunks_file):
        # Extract on the fly using same logic as main.py
        try:
            import fitz  # PyMuPDF
            from langchain.text_splitter import RecursiveCharacterTextSplitter
        except ImportError as exc:
            raise RuntimeError("Required libraries missing for on-the-fly extraction") from exc

        with open(os.path.join(collection_path, "challenge1b_input.json"), "r", encoding="utf-8") as f:
            config = json.load(f)
        meta_cfg = config.get("metadata", config)
        pdf_files = meta_cfg.get("input_documents")
        if pdf_files is None:
            # Fallback: assume every PDF in pdfs/ belongs to the collection.
            pdf_dir = os.path.join(collection_path, "pdfs")
            pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, length_function=len,
                                                  separators=["\n\n", "\n", ". ", " ", ""])
        chunks = []
        for pdf in pdf_files:
            pdf_path = os.path.join(collection_path, "pdfs", pdf)
            doc = fitz.open(pdf_path)
            for page_index in range(doc.page_count):
                page_text = doc.load_page(page_index).get_text("text")  # type: ignore[attr-defined]
                if not page_text or len(page_text.strip()) < 50:
                    continue
                for ch in splitter.split_text(page_text):
                    chunks.append({
                        "source": pdf,
                        "page_number": page_index + 1,
                        "text": ch
                    })
            doc.close()
        # Cache for future runs
        with open(chunks_file, "w", encoding="utf-8") as f_out:
            json.dump(chunks, f_out)
    else:
        with open(chunks_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)

    pairs = []
    for ch in chunks:
        key = (ch["page_number"], ch["source"])
        label = 1.0 if key in positives else 0.0
        pairs.append(InputExample(texts=[query, ch["text"]], label=label))
    return pairs


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    collections = discover_collections(TRAIN_ROOT)
    if MAX_COLLECTIONS:
        collections = collections[:MAX_COLLECTIONS]
    if not collections:
        raise RuntimeError("No training collections found in sample_data/")

    print(f"Found {len(collections)} collections for training.")

    train_examples: List[InputExample] = []
    for coll in collections:
        query, positives = load_ground_truth(coll)
        train_examples.extend(create_training_examples(coll, query, positives))

    print(f"Generated {len(train_examples)} training pairs.")

    model = SentenceTransformer(MODEL_NAME)
    # mypy/pylance complain that `DataLoader` expects a torch Dataset, but
    # a plain Python list works fine at runtime because it implements
    # `__len__` and `__getitem__`. We silence the type checker accordingly.
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)  # type: ignore[arg-type]
    loss_fn = losses.CosineSimilarityLoss(model)

    warmup_steps = int(len(train_dataloader) * NUM_EPOCHS * 0.1)

    model.fit(
        train_objectives=[(train_dataloader, loss_fn)],
        epochs=NUM_EPOCHS,
        warmup_steps=warmup_steps,
        use_amp=False,
        weight_decay=0.01,
        output_path=OUTPUT_DIR,
        optimizer_params={"lr": LR}
    )

    print(f"Fine-tuned model saved to {OUTPUT_DIR}") 