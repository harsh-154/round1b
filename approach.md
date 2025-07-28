Approach Explanation: Persona-Driven Document Intelligence
Our system is designed as a modular, four-stage pipeline to intelligently extract and rank document sections based on a user persona and their specific "job-to-be-done." This approach is optimized for performance and accuracy while adhering to strict execution constraints (CPU-only, ≤ 1GB model, ≤ 60s processing, offline).

Stage 1: Document Ingestion & Semantic Chunking

To begin, we extract clean text from the input PDFs. For this, we selected the PyMuPDF library due to its exceptional speed and efficiency, which is critical for meeting the tight processing time limit. After extraction, we employ a RecursiveCharacterTextSplitter. This method is superior to fixed-size chunking as it respects the document's natural structure by attempting to split text along semantic boundaries like paragraphs (\n\n) and sentences first. This preserves the contextual integrity of the information, which is vital for accurate relevance ranking.

Stage 2: Semantic Representation (Embedding)

The core of our machine learning approach is converting text into meaningful numerical vectors (embeddings). We chose the all-MiniLM-L6-v2 sentence transformer model for this task. This model offers an outstanding balance of performance, speed, and size (~80MB), making it perfect for the competition's constraints. We create a single "query vector" by concatenating the persona description and the job-to-be-done, representing the user's complete intent. Then, we generate an embedding for each document chunk. This process maps all textual data into a unified vector space where semantic similarity can be measured.

Stage 3: Relevance Ranking

With all text represented as vectors, we calculate the cosine similarity between the query vector and each document chunk vector. This mathematical operation efficiently measures the semantic relevance of each chunk to the user's query. The chunks are then sorted in descending order of their similarity scores to create a final importance rank, directly addressing the primary scoring criterion.

Stage 4: Sub-section Analysis & Output Generation

For the highest-ranked sections, we perform a more granular analysis to generate the required Refined Text. We use an extractive summarization model (bert-extractive-summarizer) configured to leverage the same all-MiniLM-L6-v2 model. This is highly efficient as it avoids loading a second large model. The summarizer extracts the most salient sentences from each top-ranked chunk. Finally, all metadata, ranked sections, and sub-section analyses are compiled and formatted into the required JSON output structure. This end-to-end pipeline ensures a robust, efficient, and highly relevant result.