# Use a Python 3.10 slim base image to match the required version.
FROM python:3.10-slim

# Set the working directory inside the container.
WORKDIR /app

# Copy the requirements file first to leverage Docker layer caching.
# This layer will only be rebuilt if requirements.txt changes.
COPY requirements.txt .

# --- Install Dependencies and Download Models ---
# This is a critical step. We install all Python libraries and then
# run a command to pre-download and cache the sentence-transformer model
# and the spacy model required by the summarizer. This ensures that when
# the container runs for analysis, it has everything it needs locally and
# does not require internet access.
RUN pip install --no-cache-dir -r requirements.txt && \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy the rest of the application code into the container.
COPY . .

# Define the command to run when the container starts.
# This executes the main Python script to perform the analysis.
CMD ["python", "main.py"]
