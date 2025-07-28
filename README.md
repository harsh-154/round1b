Persona-Driven Document Intelligence System
This project is a solution for the Adobe India Hackathon 2025, Round 1B. It acts as an intelligent document analyst, extracting and prioritizing relevant sections from a collection of PDFs based on a specific persona and their job-to-be-done.

Prerequisites
Docker must be installed on your system.

Setup
Place Documents: Add your input PDF files (3-10 documents) into the sample_data/input/pdfs/ directory.

Configure Input: Open the sample_data/input/input.json file and update the following fields:

document_collection: List the exact filenames of the PDFs you placed in the pdfs folder.

persona_definition: Describe the user persona.

job_to_be_done: Describe the task the persona needs to accomplish.

Execution
The entire application is containerized with Docker to ensure it runs in a consistent, offline environment with all dependencies and models included.

Step 1: Build the Docker Image

Navigate to the root directory of the project (where the Dockerfile is located) and run the following command in your terminal. This will build the Docker image, download all necessary libraries, and cache the machine learning models inside the image.

Bash

docker build -t adobe-challenge-1b.
Step 2: Run the Application

Once the image is built, run the following command. This will start a container from the image, run the analysis script, and then automatically remove the container when finished.

The -v flag is used to mount your local sample_data directory into the container. This allows the script inside the container to read your input files and, most importantly, write the output file back to your local machine.

Bash

docker run --rm -v "$(pwd)/sample_data:/app/sample_data" adobe-challenge-1b
Note for Windows users: You may need to replace $(pwd) with %cd% in Command Prompt or ${PWD} in PowerShell.

Output
The processing will take a moment (designed to be under 60 seconds). Once complete, the results will be saved to:

sample_data/output/challenge1b_output.json

This file will contain the ranked sections and sub-section analysis based on your input.