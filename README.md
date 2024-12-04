# Semantic Graph Chunking for RAG Pipelines

Semantic Graph Chunking is an innovative method for chunking text data, designed to enhance the performance of Retrieval-Augmented Generation (RAG) pipelines. By leveraging graph-based techniques, this approach creates semantically meaningful chunks that improve information retrieval and downstream tasks.

## Overview

This repository contains the implementation of the Semantic Graph Chunking method. The pipeline includes:
1. Splitting documents into sentences.
2. Embedding sentences using a pretrained embedding model.
3. Building a semantic similarity graph based on sentence embeddings.
4. Detecting semantic communities using the Louvain algorithm.
5. Generating coherent text chunks from identified communities.

This approach ensures that the created chunks maintain semantic coherence, which is critical for effective document segmentation and retrieval.

## Prerequisites

Before running the project, ensure you have:
- A **Cohere API key**.
- A **Pinecone API key**.
- The ground truth file (`ground_truth.xlsx` provided in this repository).
- Python 3.8+ installed.

## Installation

Clone this repository and install the required dependencies:

```bash
git clone <repository_url>
cd <repository_name>
pip install -r requirements.txt
```
---
## Running the Project

### Environment Setup

1. **Set up API keys**:
   - Obtain your API keys from [Cohere](https://cohere.ai) and [Pinecone](https://www.pinecone.io).

2. **Prepare the ground truth file**:
   - Ensure the `ground_truth.xlsx` file is available in the repository or provide its absolute path.

3. **Configure environment variables**:
   - Define the following variables in your script or notebook:
     ```python
     COHERE_API_KEY = "your-cohere-api-key"
     PINECONE_API_KEY = "your-pinecone-api-key"
     GROUND_TRUTH_PATH = "/path/to/ground_truth.xlsx"
     ```

### Execution

#### Option 1: Using the Jupyter Notebook
1. Open the `execute_evaluation.ipynb` file in your preferred Jupyter environment.
2. Update the environment variables (`COHERE_API_KEY`, `PINECONE_API_KEY`, `GROUND_TRUTH_PATH`) in the first cell.
3. Execute the notebook cell by cell to evaluate the Semantic Graph Chunking method.

#### Option 2: Running the Python Script
1. Open the `semantic_graph_chunker.py` file and modify it to include your environment variables.
2. Run the script using the following command:
   ```bash
   python semantic_graph_chunker.py
   ```

The script will process the input data using the Semantic Graph Chunking method and evaluate the results against the ground truth. The output will include metrics comparing the generated chunks to the ground truth data.

---

## Repository Structure

- **`semantic_graph_chunker.py`**: The main Python implementation of the Semantic Graph Chunking method.
- **`execute_evaluation.ipynb`**: A Jupyter Notebook for demonstrating the chunking process and evaluating results.
- **`ground_truth.xlsx`**: The ground truth data used for evaluation.
- **`requirements.txt`**: A file listing all dependencies required for this project.

---


