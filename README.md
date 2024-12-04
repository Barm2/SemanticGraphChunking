# Semantic Graph Chunking for RAG Pipelines

Semantic Graph Chunking is an innovative chunking method designed to improve document processing in Retrieval-Augmented Generation (RAG) pipelines. By leveraging graph-based approaches, this method creates semantically coherent chunks, enhancing the efficiency and accuracy of information retrieval and downstream tasks.


## Overview

This repository implements the Semantic Graph Chunking method. The process includes:

1. Splitting documents into sentences.
2. Embedding sentences using pretrained models.
3. Constructing a semantic similarity graph based on embeddings.
4. Identifying semantic communities using the Louvain algorithm.
5. Generating coherent text chunks from the identified communities.

This method is particularly useful in scenarios where semantic coherence is critical for document segmentation and retrieval.


## Requirements

To use this repository, install the required Python dependencies using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Dependencies

Key libraries used in this project include:

- PyTorch
- Sentence Transformers
- Pinecone
- LangChain and its extensions
- Cohere API

For the complete list of dependencies, refer to the [`requirements.txt`](requirements.txt) file.

---

## How to Run the Project

### Prerequisites

Before running the project, ensure you have the following:

1. A **Cohere API key**.
2. A **Pinecone API key**.
3. A **path to the ground truth file** (`ground_truth.xlsx` provided in this repository).

### Instructions

1. Clone the repository to your local machine.

2. Install all the dependencies.

3. Open the Jupyter Notebook `execute_evaluation.ipynb`.

4. In the first cell, configure the following environment variables:
   - `COHERE_API_KEY`: Your Cohere API key.
   - `PINECONE_API_KEY`: Your Pinecone API key.
   - `GROUND_TRUTH_PATH`: The path to the ground truth Excel file.

5. Run the notebook cell-by-cell to evaluate the Semantic Graph Chunking method.

### Example Configuration

In the notebook:

```python
COHERE_API_KEY = "your-cohere-api-key"
PINECONE_API_KEY = "your-pinecone-api-key"
GROUND_TRUTH_PATH = "/path/to/ground_truth.xlsx"
```

## Project Files

- **`semantic_graph_chunker.py`**: Contains the implementation of the Semantic Graph Chunking method.
- **`execute_evaluation.ipynb`**: A Jupyter Notebook demonstrating how to use the chunker and evaluate its performance.
- **`ground_truth.xlsx`**: The ground truth data used for evaluation.
- **`requirements.txt`**: Lists the required dependencies.
