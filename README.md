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
