import numpy as np
import scipy.sparse as sp
import scipy.spatial.distance as distance
import networkx as nx
import community as community_louvain
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
import nltk
import torch
from typing import List, Optional, Dict, Any
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom Exceptions
class ChunkerError(Exception):
    """Base exception class for SemanticGraphChunker."""
    pass

class ModelInitializationError(ChunkerError):
    """Raised when there are issues initializing the embedding model."""
    pass

class EmbeddingComputationError(ChunkerError):
    """Raised when there are issues computing embeddings."""
    pass

class GraphConstructionError(ChunkerError):
    """Raised when there are issues constructing or analyzing the semantic graph."""
    pass

class SemanticGraphChunker:
    """
    A class for splitting documents into semantically coherent chunks using graph-based methods.

    The process involves:
    1. Splitting text into sentences.
    2. Computing embeddings for each sentence.
    3. Building a semantic similarity graph based on sentence embeddings using cosine similarity.
    4. Detecting semantic communities (clusters) in the graph using Louvain algorithm
    5. Merging small communities into larger ones (if needed).
    6. Producing text chunks that represent semantically coherent groups of sentences - a chunk in a community in the graph
    """

    def __init__(
            self,
            embedding_model: Optional[Any] = None,
            similarity_threshold: Optional[float] = None,
            locality_window: int = 4,
            distance_penalty_alpha: float = 0.1,
            batch_size: Optional[int] = None,
            device: Optional[str] = None
    ):
        """
        Initialize the SemanticGraphChunker with configurable parameters and dependencies.

        Args:
            embedding_model: Pretrained embedding model. Defaults to 'all-MiniLM-L6-v2'.
            similarity_threshold: Minimum similarity score to create a graph edge. If None, dynamic thresholding is used.
            locality_window: Number of neighboring sentences to consider for similarity comparison.
            distance_penalty_alpha: Factor to penalize similarity based on sentence distance.
            batch_size: Number of sentences to process in a batch for embeddings. Defaults to processing all sentences at once.
            device: Computation device ('cuda', 'mps', 'cpu', or None for auto-detection).
        """
        try:
            # Validate input parameters
            if locality_window < 1:
                raise ValueError("locality_window must be at least 1")
            if distance_penalty_alpha < 0:
                raise ValueError("distance_penalty_alpha must be non-negative")
            if similarity_threshold is not None and not (0 <= similarity_threshold <= 1):
                raise ValueError("similarity_threshold must be between 0 and 1")

            # Configure the computation device
            self.device = self._configure_device(device)

            # Initialize the NLTK tokenizer
            try:
                nltk.download('punkt', quiet=True)
            except Exception as e:
                raise ModelInitializationError(f"Failed to download NLTK punkt tokenizer: {str(e)}")

            # Initialize the embedding model
            self.embedder = self._initialize_embedder(embedding_model)

            # Save chunking parameters
            self.similarity_threshold = similarity_threshold
            self.locality_window = locality_window
            self.distance_penalty_alpha = distance_penalty_alpha
            self.batch_size = batch_size

            logger.info(f"SemanticGraphChunker initialized on device: {self.device}")

        except Exception as e:
            raise ModelInitializationError(f"Failed to initialize SemanticGraphChunker: {str(e)}")

    def _initialize_embedder(self, embedding_model: Optional[Any]) -> Any:
        """
        Load or initialize the embedding model.

        Args:
            embedding_model: User-provided embedding model. Defaults to 'all-MiniLM-L6-v2'.

        Returns:
            An embedding model instance.

        Raises:
            ModelInitializationError: If the embedding model fails to load.
        """
        try:
            if embedding_model is None:
                # Default to SentenceTransformer's MiniLM model if no model is provided
                return SentenceTransformer(model_name_or_path='all-MiniLM-L6-v2', device=self.device)
            else:
                # Configure the device for a user-provided model
                if hasattr(embedding_model, 'device'):
                    embedding_model.device = self.device
                return embedding_model
        except Exception as e:
            raise ModelInitializationError(f"Failed to initialize embedding model: {str(e)}")

    def _configure_device(self, device: Optional[str] = None) -> str:
        """
        Detect and configure the computation device.

        Args:
            device: User-specified device ('cuda', 'mps', 'cpu').

        Returns:
            The selected device as a string.

        Raises:
            ValueError: If an unsupported device is specified.
        """
        if device:
            if device not in ['cuda', 'mps', 'cpu']:
                raise ValueError(f"Unsupported device: {device}")
            return device

        try:
            # Auto-detect the best available device
            if torch.cuda.is_available():
                return 'cuda'
            elif torch.backends.mps.is_available():
                return 'mps'
            return 'cpu'
        except Exception as e:
            logger.warning(f"Device detection failed. Falling back to CPU: {str(e)}")
            return 'cpu'

    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using NLTK.

        Args:
            text: Input text to be split.

        Returns:
            A list of sentences.

        Raises:
            ValueError: If the input text is invalid.
            ChunkerError: If sentence splitting fails.
        """
        if not isinstance(text, str):
            raise ValueError("Input text must be a string")
        if not text.strip():
            raise ValueError("Input text cannot be empty")

        try:
            # Use NLTK to tokenize the text into sentences
            sentences = nltk.sent_tokenize(text)
            if not sentences:
                raise ValueError("No sentences detected in input text")
            return sentences
        except Exception as e:
            raise ChunkerError(f"Error splitting text into sentences: {str(e)}")

    def _compute_embeddings(self, sentences: List[str]) -> np.ndarray:
        """
        Compute sentence embeddings using the embedding model.

        Args:
            sentences: List of sentences to embed.

        Returns:
            A NumPy array of embeddings.

        Raises:
            EmbeddingComputationError: If embedding computation fails.
        """
        if not sentences:
            raise ValueError("Empty sentence list provided")

        try:
            # Set the batch size to process all sentences at once if not provided
            batch_size = self.batch_size or len(sentences)

            all_embeddings = []

            # Process sentences in batches
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i + batch_size]

                try:
                    # Compute embeddings for the current batch
                    with torch.no_grad():
                        batch_embeddings = self.embedder.encode(
                            batch,
                            convert_to_numpy=True,
                            device=self.device,
                            show_progress_bar=False
                        )

                    # Ensure embeddings are a valid NumPy array
                    if not isinstance(batch_embeddings, np.ndarray):
                        raise TypeError(f"Expected numpy array, got {type(batch_embeddings)}")

                    all_embeddings.append(batch_embeddings)

                except Exception as e:
                    raise EmbeddingComputationError(
                        f"Error computing embeddings for batch {i // batch_size}: {str(e)}")

            # Concatenate all batch embeddings into a single array
            return np.vstack(all_embeddings)

        except Exception as e:
            raise EmbeddingComputationError(f"Failed to compute embeddings: {str(e)}")

    def _compute_local_similarities(self, embeddings: np.ndarray) -> sp.csr_matrix:
        """
        Compute local semantic similarities between sentences.

        Each sentence is compared with its neighbors within a defined locality window.
        A distance-based penalty is applied to account for the positional distance between sentences.
        The output is a sparse similarity matrix where edges represent significant semantic similarities.

        Args:
            embeddings: A NumPy array of sentence embeddings (shape: [num_sentences, embedding_dim]).

        Returns:
            A sparse matrix (scipy.sparse.csr_matrix) representing the semantic similarity graph.

        Raises:
            ValueError: If input embeddings are invalid.
            GraphConstructionError: If similarity computation fails.
        """
        if not isinstance(embeddings, np.ndarray):
            raise ValueError("Embeddings must be a numpy array")
        if len(embeddings.shape) != 2:
            raise ValueError("Embeddings must be a 2D array")
        if not embeddings.size:
            raise ValueError("Embeddings array is empty")

        try:
            n = len(embeddings)
            # Normalize each embedding to have unit length (required for cosine similarity)
            norms = np.linalg.norm(embeddings, axis=1)
            if np.any(norms == 0):
                raise ValueError("Zero-length embedding vectors detected")
            normalized_embeddings = embeddings / norms[:, np.newaxis]

            # Variables to store the indices and values for the sparse matrix
            row_indices, col_indices, similarity_values = [], [], []

            # Iterate through each sentence to compute similarities within the locality window
            for i in range(n):
                # Define the start and end of the locality window
                window_start = max(0, i - self.locality_window)
                window_end = min(n, i + self.locality_window + 1)

                # Exclude the current sentence index from the window
                window_indices = [j for j in range(window_start, window_end) if j != i]

                if not window_indices:
                    continue  # Skip if no neighbors exist

                # Extract the embeddings of the neighboring sentences
                window_embeddings = normalized_embeddings[window_indices]

                # Compute cosine similarities between the current sentence and its neighbors
                similarities = 1 - distance.cdist(
                    normalized_embeddings[i:i + 1],
                    window_embeddings,
                    metric='cosine'
                )[0]

                # Filter out invalid similarity values (e.g., negative ones due to numerical precision)
                similarities = np.clip(similarities, 0, 1)

                # Compute the positional distances between the current sentence and its neighbors
                distances = np.abs(np.array(window_indices) - i)

                # Apply a penalty based on positional distance
                distance_penalties = np.exp(-self.distance_penalty_alpha * distances)
                weighted_similarities = similarities * distance_penalties

                # Determine a similarity threshold (dynamic or fixed)
                if self.similarity_threshold is None:
                    # Use the 50th percentile of the weighted similarities as the threshold
                    local_threshold = np.percentile(weighted_similarities, 50)
                else:
                    local_threshold = self.similarity_threshold

                # Identify significant similarities (above the threshold)
                significant_mask = weighted_similarities > local_threshold

                # Add significant edges to the sparse matrix representation
                for j, (sim, is_significant) in enumerate(zip(weighted_similarities, significant_mask)):
                    if is_significant:
                        col_idx = window_indices[j]
                        row_indices.append(i)  # Source node index
                        col_indices.append(col_idx)  # Target node index
                        similarity_values.append(float(sim))  # Similarity score

            # Create a sparse matrix to represent the semantic similarity graph
            return sp.csr_matrix(
                (similarity_values, (row_indices, col_indices)),
                shape=(n, n)
            )

        except Exception as e:
            raise GraphConstructionError(f"Failed to compute similarities: {str(e)}")

    def _build_semantic_graph(self, sentences: List[str], embeddings: np.ndarray) -> nx.Graph:
        """
        Construct a semantic similarity graph.

        Nodes represent sentences, and edges represent semantic proximity based on the computed similarities.

        Args:
            sentences: A list of sentences from the input text.
            embeddings: Sentence embeddings corresponding to the sentences.

        Returns:
            A NetworkX graph where edges are weighted by similarity scores.

        Raises:
            GraphConstructionError: If graph construction fails.
        """
        try:
            # Compute the sparse similarity matrix
            sparse_similarities = self._compute_local_similarities(embeddings)

            # Initialize an empty graph
            G = nx.Graph()

            # Add nodes to the graph
            for i, sentence in enumerate(sentences):
                G.add_node(i, sentence=sentence)  # Each node is a sentence index with its content as metadata

            # Add edges to the graph based on the similarity matrix
            for i, j in zip(*sparse_similarities.nonzero()):
                G.add_edge(i, j, weight=float(sparse_similarities[i, j]))

            return G

        except Exception as e:
            raise GraphConstructionError(f"Failed to build semantic graph: {str(e)}")

    def _detect_communities(self, G: nx.Graph) -> List[List[int]]:
        """
        Detect semantic communities in the graph.

        Communities are identified using the Louvain method, which groups nodes (sentences) that are closely connected.

        Args:
            G: The semantic similarity graph.

        Returns:
            A list of communities, where each community is a list of sentence indices.

        Raises:
            GraphConstructionError: If community detection fails.
        """
        if not isinstance(G, nx.Graph):
            raise ValueError("Input must be a NetworkX graph")
        if not G.number_of_nodes():
            raise ValueError("Empty graph provided")

        try:
            # Use the Louvain algorithm to partition the graph into communities
            partition = community_louvain.best_partition(G)

            # Group nodes by their community IDs
            communities: Dict[int, List[int]] = {}
            for node, community_id in partition.items():
                if community_id not in communities:
                    communities[community_id] = []
                communities[community_id].append(node)

            # Return the communities as sorted lists of node indices
            return [
                sorted(comm) for comm in
                sorted(communities.values(), key=lambda x: min(x))
            ]

        except Exception as e:
            raise GraphConstructionError(f"Failed to detect communities: {str(e)}")

    def _merge_small_communities(self, communities: List[List[int]], min_community_size: int = 2) -> List[List[int]]:
        """
        Merge small communities to ensure each has at least a minimum size.

        Args:
            communities: A list of communities (each a list of sentence indices).
            min_community_size: The minimum allowed size for a community.

        Returns:
            A list of merged communities.

        Raises:
            ValueError: If input is invalid or merging fails.
        """
        if min_community_size < 1:
            raise ValueError("min_community_size must be at least 1")
        if not communities:
            raise ValueError("Empty communities list provided")

        try:
            merged_communities = []
            i = 0

            # Iterate through the communities to merge small ones
            while i < len(communities):
                current_community = communities[i]

                if len(current_community) < min_community_size:
                    # Merge with the previous community if one exists
                    if i > 0:
                        merged_communities[-1].extend(current_community)
                    # Otherwise, merge with the next community
                    elif i < len(communities) - 1:
                        communities[i + 1] = current_community + communities[i + 1]
                else:
                    merged_communities.append(current_community)

                i += 1

            # If all communities were merged, return a single community
            if not merged_communities:
                logger.warning("All communities were merged into one.")
                return [sum(communities, [])]

            return merged_communities

        except Exception as e:
            raise GraphConstructionError(f"Failed to merge communities: {str(e)}")

    def split_text(self, text: str, min_community_size: int = 2) -> List[str]:
        """
        Main document chunking method with comprehensive error handling.

        Args:
            text: Input text to be chunked
            min_community_size: Minimum number of sentences per chunk

        Returns:
            List of semantically coherent text chunks

        Raises:
            ValueError: If input validation fails
            ChunkerError: If chunking process fails
        """
        if not isinstance(text, str):
            raise ValueError("Input text must be a string")
        if not text.strip():
            raise ValueError("Input text cannot be empty")
        if min_community_size < 1:
            raise ValueError("min_community_size must be at least 1")

        try:
            # Split the text into sentences
            sentences = self._split_sentences(text)
            if len(sentences) <= min_community_size:
                return [text]  # If text is too short, return it as a single chunk

            # Compute embeddings for the sentences
            embeddings = self._compute_embeddings(sentences)

            # Build a semantic graph using embeddings
            G = self._build_semantic_graph(sentences, embeddings)

            # Detect communities (clusters of semantically related sentences)
            communities = self._detect_communities(G)

            # Merge small communities to meet the minimum community size
            merged_communities = self._merge_small_communities(communities, min_community_size)

            # Convert each community (group of indices) into a text chunk
            chunks = [' '.join([sentences[i] for i in community]) for community in merged_communities]

            return chunks

        except Exception as e:
            raise ChunkerError(f"Failed to split text: {str(e)}")

    def split_documents(self, documents: List[Document], min_community_size: int = 2) -> List[Document]:
        """
        Split documents with error handling and progress tracking.

        Args:
            documents: List of Document objects
            min_community_size: Minimum number of sentences per chunk

        Returns:
            List of new Document objects

        Raises:
            ValueError: If input validation fails
            ChunkerError: If document splitting fails
        """
        if not isinstance(documents, list):
            raise ValueError("Input must be a list of Documents")
        if not documents:
            raise ValueError("Document list cannot be empty")
        if not all(isinstance(doc, Document) for doc in documents):
            raise ValueError("All elements in the list must be Document objects")

        try:
            split_documents = []

            # Use tqdm to show progress
            for document in tqdm(documents, desc="Splitting documents", unit="doc"):
                # Extract the text content of the document
                content = document.page_content

                # Split the text into chunks
                chunks = self.split_text(content, min_community_size)

                # Create a new Document object for each chunk
                for chunk in chunks:
                    split_documents.append(
                        Document(page_content=chunk, metadata=document.metadata)
                    )

            return split_documents

        except Exception as e:
            raise ChunkerError(f"Failed to split documents: {str(e)}")
