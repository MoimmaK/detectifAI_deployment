"""
Sentence-BERT embedding generation for captions
"""

import logging
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
import asyncio
from concurrent.futures import ThreadPoolExecutor

try:
    from .config import CaptioningConfig
except ImportError:
    from config import CaptioningConfig


class EmbeddingGenerator:
    """Handles Sentence-BERT embedding generation"""
    
    def __init__(self, config: CaptioningConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize Sentence-BERT model
        self._load_model()
        
        # Thread pool for async processing
        self.executor = ThreadPoolExecutor(max_workers=config.max_concurrent_requests)
    
    def _load_model(self):
        """Load the Sentence-BERT model"""
        try:
            self.logger.info(f"Loading embedding model: {self.config.embedding_model_name}")
            self.model = SentenceTransformer(
                self.config.embedding_model_name,
                device=self.config.embedding_device
            )
            self.logger.info("Embedding model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        try:
            # Generate embedding
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=self.config.embedding_normalize
            )
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"Failed to generate embedding: {e}")
            # Return zero vector as fallback
            return np.zeros(self.model.get_sentence_embedding_dimension())
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for a batch of texts"""
        try:
            # Generate embeddings in batch for efficiency
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=self.config.embedding_normalize,
                batch_size=32,  # Optimize batch size for memory
                show_progress_bar=False
            )
            
            # Convert to list of arrays
            return [embedding for embedding in embeddings]
            
        except Exception as e:
            self.logger.error(f"Failed to generate batch embeddings: {e}")
            # Return zero vectors as fallback
            dim = self.model.get_sentence_embedding_dimension()
            return [np.zeros(dim) for _ in texts]
    
    async def generate_embeddings_async(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings asynchronously"""
        if not self.config.enable_async_processing:
            return self.generate_embeddings_batch(texts)
        
        loop = asyncio.get_event_loop()
        
        # Run in thread pool
        embeddings = await loop.run_in_executor(
            self.executor,
            self.generate_embeddings_batch,
            texts
        )
        
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        return self.model.get_sentence_embedding_dimension()
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings"""
        try:
            # Normalize if not already normalized
            if not self.config.embedding_normalize:
                embedding1 = embedding1 / np.linalg.norm(embedding1)
                embedding2 = embedding2 / np.linalg.norm(embedding2)
            
            # Compute cosine similarity
            similarity = np.dot(embedding1, embedding2)
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"Failed to compute similarity: {e}")
            return 0.0
    
    def find_similar_embeddings(self, query_embedding: np.ndarray, 
                              embeddings: List[np.ndarray], 
                              top_k: int = 5) -> List[tuple]:
        """Find most similar embeddings to query"""
        try:
            similarities = []
            
            for i, embedding in enumerate(embeddings):
                similarity = self.compute_similarity(query_embedding, embedding)
                similarities.append((i, similarity))
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            return similarities[:top_k]
            
        except Exception as e:
            self.logger.error(f"Failed to find similar embeddings: {e}")
            return []
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)