"""
OpenRouter Embeddings Helper
Provides a SentenceTransformer-like interface for OpenRouter embeddings API
"""

import os
import requests
import numpy as np
from typing import List, Union
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class OpenRouterEmbedder:
    """
    Wrapper for OpenRouter embeddings API with SentenceTransformer-like interface
    """
    
    def __init__(self, api_key: str = None, model: str = "nvidia/llama-nemotron-embed-vl-1b-v2:free"):
        """
        Initialize the OpenRouter embedder
        
        Args:
            api_key: OpenRouter API key (uses OPENROUTER_API_KEY env var if not provided)
            model: The embedding model to use
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not provided and not found in environment variables")
        
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1/embeddings"
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })
        
        # Detect embedding dimension by encoding a sample
        print(f"🔍 Detecting embedding dimension for {model}...")
        sample_embedding = self._get_embedding("test")
        self.embedding_dim = len(sample_embedding)
        print(f"✅ Embedding dimension: {self.embedding_dim}")
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text"""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                response = self.session.post(
                    self.base_url,
                    json={
                        "model": self.model,
                        "input": text,
                        "encoding_format": "float"
                    },
                    timeout=30
                )
                response.raise_for_status()
                data = response.json()
                
                if "data" in data and len(data["data"]) > 0:
                    return data["data"][0]["embedding"]
                else:
                    raise ValueError(f"Unexpected response format: {data}")
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    print(f"⚠️ Retry {attempt + 1}/{max_retries - 1} after {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print(f"❌ Error calling OpenRouter API: {e}")
                    raise
    
    def encode(
        self,
        sentences: Union[str, List[str]],
        convert_to_numpy: bool = True,
        batch_size: int = 8,
        show_progress_bar: bool = False
    ) -> np.ndarray:
        """
        Encode sentences to embeddings (compatible with SentenceTransformer API)
        
        Args:
            sentences: String or list of strings to encode
            convert_to_numpy: If True, return numpy array
            batch_size: Number of sentences per batch (for rate limiting) - reduced to 8
            show_progress_bar: Show progress bar during encoding
            
        Returns:
            numpy array of embeddings
        """
        # Handle single string
        if isinstance(sentences, str):
            sentences = [sentences]
        
        embeddings = []
        total = len(sentences)
        
        # Process in batches
        for i in range(0, total, batch_size):
            batch = sentences[i : i + batch_size]
            
            if show_progress_bar:
                progress = min(i + batch_size, total)
                print(f"Encoding: {progress}/{total}", end="\r")
            
            # Encode batch with retry logic
            max_retries = 3
            retry_delay = 2
            
            for attempt in range(max_retries):
                try:
                    response = self.session.post(
                        self.base_url,
                        json={
                            "model": self.model,
                            "input": batch,
                            "encoding_format": "float"
                        },
                        timeout=60
                    )
                    response.raise_for_status()
                    data = response.json()
                    
                    # Extract embeddings from response
                    for item in data["data"]:
                        embeddings.append(item["embedding"])
                    
                    # Small delay to avoid rate limiting
                    if i + batch_size < total:
                        time.sleep(0.5)
                    
                    break  # Success, exit retry loop
                    
                except requests.exceptions.RequestException as e:
                    if attempt < max_retries - 1:
                        print(f"\n⚠️ Batch {i//batch_size + 1} - Retry {attempt + 1}/{max_retries - 1} after {retry_delay}s...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        print(f"\n❌ Error encoding batch {i//batch_size + 1}: {e}")
                        raise
        
        if show_progress_bar:
            print(f"Encoding: {total}/{total} ✅")
        
        # Convert to numpy if requested
        if convert_to_numpy:
            embeddings = np.array(embeddings, dtype=np.float32)
        
        return embeddings

    def __del__(self):
        """Clean up session"""
        if hasattr(self, 'session'):
            self.session.close()
