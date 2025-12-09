"""
Cache key generator - simplified version
"""

import hashlib
import json
from .config import KeyGeneratorConfig


class CacheKeyGenerator:
    """Generate cache keys for generator calls"""
    
    def __init__(self, config: KeyGeneratorConfig):
        self.config = config
        self.cached_parameters = set(config.cached_parameters)
    
    def generate_key(self, model_name: str, question: str, **kwargs) -> str:
        """Generate cache key from generator parameters"""
        # Detect API type based on cached_parameters
        # Embedding API uses "input" field, Chat API uses "messages" field
        is_embedding_api = "input" in self.cached_parameters

        # Build payload based on API type
        if is_embedding_api:
            # Embedding API format: {"model": ..., "input": text}
            payload = {
                "model": model_name,
                "input": question
            }
        else:
            # Chat API format: {"model": ..., "messages": [...]}
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": question}]
            }

        # Add parameters from kwargs
        payload.update(kwargs)

        # Only keep parameters specified in cached_parameters, excluding None values
        filtered = {k: payload[k] for k in self.cached_parameters
                   if k in payload and payload[k] is not None}
        
        # Special handling: always include 'images' parameter if present in kwargs
        # This ensures multimodal requests with different images get different cache keys
        if "images" in kwargs and kwargs["images"] is not None:
            filtered["images"] = kwargs["images"]

        # Encode and hash
        encoded_body = json.dumps(
            filtered, separators=(",", ":"), sort_keys=True, ensure_ascii=False
        ).encode()

        hasher = self._create_hasher()
        hasher.update(encoded_body)
        return hasher.hexdigest()
    
    def _create_hasher(self):
        """Create hasher based on configuration"""
        algorithm = self.config.hash_algorithm.lower()
        
        if algorithm == 'blake2b':
            try:
                return hashlib.blake2b(
                    digest_size=self.config.hash_digest_size, 
                    usedforsecurity=False
                )
            except TypeError:
                try:
                    return hashlib.blake2b(digest_size=self.config.hash_digest_size)
                except Exception:
                    return hashlib.sha256()
        elif algorithm == 'sha256':
            return hashlib.sha256()
        elif algorithm == 'sha1':
            return hashlib.sha1()
        elif algorithm == 'md5':
            return hashlib.md5()
        else:
            return hashlib.sha256()


def create_cache_key_generator(config: KeyGeneratorConfig) -> CacheKeyGenerator:
    """Factory function to create cache key generator"""
    return CacheKeyGenerator(config)