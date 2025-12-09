"""
GeneratorCacheDecorator - Decorator class for caching GeneratorOutput objects
"""

import functools
from typing import Callable, Dict, Any, Optional
from dataclasses import asdict

from loguru import logger
from .config import CacheConfig
from .mysql_store import MySQLCacheStore
from .key_generator import create_cache_key_generator


class GeneratorCacheDecorator:
    """Decorator class for caching GeneratorOutput objects in MySQL"""
    
    def __init__(self, cache_config: CacheConfig):
        self.config = cache_config
        self.store = None
        self.key_generator = None
        
        if cache_config.enabled:
            try:
                self.store = MySQLCacheStore(cache_config.mysql)
                self.key_generator = create_cache_key_generator(cache_config.key_generator)
                logger.info("GeneratorCacheDecorator initialized with MySQL backend")
            except Exception as e:
                logger.error(f"Failed to initialize GeneratorCacheDecorator: {e}")
                self.config.enabled = False
    
    def __call__(self, func: Callable, generator_instance=None) -> Callable:
        """Decorator that adds caching to _generate method"""
        if not self.config.enabled:
            # If caching is disabled, return original function
            return func
        
        # Resolve the generator instance so we can access its attributes in the wrapper
        bound_instance = generator_instance or getattr(func, "__self__", None)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # When method is replaced, args[0] is the question, not self
            if len(args) >= 1:
                question = args[0]  # First argument is question
            elif "question" in kwargs:
                question = kwargs["question"]
            else:
                # Missing question argument, call original function directly
                return func(*args, **kwargs)
            
            # Extract additional arguments for cache key generation
            # For multimodal generators, we need to include images parameter
            cache_kwargs = {}
            # Only extract images parameter for MultimodalGenerator to avoid incorrect cache keys
            is_multimodal = bound_instance and type(bound_instance).__name__ == 'MultimodalGenerator'
            if is_multimodal:
                if len(args) >= 2:
                    # Second argument is images for multimodal generator
                    cache_kwargs['images'] = args[1]
                if 'images' in kwargs:
                    cache_kwargs['images'] = kwargs['images']
            
            # Generate cache key
            cache_key = self._generate_cache_key(bound_instance, question, **cache_kwargs)
            
            # Check if cache override is enabled
            if not self.config.force_override_cache:
                # Try to get from cache first
                try:
                    cached_data = self._get_from_cache(cache_key)
                    if cached_data is not None:
                        # Reconstruct appropriate output type from cached data
                        # Returns None if refresh is needed (e.g., missing raw_response)
                        result = self._reconstruct_output_from_cache(cached_data, bound_instance)
                        if result is not None:
                            # logger.debug(f"GeneratorCache: Successfully reconstructed from cache")
                            return result
                        # result is None means we need to refresh, continue to API call
                except Exception as e:
                    # Cache data reconstruction failed, log and fallback to API call
                    logger.warning(f"GeneratorCache: Cache data reconstruction failed for key {cache_key}: {e}")
                    logger.info("GeneratorCache: Falling back to API call with retry logic")
            else:
                logger.info(f"GeneratorCache: Cache override enabled, skipping cache lookup for key {cache_key}")
            
            # Cache miss - call original function
            # Call original function (already bound to the generator instance if applicable)
            result = func(*args, **kwargs)
            
            # Store result in cache if it meets conditions
            self._store_to_cache_hook(cache_key, result)
            
            return result
        
        return wrapper

    def _should_refresh_for_missing_raw_response(self, cached_data: Dict[str, Any]) -> bool:
        """Check if cache should be refreshed due to missing raw_response"""
        if not self.config.conditions.cache_raw_response:
            return False
        if not self.config.conditions.refresh_if_missing_raw_response:
            return False
        return 'raw_response' not in cached_data or cached_data.get('raw_response') is None

    def _reconstruct_output_from_cache(self, cached_data: Dict[str, Any], generator_instance) -> Optional[Any]:
        """Reconstruct appropriate output type from cached data based on generator type.
        Returns None if cache should be refreshed (e.g., missing raw_response when configured)."""
        try:
            # Check if we need to refresh cache due to missing raw_response
            if self._should_refresh_for_missing_raw_response(cached_data):
                logger.info("GeneratorCache: Cache missing raw_response, triggering refresh")
                return None

            # Determine the generator type and import appropriate class
            from generators.generator import GeneratorOutput, EmbeddingOutput

            # Check if this is an EmbeddingGenerator by looking at the class name
            generator_class_name = type(generator_instance).__name__

            if generator_class_name == "EmbeddingGenerator":
                # Reconstruct EmbeddingOutput
                # Only include fields that exist in EmbeddingOutput
                embedding_fields = {"embeddings", "prompt_tokens"}
                filtered_data = {k: v for k, v in cached_data.items() if k in embedding_fields}
                return EmbeddingOutput(**filtered_data)
            else:
                # Default to GeneratorOutput for DirectGenerator and MultimodalGenerator
                output = GeneratorOutput(**cached_data)

                # Smart cost recalculation: if cached cost is 0 and pricing_config is available,
                # recalculate cost using current pricing configuration
                if output.cost == 0.0 and hasattr(generator_instance, 'pricing_config'):
                    pricing_config = generator_instance.pricing_config
                    if pricing_config:
                        prompt_price = pricing_config.get('prompt_price_per_million', 0.0)
                        completion_price = pricing_config.get('completion_price_per_million', 0.0)

                        if prompt_price > 0 or completion_price > 0:
                            # Recalculate cost using cached token counts and current pricing
                            prompt_cost = (output.prompt_tokens / 1_000_000) * prompt_price
                            completion_cost = (output.completion_tokens / 1_000_000) * completion_price
                            recalculated_cost = prompt_cost + completion_cost

                            # Create new GeneratorOutput with recalculated cost, preserving raw_response
                            output = GeneratorOutput(
                                output=output.output,
                                prompt_tokens=output.prompt_tokens,
                                completion_tokens=output.completion_tokens,
                                cost=recalculated_cost,
                                raw_response=output.raw_response
                            )
                            logger.debug(f"Recalculated cost from 0.0 to {recalculated_cost:.6f} using current pricing config")

                return output

        except Exception as e:
            logger.error(f"Failed to reconstruct output from cache: {e}")
            raise
    
    def _generate_cache_key(self, generator_instance, question: str, **kwargs) -> str:
        """Generate cache key for the request"""
        if not self.key_generator:
            return ""
        
        try:
            # Extract generator parameters for key generation
            params = {}
            
            # Get all parameters specified in cached_parameters from the generator instance
            for param in self.key_generator.cached_parameters:
                if param in ['model', 'messages']:
                    # These are handled specially (model via model_name, messages via question)
                    continue
                
                # Try to get the parameter value from generator instance
                value = getattr(generator_instance, param, None)
                if value is not None:
                    params[param] = value
            
            # Add any additional parameters from kwargs
            params.update(kwargs)
            
            return self.key_generator.generate_key(
                model_name=getattr(generator_instance, 'config_name', getattr(generator_instance, 'model_name', '')),
                question=question,
                **params
            )
        except Exception as e:
            logger.debug(f"Cache key generation failed: {e}")
            return ""
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached result"""
        if not self.store or not cache_key:
            return None
        
        try:
            return self.store.get(cache_key)
        except Exception as e:
            logger.debug(f"Cache get failed for key {cache_key}: {e}")
            return None
    
    def _store_to_cache_hook(self, cache_key: str, result) -> None:
        """Hook method to store result in cache - to be called from _generate method"""
        if not self.store or not cache_key or not self._should_cache_result(result):
            return
        
        try:
            # Convert GeneratorOutput to dict for storage
            result_data = asdict(result)
            self.store.put(cache_key, result_data)
        except Exception as e:
            logger.debug(f"Cache put failed for key {cache_key}: {e}")
    
    def _should_cache_result(self, result) -> bool:
        """Check if result should be cached based on conditions"""
        if not self.config.conditions.cache_successful_only:
            return True

        # Check if it's a successful response based on result type
        if hasattr(result, 'output'):
            # GeneratorOutput type - check for generation failure (specific error message + no completion tokens)
            output_str = str(result.output)
            if (output_str.startswith("Generation failed:") or output_str.startswith("Multimodal generation failed:")) and result.completion_tokens == 0:
                return False
        elif hasattr(result, 'embeddings'):
            # EmbeddingOutput type - check if embeddings is empty
            if not result.embeddings:
                return False

        # Check minimum completion tokens (only applies to GeneratorOutput)
        if hasattr(result, 'completion_tokens'):
            if result.completion_tokens < self.config.conditions.min_completion_tokens:
                return False

        # For EmbeddingOutput, we don't check completion_tokens since it doesn't exist
        # The prompt_tokens check could be added if needed

        return True


def create_cache_decorator(cache_config: Optional[Dict[str, Any]] = None) -> GeneratorCacheDecorator:
    """Factory function to create cache decorator from config dict"""
    if cache_config is None:
        config = CacheConfig(enabled=False)
    else:
        config = CacheConfig.from_dict(cache_config)
    
    return GeneratorCacheDecorator(config)
