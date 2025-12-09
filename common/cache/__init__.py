"""
Generator Output Cache System

A decorator-based caching system that caches GeneratorOutput objects in MySQL.
Provides optional caching enhancement for generator methods through configuration.
"""

from .decorator import GeneratorCacheDecorator
from .config import CacheConfig
from .mysql_store import MySQLCacheStore
from .key_generator import CacheKeyGenerator, create_cache_key_generator

__version__ = "1.0.0"
__all__ = [
    "GeneratorCacheDecorator",
    "CacheConfig", 
    "MySQLCacheStore",
    "CacheKeyGenerator",
    "create_cache_key_generator"
]