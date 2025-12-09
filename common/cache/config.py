"""
Cache configuration for GeneratorOutput caching
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import os


@dataclass
class MySQLConfig:
    """MySQL database configuration for caching"""
    host: str = "localhost"
    port: int = 3306
    user: Optional[str] = None
    password: Optional[str] = None
    database: str = "avengers_cache"
    table_name: str = "generator_output_cache"
    charset: str = "utf8mb4"
    autocommit: bool = True
    ttl_seconds: Optional[int] = None
    
    # Connection pool settings
    use_connection_pool: bool = False
    pool_size: int = 4
    max_overflow: int = 2
    pool_timeout: int = 10
    pool_recycle: int = 3600

    def __post_init__(self):
        # Get from environment variables if not provided or if value is env var name
        if self.host is None or self.host == "MYSQL_HOST":
            self.host = os.getenv('MYSQL_HOST', 'localhost')
        if self.port is None or str(self.port) == "MYSQL_PORT":
            self.port = int(os.getenv('MYSQL_PORT', '3306'))
        if self.user is None or self.user == "MYSQL_USER":
            self.user = os.getenv('MYSQL_USER', 'root')
        if self.password is None or self.password == "MYSQL_PASSWORD":
            self.password = os.getenv('MYSQL_PASSWORD', '')


@dataclass
class KeyGeneratorConfig:
    """Cache key generation configuration"""
    cached_parameters: List[str] = field(default_factory=lambda: [
        "model", "temperature", "top_p", "messages", "reasoning_effort"
    ])
    hash_algorithm: str = "blake2b"
    hash_digest_size: int = 16


@dataclass
class CacheConditionsConfig:
    """Conditions for what to cache"""
    cache_successful_only: bool = True  # Only cache successful responses
    min_completion_tokens: int = 0  # Minimum completion tokens to cache
    cache_raw_response: bool = False  # Cache complete API response JSON
    refresh_if_missing_raw_response: bool = False  # Re-fetch if cached data missing raw_response


@dataclass
class CacheConfig:
    """Complete cache configuration"""
    enabled: bool = False
    force_override_cache: bool = False  # When True, always call API but still cache results
    mysql: MySQLConfig = field(default_factory=MySQLConfig)
    key_generator: KeyGeneratorConfig = field(default_factory=KeyGeneratorConfig)
    conditions: CacheConditionsConfig = field(default_factory=CacheConditionsConfig)
    log_level: str = "INFO"
    enable_stats: bool = True

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CacheConfig':
        """Create CacheConfig from dictionary"""
        mysql_config = MySQLConfig(**config_dict.get('mysql', {}))
        key_config = KeyGeneratorConfig(**config_dict.get('key_generator', {}))
        conditions_config = CacheConditionsConfig(**config_dict.get('conditions', {}))
        
        return cls(
            enabled=config_dict.get('enabled', False),
            force_override_cache=config_dict.get('force_override_cache', False),
            mysql=mysql_config,
            key_generator=key_config,
            conditions=conditions_config,
            log_level=config_dict.get('log_level', 'INFO'),
            enable_stats=config_dict.get('enable_stats', True)
        )