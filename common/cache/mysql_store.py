"""
MySQL storage for GeneratorOutput cache - adapted from cached_openai mysql_cache.py
"""

import json
import time
import threading
import queue
import pickle
from typing import Optional, Dict, Any
from dataclasses import asdict

from loguru import logger
from .config import MySQLConfig

try:
    import pymysql
except ImportError:
    pymysql = None


class MySQLConnectionPool:
    """Simple MySQL connection pool implementation"""
    
    def __init__(self, connect_func, pool_size=10, max_overflow=5, pool_timeout=30, pool_recycle=3600):
        self._connect_func = connect_func
        self._pool_size = pool_size
        self._max_overflow = max_overflow
        self._pool_timeout = pool_timeout
        self._pool_recycle = pool_recycle
        
        self._pool = queue.Queue(maxsize=pool_size)
        self._overflow_connections = set()
        self._created_connections = 0
        self._lock = threading.Lock()
        
        # Pre-create initial connections
        try:
            for _ in range(pool_size):
                conn = self._create_connection()
                self._pool.put(conn)
        except Exception as e:
            logger.warning(f"Failed to pre-create connections: {e}")
    
    def _create_connection(self):
        """Create a new database connection"""
        conn = self._connect_func()
        conn._created_at = time.time()
        return conn
    
    def _is_connection_stale(self, conn):
        """Check if connection is stale and needs recycling"""
        if not hasattr(conn, '_created_at'):
            return True
        return time.time() - conn._created_at > self._pool_recycle
    
    def get_connection(self):
        """Get a connection from the pool"""
        try:
            # Try to get from pool first
            while True:
                try:
                    conn = self._pool.get_nowait()
                    # Check if connection is still valid and not stale
                    if self._is_connection_stale(conn):
                        try:
                            conn.close()
                        except Exception:
                            pass
                        continue
                    
                    try:
                        conn.ping(reconnect=False)
                        return conn
                    except Exception:
                        # Connection is dead, try again
                        continue
                except queue.Empty:
                    break
            
            # Pool is empty, try to create overflow connection
            with self._lock:
                if len(self._overflow_connections) < self._max_overflow:
                    conn = self._create_connection()
                    self._overflow_connections.add(conn)
                    return conn
            
            # Wait for a connection to be returned
            conn = self._pool.get(timeout=self._pool_timeout)
            if self._is_connection_stale(conn):
                try:
                    conn.close()
                except Exception:
                    pass
                conn = self._create_connection()
            
            return conn
            
        except queue.Empty:
            raise RuntimeError(f"Failed to get connection within {self._pool_timeout} seconds")
        except Exception as e:
            # Fallback: create new connection
            logger.warning(f"Pool connection failed, creating new: {e}")
            return self._create_connection()
    
    def return_connection(self, conn):
        """Return a connection to the pool"""
        if not conn:
            return
            
        try:
            # Check if it's an overflow connection
            with self._lock:
                if conn in self._overflow_connections:
                    self._overflow_connections.remove(conn)
                    try:
                        conn.close()
                    except Exception:
                        pass
                    return
            
            # Try to return to pool
            try:
                self._pool.put_nowait(conn)
            except queue.Full:
                # Pool is full, close the connection
                try:
                    conn.close()
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"Error returning connection to pool: {e}")
            try:
                conn.close()
            except Exception:
                pass
    
    def close_all(self):
        """Close all connections in the pool"""
        # Close pool connections
        while True:
            try:
                conn = self._pool.get_nowait()
                try:
                    conn.close()
                except Exception:
                    pass
            except queue.Empty:
                break
        
        # Close overflow connections
        with self._lock:
            for conn in list(self._overflow_connections):
                try:
                    conn.close()
                except Exception:
                    pass
            self._overflow_connections.clear()


class MySQLCacheStore:
    """MySQL storage backend for GeneratorOutput cache"""
    
    def __init__(self, config: MySQLConfig):
        if pymysql is None:
            raise RuntimeError("PyMySQL package is required for MySQL cache storage")
        
        self.config = config
        self._connection_pool = None
        self._local = threading.local()
        self._stats_lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._reconnects = 0
        
        # Initialize database and table
        self._initialize_database()
        
        # Set up connection pool if enabled
        if config.use_connection_pool:
            self._initialize_pool()
        
        # Register cleanup
        import atexit, weakref
        _selfref = weakref.ref(self)
        def _cleanup():
            s = _selfref()
            if s is None:
                return
            try:
                with s._stats_lock:
                    logger.info(f"GeneratorCache Summary: hits={s._hits} misses={s._misses} reconnects={s._reconnects}")
                if s._connection_pool:
                    s._connection_pool.close_all()
            except Exception:
                pass
        atexit.register(_cleanup)
    
    def _initialize_database(self):
        """Initialize database and table"""
        logger.info(f"MySQLCacheStore: connecting host={self.config.host} port={self.config.port} db={self.config.database}")
        
        # Ensure database exists
        try:
            admin_conn = pymysql.connect(
                host=self.config.host, 
                port=self.config.port, 
                user=self.config.user, 
                password=self.config.password, 
                autocommit=self.config.autocommit
            )
            with admin_conn.cursor() as cur:
                cur.execute(f"CREATE DATABASE IF NOT EXISTS `{self.config.database}` DEFAULT CHARACTER SET {self.config.charset}")
            logger.debug("MySQLCacheStore: ensured database exists")
        except Exception as e:
            logger.warning(f"Could not ensure database exists: {e}")
        finally:
            try:
                admin_conn.close()
            except Exception:
                pass
        
        # Ensure table exists
        conn = self._create_connection()
        try:
            with conn.cursor() as cur:
                table_sql = self._get_table_sql()
                cur.execute(table_sql)
            logger.debug(f"MySQLCacheStore: ensured {self.config.table_name} table exists")
        finally:
            try:
                conn.close()
            except Exception:
                pass
    
    def _initialize_pool(self):
        """Initialize connection pool"""
        try:
            self._connection_pool = MySQLConnectionPool(
                connect_func=self._create_connection,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle
            )
            logger.info(f"MySQLCacheStore: initialized connection pool (size={self.config.pool_size}, max_overflow={self.config.max_overflow})")
        except Exception as e:
            logger.warning(f"Failed to initialize connection pool, falling back to per-thread connections: {e}")
            self._connection_pool = None
    
    def _create_connection(self):
        """Create a new MySQL connection"""
        return pymysql.connect(
            host=self.config.host,
            port=self.config.port,
            user=self.config.user,
            password=self.config.password,
            database=self.config.database,
            autocommit=self.config.autocommit,
            charset=self.config.charset,
        )
    
    def _get_connection(self):
        """Get a MySQL connection"""
        if self._connection_pool:
            # Use connection pool
            try:
                return self._connection_pool.get_connection()
            except Exception as e:
                logger.warning(f"MySQLCacheStore: pool connection error, creating direct connection: {e}")
                with self._stats_lock:
                    self._reconnects += 1
                return self._create_connection()
        else:
            # Fallback to per-thread connections
            try:
                c = getattr(self._local, "conn", None)
                if c is None:
                    c = self._create_connection()
                    self._local.conn = c
                try:
                    # If ping fails, reconnect and count it
                    c.ping(reconnect=True)
                except Exception:
                    c = self._create_connection()
                    with self._stats_lock:
                        self._reconnects += 1
                    self._local.conn = c
                return c
            except Exception as e:
                logger.warning(f"MySQLCacheStore: connection error, creating new: {e}")
                with self._stats_lock:
                    self._reconnects += 1
                c = self._create_connection()
                self._local.conn = c
                return c
    
    def _return_connection(self, conn):
        """Return connection to pool if using pool"""
        if self._connection_pool and conn:
            self._connection_pool.return_connection(conn)
    
    def _get_table_sql(self) -> str:
        """Generate table creation SQL"""
        return (
            f"CREATE TABLE IF NOT EXISTS {self.config.table_name} ("
            "  cache_key VARCHAR(128) PRIMARY KEY,"
            "  generator_output LONGBLOB NOT NULL,"
            "  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,"
            "  ttl_sec INT NULL,"
            "  expires_at TIMESTAMP NULL,"
            "  INDEX (expires_at)"
            f") ENGINE=InnoDB DEFAULT CHARSET={self.config.charset};"
        )
    
    def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached GeneratorOutput data"""
        conn = None
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT generator_output, expires_at FROM {self.config.table_name} WHERE cache_key=%s",
                    (cache_key,),
                )
                row = cur.fetchone()
                if not row:
                    logger.debug(f"GeneratorCache MISS key={cache_key}")
                    with self._stats_lock:
                        self._misses += 1
                    return None
                
                generator_output_blob, expires_at = row
                
                # Expiry check
                if expires_at is not None:
                    cur.execute("SELECT NOW() < %s", (expires_at,))
                    result = cur.fetchone()
                    fresh = bool(result[0]) if result is not None else False
                    if not fresh:
                        logger.debug(f"GeneratorCache EXPIRED key={cache_key}")
                        with self._stats_lock:
                            self._misses += 1
                        return None
                
                # Deserialize GeneratorOutput
                try:
                    generator_output_data = pickle.loads(generator_output_blob)
                    # logger.debug(f"GeneratorCache HIT key={cache_key}")
                    with self._stats_lock:
                        self._hits += 1
                    return generator_output_data
                except Exception as e:
                    logger.warning(f"Failed to deserialize cached data for key={cache_key}: {e}")
                    with self._stats_lock:
                        self._misses += 1
                    return None
                    
        except Exception as e:
            logger.debug(f"MySQL cache get failed, returning None: {e}")
            with self._stats_lock:
                self._misses += 1
            return None
        finally:
            self._return_connection(conn)
    
    def put(self, cache_key: str, generator_output_data: Dict[str, Any]) -> bool:
        """Store GeneratorOutput data in cache"""
        conn = None
        try:
            conn = self._get_connection()
            
            # Serialize GeneratorOutput
            try:
                generator_output_blob = pickle.dumps(generator_output_data)
            except Exception as e:
                logger.warning(f"Failed to serialize GeneratorOutput for key={cache_key}: {e}")
                return False
            
            ttl_sec = self.config.ttl_seconds
            if ttl_sec is not None:
                # Let MySQL compute expiration timestamp
                with conn.cursor() as cur:
                    cur.execute(
                        f"INSERT INTO {self.config.table_name} (cache_key, generator_output, ttl_sec, expires_at)"
                        " VALUES (%s, %s, %s, NOW() + INTERVAL %s SECOND)"
                        " ON DUPLICATE KEY UPDATE generator_output=VALUES(generator_output), ttl_sec=VALUES(ttl_sec), expires_at=VALUES(expires_at)",
                        (cache_key, generator_output_blob, ttl_sec, ttl_sec),
                    )
            else:
                with conn.cursor() as cur:
                    cur.execute(
                        f"INSERT INTO {self.config.table_name} (cache_key, generator_output, ttl_sec, expires_at)"
                        " VALUES (%s, %s, %s, NULL)"
                        " ON DUPLICATE KEY UPDATE generator_output=VALUES(generator_output), ttl_sec=VALUES(ttl_sec), expires_at=NULL",
                        (cache_key, generator_output_blob, None),
                    )
            
            logger.debug(f"GeneratorCache STORE key={cache_key} size={len(generator_output_blob)} ttl={ttl_sec}")
            return True
            
        except Exception as e:
            logger.warning(f"MySQL cache put failed, ignoring: {e}")
            return False
        finally:
            self._return_connection(conn)