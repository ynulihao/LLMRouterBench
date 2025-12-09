from typing import List
from pathlib import Path
import sqlite3
import hashlib
import json
import os
import time
import threading
from loguru import logger
from openai import OpenAI, RateLimitError, APIError

__all__ = ["EmbeddingCache"]


class EmbeddingCache:
    """Thin wrapper around the OpenAI embedding endpoint with SQLite caching."""
    def __init__(
        self,
        base_url: str = "http://api.openai.com/v1",
        api_key: str = "sk-12344",
        model_name: str = "text-embedding-3-large",
        cache_dir: str | os.PathLike = ".cache",
        max_retries: int = 5,
        initial_delay: float = 1.0,
    ) -> None:
        self.model_name = model_name
        self.max_retries = max_retries
        self.initial_delay = initial_delay

        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        self.db_path = cache_path / "embeddings.db"
        # —— 单一持久连接 ——
        self._conn = self._open_conn()
        self._init_db()

        # 写锁，保证一次只写一条
        self._w_lock = threading.Lock()
        
        self._init_db()

    # ---------------------------------------------------------------------
    # public helpers
    # ---------------------------------------------------------------------

    def get(self, text: str) -> List[float]:
        """Return the embedding for *text*, fetching from cache or remote."""
        text_hash = hashlib.md5(text.encode()).hexdigest()

        # 1. try cache ------------------------------------------------------
        row = self._select(text_hash)
        if row is not None:
            return row

        # 2. call OpenAI ----------------------------------------------------
        delay = self.initial_delay
        for attempt in range(self.max_retries):
            try:
                rsp = self._client.embeddings.create(input=text, model=self.model_name)
                emb: List[float] = rsp.data[0].embedding  # type: ignore[index]
                self._insert(text_hash, text, emb)
                return emb

            except RateLimitError as e:
                logger.warning(
                    f"Rate limited (attempt {attempt+1}/{self.max_retries}). Retry in {delay:.1f}s"
                )
            except APIError as e:
                logger.warning(
                    f"OpenAI API error (attempt {attempt+1}/{self.max_retries}): {e}. Retry in {delay:.1f}s"
                )
            except Exception as e:
                logger.error(f"Unexpected error — abort: {e}")
                raise

            time.sleep(delay)
            delay *= 2

        raise RuntimeError("Failed to get embedding after multiple retries.")

    def batch(self, texts: List[str], max_batch_size: int = 100) -> List[List[float]]:
        """Return embeddings for a list of texts (keeps order).
        
        Args:
            texts: List of texts to get embeddings for
            max_batch_size: Maximum number of texts to process in a single API call
        """
        all_embeddings: List[List[float]] = []
        
        # Process texts in chunks
        for i in range(0, len(texts), max_batch_size):
            chunk = texts[i:i + max_batch_size]
            
            # 原有的缓存查询和API调用逻辑
            hits: List[List[float]] = []
            misses: List[str] = []
            mapping: dict[str, int] = {}

            for idx, t in enumerate(chunk):
                h = hashlib.md5(t.encode()).hexdigest()
                row = self._select(h)
                if row is not None:
                    hits.append(row)
                else:
                    mapping[t] = idx
                    misses.append(t)

            if misses:
                rsp = self._client.embeddings.create(input=misses, model=self.model_name)
                for text, record in zip(misses, rsp.data):
                    emb: List[float] = record.embedding
                    self._insert(hashlib.md5(text.encode()).hexdigest(), text, emb)
                    hits.insert(mapping[text], emb)

            all_embeddings.extend(hits)
            
        return all_embeddings

    # ------------------------------------------------------------------
    # private db helpers
    # ------------------------------------------------------------------
    def _open_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.db_path,
            timeout=30,               # 等锁 30s
            check_same_thread=False,  # 允许跨线程
            isolation_level=None      # autocommit
        )
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn
    
    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    text_hash TEXT,
                    model     TEXT,
                    embedding TEXT NOT NULL,
                    text      TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(text_hash, model)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_model ON embeddings(model);")

    def _select(self, text_hash: str) -> List[float] | None:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT embedding FROM embeddings WHERE text_hash=? AND model=?",
                (text_hash, self.model_name),
            ).fetchone()
            if row:
                return json.loads(row[0])  # stored as JSON string for readability
            return None

    def _insert(self, text_hash: str, text: str, embedding: List[float]) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO embeddings (text_hash, model, embedding, text) VALUES (?,?,?,?)",
                (text_hash, self.model_name, json.dumps(embedding), text),
            )