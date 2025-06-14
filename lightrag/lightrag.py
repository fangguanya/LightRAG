from __future__ import annotations

import traceback
import asyncio
import configparser
import os
import time
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import partial
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Iterator,
    cast,
    final,
    Literal,
    Optional,
    List,
    Dict,
)
from lightrag.constants import (
    DEFAULT_MAX_TOKEN_SUMMARY,
    DEFAULT_FORCE_LLM_SUMMARY_ON_MERGE,
)
from lightrag.utils import get_env_value

from lightrag.kg import (
    STORAGES,
    verify_storage_implementation,
)

from lightrag.kg.shared_storage import (
    get_namespace_data,
    get_pipeline_status_lock,
)

from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    DocProcessingStatus,
    DocStatus,
    DocStatusStorage,
    QueryParam,
    StorageNameSpace,
    StoragesStatus,
)
from .namespace import NameSpace, make_namespace
from .operate import (
    chunking_by_token_size,
    extract_entities,
    merge_nodes_and_edges,
    kg_query,
    naive_query,
    query_with_keywords,
)
from .prompt import GRAPH_FIELD_SEP
from .utils import (
    Tokenizer,
    TiktokenTokenizer,
    EmbeddingFunc,
    always_get_an_event_loop,
    compute_mdhash_id,
    convert_response_to_json,
    lazy_external_import,
    priority_limit_async_func_call,
    get_content_summary,
    clean_text,
    check_storage_env_vars,
    logger,
)
from .types import KnowledgeGraph
from dotenv import load_dotenv

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)

# TODO: TO REMOVE @Yannick
config = configparser.ConfigParser()
config.read("config.ini", "utf-8")


@final
@dataclass
class LightRAG:
    """LightRAG: Simple and Fast Retrieval-Augmented Generation."""

    # Directory
    # ---

    working_dir: str = field(
        default=f"./lightrag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )
    """Directory where cache and temporary files are stored."""

    # Storage
    # ---

    kv_storage: str = field(default="JsonKVStorage")
    """Storage backend for key-value data."""

    vector_storage: str = field(default="NanoVectorDBStorage")
    """Storage backend for vector embeddings."""

    graph_storage: str = field(default="NetworkXStorage")
    """Storage backend for knowledge graphs."""

    doc_status_storage: str = field(default="JsonDocStatusStorage")
    """Storage type for tracking document processing statuses."""

    # Logging (Deprecated, use setup_logger in utils.py instead)
    # ---
    log_level: int | None = field(default=None)
    log_file_path: str | None = field(default=None)

    # Entity extraction
    # ---

    entity_extract_max_gleaning: int = field(default=1)
    """Maximum number of entity extraction attempts for ambiguous content."""

    summary_to_max_tokens: int = field(
        default=get_env_value("MAX_TOKEN_SUMMARY", DEFAULT_MAX_TOKEN_SUMMARY, int)
    )

    force_llm_summary_on_merge: int = field(
        default=get_env_value(
            "FORCE_LLM_SUMMARY_ON_MERGE", DEFAULT_FORCE_LLM_SUMMARY_ON_MERGE, int
        )
    )

    # Text chunking
    # ---

    chunk_token_size: int = field(default=int(os.getenv("CHUNK_SIZE", 1200)))
    """Maximum number of tokens per text chunk when splitting documents."""

    chunk_overlap_token_size: int = field(
        default=int(os.getenv("CHUNK_OVERLAP_SIZE", 100))
    )
    """Number of overlapping tokens between consecutive text chunks to preserve context."""

    tokenizer: Optional[Tokenizer] = field(default=None)
    """
    A function that returns a Tokenizer instance.
    If None, and a `tiktoken_model_name` is provided, a TiktokenTokenizer will be created.
    If both are None, the default TiktokenTokenizer is used.
    """

    tiktoken_model_name: str = field(default="gpt-4o-mini")
    """Model name used for tokenization when chunking text with tiktoken. Defaults to `gpt-4o-mini`."""

    chunking_func: Callable[
        [
            Tokenizer,
            str,
            Optional[str],
            bool,
            int,
            int,
        ],
        List[Dict[str, Any]],
    ] = field(default_factory=lambda: chunking_by_token_size)
    """
    Custom chunking function for splitting text into chunks before processing.

    The function should take the following parameters:

        - `tokenizer`: A Tokenizer instance to use for tokenization.
        - `content`: The text to be split into chunks.
        - `split_by_character`: The character to split the text on. If None, the text is split into chunks of `chunk_token_size` tokens.
        - `split_by_character_only`: If True, the text is split only on the specified character.
        - `chunk_token_size`: The maximum number of tokens per chunk.
        - `chunk_overlap_token_size`: The number of overlapping tokens between consecutive chunks.

    The function should return a list of dictionaries, where each dictionary contains the following keys:
        - `tokens`: The number of tokens in the chunk.
        - `content`: The text content of the chunk.

    Defaults to `chunking_by_token_size` if not specified.
    """

    # Embedding
    # ---

    embedding_func: EmbeddingFunc | None = field(default=None)
    """Function for computing text embeddings. Must be set before use."""

    embedding_batch_num: int = field(default=int(os.getenv("EMBEDDING_BATCH_NUM", 32)))
    """Batch size for embedding computations."""

    embedding_func_max_async: int = field(
        default=int(os.getenv("EMBEDDING_FUNC_MAX_ASYNC", 16))
    )
    """Maximum number of concurrent embedding function calls."""

    embedding_cache_config: dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": False,
            "similarity_threshold": 0.95,
            "use_llm_check": False,
        }
    )
    """Configuration for embedding cache.
    - enabled: If True, enables caching to avoid redundant computations.
    - similarity_threshold: Minimum similarity score to use cached embeddings.
    - use_llm_check: If True, validates cached embeddings using an LLM.
    """

    # LLM Configuration
    # ---

    llm_model_func: Callable[..., object] | None = field(default=None)
    """Function for interacting with the large language model (LLM). Must be set before use."""

    llm_model_name: str = field(default="gpt-4o-mini")
    """Name of the LLM model used for generating responses."""

    llm_model_max_token_size: int = field(default=int(os.getenv("MAX_TOKENS", 32768)))
    """Maximum number of tokens allowed per LLM response."""

    llm_model_max_async: int = field(default=int(os.getenv("MAX_ASYNC", 4)))
    """Maximum number of concurrent LLM calls."""

    llm_model_kwargs: dict[str, Any] = field(default_factory=dict)
    """Additional keyword arguments passed to the LLM model function."""

    # Storage
    # ---

    vector_db_storage_cls_kwargs: dict[str, Any] = field(default_factory=dict)
    """Additional parameters for vector database storage."""

    # TODO：deprecated, remove in the future, use WORKSPACE instead
    namespace_prefix: str = field(default="")
    """Prefix for namespacing stored data across different environments."""

    enable_llm_cache: bool = field(default=True)
    """Enables caching for LLM responses to avoid redundant computations."""

    enable_llm_cache_for_entity_extract: bool = field(default=True)
    """If True, enables caching for entity extraction steps to reduce LLM costs."""

    # Extensions
    # ---

    max_parallel_insert: int = field(default=int(os.getenv("MAX_PARALLEL_INSERT", 2)))
    """Maximum number of parallel insert operations."""

    addon_params: dict[str, Any] = field(
        default_factory=lambda: {
            "language": get_env_value("SUMMARY_LANGUAGE", "English", str)
        }
    )

    # Storages Management
    # ---

    auto_manage_storages_states: bool = field(default=True)
    """If True, lightrag will automatically calls initialize_storages and finalize_storages at the appropriate times."""

    # Storages Management
    # ---

    convert_response_to_json_func: Callable[[str], dict[str, Any]] = field(
        default_factory=lambda: convert_response_to_json
    )
    """
    Custom function for converting LLM responses to JSON format.

    The default function is :func:`.utils.convert_response_to_json`.
    """

    cosine_better_than_threshold: float = field(
        default=float(os.getenv("COSINE_THRESHOLD", 0.2))
    )

    _storages_status: StoragesStatus = field(default=StoragesStatus.NOT_CREATED)

    def __post_init__(self):
        from lightrag.kg.shared_storage import (
            initialize_share_data,
        )

        # Handle deprecated parameters
        if self.log_level is not None:
            warnings.warn(
                "WARNING: log_level parameter is deprecated, use setup_logger in utils.py instead",
                UserWarning,
                stacklevel=2,
            )
        if self.log_file_path is not None:
            warnings.warn(
                "WARNING: log_file_path parameter is deprecated, use setup_logger in utils.py instead",
                UserWarning,
                stacklevel=2,
            )

        # Remove these attributes to prevent their use
        if hasattr(self, "log_level"):
            delattr(self, "log_level")
        if hasattr(self, "log_file_path"):
            delattr(self, "log_file_path")

        initialize_share_data()

        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)

        # Verify storage implementation compatibility and environment variables
        storage_configs = [
            ("KV_STORAGE", self.kv_storage),
            ("VECTOR_STORAGE", self.vector_storage),
            ("GRAPH_STORAGE", self.graph_storage),
            ("DOC_STATUS_STORAGE", self.doc_status_storage),
        ]

        for storage_type, storage_name in storage_configs:
            # Verify storage implementation compatibility
            verify_storage_implementation(storage_type, storage_name)
            # Check environment variables
            check_storage_env_vars(storage_name)

        # Ensure vector_db_storage_cls_kwargs has required fields
        self.vector_db_storage_cls_kwargs = {
            "cosine_better_than_threshold": self.cosine_better_than_threshold,
            **self.vector_db_storage_cls_kwargs,
        }

        # Init Tokenizer
        # Post-initialization hook to handle backward compatabile tokenizer initialization based on provided parameters
        if self.tokenizer is None:
            if self.tiktoken_model_name:
                self.tokenizer = TiktokenTokenizer(self.tiktoken_model_name)
            else:
                self.tokenizer = TiktokenTokenizer()

        # Fix global_config now
        global_config = asdict(self)
        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in global_config.items()])
        logger.debug(f"LightRAG init with param:\n  {_print_config}\n")

        # Init Embedding
        self.embedding_func = priority_limit_async_func_call(
            self.embedding_func_max_async
        )(self.embedding_func)

        # Initialize all storages
        self.key_string_value_json_storage_cls: type[BaseKVStorage] = (
            self._get_storage_class(self.kv_storage)
        )  # type: ignore
        self.vector_db_storage_cls: type[BaseVectorStorage] = self._get_storage_class(
            self.vector_storage
        )  # type: ignore
        self.graph_storage_cls: type[BaseGraphStorage] = self._get_storage_class(
            self.graph_storage
        )  # type: ignore
        self.key_string_value_json_storage_cls = partial(  # type: ignore
            self.key_string_value_json_storage_cls, global_config=global_config
        )
        self.vector_db_storage_cls = partial(  # type: ignore
            self.vector_db_storage_cls, global_config=global_config
        )
        self.graph_storage_cls = partial(  # type: ignore
            self.graph_storage_cls, global_config=global_config
        )

        # Initialize document status storage
        self.doc_status_storage_cls = self._get_storage_class(self.doc_status_storage)

        self.llm_response_cache: BaseKVStorage = self.key_string_value_json_storage_cls(  # type: ignore
            namespace=make_namespace(
                self.namespace_prefix, NameSpace.KV_STORE_LLM_RESPONSE_CACHE
            ),
            global_config=asdict(
                self
            ),  # Add global_config to ensure cache works properly
            embedding_func=self.embedding_func,
        )

        self.full_docs: BaseKVStorage = self.key_string_value_json_storage_cls(  # type: ignore
            namespace=make_namespace(
                self.namespace_prefix, NameSpace.KV_STORE_FULL_DOCS
            ),
            embedding_func=self.embedding_func,
        )

        # TODO: deprecating, text_chunks is redundant with chunks_vdb
        self.text_chunks: BaseKVStorage = self.key_string_value_json_storage_cls(  # type: ignore
            namespace=make_namespace(
                self.namespace_prefix, NameSpace.KV_STORE_TEXT_CHUNKS
            ),
            embedding_func=self.embedding_func,
        )
        self.chunk_entity_relation_graph: BaseGraphStorage = self.graph_storage_cls(  # type: ignore
            namespace=make_namespace(
                self.namespace_prefix, NameSpace.GRAPH_STORE_CHUNK_ENTITY_RELATION
            ),
            embedding_func=self.embedding_func,
        )

        self.entities_vdb: BaseVectorStorage = self.vector_db_storage_cls(  # type: ignore
            namespace=make_namespace(
                self.namespace_prefix, NameSpace.VECTOR_STORE_ENTITIES
            ),
            embedding_func=self.embedding_func,
            meta_fields={"entity_name", "source_id", "content", "file_path"},
        )
        self.relationships_vdb: BaseVectorStorage = self.vector_db_storage_cls(  # type: ignore
            namespace=make_namespace(
                self.namespace_prefix, NameSpace.VECTOR_STORE_RELATIONSHIPS
            ),
            embedding_func=self.embedding_func,
            meta_fields={"src_id", "tgt_id", "source_id", "content", "file_path"},
        )
        self.chunks_vdb: BaseVectorStorage = self.vector_db_storage_cls(  # type: ignore
            namespace=make_namespace(
                self.namespace_prefix, NameSpace.VECTOR_STORE_CHUNKS
            ),
            embedding_func=self.embedding_func,
            meta_fields={"full_doc_id", "content", "file_path"},
        )

        # Initialize document status storage
        self.doc_status: DocStatusStorage = self.doc_status_storage_cls(
            namespace=make_namespace(self.namespace_prefix, NameSpace.DOC_STATUS),
            global_config=global_config,
            embedding_func=None,
        )

        # Directly use llm_response_cache, don't create a new object
        hashing_kv = self.llm_response_cache

        self.llm_model_func = priority_limit_async_func_call(self.llm_model_max_async)(
            partial(
                self.llm_model_func,  # type: ignore
                hashing_kv=hashing_kv,
                **self.llm_model_kwargs,
            )
        )

        self._storages_status = StoragesStatus.CREATED

        if self.auto_manage_storages_states:
            self._run_async_safely(self.initialize_storages, "Storage Initialization")

    def __del__(self):
        if self.auto_manage_storages_states:
            self._run_async_safely(self.finalize_storages, "Storage Finalization")

    def _run_async_safely(self, async_func, action_name=""):
        """Safely execute an async function, avoiding event loop conflicts."""
        try:
            loop = always_get_an_event_loop()
            if loop.is_running():
                task = loop.create_task(async_func())
                task.add_done_callback(
                    lambda t: logger.info(f"{action_name} completed!")
                )
            else:
                loop.run_until_complete(async_func())
        except RuntimeError:
            logger.warning(
                f"No running event loop, creating a new loop for {action_name}."
            )
            loop = asyncio.new_event_loop()
            loop.run_until_complete(async_func())
            loop.close()

    async def initialize_storages(self):
        """Asynchronously initialize the storages"""
        if self._storages_status == StoragesStatus.CREATED:
            tasks = []

            for storage in (
                self.full_docs,
                self.text_chunks,
                self.entities_vdb,
                self.relationships_vdb,
                self.chunks_vdb,
                self.chunk_entity_relation_graph,
                self.llm_response_cache,
                self.doc_status,
            ):
                if storage:
                    tasks.append(storage.initialize())

            await asyncio.gather(*tasks)

            self._storages_status = StoragesStatus.INITIALIZED
            logger.debug("Initialized Storages")

    async def finalize_storages(self):
        """Asynchronously finalize the storages"""
        if self._storages_status == StoragesStatus.INITIALIZED:
            tasks = []

            for storage in (
                self.full_docs,
                self.text_chunks,
                self.entities_vdb,
                self.relationships_vdb,
                self.chunks_vdb,
                self.chunk_entity_relation_graph,
                self.llm_response_cache,
                self.doc_status,
            ):
                if storage:
                    tasks.append(storage.finalize())

            await asyncio.gather(*tasks)

            self._storages_status = StoragesStatus.FINALIZED
            logger.debug("Finalized Storages")

    async def get_graph_labels(self):
        text = await self.chunk_entity_relation_graph.get_all_labels()
        return text

    async def get_knowledge_graph(
        self,
        node_label: str,
        max_depth: int = 3,
        max_nodes: int = 1000,
    ) -> KnowledgeGraph:
        """Get knowledge graph for a given label

        Args:
            node_label (str): Label to get knowledge graph for
            max_depth (int): Maximum depth of graph
            max_nodes (int, optional): Maximum number of nodes to return. Defaults to 1000.

        Returns:
            KnowledgeGraph: Knowledge graph containing nodes and edges
        """

        return await self.chunk_entity_relation_graph.get_knowledge_graph(
            node_label, max_depth, max_nodes
        )

    def _get_storage_class(self, storage_name: str) -> Callable[..., Any]:
        import_path = STORAGES[storage_name]
        storage_class = lazy_external_import(import_path, storage_name)
        return storage_class

    def insert(
        self,
        input: str | list[str],
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        ids: str | list[str] | None = None,
        file_paths: str | list[str] | None = None,
    ) -> None:
        """Sync Insert documents with checkpoint support

        Args:
            input: Single document string or list of document strings
            split_by_character: if split_by_character is not None, split the string by character, if chunk longer than
            chunk_token_size, it will be split again by token size.
            split_by_character_only: if split_by_character_only is True, split the string by character only, when
            split_by_character is None, this parameter is ignored.
            ids: single string of the document ID or list of unique document IDs, if not provided, MD5 hash IDs will be generated
            file_paths: single string of the file path or list of file paths, used for citation
        """
        loop = always_get_an_event_loop()
        loop.run_until_complete(
            self.ainsert(
                input, split_by_character, split_by_character_only, ids, file_paths
            )
        )

    async def ainsert(
        self,
        input: str | list[str],
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        ids: str | list[str] | None = None,
        file_paths: str | list[str] | None = None,
    ) -> None:
        """Async Insert documents with checkpoint support

        Args:
            input: Single document string or list of document strings
            split_by_character: if split_by_character is not None, split the string by character, if chunk longer than
            chunk_token_size, it will be split again by token size.
            split_by_character_only: if split_by_character_only is True, split the string by character only, when
            split_by_character is None, this parameter is ignored.
            ids: list of unique document IDs, if not provided, MD5 hash IDs will be generated
            file_paths: list of file paths corresponding to each document, used for citation
        """
        await self.apipeline_enqueue_documents(input, ids, file_paths)
        await self.apipeline_process_enqueue_documents(
            split_by_character, split_by_character_only
        )

    # TODO: deprecated, use insert instead
    def insert_custom_chunks(
        self,
        full_text: str,
        text_chunks: list[str],
        doc_id: str | list[str] | None = None,
    ) -> None:
        loop = always_get_an_event_loop()
        loop.run_until_complete(
            self.ainsert_custom_chunks(full_text, text_chunks, doc_id)
        )

    # TODO: deprecated, use ainsert instead
    async def ainsert_custom_chunks(
        self, full_text: str, text_chunks: list[str], doc_id: str | None = None
    ) -> None:
        update_storage = False
        try:
            # Clean input texts
            full_text = clean_text(full_text)
            text_chunks = [clean_text(chunk) for chunk in text_chunks]
            file_path = ""

            # Process cleaned texts
            if doc_id is None:
                doc_key = compute_mdhash_id(full_text, prefix="doc-")
            else:
                doc_key = doc_id
            new_docs = {doc_key: {"content": full_text}}

            _add_doc_keys = await self.full_docs.filter_keys({doc_key})
            new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
            if not len(new_docs):
                logger.warning("This document is already in the storage.")
                return

            update_storage = True
            logger.info(f"Inserting {len(new_docs)} docs")

            inserting_chunks: dict[str, Any] = {}
            for index, chunk_text in enumerate(text_chunks):
                chunk_key = compute_mdhash_id(chunk_text, prefix="chunk-")
                tokens = len(self.tokenizer.encode(chunk_text))
                inserting_chunks[chunk_key] = {
                    "content": chunk_text,
                    "full_doc_id": doc_key,
                    "tokens": tokens,
                    "chunk_order_index": index,
                    "file_path": file_path,
                }

            doc_ids = set(inserting_chunks.keys())
            add_chunk_keys = await self.text_chunks.filter_keys(doc_ids)
            inserting_chunks = {
                k: v for k, v in inserting_chunks.items() if k in add_chunk_keys
            }
            if not len(inserting_chunks):
                logger.warning("All chunks are already in the storage.")
                return

            tasks = [
                self.chunks_vdb.upsert(inserting_chunks),
                self._process_entity_relation_graph(inserting_chunks),
                self.full_docs.upsert(new_docs),
                self.text_chunks.upsert(inserting_chunks),
            ]
            await asyncio.gather(*tasks)

        finally:
            if update_storage:
                await self._insert_done()

    async def apipeline_enqueue_documents(
        self,
        input: str | list[str],
        ids: list[str] | None = None,
        file_paths: str | list[str] | None = None,
    ) -> None:
        """
        Pipeline for Processing Documents

        1. Validate ids if provided or generate MD5 hash IDs
        2. Remove duplicate contents
        3. Generate document initial status
        4. Filter out already processed documents
        5. Enqueue document in status

        Args:
            input: Single document string or list of document strings
            ids: list of unique document IDs, if not provided, MD5 hash IDs will be generated
            file_paths: list of file paths corresponding to each document, used for citation
        """
        if isinstance(input, str):
            input = [input]
        if isinstance(ids, str):
            ids = [ids]
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        # If file_paths is provided, ensure it matches the number of documents
        if file_paths is not None:
            if isinstance(file_paths, str):
                file_paths = [file_paths]
            if len(file_paths) != len(input):
                raise ValueError(
                    "Number of file paths must match the number of documents"
                )
        else:
            # If no file paths provided, use placeholder
            file_paths = ["unknown_source"] * len(input)

        # 1. Validate ids if provided or generate MD5 hash IDs
        if ids is not None:
            # Check if the number of IDs matches the number of documents
            if len(ids) != len(input):
                raise ValueError("Number of IDs must match the number of documents")

            # Check if IDs are unique
            if len(ids) != len(set(ids)):
                raise ValueError("IDs must be unique")

            # Generate contents dict of IDs provided by user and documents
            contents = {
                id_: {"content": doc, "file_path": path}
                for id_, doc, path in zip(ids, input, file_paths)
            }
        else:
            # Clean input text and remove duplicates
            cleaned_input = [
                (clean_text(doc), path) for doc, path in zip(input, file_paths)
            ]
            unique_content_with_paths = {}

            # Keep track of unique content and their paths
            for content, path in cleaned_input:
                if content not in unique_content_with_paths:
                    unique_content_with_paths[content] = path

            # Generate contents dict of MD5 hash IDs and documents with paths
            contents = {
                compute_mdhash_id(content, prefix="doc-"): {
                    "content": content,
                    "file_path": path,
                }
                for content, path in unique_content_with_paths.items()
            }

        # 2. Remove duplicate contents
        unique_contents = {}
        for id_, content_data in contents.items():
            content = content_data["content"]
            file_path = content_data["file_path"]
            if content not in unique_contents:
                unique_contents[content] = (id_, file_path)

        # Reconstruct contents with unique content
        contents = {
            id_: {"content": content, "file_path": file_path}
            for content, (id_, file_path) in unique_contents.items()
        }

        # 3. Generate document initial status
        new_docs: dict[str, Any] = {
            id_: {
                "status": DocStatus.PENDING,
                "content": content_data["content"],
                "content_summary": get_content_summary(content_data["content"]),
                "content_length": len(content_data["content"]),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "file_path": content_data[
                    "file_path"
                ],  # Store file path in document status
            }
            for id_, content_data in contents.items()
        }

        # 4. Filter out already processed documents
        # Get docs ids
        all_new_doc_ids = set(new_docs.keys())
        # Exclude IDs of documents that are already in progress
        unique_new_doc_ids = await self.doc_status.filter_keys(all_new_doc_ids)

        # Log ignored document IDs
        ignored_ids = [
            doc_id for doc_id in unique_new_doc_ids if doc_id not in new_docs
        ]
        if ignored_ids:
            logger.warning(
                f"Ignoring {len(ignored_ids)} document IDs not found in new_docs"
            )
            for doc_id in ignored_ids:
                logger.warning(f"Ignored document ID: {doc_id}")

        # Filter new_docs to only include documents with unique IDs
        new_docs = {
            doc_id: new_docs[doc_id]
            for doc_id in unique_new_doc_ids
            if doc_id in new_docs
        }

        if not new_docs:
            logger.info("No new unique documents were found.")
            return

        # 5. Store status document
        await self.doc_status.upsert(new_docs)
        logger.info(f"Stored {len(new_docs)} new unique documents")

    async def apipeline_process_enqueue_documents(
        self,
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
    ) -> None:
        """
        Process pending documents by splitting them into chunks, processing
        each chunk for entity and relation extraction, and updating the
        document status.

        1. Get all pending, failed, and abnormally terminated processing documents.
        2. Split document content into chunks
        3. Process each chunk for entity and relation extraction
        4. Update the document status
        """

        # Get pipeline status shared data and lock
        pipeline_status = await get_namespace_data("pipeline_status")
        pipeline_status_lock = get_pipeline_status_lock()

        # Check if another process is already processing the queue
        async with pipeline_status_lock:
            # Ensure only one worker is processing documents
            if not pipeline_status.get("busy", False):
                processing_docs, failed_docs, pending_docs = await asyncio.gather(
                    self.doc_status.get_docs_by_status(DocStatus.PROCESSING),
                    self.doc_status.get_docs_by_status(DocStatus.FAILED),
                    self.doc_status.get_docs_by_status(DocStatus.PENDING),
                )

                to_process_docs: dict[str, DocProcessingStatus] = {}
                to_process_docs.update(processing_docs)
                to_process_docs.update(failed_docs)
                to_process_docs.update(pending_docs)

                if not to_process_docs:
                    logger.info("No documents to process")
                    return

                pipeline_status.update(
                    {
                        "busy": True,
                        "job_name": "Default Job",
                        "job_start": datetime.now(timezone.utc).isoformat(),
                        "docs": 0,
                        "batchs": 0,  # Total number of files to be processed
                        "cur_batch": 0,  # Number of files already processed
                        "request_pending": False,  # Clear any previous request
                        "latest_message": "",
                    }
                )
                # Cleaning history_messages without breaking it as a shared list object
                del pipeline_status["history_messages"][:]
            else:
                # Another process is busy, just set request flag and return
                pipeline_status["request_pending"] = True
                logger.info(
                    "Another process is already processing the document queue. Request queued."
                )
                return

        try:
            # Process documents until no more documents or requests
            while True:
                if not to_process_docs:
                    log_message = "All documents have been processed or are duplicates"
                    logger.info(log_message)
                    pipeline_status["latest_message"] = log_message
                    pipeline_status["history_messages"].append(log_message)
                    break

                log_message = f"Processing {len(to_process_docs)} document(s)"
                logger.info(log_message)

                # Update pipeline_status, batchs now represents the total number of files to be processed
                pipeline_status["docs"] = len(to_process_docs)
                pipeline_status["batchs"] = len(to_process_docs)
                pipeline_status["cur_batch"] = 0
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

                # Get first document's file path and total count for job name
                first_doc_id, first_doc = next(iter(to_process_docs.items()))
                first_doc_path = first_doc.file_path
                path_prefix = first_doc_path[:20] + (
                    "..." if len(first_doc_path) > 20 else ""
                )
                total_files = len(to_process_docs)
                job_name = f"{path_prefix}[{total_files} files]"
                pipeline_status["job_name"] = job_name

                # Create a counter to track the number of processed files
                processed_count = 0
                # Create a semaphore to limit the number of concurrent file processing
                semaphore = asyncio.Semaphore(self.max_parallel_insert)

                async def process_document(
                    doc_id: str,
                    status_doc: DocProcessingStatus,
                    split_by_character: str | None,
                    split_by_character_only: bool,
                    pipeline_status: dict,
                    pipeline_status_lock: asyncio.Lock,
                    semaphore: asyncio.Semaphore,
                ) -> None:
                    """Process single document"""
                    file_extraction_stage_ok = False
                    async with semaphore:
                        nonlocal processed_count
                        current_file_number = 0
                        try:
                            # Get file path from status document
                            file_path = getattr(
                                status_doc, "file_path", "unknown_source"
                            )

                            async with pipeline_status_lock:
                                # Update processed file count and save current file number
                                processed_count += 1
                                current_file_number = (
                                    processed_count  # Save the current file number
                                )
                                pipeline_status["cur_batch"] = processed_count

                                log_message = f"Extracting stage {current_file_number}/{total_files}: {file_path}"
                                logger.info(log_message)
                                pipeline_status["history_messages"].append(log_message)
                                log_message = f"Processing d-id: {doc_id}"
                                logger.info(log_message)
                                pipeline_status["latest_message"] = log_message
                                pipeline_status["history_messages"].append(log_message)

                            # Generate chunks from document
                            logger.info(f"开始对文档 {doc_id} 进行分块处理...")
                            chunks: dict[str, Any] = {
                                compute_mdhash_id(dp["content"], prefix="chunk-"): {
                                    **dp,
                                    "full_doc_id": doc_id,
                                    "file_path": file_path,  # Add file path to each chunk
                                }
                                for dp in self.chunking_func(
                                    self.tokenizer,
                                    status_doc.content,
                                    split_by_character,
                                    split_by_character_only,
                                    self.chunk_overlap_token_size,
                                    self.chunk_token_size,
                                )
                            }
                            logger.info(f"文档 {doc_id} 分块完成，生成 {len(chunks)} 个文本块")

                            # Process document (text chunks and full docs) in parallel
                            logger.info(f"开始并行处理文档 {doc_id}：文档状态更新、块向量化、实体关系抽取、原文存储")
                            
                            logger.info("===== 创建并行任务 =====")
                            logger.info("创建文档状态更新任务...")
                            doc_status_task = asyncio.create_task(
                                self.doc_status.upsert(
                                    {
                                        doc_id: {
                                            "status": DocStatus.PROCESSING,
                                            "chunks_count": len(chunks),
                                            "content": status_doc.content,
                                            "content_summary": status_doc.content_summary,
                                            "content_length": status_doc.content_length,
                                            "created_at": status_doc.created_at,
                                            "updated_at": datetime.now(
                                                timezone.utc
                                            ).isoformat(),
                                            "file_path": file_path,
                                        }
                                    }
                                )
                            )
                            logger.info("文档状态更新任务创建完成")
                            
                            logger.info(f"创建块向量化任务，处理 {len(chunks)} 个文本块...")
                            # 为chunks_vdb_task添加特殊的监控包装
                            async def chunks_vdb_task_with_monitoring():
                                logger.info(f"===== 开始执行chunks_vdb.upsert =====")
                                logger.info(f"待处理的文本块数量: {len(chunks)}")
                                logger.info(f"当前embedding_batch_num配置: {getattr(self, 'embedding_batch_num', '未设置')}")
                                
                                # 检查数据大小
                                total_content_length = sum(len(chunk.get('content', '')) for chunk in chunks.values())
                                avg_content_length = total_content_length / len(chunks) if chunks else 0
                                logger.info(f"文本内容统计 - 总长度: {total_content_length:,} 字符, 平均长度: {avg_content_length:.0f} 字符")
                                
                                import time
                                start_time = time.time()
                                
                                try:
                                    # 分批处理优化
                                    if len(chunks) > 1000:  # 如果超过1000个块，分批处理
                                        logger.info("数据量较大，启用分批处理模式...")
                                        batch_size = 500
                                        chunk_items = list(chunks.items())
                                        
                                        for i in range(0, len(chunk_items), batch_size):
                                            batch_chunks = dict(chunk_items[i:i + batch_size])
                                            batch_num = i // batch_size + 1
                                            total_batches = (len(chunk_items) + batch_size - 1) // batch_size
                                            
                                            logger.info(f"处理批次 {batch_num}/{total_batches}，包含 {len(batch_chunks)} 个文本块...")
                                            
                                            try:
                                                memory_before = process.memory_info().rss / 1024 / 1024  # MB
                                            except:
                                                memory_before = 0
                                            logger.info(f"批次 {batch_num} 开始前内存使用: {memory_before:.1f} MB")
                                            
                                            batch_start_time = time.time()
                                            
                                            # 详细监控批次内部处理过程
                                            logger.info(f"批次 {batch_num}: 开始调用 chunks_vdb.upsert...")
                                            logger.info(f"批次 {batch_num}: 数据详情 - 包含 {len(batch_chunks)} 个块")
                                            
                                            # 分析批次数据内容
                                            total_content_length = sum(len(chunk.get('content', '')) for chunk in batch_chunks.values())
                                            avg_content_length = total_content_length / len(batch_chunks) if batch_chunks else 0
                                            logger.info(f"批次 {batch_num}: 平均文本长度 {avg_content_length:.0f} 字符，总长度 {total_content_length:,} 字符")
                                            
                                            # 检查是否有特殊的长文本
                                            long_chunks = [(k, len(v.get('content', ''))) for k, v in batch_chunks.items() if len(v.get('content', '')) > 5000]
                                            if long_chunks:
                                                logger.info(f"批次 {batch_num}: 发现 {len(long_chunks)} 个长文本块 (>5000字符)")
                                                for chunk_id, length in long_chunks[:3]:  # 只显示前3个
                                                    logger.info(f"  长文本块 {chunk_id}: {length} 字符")
                                            
                                            # 添加超时机制的upsert调用
                                            upsert_timeout = 60  # 5分钟超时
                                            logger.info(f"批次 {batch_num}: 开始执行upsert操作，超时设置: {upsert_timeout} 秒...")
                                            
                                            try:
                                                await asyncio.wait_for(
                                                    self.chunks_vdb.upsert(batch_chunks),
                                                    timeout=upsert_timeout
                                                )
                                                logger.info(f"批次 {batch_num}: upsert操作成功完成")
                                            except asyncio.TimeoutError:
                                                logger.error(f"批次 {batch_num}: upsert操作超时 ({upsert_timeout} 秒)")
                                                
                                                # 对于超时的批次，尝试分解为更小的子批次
                                                if len(batch_chunks) > 50:
                                                    logger.info(f"批次 {batch_num}: 尝试分解为更小的子批次处理...")
                                                    sub_batch_size = 50
                                                    batch_items = list(batch_chunks.items())
                                                    success_count = 0
                                                    
                                                    for j in range(0, len(batch_items), sub_batch_size):
                                                        sub_batch = dict(batch_items[j:j + sub_batch_size])
                                                        sub_batch_num = j // sub_batch_size + 1
                                                        total_sub_batches = (len(batch_items) + sub_batch_size - 1) // sub_batch_size
                                                        
                                                        try:
                                                            logger.info(f"批次 {batch_num}-子批次 {sub_batch_num}/{total_sub_batches}: 处理 {len(sub_batch)} 个块...")
                                                            await asyncio.wait_for(
                                                                self.chunks_vdb.upsert(sub_batch),
                                                                timeout=60  # 1分钟子批次超时
                                                            )
                                                            success_count += len(sub_batch)
                                                            logger.info(f"批次 {batch_num}-子批次 {sub_batch_num}: 成功")
                                                        except Exception as sub_e:
                                                            logger.error(f"批次 {batch_num}-子批次 {sub_batch_num}: 失败 - {str(sub_e)}")
                                                            continue  # 跳过失败的子批次，继续处理下一个
                                                    
                                                    logger.info(f"批次 {batch_num}: 分解处理完成，成功处理 {success_count}/{len(batch_chunks)} 个块")
                                                    if success_count == 0:
                                                        logger.error(f"批次 {batch_num}: 所有子批次都失败，跳过此批次")
                                                        continue  # 跳过整个批次
                                                else:
                                                    logger.error(f"批次 {batch_num}: 批次太小无法分解，跳过此批次")
                                                    continue  # 跳过此批次
                                                    
                                            except Exception as upsert_error:
                                                logger.error(f"批次 {batch_num}: upsert操作异常: {str(upsert_error)}")
                                                logger.error(f"批次 {batch_num}: 异常类型: {type(upsert_error).__name__}")
                                                logger.error(f"批次 {batch_num}: 尝试跳过此批次继续处理...")
                                                continue  # 跳过失败的批次，继续处理下一个
                                            
                                            batch_elapsed = time.time() - batch_start_time
                                            
                                            try:
                                                memory_after = process.memory_info().rss / 1024 / 1024  # MB
                                                logger.info(f"批次 {batch_num} 完成，耗时: {batch_elapsed:.1f} 秒，内存使用: {memory_after:.1f} MB (+{memory_after-memory_before:.1f} MB)")
                                            except:
                                                logger.info(f"批次 {batch_num} 完成，耗时: {batch_elapsed:.1f} 秒")
                                            
                                            # 强制垃圾回收
                                            import gc
                                            gc.collect()
                                            
                                            # 给系统一些时间
                                            await asyncio.sleep(0.1)
                                    else:
                                        # 正常处理
                                        logger.info("正常处理模式...")
                                        await self.chunks_vdb.upsert(chunks)
                                    
                                    elapsed_time = time.time() - start_time
                                    logger.info(f"===== chunks_vdb.upsert 执行完成，总耗时: {elapsed_time:.1f} 秒 =====")
                                    
                                except Exception as e:
                                    elapsed_time = time.time() - start_time
                                    logger.error(f"chunks_vdb.upsert 执行失败，已运行: {elapsed_time:.1f} 秒")
                                    logger.error(f"错误详情: {str(e)}")
                                    logger.error(traceback.format_exc())
                                    
                                    # 检查是否是embedding相关的错误
                                    error_msg = str(e).lower()
                                    if any(keyword in error_msg for keyword in ['embedding', 'timeout', 'connection', 'network']):
                                        logger.error("检测到embedding或网络相关错误，尝试降级处理...")
                                        
                                        # 尝试更保守的处理方式
                                        try:
                                            logger.info("尝试使用更小的批次重新处理...")
                                            conservative_batch_size = 100
                                            chunk_items = list(chunks.items())
                                            successful_chunks = 0
                                            
                                            for i in range(0, len(chunk_items), conservative_batch_size):
                                                conservative_batch = dict(chunk_items[i:i + conservative_batch_size])
                                                batch_num = i // conservative_batch_size + 1
                                                total_conservative_batches = (len(chunk_items) + conservative_batch_size - 1) // conservative_batch_size
                                                
                                                try:
                                                    logger.info(f"保守处理批次 {batch_num}/{total_conservative_batches}: {len(conservative_batch)} 个块")
                                                    await self.chunks_vdb.upsert(conservative_batch)
                                                    successful_chunks += len(conservative_batch)
                                                    logger.info(f"保守处理批次 {batch_num}: 成功")
                                                    
                                                    # 每个保守批次后休息一下
                                                    await asyncio.sleep(1)
                                                    
                                                except Exception as conservative_e:
                                                    logger.error(f"保守处理批次 {batch_num} 失败: {str(conservative_e)}")
                                                    continue
                                            
                                            if successful_chunks > 0:
                                                logger.info(f"保守处理完成，成功处理 {successful_chunks}/{len(chunks)} 个块")
                                                logger.info("虽然有部分失败，但继续执行后续流程...")
                                            else:
                                                logger.error("保守处理也完全失败，但仍继续执行后续流程以避免整体中断...")
                                                
                                        except Exception as conservative_error:
                                            logger.error(f"保守处理也失败: {str(conservative_error)}")
                                            logger.error("将继续执行后续流程，尽量保持系统稳定...")
                                    else:
                                        logger.error("非embedding错误，继续执行后续流程...")
                                    
                                    # 不再抛出异常，而是继续执行
                                    logger.info("尽管chunks_vdb处理有问题，但继续执行后续流程以保持系统稳定性...")
                            
                            chunks_vdb_task = asyncio.create_task(chunks_vdb_task_with_monitoring())
                            logger.info("块向量化任务创建完成")
                            
                            logger.info(f"创建实体关系抽取任务，处理 {len(chunks)} 个文本块...")
                            entity_relation_task = asyncio.create_task(
                                self._process_entity_relation_graph(
                                    chunks, pipeline_status, pipeline_status_lock
                                )
                            )
                            logger.info("实体关系抽取任务创建完成")
                            
                            logger.info("创建原文存储任务...")
                            full_docs_task = asyncio.create_task(
                                self.full_docs.upsert(
                                    {doc_id: {"content": status_doc.content}}
                                )
                            )
                            logger.info("原文存储任务创建完成")
                            
                            logger.info(f"创建文本块存储任务，处理 {len(chunks)} 个文本块...")
                            text_chunks_task = asyncio.create_task(
                                self.text_chunks.upsert(chunks)
                            )
                            logger.info("文本块存储任务创建完成")
                            
                            tasks = [
                                doc_status_task,
                                chunks_vdb_task,
                                entity_relation_task,
                                full_docs_task,
                                text_chunks_task,
                            ]
                            logger.info(f"启动文档 {doc_id} 的 {len(tasks)} 个并行处理任务")
                            
                            # 监控每个任务的状态
                            task_names = [
                                "doc_status_task", 
                                "chunks_vdb_task", 
                                "entity_relation_task", 
                                "full_docs_task", 
                                "text_chunks_task"
                            ]
                            
                            logger.info("===== 并行任务状态监控 =====")
                            for i, (task, name) in enumerate(zip(tasks, task_names)):
                                logger.info(f"任务 {i+1} ({name}): 完成状态 = {task.done()}")
                            
                            logger.info("准备执行 asyncio.gather(*tasks)...")
                            try:
                                # 使用asyncio.wait代替gather来更好地监控任务状态
                                import time
                                start_time = time.time()
                                timeout_duration = 3600  # 1小时超时
                                
                                logger.info(f"开始等待所有任务完成，超时时间: {timeout_duration} 秒...")
                                
                                # 每30秒检查一次任务状态
                                check_interval = 30
                                while True:
                                    done, pending = await asyncio.wait(tasks, timeout=check_interval, return_when=asyncio.FIRST_COMPLETED)
                                    
                                    elapsed_time = time.time() - start_time
                                    #logger.info(f"===== 任务进度检查 (已运行 {elapsed_time:.1f} 秒) =====")
                                    
                                    # 检查每个任务的状态
                                    all_tasks_done = True
                                    for i, (task, name) in enumerate(zip(tasks, task_names)):
                                        status = "已完成" if task.done() else "运行中"
                                        if task.done() and task.exception():
                                            status = f"异常: {task.exception()}"
                                        #logger.info(f"任务 {i+1} ({name}): {status}")
                                        
                                        # 如果任务未完成，标记all_tasks_done为False
                                        if not task.done():
                                            all_tasks_done = False
                                    
                                    # 检查是否所有任务都完成了 - 使用我们的计数器而不是all()函数
                                    if all_tasks_done:
                                        logger.info("检测到所有任务已完成！准备退出监控循环...")
                                        break
                                    
                                    # 检查是否超时
                                    if elapsed_time > timeout_duration:
                                        logger.error(f"任务执行超时 ({timeout_duration} 秒)，取消未完成的任务...")
                                        for task in tasks:
                                            if not task.done():
                                                task.cancel()
                                        raise asyncio.TimeoutError(f"任务执行超时 ({timeout_duration} 秒)")
                                    
                                    # 如果有任务异常，立即处理
                                    for task in done:
                                        if task.exception():
                                            logger.error(f"任务异常: {task.exception()}")
                                            # 取消其他任务
                                            for other_task in tasks:
                                                if not other_task.done():
                                                    other_task.cancel()
                                            raise task.exception()
                                
                                logger.info("任务监控循环已退出，准备获取所有任务结果...")
                                # 获取所有任务的结果
                                results = []
                                for i, task in enumerate(tasks):
                                    try:
                                        result = task.result()
                                        results.append(result)
                                        logger.info(f"任务 {i+1} ({task_names[i]}) 结果获取成功")
                                    except Exception as e:
                                        logger.error(f"任务 {i+1} ({task_names[i]}) 结果获取失败: {e}")
                                        raise
                                
                                logger.info(f"所有任务成功完成，总耗时: {time.time() - start_time:.1f} 秒")
                                
                            except Exception as e:
                                logger.error(f"并行任务执行出现异常: {str(e)}")
                                logger.error(traceback.format_exc())
                                raise
                            
                            logger.info("asyncio.gather(*tasks) 执行完成")
                            logger.info("===== 检查所有任务最终状态 =====")
                            for i, (task, name) in enumerate(zip(tasks, task_names)):
                                logger.info(f"任务 {i+1} ({name}): 完成状态 = {task.done()}, 异常 = {task.exception() if task.done() else 'N/A'}")
                            
                            logger.info(f"文档 {doc_id} 的所有并行任务完成")
                            file_extraction_stage_ok = True

                        except Exception as e:
                            # Log error and update pipeline status
                            logger.error(traceback.format_exc())
                            error_msg = f"Failed to extrat document {current_file_number}/{total_files}: {file_path}"
                            logger.error(error_msg)
                            async with pipeline_status_lock:
                                pipeline_status["latest_message"] = error_msg
                                pipeline_status["history_messages"].append(
                                    traceback.format_exc()
                                )
                                pipeline_status["history_messages"].append(error_msg)

                            # Persistent llm cache
                            if self.llm_response_cache:
                                await self.llm_response_cache.index_done_callback()

                            # Update document status to failed
                            await self.doc_status.upsert(
                                {
                                    doc_id: {
                                        "status": DocStatus.FAILED,
                                        "error": str(e),
                                        "content": status_doc.content,
                                        "content_summary": status_doc.content_summary,
                                        "content_length": status_doc.content_length,
                                        "created_at": status_doc.created_at,
                                        "updated_at": datetime.now(
                                            timezone.utc
                                        ).isoformat(),
                                        "file_path": file_path,
                                    }
                                }
                            )

                    # Semphore released, concurrency controlled by graph_db_lock in merge_nodes_and_edges instead

                    if file_extraction_stage_ok:
                        try:
                            logger.info(f"文档 {doc_id} 开始进入合并阶段...")
                            # Get chunk_results from entity_relation_task
                            logger.info(f"文档 {doc_id} 准备等待实体抽取任务完成...")
                            logger.info(f"实体抽取任务状态: {entity_relation_task}")
                            logger.info(f"实体抽取任务是否完成: {entity_relation_task.done()}")
                            
                            chunk_results = await entity_relation_task
                            logger.info(f"文档 {doc_id} 成功获取到实体抽取结果")
                            logger.info(f"文档 {doc_id} 实体抽取完成，开始合并节点和边...")
                            logger.info(f"准备合并 {len(chunk_results)} 个块的实体关系结果...")
                            
                            # 分析chunk_results内容
                            logger.info("===== 分析chunk_results详细内容 =====")
                            if isinstance(chunk_results, list):
                                logger.info(f"chunk_results是列表，长度: {len(chunk_results)}")
                                for i, item in enumerate(chunk_results[:5]):  # 只显示前5个
                                    logger.info(f"chunk_results[{i}] 类型: {type(item)}, 内容长度: {len(str(item))}")
                            else:
                                logger.info(f"chunk_results类型: {type(chunk_results)}")
                            logger.info("===== 开始调用merge_nodes_and_edges =====")
                            
                            try:
                                await merge_nodes_and_edges(
                                    chunk_results=chunk_results,  # result collected from entity_relation_task
                                    knowledge_graph_inst=self.chunk_entity_relation_graph,
                                    entity_vdb=self.entities_vdb,
                                    relationships_vdb=self.relationships_vdb,
                                    global_config=asdict(self),
                                    pipeline_status=pipeline_status,
                                    pipeline_status_lock=pipeline_status_lock,
                                    llm_response_cache=self.llm_response_cache,
                                    current_file_number=current_file_number,
                                    total_files=total_files,
                                    file_path=file_path,
                                )
                                logger.info(f"文档 {doc_id} 节点和边合并完成")
                            except Exception as merge_error:
                                logger.error(f"合并节点和边时发生错误: {str(merge_error)}")
                                logger.error(traceback.format_exc())
                                raise merge_error

                            await self.doc_status.upsert(
                                {
                                    doc_id: {
                                        "status": DocStatus.PROCESSED,
                                        "chunks_count": len(chunks),
                                        "content": status_doc.content,
                                        "content_summary": status_doc.content_summary,
                                        "content_length": status_doc.content_length,
                                        "created_at": status_doc.created_at,
                                        "updated_at": datetime.now(
                                            timezone.utc
                                        ).isoformat(),
                                        "file_path": file_path,
                                    }
                                }
                            )

                            # Call _insert_done after processing each file
                            logger.info(f"文档 {doc_id} 调用最终处理完成回调...")
                            await self._insert_done()
                            logger.info(f"文档 {doc_id} 处理完全完成")

                            async with pipeline_status_lock:
                                log_message = f"Completed processing file {current_file_number}/{total_files}: {file_path}"
                                logger.info(log_message)
                                pipeline_status["latest_message"] = log_message
                                pipeline_status["history_messages"].append(log_message)

                        except Exception as e:
                            # Log error and update pipeline status
                            logger.error(traceback.format_exc())
                            error_msg = f"Merging stage failed in document {current_file_number}/{total_files}: {file_path}"
                            logger.error(error_msg)
                            async with pipeline_status_lock:
                                pipeline_status["latest_message"] = error_msg
                                pipeline_status["history_messages"].append(
                                    traceback.format_exc()
                                )
                                pipeline_status["history_messages"].append(error_msg)

                            # Persistent llm cache
                            if self.llm_response_cache:
                                await self.llm_response_cache.index_done_callback()

                            # Update document status to failed
                            await self.doc_status.upsert(
                                {
                                    doc_id: {
                                        "status": DocStatus.FAILED,
                                        "error": str(e),
                                        "content": status_doc.content,
                                        "content_summary": status_doc.content_summary,
                                        "content_length": status_doc.content_length,
                                        "created_at": status_doc.created_at,
                                        "updated_at": datetime.now().isoformat(),
                                        "file_path": file_path,
                                    }
                                }
                            )

                # Create processing tasks for all documents
                doc_tasks = []
                for doc_id, status_doc in to_process_docs.items():
                    doc_tasks.append(
                        process_document(
                            doc_id,
                            status_doc,
                            split_by_character,
                            split_by_character_only,
                            pipeline_status,
                            pipeline_status_lock,
                            semaphore,
                        )
                    )

                # Wait for all document processing to complete
                await asyncio.gather(*doc_tasks)

                # Check if there's a pending request to process more documents (with lock)
                has_pending_request = False
                async with pipeline_status_lock:
                    has_pending_request = pipeline_status.get("request_pending", False)
                    if has_pending_request:
                        # Clear the request flag before checking for more documents
                        pipeline_status["request_pending"] = False

                if not has_pending_request:
                    break

                log_message = "Processing additional documents due to pending request"
                logger.info(log_message)
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

                # Check for pending documents again
                processing_docs, failed_docs, pending_docs = await asyncio.gather(
                    self.doc_status.get_docs_by_status(DocStatus.PROCESSING),
                    self.doc_status.get_docs_by_status(DocStatus.FAILED),
                    self.doc_status.get_docs_by_status(DocStatus.PENDING),
                )

                to_process_docs = {}
                to_process_docs.update(processing_docs)
                to_process_docs.update(failed_docs)
                to_process_docs.update(pending_docs)

        finally:
            log_message = "Document processing pipeline completed"
            logger.info(log_message)
            # Always reset busy status when done or if an exception occurs (with lock)
            async with pipeline_status_lock:
                pipeline_status["busy"] = False
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

    async def _process_entity_relation_graph(
        self, chunks: dict[str, Any], pipeline_status=None, pipeline_status_lock=None
    ) -> list:
        logger.info("===== 开始执行_process_entity_relation_graph函数 =====")
        logger.info(f"输入参数检查 - chunks类型: {type(chunks)}, 数量: {len(chunks)}")
        try:
            logger.info(f"开始处理 {len(chunks)} 个文本块的实体关系抽取...")
            logger.info("准备调用extract_entities函数...")
            
            chunk_results = await extract_entities(
                chunks,
                global_config=asdict(self),
                pipeline_status=pipeline_status,
                pipeline_status_lock=pipeline_status_lock,
                llm_response_cache=self.llm_response_cache,
            )
            
            logger.info(f"extract_entities函数执行完成")
            logger.info(f"实体关系抽取完成，返回 {len(chunk_results)} 个块的结果")
            logger.info(f"返回结果类型: {type(chunk_results)}")
            logger.info("===== _process_entity_relation_graph函数执行完成 =====")
            return chunk_results
        except Exception as e:
            error_msg = f"Failed to extract entities and relationships: {str(e)}"
            logger.error(error_msg)
            logger.error(f"错误详情: {traceback.format_exc()}")
            if pipeline_status_lock:
                async with pipeline_status_lock:
                    pipeline_status["latest_message"] = error_msg
                    pipeline_status["history_messages"].append(error_msg)
            raise e

    async def _insert_done(
        self, pipeline_status=None, pipeline_status_lock=None
    ) -> None:
        logger.info("===== 开始执行_insert_done函数 =====")
        logger.info("准备收集存储实例列表...")
        
        storage_instances = [  # type: ignore
            self.full_docs,
            self.text_chunks,
            self.llm_response_cache,
            self.entities_vdb,
            self.relationships_vdb,
            self.chunks_vdb,
            self.chunk_entity_relation_graph,
        ]
        
        # 检查每个存储实例
        logger.info("检查存储实例状态:")
        for i, storage_inst in enumerate(storage_instances):
            if storage_inst is not None:
                logger.info(f"  存储实例 {i}: {type(storage_inst).__name__} - 有效")
            else:
                logger.info(f"  存储实例 {i}: None - 跳过")
        
        tasks = [
            cast(StorageNameSpace, storage_inst).index_done_callback()
            for storage_inst in storage_instances
            if storage_inst is not None
        ]
        
        logger.info(f"准备执行 {len(tasks)} 个index_done_callback任务...")
        try:
            await asyncio.gather(*tasks)
            logger.info("所有index_done_callback任务执行完成")
        except Exception as e:
            logger.error(f"执行index_done_callback任务时发生错误: {str(e)}")
            logger.error(traceback.format_exc())
            raise

        log_message = "In memory DB persist to disk"
        logger.info(log_message)

        if pipeline_status is not None and pipeline_status_lock is not None:
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)
        
        logger.info("===== _insert_done函数执行完成 =====")

    def insert_custom_kg(
        self, custom_kg: dict[str, Any], full_doc_id: str = None
    ) -> None:
        loop = always_get_an_event_loop()
        loop.run_until_complete(self.ainsert_custom_kg(custom_kg, full_doc_id))

    async def ainsert_custom_kg(
        self,
        custom_kg: dict[str, Any],
        full_doc_id: str = None,
    ) -> None:
        update_storage = False
        try:
            # Insert chunks into vector storage
            all_chunks_data: dict[str, dict[str, str]] = {}
            chunk_to_source_map: dict[str, str] = {}
            for chunk_data in custom_kg.get("chunks", []):
                chunk_content = clean_text(chunk_data["content"])
                source_id = chunk_data["source_id"]
                file_path = chunk_data.get("file_path", "custom_kg")
                tokens = len(self.tokenizer.encode(chunk_content))
                chunk_order_index = (
                    0
                    if "chunk_order_index" not in chunk_data.keys()
                    else chunk_data["chunk_order_index"]
                )
                chunk_id = compute_mdhash_id(chunk_content, prefix="chunk-")

                chunk_entry = {
                    "content": chunk_content,
                    "source_id": source_id,
                    "tokens": tokens,
                    "chunk_order_index": chunk_order_index,
                    "full_doc_id": full_doc_id
                    if full_doc_id is not None
                    else source_id,
                    "file_path": file_path,
                    "status": DocStatus.PROCESSED,
                }
                all_chunks_data[chunk_id] = chunk_entry
                chunk_to_source_map[source_id] = chunk_id
                update_storage = True

            if all_chunks_data:
                await asyncio.gather(
                    self.chunks_vdb.upsert(all_chunks_data),
                    self.text_chunks.upsert(all_chunks_data),
                )

            # Insert entities into knowledge graph
            all_entities_data: list[dict[str, str]] = []
            for entity_data in custom_kg.get("entities", []):
                entity_name = entity_data["entity_name"]
                entity_type = entity_data.get("entity_type", "UNKNOWN")
                description = entity_data.get("description", "No description provided")
                source_chunk_id = entity_data.get("source_id", "UNKNOWN")
                source_id = chunk_to_source_map.get(source_chunk_id, "UNKNOWN")
                file_path = entity_data.get("file_path", "custom_kg")

                # Log if source_id is UNKNOWN
                if source_id == "UNKNOWN":
                    logger.warning(
                        f"Entity '{entity_name}' has an UNKNOWN source_id. Please check the source mapping."
                    )

                # Prepare node data
                node_data: dict[str, str] = {
                    "entity_id": entity_name,
                    "entity_type": entity_type,
                    "description": description,
                    "source_id": source_id,
                    "file_path": file_path,
                    "created_at": int(time.time()),
                }
                # Insert node data into the knowledge graph
                await self.chunk_entity_relation_graph.upsert_node(
                    entity_name, node_data=node_data
                )
                node_data["entity_name"] = entity_name
                all_entities_data.append(node_data)
                update_storage = True

            # Insert relationships into knowledge graph
            all_relationships_data: list[dict[str, str]] = []
            for relationship_data in custom_kg.get("relationships", []):
                src_id = relationship_data["src_id"]
                tgt_id = relationship_data["tgt_id"]
                description = relationship_data["description"]
                keywords = relationship_data["keywords"]
                weight = relationship_data.get("weight", 1.0)
                source_chunk_id = relationship_data.get("source_id", "UNKNOWN")
                source_id = chunk_to_source_map.get(source_chunk_id, "UNKNOWN")
                file_path = relationship_data.get("file_path", "custom_kg")

                # Log if source_id is UNKNOWN
                if source_id == "UNKNOWN":
                    logger.warning(
                        f"Relationship from '{src_id}' to '{tgt_id}' has an UNKNOWN source_id. Please check the source mapping."
                    )

                # Check if nodes exist in the knowledge graph
                for need_insert_id in [src_id, tgt_id]:
                    if not (
                        await self.chunk_entity_relation_graph.has_node(need_insert_id)
                    ):
                        await self.chunk_entity_relation_graph.upsert_node(
                            need_insert_id,
                            node_data={
                                "entity_id": need_insert_id,
                                "source_id": source_id,
                                "description": "UNKNOWN",
                                "entity_type": "UNKNOWN",
                                "file_path": file_path,
                                "created_at": int(time.time()),
                            },
                        )

                # Insert edge into the knowledge graph
                await self.chunk_entity_relation_graph.upsert_edge(
                    src_id,
                    tgt_id,
                    edge_data={
                        "weight": weight,
                        "description": description,
                        "keywords": keywords,
                        "source_id": source_id,
                        "file_path": file_path,
                        "created_at": int(time.time()),
                    },
                )

                edge_data: dict[str, str] = {
                    "src_id": src_id,
                    "tgt_id": tgt_id,
                    "description": description,
                    "keywords": keywords,
                    "source_id": source_id,
                    "weight": weight,
                    "file_path": file_path,
                    "created_at": int(time.time()),
                }
                all_relationships_data.append(edge_data)
                update_storage = True

            # Insert entities into vector storage with consistent format
            data_for_vdb = {
                compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                    "content": dp["entity_name"] + "\n" + dp["description"],
                    "entity_name": dp["entity_name"],
                    "source_id": dp["source_id"],
                    "description": dp["description"],
                    "entity_type": dp["entity_type"],
                    "file_path": dp.get("file_path", "custom_kg"),
                }
                for dp in all_entities_data
            }
            await self.entities_vdb.upsert(data_for_vdb)

            # Insert relationships into vector storage with consistent format
            data_for_vdb = {
                compute_mdhash_id(dp["src_id"] + dp["tgt_id"], prefix="rel-"): {
                    "src_id": dp["src_id"],
                    "tgt_id": dp["tgt_id"],
                    "source_id": dp["source_id"],
                    "content": f"{dp['keywords']}\t{dp['src_id']}\n{dp['tgt_id']}\n{dp['description']}",
                    "keywords": dp["keywords"],
                    "description": dp["description"],
                    "weight": dp["weight"],
                    "file_path": dp.get("file_path", "custom_kg"),
                }
                for dp in all_relationships_data
            }
            await self.relationships_vdb.upsert(data_for_vdb)

        except Exception as e:
            logger.error(f"Error in ainsert_custom_kg: {e}")
            raise
        finally:
            if update_storage:
                await self._insert_done()

    def query(
        self,
        query: str,
        param: QueryParam = QueryParam(),
        system_prompt: str | None = None,
    ) -> str | Iterator[str]:
        """
        Perform a sync query.

        Args:
            query (str): The query to be executed.
            param (QueryParam): Configuration parameters for query execution.
            prompt (Optional[str]): Custom prompts for fine-tuned control over the system's behavior. Defaults to None, which uses PROMPTS["rag_response"].

        Returns:
            str: The result of the query execution.
        """
        loop = always_get_an_event_loop()

        return loop.run_until_complete(self.aquery(query, param, system_prompt))  # type: ignore

    async def aquery(
        self,
        query: str,
        param: QueryParam = QueryParam(),
        system_prompt: str | None = None,
    ) -> str | AsyncIterator[str]:
        """
        Perform a async query.

        Args:
            query (str): The query to be executed.
            param (QueryParam): Configuration parameters for query execution.
                If param.model_func is provided, it will be used instead of the global model.
            prompt (Optional[str]): Custom prompts for fine-tuned control over the system's behavior. Defaults to None, which uses PROMPTS["rag_response"].

        Returns:
            str: The result of the query execution.
        """
        # If a custom model is provided in param, temporarily update global config
        global_config = asdict(self)
        # Save original query for vector search
        param.original_query = query

        if param.mode in ["local", "global", "hybrid", "mix"]:
            response = await kg_query(
                query.strip(),
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.relationships_vdb,
                self.text_chunks,
                param,
                global_config,
                hashing_kv=self.llm_response_cache,
                system_prompt=system_prompt,
                chunks_vdb=self.chunks_vdb,
            )
        elif param.mode == "naive":
            response = await naive_query(
                query.strip(),
                self.chunks_vdb,
                param,
                global_config,
                hashing_kv=self.llm_response_cache,
                system_prompt=system_prompt,
            )
        elif param.mode == "bypass":
            # Bypass mode: directly use LLM without knowledge retrieval
            use_llm_func = param.model_func or global_config["llm_model_func"]
            # Apply higher priority (8) to entity/relation summary tasks
            use_llm_func = partial(use_llm_func, _priority=8)

            param.stream = True if param.stream is None else param.stream
            response = await use_llm_func(
                query.strip(),
                system_prompt=system_prompt,
                history_messages=param.conversation_history,
                stream=param.stream,
            )
        else:
            raise ValueError(f"Unknown mode {param.mode}")
        await self._query_done()
        return response

    # TODO: Deprecated, use user_prompt in QueryParam instead
    def query_with_separate_keyword_extraction(
        self, query: str, prompt: str, param: QueryParam = QueryParam()
    ):
        """
        Query with separate keyword extraction step.

        This method extracts keywords from the query first, then uses them for the query.

        Args:
            query: User query
            prompt: Additional prompt for the query
            param: Query parameters

        Returns:
            Query response
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.aquery_with_separate_keyword_extraction(query, prompt, param)
        )

    # TODO: Deprecated, use user_prompt in QueryParam instead
    async def aquery_with_separate_keyword_extraction(
        self, query: str, prompt: str, param: QueryParam = QueryParam()
    ) -> str | AsyncIterator[str]:
        """
        Async version of query_with_separate_keyword_extraction.

        Args:
            query: User query
            prompt: Additional prompt for the query
            param: Query parameters

        Returns:
            Query response or async iterator
        """
        response = await query_with_keywords(
            query=query,
            prompt=prompt,
            param=param,
            knowledge_graph_inst=self.chunk_entity_relation_graph,
            entities_vdb=self.entities_vdb,
            relationships_vdb=self.relationships_vdb,
            chunks_vdb=self.chunks_vdb,
            text_chunks_db=self.text_chunks,
            global_config=asdict(self),
            hashing_kv=self.llm_response_cache,
        )

        await self._query_done()
        return response

    async def _query_done(self):
        await self.llm_response_cache.index_done_callback()

    async def aclear_cache(self, modes: list[str] | None = None) -> None:
        """Clear cache data from the LLM response cache storage.

        Args:
            modes (list[str] | None): Modes of cache to clear. Options: ["default", "naive", "local", "global", "hybrid", "mix"].
                             "default" represents extraction cache.
                             If None, clears all cache.

        Example:
            # Clear all cache
            await rag.aclear_cache()

            # Clear local mode cache
            await rag.aclear_cache(modes=["local"])

            # Clear extraction cache
            await rag.aclear_cache(modes=["default"])
        """
        if not self.llm_response_cache:
            logger.warning("No cache storage configured")
            return

        valid_modes = ["default", "naive", "local", "global", "hybrid", "mix"]

        # Validate input
        if modes and not all(mode in valid_modes for mode in modes):
            raise ValueError(f"Invalid mode. Valid modes are: {valid_modes}")

        try:
            # Reset the cache storage for specified mode
            if modes:
                success = await self.llm_response_cache.drop_cache_by_modes(modes)
                if success:
                    logger.info(f"Cleared cache for modes: {modes}")
                else:
                    logger.warning(f"Failed to clear cache for modes: {modes}")
            else:
                # Clear all modes
                success = await self.llm_response_cache.drop_cache_by_modes(valid_modes)
                if success:
                    logger.info("Cleared all cache")
                else:
                    logger.warning("Failed to clear all cache")

            await self.llm_response_cache.index_done_callback()

        except Exception as e:
            logger.error(f"Error while clearing cache: {e}")

    def clear_cache(self, modes: list[str] | None = None) -> None:
        """Synchronous version of aclear_cache."""
        return always_get_an_event_loop().run_until_complete(self.aclear_cache(modes))

    async def get_docs_by_status(
        self, status: DocStatus
    ) -> dict[str, DocProcessingStatus]:
        """Get documents by status

        Returns:
            Dict with document id is keys and document status is values
        """
        return await self.doc_status.get_docs_by_status(status)

    async def aget_docs_by_ids(
        self, ids: str | list[str]
    ) -> dict[str, DocProcessingStatus]:
        """Retrieves the processing status for one or more documents by their IDs.

        Args:
            ids: A single document ID (string) or a list of document IDs (list of strings).

        Returns:
            A dictionary where keys are the document IDs for which a status was found,
            and values are the corresponding DocProcessingStatus objects. IDs that
            are not found in the storage will be omitted from the result dictionary.
        """
        if isinstance(ids, str):
            # Ensure input is always a list of IDs for uniform processing
            id_list = [ids]
        elif (
            ids is None
        ):  # Handle potential None input gracefully, although type hint suggests str/list
            logger.warning(
                "aget_docs_by_ids called with None input, returning empty dict."
            )
            return {}
        else:
            # Assume input is already a list if not a string
            id_list = ids

        # Return early if the final list of IDs is empty
        if not id_list:
            logger.debug("aget_docs_by_ids called with an empty list of IDs.")
            return {}

        # Create tasks to fetch document statuses concurrently using the doc_status storage
        tasks = [self.doc_status.get_by_id(doc_id) for doc_id in id_list]
        # Execute tasks concurrently and gather the results. Results maintain order.
        # Type hint indicates results can be DocProcessingStatus or None if not found.
        results_list: list[Optional[DocProcessingStatus]] = await asyncio.gather(*tasks)

        # Build the result dictionary, mapping found IDs to their statuses
        found_statuses: dict[str, DocProcessingStatus] = {}
        # Keep track of IDs for which no status was found (for logging purposes)
        not_found_ids: list[str] = []

        # Iterate through the results, correlating them back to the original IDs
        for i, status_obj in enumerate(results_list):
            doc_id = id_list[
                i
            ]  # Get the original ID corresponding to this result index
            if status_obj:
                # If a status object was returned (not None), add it to the result dict
                found_statuses[doc_id] = status_obj
            else:
                # If status_obj is None, the document ID was not found in storage
                not_found_ids.append(doc_id)

        # Log a warning if any of the requested document IDs were not found
        if not_found_ids:
            logger.warning(
                f"Document statuses not found for the following IDs: {not_found_ids}"
            )

        # Return the dictionary containing statuses only for the found document IDs
        return found_statuses

    # TODO: Deprecated (Deleting documents can cause hallucinations in RAG.)
    # Document delete is not working properly for most of the storage implementations.
    async def adelete_by_doc_id(self, doc_id: str) -> None:
        """Delete a document and all its related data

        Args:
            doc_id: Document ID to delete
        """
        try:
            # 1. Get the document status and related data
            if not await self.doc_status.get_by_id(doc_id):
                logger.warning(f"Document {doc_id} not found")
                return

            logger.debug(f"Starting deletion for document {doc_id}")

            # 2. Get all chunks related to this document
            # Find all chunks where full_doc_id equals the current doc_id
            all_chunks = await self.text_chunks.get_all()
            related_chunks = {
                chunk_id: chunk_data
                for chunk_id, chunk_data in all_chunks.items()
                if isinstance(chunk_data, dict)
                and chunk_data.get("full_doc_id") == doc_id
            }

            if not related_chunks:
                logger.warning(f"No chunks found for document {doc_id}")
                return

            # Get all related chunk IDs
            chunk_ids = set(related_chunks.keys())
            logger.debug(f"Found {len(chunk_ids)} chunks to delete")

            # TODO: self.entities_vdb.client_storage only works for local storage, need to fix this

            # 3. Before deleting, check the related entities and relationships for these chunks
            for chunk_id in chunk_ids:
                # Check entities
                entities_storage = await self.entities_vdb.client_storage
                entities = [
                    dp
                    for dp in entities_storage["data"]
                    if chunk_id in dp.get("source_id")
                ]
                logger.debug(f"Chunk {chunk_id} has {len(entities)} related entities")

                # Check relationships
                relationships_storage = await self.relationships_vdb.client_storage
                relations = [
                    dp
                    for dp in relationships_storage["data"]
                    if chunk_id in dp.get("source_id")
                ]
                logger.debug(f"Chunk {chunk_id} has {len(relations)} related relations")

            # Continue with the original deletion process...

            # 4. Delete chunks from vector database
            if chunk_ids:
                await self.chunks_vdb.delete(chunk_ids)
                await self.text_chunks.delete(chunk_ids)

            # 5. Find and process entities and relationships that have these chunks as source
            # Get all nodes and edges from the graph storage using storage-agnostic methods
            entities_to_delete = set()
            entities_to_update = {}  # entity_name -> new_source_id
            relationships_to_delete = set()
            relationships_to_update = {}  # (src, tgt) -> new_source_id

            # Process entities - use storage-agnostic methods
            all_labels = await self.chunk_entity_relation_graph.get_all_labels()
            for node_label in all_labels:
                node_data = await self.chunk_entity_relation_graph.get_node(node_label)
                if node_data and "source_id" in node_data:
                    # Split source_id using GRAPH_FIELD_SEP
                    sources = set(node_data["source_id"].split(GRAPH_FIELD_SEP))
                    sources.difference_update(chunk_ids)
                    if not sources:
                        entities_to_delete.add(node_label)
                        logger.debug(
                            f"Entity {node_label} marked for deletion - no remaining sources"
                        )
                    else:
                        new_source_id = GRAPH_FIELD_SEP.join(sources)
                        entities_to_update[node_label] = new_source_id
                        logger.debug(
                            f"Entity {node_label} will be updated with new source_id: {new_source_id}"
                        )

            # Process relationships
            for node_label in all_labels:
                node_edges = await self.chunk_entity_relation_graph.get_node_edges(
                    node_label
                )
                if node_edges:
                    for src, tgt in node_edges:
                        edge_data = await self.chunk_entity_relation_graph.get_edge(
                            src, tgt
                        )
                        if edge_data and "source_id" in edge_data:
                            # Split source_id using GRAPH_FIELD_SEP
                            sources = set(edge_data["source_id"].split(GRAPH_FIELD_SEP))
                            sources.difference_update(chunk_ids)
                            if not sources:
                                relationships_to_delete.add((src, tgt))
                                logger.debug(
                                    f"Relationship {src}-{tgt} marked for deletion - no remaining sources"
                                )
                            else:
                                new_source_id = GRAPH_FIELD_SEP.join(sources)
                                relationships_to_update[(src, tgt)] = new_source_id
                                logger.debug(
                                    f"Relationship {src}-{tgt} will be updated with new source_id: {new_source_id}"
                                )

            # Delete entities
            if entities_to_delete:
                for entity in entities_to_delete:
                    await self.entities_vdb.delete_entity(entity)
                    logger.debug(f"Deleted entity {entity} from vector DB")
                await self.chunk_entity_relation_graph.remove_nodes(
                    list(entities_to_delete)
                )
                logger.debug(f"Deleted {len(entities_to_delete)} entities from graph")

            # Update entities
            for entity, new_source_id in entities_to_update.items():
                node_data = await self.chunk_entity_relation_graph.get_node(entity)
                if node_data:
                    node_data["source_id"] = new_source_id
                    await self.chunk_entity_relation_graph.upsert_node(
                        entity, node_data
                    )
                    logger.debug(
                        f"Updated entity {entity} with new source_id: {new_source_id}"
                    )

            # Delete relationships
            if relationships_to_delete:
                for src, tgt in relationships_to_delete:
                    rel_id_0 = compute_mdhash_id(src + tgt, prefix="rel-")
                    rel_id_1 = compute_mdhash_id(tgt + src, prefix="rel-")
                    await self.relationships_vdb.delete([rel_id_0, rel_id_1])
                    logger.debug(f"Deleted relationship {src}-{tgt} from vector DB")
                await self.chunk_entity_relation_graph.remove_edges(
                    list(relationships_to_delete)
                )
                logger.debug(
                    f"Deleted {len(relationships_to_delete)} relationships from graph"
                )

            # Update relationships
            for (src, tgt), new_source_id in relationships_to_update.items():
                edge_data = await self.chunk_entity_relation_graph.get_edge(src, tgt)
                if edge_data:
                    edge_data["source_id"] = new_source_id
                    await self.chunk_entity_relation_graph.upsert_edge(
                        src, tgt, edge_data
                    )
                    logger.debug(
                        f"Updated relationship {src}-{tgt} with new source_id: {new_source_id}"
                    )

            # 6. Delete original document and status
            await self.full_docs.delete([doc_id])
            await self.doc_status.delete([doc_id])

            # 7. Ensure all indexes are updated
            await self._insert_done()

            logger.info(
                f"Successfully deleted document {doc_id} and related data. "
                f"Deleted {len(entities_to_delete)} entities and {len(relationships_to_delete)} relationships. "
                f"Updated {len(entities_to_update)} entities and {len(relationships_to_update)} relationships."
            )

            async def process_data(data_type, vdb, chunk_id):
                # Check data (entities or relationships)
                storage = await vdb.client_storage
                data_with_chunk = [
                    dp
                    for dp in storage["data"]
                    if chunk_id in (dp.get("source_id") or "").split(GRAPH_FIELD_SEP)
                ]

                data_for_vdb = {}
                if data_with_chunk:
                    logger.warning(
                        f"found {len(data_with_chunk)} {data_type} still referencing chunk {chunk_id}"
                    )

                    for item in data_with_chunk:
                        old_sources = item["source_id"].split(GRAPH_FIELD_SEP)
                        new_sources = [src for src in old_sources if src != chunk_id]

                        if not new_sources:
                            logger.info(
                                f"{data_type} {item.get('entity_name', 'N/A')} is deleted because source_id is not exists"
                            )
                            await vdb.delete_entity(item)
                        else:
                            item["source_id"] = GRAPH_FIELD_SEP.join(new_sources)
                            item_id = item["__id__"]
                            data_for_vdb[item_id] = item.copy()
                            if data_type == "entities":
                                data_for_vdb[item_id]["content"] = data_for_vdb[
                                    item_id
                                ].get("content") or (
                                    item.get("entity_name", "")
                                    + (item.get("description") or "")
                                )
                            else:  # relationships
                                data_for_vdb[item_id]["content"] = data_for_vdb[
                                    item_id
                                ].get("content") or (
                                    (item.get("keywords") or "")
                                    + (item.get("src_id") or "")
                                    + (item.get("tgt_id") or "")
                                    + (item.get("description") or "")
                                )

                    if data_for_vdb:
                        await vdb.upsert(data_for_vdb)
                        logger.info(f"Successfully updated {data_type} in vector DB")

            # Add verification step
            async def verify_deletion():
                # Verify if the document has been deleted
                if await self.full_docs.get_by_id(doc_id):
                    logger.warning(f"Document {doc_id} still exists in full_docs")

                # Verify if chunks have been deleted
                all_remaining_chunks = await self.text_chunks.get_all()
                remaining_related_chunks = {
                    chunk_id: chunk_data
                    for chunk_id, chunk_data in all_remaining_chunks.items()
                    if isinstance(chunk_data, dict)
                    and chunk_data.get("full_doc_id") == doc_id
                }

                if remaining_related_chunks:
                    logger.warning(
                        f"Found {len(remaining_related_chunks)} remaining chunks"
                    )

                # Verify entities and relationships
                for chunk_id in chunk_ids:
                    await process_data("entities", self.entities_vdb, chunk_id)
                    await process_data(
                        "relationships", self.relationships_vdb, chunk_id
                    )

            await verify_deletion()

        except Exception as e:
            logger.error(f"Error while deleting document {doc_id}: {e}")

    async def adelete_by_entity(self, entity_name: str) -> None:
        """Asynchronously delete an entity and all its relationships.

        Args:
            entity_name: Name of the entity to delete
        """
        from .utils_graph import adelete_by_entity

        return await adelete_by_entity(
            self.chunk_entity_relation_graph,
            self.entities_vdb,
            self.relationships_vdb,
            entity_name,
        )

    def delete_by_entity(self, entity_name: str) -> None:
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.adelete_by_entity(entity_name))

    async def adelete_by_relation(self, source_entity: str, target_entity: str) -> None:
        """Asynchronously delete a relation between two entities.

        Args:
            source_entity: Name of the source entity
            target_entity: Name of the target entity
        """
        from .utils_graph import adelete_by_relation

        return await adelete_by_relation(
            self.chunk_entity_relation_graph,
            self.relationships_vdb,
            source_entity,
            target_entity,
        )

    def delete_by_relation(self, source_entity: str, target_entity: str) -> None:
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.adelete_by_relation(source_entity, target_entity)
        )

    async def get_processing_status(self) -> dict[str, int]:
        """Get current document processing status counts

        Returns:
            Dict with counts for each status
        """
        return await self.doc_status.get_status_counts()

    async def get_entity_info(
        self, entity_name: str, include_vector_data: bool = False
    ) -> dict[str, str | None | dict[str, str]]:
        """Get detailed information of an entity"""
        from .utils_graph import get_entity_info

        return await get_entity_info(
            self.chunk_entity_relation_graph,
            self.entities_vdb,
            entity_name,
            include_vector_data,
        )

    async def get_relation_info(
        self, src_entity: str, tgt_entity: str, include_vector_data: bool = False
    ) -> dict[str, str | None | dict[str, str]]:
        """Get detailed information of a relationship"""
        from .utils_graph import get_relation_info

        return await get_relation_info(
            self.chunk_entity_relation_graph,
            self.relationships_vdb,
            src_entity,
            tgt_entity,
            include_vector_data,
        )

    async def aedit_entity(
        self, entity_name: str, updated_data: dict[str, str], allow_rename: bool = True
    ) -> dict[str, Any]:
        """Asynchronously edit entity information.

        Updates entity information in the knowledge graph and re-embeds the entity in the vector database.

        Args:
            entity_name: Name of the entity to edit
            updated_data: Dictionary containing updated attributes, e.g. {"description": "new description", "entity_type": "new type"}
            allow_rename: Whether to allow entity renaming, defaults to True

        Returns:
            Dictionary containing updated entity information
        """
        from .utils_graph import aedit_entity

        return await aedit_entity(
            self.chunk_entity_relation_graph,
            self.entities_vdb,
            self.relationships_vdb,
            entity_name,
            updated_data,
            allow_rename,
        )

    def edit_entity(
        self, entity_name: str, updated_data: dict[str, str], allow_rename: bool = True
    ) -> dict[str, Any]:
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.aedit_entity(entity_name, updated_data, allow_rename)
        )

    async def aedit_relation(
        self, source_entity: str, target_entity: str, updated_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Asynchronously edit relation information.

        Updates relation (edge) information in the knowledge graph and re-embeds the relation in the vector database.

        Args:
            source_entity: Name of the source entity
            target_entity: Name of the target entity
            updated_data: Dictionary containing updated attributes, e.g. {"description": "new description", "keywords": "new keywords"}

        Returns:
            Dictionary containing updated relation information
        """
        from .utils_graph import aedit_relation

        return await aedit_relation(
            self.chunk_entity_relation_graph,
            self.entities_vdb,
            self.relationships_vdb,
            source_entity,
            target_entity,
            updated_data,
        )

    def edit_relation(
        self, source_entity: str, target_entity: str, updated_data: dict[str, Any]
    ) -> dict[str, Any]:
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.aedit_relation(source_entity, target_entity, updated_data)
        )

    async def acreate_entity(
        self, entity_name: str, entity_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Asynchronously create a new entity.

        Creates a new entity in the knowledge graph and adds it to the vector database.

        Args:
            entity_name: Name of the new entity
            entity_data: Dictionary containing entity attributes, e.g. {"description": "description", "entity_type": "type"}

        Returns:
            Dictionary containing created entity information
        """
        from .utils_graph import acreate_entity

        return await acreate_entity(
            self.chunk_entity_relation_graph,
            self.entities_vdb,
            self.relationships_vdb,
            entity_name,
            entity_data,
        )

    def create_entity(
        self, entity_name: str, entity_data: dict[str, Any]
    ) -> dict[str, Any]:
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.acreate_entity(entity_name, entity_data))

    async def acreate_relation(
        self, source_entity: str, target_entity: str, relation_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Asynchronously create a new relation between entities.

        Creates a new relation (edge) in the knowledge graph and adds it to the vector database.

        Args:
            source_entity: Name of the source entity
            target_entity: Name of the target entity
            relation_data: Dictionary containing relation attributes, e.g. {"description": "description", "keywords": "keywords"}

        Returns:
            Dictionary containing created relation information
        """
        from .utils_graph import acreate_relation

        return await acreate_relation(
            self.chunk_entity_relation_graph,
            self.entities_vdb,
            self.relationships_vdb,
            source_entity,
            target_entity,
            relation_data,
        )

    def create_relation(
        self, source_entity: str, target_entity: str, relation_data: dict[str, Any]
    ) -> dict[str, Any]:
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.acreate_relation(source_entity, target_entity, relation_data)
        )

    async def amerge_entities(
        self,
        source_entities: list[str],
        target_entity: str,
        merge_strategy: dict[str, str] = None,
        target_entity_data: dict[str, Any] = None,
    ) -> dict[str, Any]:
        """Asynchronously merge multiple entities into one entity.

        Merges multiple source entities into a target entity, handling all relationships,
        and updating both the knowledge graph and vector database.

        Args:
            source_entities: List of source entity names to merge
            target_entity: Name of the target entity after merging
            merge_strategy: Merge strategy configuration, e.g. {"description": "concatenate", "entity_type": "keep_first"}
                Supported strategies:
                - "concatenate": Concatenate all values (for text fields)
                - "keep_first": Keep the first non-empty value
                - "keep_last": Keep the last non-empty value
                - "join_unique": Join all unique values (for fields separated by delimiter)
            target_entity_data: Dictionary of specific values to set for the target entity,
                overriding any merged values, e.g. {"description": "custom description", "entity_type": "PERSON"}

        Returns:
            Dictionary containing the merged entity information
        """
        from .utils_graph import amerge_entities

        return await amerge_entities(
            self.chunk_entity_relation_graph,
            self.entities_vdb,
            self.relationships_vdb,
            source_entities,
            target_entity,
            merge_strategy,
            target_entity_data,
        )

    def merge_entities(
        self,
        source_entities: list[str],
        target_entity: str,
        merge_strategy: dict[str, str] = None,
        target_entity_data: dict[str, Any] = None,
    ) -> dict[str, Any]:
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.amerge_entities(
                source_entities, target_entity, merge_strategy, target_entity_data
            )
        )

    async def aexport_data(
        self,
        output_path: str,
        file_format: Literal["csv", "excel", "md", "txt"] = "csv",
        include_vector_data: bool = False,
    ) -> None:
        """
        Asynchronously exports all entities, relations, and relationships to various formats.
        Args:
            output_path: The path to the output file (including extension).
            file_format: Output format - "csv", "excel", "md", "txt".
                - csv: Comma-separated values file
                - excel: Microsoft Excel file with multiple sheets
                - md: Markdown tables
                - txt: Plain text formatted output
                - table: Print formatted tables to console
            include_vector_data: Whether to include data from the vector database.
        """
        from .utils import aexport_data as utils_aexport_data

        await utils_aexport_data(
            self.chunk_entity_relation_graph,
            self.entities_vdb,
            self.relationships_vdb,
            output_path,
            file_format,
            include_vector_data,
        )

    def export_data(
        self,
        output_path: str,
        file_format: Literal["csv", "excel", "md", "txt"] = "csv",
        include_vector_data: bool = False,
    ) -> None:
        """
        Synchronously exports all entities, relations, and relationships to various formats.
        Args:
            output_path: The path to the output file (including extension).
            file_format: Output format - "csv", "excel", "md", "txt".
                - csv: Comma-separated values file
                - excel: Microsoft Excel file with multiple sheets
                - md: Markdown tables
                - txt: Plain text formatted output
                - table: Print formatted tables to console
            include_vector_data: Whether to include data from the vector database.
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        loop.run_until_complete(
            self.aexport_data(output_path, file_format, include_vector_data)
        )
