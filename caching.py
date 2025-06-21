# caching.py
import os
import hashlib
import joblib
import shutil
import tempfile
from typing import Dict, List, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# --- Caching Utilities ---
CACHE_DIR = "rag_cache"
os.makedirs(CACHE_DIR, exist_ok=True) # Ensure the base cache directory exists on import

# Define specific subdirectories for different cache types, if desired
# This adds a layer of organization within rag_cache
VECTOR_CACHE_DIR = os.path.join(CACHE_DIR, "vectors")
SUMMARY_CACHE_DIR = os.path.join(CACHE_DIR, "summaries")
PDF_HASH_CACHE_DIR = os.path.join(CACHE_DIR, "pdf_hashes") # For PDF file hashes

os.makedirs(VECTOR_CACHE_DIR, exist_ok=True)
os.makedirs(SUMMARY_CACHE_DIR, exist_ok=True)
os.makedirs(PDF_HASH_CACHE_DIR, exist_ok=True)


# --- Hashing Functions (Kept Separate) ---

def get_doc_content_hash(doc_text: str) -> str:
    """Generates an MD5 hash of the document content."""
    return hashlib.md5(doc_text.encode('utf-8')).hexdigest()

def get_document_summary_hash(doc_text: str, metadata: Dict) -> str:
    """
    Generates a deterministic hash for a document's content and key metadata for summarization.
    This ensures that if the document content or relevant metadata changes,
    the summary cache is invalidated.
    """
    relevant_metadata_str = f"{metadata.get('source', '')}-{metadata.get('title', '')}"
    combined_content = doc_text[:8000] + relevant_metadata_str # Hash only the part sent to LLM for summary
    return hashlib.md5(combined_content.encode('utf-8')).hexdigest()

def get_file_hash(filepath: str) -> str:
    """Generates an MD5 hash for a file's binary content."""
    hasher = hashlib.md5()
    try:
        with open(filepath, 'rb') as afile:
            while True:
                buf = afile.read(65536) # Read 64KB chunks
                if not buf:
                    break
                hasher.update(buf)
    except Exception as e:
        logger.error(f"Error reading file for hashing {filepath}: {e}")
        return "" # Return empty hash on error
    return hasher.hexdigest()

def get_doc_identifier(documents: List[str], metadata: List[Dict]) -> str:
    """
    Generates a unique identifier for the document set based on content hashes and titles/sources.
    This helps in determining if the corpus has changed.
    """
    doc_info_list = []
    for doc, m in zip(documents, metadata):
        doc_content_hash = get_doc_content_hash(doc)
        identifier_part = m.get('title') or os.path.basename(m.get('source', 'untitled_source'))
        doc_info_list.append(f"{identifier_part}-{doc_content_hash}")
    return hashlib.md5("_".join(sorted(doc_info_list)).encode('utf-8')).hexdigest()

def get_vectorizer_params_hash(vectorizer_config: Dict[str, Any]) -> str:
    """Generates a hash of the TF-IDF vectorizer parameters."""
    config_str = str(sorted(vectorizer_config.items()))
    return hashlib.md5(config_str.encode('utf-8')).hexdigest()


# --- Common Cache Load/Save Functions ---

def _get_cache_path(cache_name: str, sub_dir: str = None) -> str:
    """Helper to get the base path for a cache file."""
    if sub_dir:
        return os.path.join(CACHE_DIR, sub_dir, cache_name)
    return os.path.join(CACHE_DIR, cache_name)

def load_cache(
    cache_name: str,
    sub_dir: Optional[str] = None,
    identifier: Optional[str] = None,
    params_hash: Optional[str] = None
) -> Any:
    """
    Loads cached data from disk.
    Args:
        cache_name (str): The base name for the cache files (e.g., 'document_summaries', 'tfidf_vectors').
        sub_dir (Optional[str]): Subdirectory within CACHE_DIR (e.g., 'summaries', 'vectors').
        identifier (Optional[str]): A hash representing the data source (e.g., document corpus hash).
                                    If provided, cache validity is checked against this identifier.
        params_hash (Optional[str]): A hash representing parameters (e.g., vectorizer config hash).
                                     If provided, cache validity is checked against this hash.
    Returns:
        Any: The loaded cached data, or None if not found or invalid.
    """
    base_path = _get_cache_path(cache_name, sub_dir)
    data_path = f"{base_path}.joblib"
    identifier_path = f"{base_path}_identifier.txt"
    params_hash_path = f"{base_path}_params_hash.txt"

    files_exist = os.path.exists(data_path)

    if identifier:
        files_exist = files_exist and os.path.exists(identifier_path)
    if params_hash:
        files_exist = files_exist and os.path.exists(params_hash_path)

    if files_exist:
        try:
            is_valid = True
            if identifier:
                with open(identifier_path, 'r') as f:
                    stored_identifier = f.read().strip()
                if stored_identifier != identifier:
                    is_valid = False
                    logger.info(f"Cached '{cache_name}' data is outdated (identifier mismatch). Recomputing...")
            
            if params_hash and is_valid: # Only check params_hash if identifier is still valid (or not present)
                with open(params_hash_path, 'r') as f:
                    stored_params_hash = f.read().strip()
                if stored_params_hash != params_hash:
                    is_valid = False
                    logger.info(f"Cached '{cache_name}' data is outdated (parameters mismatch). Recomputing...")

            if is_valid:
                logger.info(f"Loading cached '{cache_name}' data from {data_path}...")
                return joblib.load(data_path)
            
        except Exception as e:
            logger.error(f"Error loading cached '{cache_name}' data: {e}. Recomputing...")
    else:
        logger.info(f"No complete cached '{cache_name}' data found or files are missing. Recomputing...")
    return None


def save_cache(
    data: Any,
    cache_name: str,
    sub_dir: Optional[str] = None,
    identifier: Optional[str] = None,
    params_hash: Optional[str] = None
):
    """
    Saves data to cache atomically.
    Args:
        data (Any): The data to be cached.
        cache_name (str): The base name for the cache files.
        sub_dir (Optional[str]): Subdirectory within CACHE_DIR.
        identifier (Optional[str]): A hash representing the data source.
        params_hash (Optional[str]): A hash representing parameters.
    """
    base_path = _get_cache_path(cache_name, sub_dir)
    data_path = f"{base_path}.joblib"
    identifier_path = f"{base_path}_identifier.txt"
    params_hash_path = f"{base_path}_params_hash.txt"

    logger.info(f"Saving '{cache_name}' data to cache (atomically) to {data_path}...")
    temp_files = [] # Keep track of temporary files for cleanup

    try:
        # Save main data
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, dir=_get_cache_dir_for_subdir(sub_dir)) as tmp_file:
            joblib.dump(data, tmp_file)
            temp_files.append(tmp_file.name)
        os.replace(tmp_file.name, data_path)

        # Save identifier if provided
        if identifier:
            with tempfile.NamedTemporaryFile(mode='w', delete=False, dir=_get_cache_dir_for_subdir(sub_dir)) as tmp_file:
                tmp_file.write(identifier)
                temp_files.append(tmp_file.name)
            os.replace(tmp_file.name, identifier_path)

        # Save params hash if provided
        if params_hash:
            with tempfile.NamedTemporaryFile(mode='w', delete=False, dir=_get_cache_dir_for_subdir(sub_dir)) as tmp_file:
                tmp_file.write(params_hash)
                temp_files.append(tmp_file.name)
            os.replace(tmp_file.name, params_hash_path)

        logger.info(f"Successfully saved '{cache_name}' data.")
    except Exception as e:
        logger.error(f"Error saving '{cache_name}' data to cache: {e}. Cleaning up partial files.")
        for p in temp_files:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError as cleanup_e:
                logger.warning(f"Error during cleanup of temporary file {p}: {cleanup_e}")


def _get_cache_dir_for_subdir(sub_dir: Optional[str]) -> str:
    """Helper to return the actual directory path for temp files."""
    if sub_dir == "summaries":
        return SUMMARY_CACHE_DIR
    elif sub_dir == "vectors":
        return VECTOR_CACHE_DIR
    elif sub_dir == "pdf_hashes":
        return PDF_HASH_CACHE_DIR
    else:
        return CACHE_DIR # Default to base cache dir


def clear_rag_cache():
    """Removes the entire rag_cache directory."""
    if os.path.exists(CACHE_DIR):
        logger.info(f"Clearing cache directory: {CACHE_DIR}...")
        shutil.rmtree(CACHE_DIR)
        logger.info("Cache cleared.")
    else:
        logger.info("No cache directory found to clear.")
    # Recreate all necessary directories
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(VECTOR_CACHE_DIR, exist_ok=True)
    os.makedirs(SUMMARY_CACHE_DIR, exist_ok=True)
    os.makedirs(PDF_HASH_CACHE_DIR, exist_ok=True)