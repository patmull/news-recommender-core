from pathlib import Path

CACHED_POSTS_FILE_PATH = "db_cache/cached_posts_dataframe.pkl"


def get_cached_posts_file_path():
    return Path(CACHED_POSTS_FILE_PATH)