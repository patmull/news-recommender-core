from pathlib import Path

CACHED_POSTS_FILE_PATH = "db_cache/cached_posts_dataframe.pkl"

W2V_MODELS_FOLDER_PATHS_AND_MODEL_NAMES = {
    'word2vec_eval_idnes_1': ['full_models/idnes/evaluated_models/word2vec_model_1/', 'idnes_1'],
    'word2vec_eval_idnes_2': ['full_models/idnes/evaluated_models/word2vec_model_2_default_parameters/', 'idnes_2'],
    'word2vec_eval_idnes_3': ['full_models/idnes/evaluated_models/word2vec_model_3/', 'idnes_3'],
    'word2vec_eval_idnes_4': ['full_models/idnes/evaluated_models/word2vec_model_4/', 'idnes_4'],
    'word2vec_eval_cswiki_1': ['full_models/cswiki/evaluated_models/word2vec_model_cswiki_1/', 'word2vec_eval_cswiki_1']
}


def get_cached_posts_file_path():
    return Path(CACHED_POSTS_FILE_PATH)