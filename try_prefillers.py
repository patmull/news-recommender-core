from src.prefillers.prefilling_all import run_prefilling

# *** HERE was also a running of a classifier
run_prefilling(skip_cache_refresh=True, methods_short_text=[], methods_full_text=['word2vec_eval_idnes_3'])
