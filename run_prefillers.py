from src.prefillers.prefilling_all import run_prefilling
from src.prefillers.user_based_prefillers.prefilling_collaborative import run_prefilling_collaborative

run_prefilling(skip_cache_refresh=True, methods_short_text=[], methods_full_text=['lda'])
# run_prefilling_collaborative(test_run=False)
