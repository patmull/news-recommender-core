import traceback

from src.prefillers.prefiller import prefilling_job_content_based


@PendingDeprecationWarning
def prefill_doc2vec_vector_representation():
    while True:
        try:
            prefilling_job_content_based("doc2vec_vectors")
        except Exception as e:
            print("Exception occurred " + str(e))
            traceback.print_exception(None, e, e.__traceback__)
