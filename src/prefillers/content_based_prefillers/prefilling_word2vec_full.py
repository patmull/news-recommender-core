import traceback

from src.prefillers.prefiller import prefilling_job_content_based


def prefill_word2vec_full():
    """
    Invoking the Word2Vec full-text prefilling.
    :return:
    """
    while True:
        try:
            prefilling_job_content_based("word2vec_eval_idnes_3", full_text=True)
        except Exception as e:
            print("Exception occurred " + str(e))
            traceback.print_exception(None, e, e.__traceback__)
