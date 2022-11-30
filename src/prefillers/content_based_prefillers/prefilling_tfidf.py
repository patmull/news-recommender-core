import traceback

from src.prefillers.prefiller import prefilling_job_content_based


def prefill_tfidf():
    """
    Invoking the TF-IDF prefilling.

    :rtype: object
    """
    while True:
        try:
            prefilling_job_content_based("tfidf", full_text=False)
        except Exception as e:
            print("Exception occurred " + str(e))
            traceback.print_exception(None, e, e.__traceback__)


if __name__ == '__main__':
    prefill_tfidf()
