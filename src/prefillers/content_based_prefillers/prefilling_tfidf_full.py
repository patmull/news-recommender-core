import traceback

from src.prefillers.prefiller import prefilling_job_content_based


def prefill_tfidf_full():
    """
    Invoking the TF-IDF full-text prefilling.
    :rtype: object
    """
    while True:
        try:
            prefilling_job_content_based("terms_frequencies", full_text=True)
        except Exception as e:
            print("Exception occurred: ")
            traceback.print_exception(None, e, e.__traceback__)


if __name__ == '__main__':
    prefill_tfidf_full()
