import traceback

from src.prefillers.prefiller import prefilling_job_content_based


def prefill_lda_full():
    """
    Invoking the LDA full-text variant prefilling.

    :rtype: object
    """
    while True:
        try:
            prefilling_job_content_based("lda", full_text=True)
        except Exception as e:
            print("Exception occurred " + str(e))
            traceback.print_exception(None, e, e.__traceback__)


if __name__ == '__main__':
    prefill_lda_full()
