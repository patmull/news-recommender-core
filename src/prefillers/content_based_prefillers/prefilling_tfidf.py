import traceback

from src.prefillers.prefiller import prefilling_job_content_based

while True:
    try:
        prefilling_job_content_based("tfidf", "pgsql")
    except Exception as e:
        print("Exception occurred " + str(e))
        traceback.print_exception(None, e, e.__traceback__)
