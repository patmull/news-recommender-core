import traceback

from src.recommender_core.recommender_algorithms.content_based_algorithms.prefiller import prefilling_job

while True:
    try:
        prefilling_job("lda", "pgsql")
    except Exception as e:
        print("Exception occured " + str(e))
        print(traceback.print_exception(type(e), e, e.__traceback__))
