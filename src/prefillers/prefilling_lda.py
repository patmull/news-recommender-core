import traceback

from src.core.recommender_algorithms.content_based_algorithms.prefiller import PreFiller

prefiller = PreFiller()

while True:
    try:
        prefiller.prefilling_job("lda", "pgsql", full_text=False, reverse=False, random=True)
    except Exception as e:
        print("Exception occured " + str(e))
        print(traceback.print_exception(type(e), e, e.__traceback__))