import traceback

from src.recommender_core.recommender_algorithms.content_based_algorithms.prefiller import PreFiller

prefiller = PreFiller()

while True:
    try:
        prefiller.prefilling_job("tfidf", "pgsql", full_text=False, reverse=False, random=True)
    except Exception as e:
        print("Exception occured " + str(e))
        print(traceback.print_exception(type(e), e, e.__traceback__))
