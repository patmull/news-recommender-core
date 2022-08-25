import traceback

from content_based_algorithms.prefiller import PreFiller

prefiller = PreFiller()

while True:
    try:
        prefiller.prefilling_job("tfidf", "pgsql", full_text=True, reverse=False, random=True)
    except Exception as e:
        print("Exception occured:")
        traceback.print_exception(type(e), e, e.__traceback__)