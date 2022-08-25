import traceback

from content_based_algorithms.prefiller import PreFiller

prefiller = PreFiller()

while True:
    try:
        prefiller.prefilling_job("lda", "pgsql", full_text=True, reverse=False, random=True)
    except Exception as e:
        print("Exception occured " + str(e))
        print(traceback.print_exception(type(e), e, e.__traceback__))
