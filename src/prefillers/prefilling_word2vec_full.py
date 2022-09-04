from src.core.recommender_algorithms.content_based_algorithms.prefiller import PreFiller

prefiller = PreFiller()

while True:
    try:
        prefiller.prefilling_job("word2vec", "pgsql", full_text=True, reverse=False, random=False)
    except Exception as e:
        print("Exception occured " + str(e))
        print(e)