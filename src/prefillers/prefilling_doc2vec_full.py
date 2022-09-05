from src.recommender_core.recommender_algorithms.content_based_algorithms.prefiller import PreFiller

prefiller = PreFiller()

while True:
    try:
        prefiller.prefilling_job("doc2vec", "pgsql", full_text=True, reverse=False, random=True)
    except Exception as e:
        print("Exception occured" + str(e))