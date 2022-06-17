from content_based_algorithms.prefiller import PreFiller

prefiller = PreFiller()

while True:
    try:
        prefiller.prefilling_job("lda", "pgsql", full_text=False, reverse=False, random=True)
    except Exception as e:
        print("Exception occured" + str(e))