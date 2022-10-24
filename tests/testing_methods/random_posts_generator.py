from src.recommender_core.data_handling.data_manipulation import DatabaseMethods


def get_three_unique_posts():
    database = DatabaseMethods()
    posts = database.get_posts_dataframe()
    random_post_1 = posts.sample()
    ranom_slug_1 = random_post_1['slug'].iloc[0]

    while True:
        random_post_2 = posts.sample()
        random_slug_2 = random_post_2['slug'].iloc[0]
        if random_slug_2 != ranom_slug_1:
            break

    while True:
        random_post_3 = posts.sample()
        random_slug_3 = random_post_3['slug'].iloc[0]
        if (random_slug_3 != random_slug_2) and (random_slug_3 != ranom_slug_1):
            break

    return ranom_slug_1, random_slug_2, random_slug_3