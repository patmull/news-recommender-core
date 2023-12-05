import json


def convert_df_to_json(dataframe):
    result = dataframe[["title", "excerpt", "body"]].to_json(orient="records", lines=True)
    parsed = json.loads(result)
    return parsed


def convert_to_json_keyword_based(post_recommendations):
    list_of_article_slugs = []
    post_recommendation_dictionary = post_recommendations.to_dict('records')
    list_of_article_slugs.append(post_recommendation_dictionary.copy())
    return list_of_article_slugs[0]


def convert_dataframe_posts_to_json(post_recommendations, slug, cosine_sim_df):
    list_of_article_slugs = []
    list_of_coefficients = []

    # finding coefficient belonging to recommended posts compared to original post
    # (for which we want to find recommendations)
    if 'post_slug' in post_recommendations:
        post_recommendations = post_recommendations.rename(columns={'post_slug': 'slug'})

    for index, row in post_recommendations.iterrows():
        list_of_coefficients.append(cosine_sim_df.at[row['slug'], slug])

    post_recommendations['coefficient'] = list_of_coefficients
    posts_recommendations_dictionary = post_recommendations.to_dict('records')
    list_of_article_slugs.append(posts_recommendations_dictionary.copy())
    return list_of_article_slugs[0]


