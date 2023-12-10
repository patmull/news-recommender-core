def assert_recommendation(similar_posts):
    assert type(similar_posts) is list
    assert len(similar_posts) > 0
    assert type(similar_posts[0]['slug']) is str
    assert type(similar_posts[0]['coefficient']) is float
    assert len(similar_posts) > 0
