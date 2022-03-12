@DeprecationWarning
def load_models():
    print("Loading Word2Vec model")
    # amazon_bucket_url = 's3://' + AWS_ACCESS_KEY_ID + ":" + AWS_SECRET_ACCESS_KEY + "@moje-clanky/w2v_embedding_all_in_one"

    global word2vec_embedding
    """
    s3 = boto3.client('s3')
    destination_file = "w2v_embedding_all_in_one"
    bucket_name = "moje-clanky"
    s3.download_file(bucket_name, destination_file, destination_file)
    """
    word2vec_embedding = KeyedVectors.load("models/w2v_model_limited")

    # amazon_bucket_url = 's3://' + AWS_ACCESS_KEY_ID + ":" + AWS_SECRET_ACCESS_KEY + "@moje-clanky/d2v_all_in_one.model"
    print("Loading Doc2Vec model")
    global doc2vec_model
    # doc2vec_model = pickle.load(smart_open.smart_open(amazon_bucket_url))
    # doc2vec_model = Doc2Vec.load("d2v_all_in_one.model")
    doc2vec_model = Doc2Vec.load("models/d2v_limited.model")

    # amazon_bucket_url = 's3://' + AWS_ACCESS_KEY_ID + ":" + AWS_SECRET_ACCESS_KEY + "@moje-clanky/lda_all_in_one"
    print("Loading LDA model")

    global lda_model
    # lda_model = pickle.load(smart_open.smart_open(amazon_bucket_url))
    # lda_model = Lda.load("lda_all_in_one")
    lda_model = LdaModel.load("models/lda_model")