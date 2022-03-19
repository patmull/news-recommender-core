import gc
import pandas as pd
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from content_based_algorithms.data_queries import RecommenderMethods
from content_based_algorithms.tfidf import TfIdf
from data_conenction import Database


class Doc2VecClass:
    # amazon_bucket_url = 's3://' + AWS_ACCESS_KEY_ID + ":" + AWS_SECRET_ACCESS_KEY + "@moje-clanky/d2v_all_in_one.model"

    def __init__(self):
        self.documents = None
        self.df = None
        self.posts_df = None
        self.categories_df = None
        self.database = Database()

    def get_posts_dataframe(self):
        # self.database.insert_post_dataframe_to_cache() # uncomment for UPDATE of DB records
        self.posts_df = self.database.get_posts_dataframe_from_cache()
        self.posts_df.drop_duplicates(subset=['title'], inplace=True)
        return self.posts_df

    def get_categories_dataframe(self):
        self.database.connect()
        self.categories_df = self.database.get_categories_dataframe(pd)
        self.database.disconnect()
        return self.categories_df

    def join_posts_ratings_categories(self):
        self.df = self.posts_df.merge(self.categories_df, left_on='category_id', right_on='id')
        # clean up from unnecessary columns
        self.df = self.df[
            ['id_x', 'title_x', 'slug_x', 'excerpt', 'body', 'views', 'keywords', 'title_y', 'description',
             'all_features_preprocessed', 'body_preprocessed']]

    def train_doc2vec(self, documents_all_features_preprocessed, body_text, limited=True):

        tagged_data = [TaggedDocument(words=_d.split(", "), tags=[str(i)]) for i, _d in
                       enumerate(documents_all_features_preprocessed)]

        print("tagged_data:")
        print(tagged_data)

        self.train_full_text(tagged_data, body_text, limited)

        # amazon loading
        # amazon_bucket_url = "..."
        # recommenderMethods = RecommenderMethods()
        # model = recommenderMethods.download_from_amazon(amazon_bucket_url)

        # to find the vector of a document which is not in training data

    def get_similar_doc2vec(self, slug, train=False, limited=True, number_of_recommended_posts=21):
        self.get_posts_dataframe()
        self.get_categories_dataframe()
        self.join_posts_ratings_categories()

        cols = ['keywords', 'all_features_preprocessed']
        documents_df = pd.DataFrame()
        self.df['all_features_preprocessed'] = self.df['all_features_preprocessed'].apply(
            lambda x: x.replace(' ', ', '))
        documents_df['all_features_preprocessed'] = self.df[cols].apply(lambda row: ' '.join(row.values.astype(str)),
                                                                        axis=1)
        documents_df['all_features_preprocessed'] = self.df['title_y'] + ', ' + documents_df[
            'all_features_preprocessed']

        documents_all_features_preprocessed = list(
            map(' '.join, documents_df[['all_features_preprocessed']].values.tolist()))

        del documents_df
        gc.collect()

        documents_slugs = self.df['slug_x'].tolist()

        """
        filename = "cz_stemmer/czech_stopwords.txt"

        with open(filename, encoding="utf-8") as file:
            cz_stopwords = file.readlines()
            cz_stopwords = [line.rstrip() for line in cz_stopwords]
        """

        if train is True:
            self.train_doc2vec(documents_all_features_preprocessed, False)

        del documents_all_features_preprocessed
        gc.collect()

        # pickle.dump(d2v_model,open("d2v_all_in_one.model", "wb" ))
        # global doc2vec_model

        if limited is True:
            doc2vec_model = Doc2Vec.load("models/d2v_limited.model")
        else:
            doc2vec_model = Doc2Vec.load("models/d2v.model")
        # model = pickle.load(smart_open.smart_open(self.amazon_bucket_url))
        # to find the vector of a document which is not in training data
        recommenderMethods = RecommenderMethods()
        # not necessary
        post_found = recommenderMethods.find_post_by_slug(slug)
        keywords_preprocessed = post_found.iloc[0]['keywords'].split(", ")
        all_features_preprocessed = post_found.iloc[0]['all_features_preprocessed'].split(" ")

        tokens = keywords_preprocessed + all_features_preprocessed

        vector_source = doc2vec_model.infer_vector(tokens)

        most_similar = doc2vec_model.dv.most_similar([vector_source], topn=number_of_recommended_posts)
        return self.get_similar_posts_slug(most_similar, documents_slugs, number_of_recommended_posts)

    def get_similar_doc2vec_with_full_text(self, slug, train=False, number_of_recommended_posts=21):
        self.get_posts_dataframe()
        self.get_categories_dataframe()
        self.join_posts_ratings_categories()

        cols = ['keywords', 'all_features_preprocessed', 'body_preprocessed']
        documents_df = pd.DataFrame()
        self.df['all_features_preprocessed'] = self.df['all_features_preprocessed'].apply(
            lambda x: x.replace(' ', ', '))

        self.df.fillna("", inplace=True)

        self.df['body_preprocessed'] = self.df['body_preprocessed'].apply(
            lambda x: x.replace(' ', ', '))
        documents_df['all_features_preprocessed'] = self.df[cols].apply(lambda row: ' '.join(row.values.astype(str)),
                                                                        axis=1)

        documents_df['all_features_preprocessed'] = self.df['title_y'] + ', ' + documents_df[
            'all_features_preprocessed'] + ", " + self.df['body_preprocessed']

        documents_all_features_preprocessed = list(
            map(' '.join, documents_df[['all_features_preprocessed']].values.tolist()))

        del documents_df
        gc.collect()

        documents_slugs = self.df['slug_x'].tolist()

        if train is True:
            self.train_doc2vec(documents_all_features_preprocessed, True)
        del documents_all_features_preprocessed
        gc.collect()

        """
        filename = "cz_stemmer/czech_stopwords.txt"
        
        with open(filename, encoding="utf-8") as file:
            cz_stopwords = file.readlines()
            cz_stopwords = [line.rstrip() for line in cz_stopwords]
        """
        doc2vec_model = Doc2Vec.load("models/d2v_full_text_limited.model")

        recommendMethods = RecommenderMethods()

        # not necessary
        post_found = recommendMethods.find_post_by_slug(slug)
        keywords_preprocessed = post_found.iloc[0]['keywords'].split(", ")
        all_features_preprocessed = post_found.iloc[0]['all_features_preprocessed'].split(" ")
        full_text = post_found.iloc[0]['body_preprocessed'].split(" ")
        tokens = keywords_preprocessed + all_features_preprocessed + full_text
        vector_source = doc2vec_model.infer_vector(tokens)

        most_similar = doc2vec_model.dv.most_similar([vector_source], topn=number_of_recommended_posts)

        return self.get_similar_posts_slug(most_similar, documents_slugs, number_of_recommended_posts)

    def get_similar_doc2vec_by_keywords(self, slug, number_of_recommended_posts=21):
        self.get_posts_dataframe()
        self.get_categories_dataframe()
        self.join_posts_ratings_categories()

        # cols = ["title_y","title_x","excerpt","keywords","slug_x"]

        cols = ["keywords"]
        documents_df = pd.DataFrame()
        documents_df['keywords'] = self.df[cols].apply(lambda row: '. '.join(row.values.astype(str)), axis=1)
        documents_slugs = self.df['slug_x'].tolist()

        filename = "cz_stemmer/czech_stopwords.txt"
        with open(filename, encoding="utf-8") as file:
            cz_stopwords = file.readlines()
            cz_stopwords = [line.rstrip() for line in cz_stopwords]

        """
        documents_df['documents_cleaned'] = documents_df.title_x.apply(lambda x: " ".join(
            re.sub(r'(?![\d_])\w', ' ', w).lower() for w in x.split() if
            # re.sub(r'[^a-zA-Z]', ' ', w).lower() not in cz_stopwords))
        documents_df['keywords_cleaned'] = documents_df.keywords.apply(lambda x: " ".join(
            w.lower() for w in x.split()
            if w.lower() not in cz_stopwords))
        documents_df['title_x'] = title_df.title_x.apply(lambda x: " ".join(
            w.lower() for w in x.split()
            if w.lower() not in cz_stopwords))
        documents_df['slug_x'] = slug_df.slug_x
        """
        # keywords_cleaned = list(map(' '.join, documents_df[['keywords_cleaned']].values.tolist()))
        # print("keywords_cleaned:")
        # print(keywords_cleaned[:20])
        """
        self.train()
        """

        # model = Doc2Vec.load()
        # to find the vector of a document which is not in training data
        tfidf = TfIdf()
        # post_preprocessed = word_tokenize("Zemřel poslední krkonošský nosič Helmut Hofer, ikona Velké Úpy. Ve věku 88 let zemřel potomek slavného rodu vysokohorských nosičů Helmut Hofer z Velké Úpy. Byl posledním žijícím nosičem v Krkonoších, starodávným řemeslem se po staletí živili generace jeho předků. Jako nosič pracoval pro Českou boudu na Sněžce mezi lety 1948 až 1953.".lower())
        # post_preprocessed = tfidf.preprocess_single_post("zemrel-posledni-krkonossky-nosic-helmut-hofer-ikona-velke-upy")

        # not necessary
        post_preprocessed = tfidf.preprocess_single_post(slug)
        post_features_to_find = post_preprocessed.iloc[0]['keywords']

        tokens = post_features_to_find.split()

        global doc2vec_model

        doc2vec_model = Doc2Vec.load("models/d2v.models")
        vector = doc2vec_model.infer_vector(tokens)

        most_similar = doc2vec_model.docvecs.most_similar([vector], topn=number_of_recommended_posts)
        return self.get_similar_posts_slug(most_similar, documents_slugs, number_of_recommended_posts)

    def get_similar_posts_slug(self, most_similar, documents_slugs, number_of_recommended_posts):
        print('\n')

        post_recommendations = pd.DataFrame()
        list_of_article_slugs = []
        list_of_coefficients = []

        most_similar = most_similar[1:number_of_recommended_posts]

        # for label, index in [('MOST', 0), ('SECOND-MOST', 1), ('THIRD-MOST', 2), ('FOURTH-MOST', 3), ('FIFTH-MOST', 4), ('MEDIAN', len(most_similar) // 2), ('LEAST', len(most_similar) - 1)]:
        for index in range(0, len(most_similar)):
            print(u'%s: %s\n' % (most_similar[index][1], documents_slugs[int(most_similar[index][0])]))
            list_of_article_slugs.append(documents_slugs[int(most_similar[index][0])])
            list_of_coefficients.append(most_similar[index][1])
        print('=====================\n')

        post_recommendations['slug'] = list_of_article_slugs
        post_recommendations['coefficient'] = list_of_coefficients

        dict = post_recommendations.to_dict('records')

        list_of_articles = []

        list_of_articles.append(dict.copy())
        # print("------------------------------------")
        # print("JSON:")
        # print("------------------------------------")
        # print(list_of_article_slugs[0])
        return self.flatten(list_of_articles)

    def train(self, tagged_data):

        max_epochs = 20
        vec_size = 150
        alpha = 0.025
        minimum_alpha = 0.0025
        reduce_alpha = 0.0002

        model = Doc2Vec(vector_size=vec_size,
                        alpha=alpha,
                        min_count=0,
                        dm=0)

        model.build_vocab(tagged_data)

        for epoch in range(max_epochs):
            print('iteration {0}'.format(epoch))
            model.train(tagged_data,
                        total_examples=model.corpus_count,
                        epochs=model.epochs)
            # decrease the learning rate
            model.alpha -= 0.0002
            # fix the learning rate, no decay
            model.min_alpha = model.alpha

        model.train(tagged_data,
                    total_examples=model.corpus_count,
                    epochs=model.epochs)
        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha

        model.save("models/d2v.model")
        print("LDA model Saved")

    def train_full_text(self, tagged_data, full_body, limited):

        max_epochs = 20
        vec_size = 150
        alpha = 0.025
        minimum_alpha = 0.0025
        reduce_alpha = 0.0002

        if limited is True:
            model = Doc2Vec(vector_size=vec_size,
                            alpha=alpha,
                            min_count=0,
                            dm=0, max_vocab_size=87000)
        else:
            model = Doc2Vec(vector_size=vec_size,
                            alpha=alpha,
                            min_count=0,
                            dm=0)

        model.build_vocab(tagged_data)

        for epoch in range(max_epochs):
            print('iteration {0}'.format(epoch))
            model.train(tagged_data,
                        total_examples=model.corpus_count,
                        epochs=model.epochs)
            # decrease the learning rate
            model.alpha -= 0.0002
            # fix the learning rate, no decay
            model.min_alpha = model.alpha

        model.train(tagged_data,
                    total_examples=model.corpus_count,
                    epochs=model.epochs)
        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha

        if full_body is True:
            if limited is True:
                model.save("models/d2v_full_text_limited.model")
            else:
                model.save("models/d2v_full_text.model")
        else:
            if limited is True:
                model.save("models/d2v_limited.model")
            else:
                model.save("models/d2v.model")
        print("LDA model Saved")

    def flatten(self, t):
        return [item for sublist in t for item in sublist]

    def most_similar(self, model, search_term):

        inferred_vector = model.infer_vector(search_term)

        sims = model.docvecs.most_similar([inferred_vector], topn=20)

        res = []
        for elem in sims:
            inner = {}
            inner['index'] = elem[0]
            inner['distance'] = elem[1]
            res.append(inner)

        return (res[:20])