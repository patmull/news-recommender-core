import gc
import json
import re
import string

import numpy as np
import pandas as pd
from nltk import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

from content_based_algorithms.data_queries import RecommenderMethods
from cz_stemmer.czech_stemmer import cz_stem
from data_conenction import Database


class TfIdf:

    def __init__(self):
        self.posts_df = None
        self.ratings_df = None
        self.categories_df = None
        self.df = None
        self.database = Database()
        self.user_categories_df = None
        self.tfidf_tuples = None
        self.tfidf_vectorizer = None
        self.cosine_sim_df = None

    def demo(self):
        # print("----------------------------------")
        # print("----------------------------------")
        # print("Multi-document comparison")
        # print("----------------------------------")
        # print("----------------------------------")
        sentence_1 = "S dnešním Mašinfírou se vrátíme do Jeseníků. Projedeme se po dráze, která začíná přímo pod úbočím Pradědu. Po povodních v roce 1997 málem zanikla, zachránili ji ale soukromníci. Prototypem řady 816 se projedeme z Vrbna pod Pradědem do Milotic nad Opavou."
        sentence_2 = "Za necelé dva týdny chce Milan Světlík začít veslovat napříč Atlantikem z New Yorku do Evropy. Pokud trasu dlouhou víc než 5 000 kilometrů zvládne, stane se prvním člověkem, který to dokázal sólo. „Bude to kombinace spousty extrémních věcí, ale nebavilo by mě dělat něco běžného, co už přede mnou někdo zvládl,“ svěřil se v rozhovoru pro magazín Víkend DNES."

        sentence_splitted_1 = sentence_1.split(" ")
        sentence_splitted_2 = sentence_2.split(" ")

        union_of_sentences = set(sentence_splitted_1).union(set(sentence_splitted_2))

        # print("union_of_sentences:")
        # print(union_of_sentences)

        wordDictA = dict.fromkeys(union_of_sentences, 0)
        wordDictB = dict.fromkeys(union_of_sentences, 0)

        for word in sentence_splitted_1:
            wordDictA[word] += 1

        for word in sentence_splitted_2:
            wordDictB[word] += 1

        # print(pd.DataFrame([wordDictA, wordDictB]))

        tf_computed1 = self.computeTF(wordDictA, sentence_splitted_1)
        tf_computed2 = self.computeTF(wordDictB, sentence_splitted_2)

        tfidf_vectorizer = pd.DataFrame([tf_computed1, tf_computed2])
        # print("tfidf_vectorizer:")
        # print(tfidf_vectorizer)

        idfs = self.computeIDF([wordDictA, wordDictB])

        idf_computed_1 = self.computeTFIDF(tf_computed1, idfs)
        idf_computed_2 = self.computeTFIDF(tf_computed2, idfs)

        idf = pd.DataFrame([idf_computed_1, idf_computed_2])

        # print("IDF")
        # print(idf)

        vectorize = TfidfVectorizer()
        vectors = vectorize.fit_transform(sentence_splitted_1)

        names = vectorize.get_feature_names()
        data = vectors.todense().tolist()
        # Compute Sparsicity = Percentage of Non-Zero cells
        print("Sparsicity: ", ((vectors.todense() > 0).sum() / vectors.todense().size) * 100, "%")
        df = pd.DataFrame(data, columns=names)
        # print("SKLEARN:")
        """
        for i in df.iterrows():
            print(i[1].sort_values(ascending=False)[:10])
        """

    def keyword_based_comparison(self, keywords):

        # print("----------------------------------")
        # print("----------------------------------")
        # print("Keyword based comparison")
        # print("----------------------------------")
        # print("----------------------------------")

        keywords_splitted_1 = keywords.split(" ")  # splitting sentence into list of keywords by space

        # creating dictionary of words
        wordDictA = dict.fromkeys(keywords_splitted_1, 0)

        for word in keywords_splitted_1:
            wordDictA[word] += 1

        # keywords_df = pd.DataFrame([wordDictA])

        recommenderMethods = RecommenderMethods()

        recommenderMethods.get_posts_dataframe()
        recommenderMethods.get_categories_dataframe()
        # tfidf.get_ratings_dataframe()
        recommenderMethods.join_posts_ratings_categories()

        # same as "classic" tf-idf
        fit_by_title_matrix = recommenderMethods.get_fit_by_feature('title_x', 'title_y')  # prepended by category
        fit_by_excerpt_matrix = recommenderMethods.get_fit_by_feature('excerpt')
        fit_by_keywords_matrix = recommenderMethods.get_fit_by_feature('keywords')

        tuple_of_fitted_matrices = (fit_by_title_matrix, fit_by_excerpt_matrix, fit_by_keywords_matrix)

        post_recommendations = recommenderMethods.recommend_by_keywords(keywords, tuple_of_fitted_matrices)

        del recommenderMethods
        return post_recommendations

    def get_ratings_dataframe(self):
        self.ratings_df = self.database.get_ratings_dataframe(pd)
        return self.ratings_df

    def get_user_categories(self):
        user_categories_df = self.database.get_user_categories(pd)
        return user_categories_df

    def join_post_ratings_categories(self, dataframe):
        recommenderMethods = RecommenderMethods()
        recommenderMethods.get_categories_dataframe()
        dataframe = dataframe.merge(self.categories_df, left_on='category_id', right_on='id')
        # clean up from unnecessary columns
        dataframe = dataframe[
            ['id_x', 'title_x', 'slug_x', 'excerpt', 'body', 'views', 'keywords', 'title_y', 'description']]
        # print("dataframe afer joining with category")
        # print(dataframe.iloc[0])
        return dataframe.iloc[0]

    """

    def get_most_favourite_categories(self):
        self.user_categories_df = self.get_user_categories()

        self.user_categories_df = self.user_categories_df.merge(self.categories_df, left_on='category_id', right_on='id')
        topic_popularity = (self.user_categories_df.title
                            .value_counts()
                            .sort_values(ascending=False))
        topic_popularity.head(10)
        return topic_popularity

    def get_weighted_average(self):
        self.compute_weighted_average_score(self.df).head(10)

    def compute_weighted_average_score(self, df, k=0.8):
        # n_views = self.df.groupby('id_x', sort=False).id_x.count()
        n_views = self.df['views']
        ratings = self.df.groupby('id_x', sort=False).value.mean()
        scores = ((1-k)*(n_views/n_views.max()) + k*(ratings/ratings.max())).to_numpy().argsort()[::-1]
        df_deduped = df.groupby('id_x', sort=False).agg({
            'title_x':'first',
            'title_y':'first', # category title
        })

        return df_deduped.assign(views=n_views).iloc[scores]

    def computeTF(self, wordDict, doc):
        tfDict = {}
        corpusCount = len(doc)
        for word, count in wordDict.items():
            tfDict[word] = count / float(corpusCount)
        return (tfDict)

    def computeIDF(self, docList):
        idfDict = {}
        N = len(docList)
        idfDict = dict.fromkeys(docList[0].keys(), 0)
        for word, val in idfDict.items():
            if val > 0:
                idfDict[word] += 1

        for word, val in idfDict.items():
            idfDict[word] = math.log10(N / (float(val) + 1));

        return idfDict

    def computeTFIDF(self, tfBow, idfs):
        tfidf = {}
        for word, val in tfBow.items():
            tfidf[word] = val * idfs[word]
        return tfidf

    """

    # https://datascience.stackexchange.com/questions/18581/same-tf-idf-vectorizer-for-2-data-inputs
    def set_tfidf_vectorizer_combine_features(self):
        tfidf_vectorizer = TfidfVectorizer()
        self.df.drop_duplicates(subset=['title_x'], inplace=True)
        tf_train_data = pd.concat([self.df['title_y'], self.df['keywords'], self.df['title_x'], self.df['excerpt']])
        tfidf_vectorizer.fit_transform(tf_train_data)

        tf_idf_title_x = tfidf_vectorizer.transform(self.df['title_x'])
        tf_idf_title_y = tfidf_vectorizer.transform(self.df['title_y'])  # category title
        tf_idf_keywords = tfidf_vectorizer.transform(self.df['keywords'])
        tf_idf_excerpt = tfidf_vectorizer.transform(self.df['excerpt'])

        model = LogisticRegression()
        model.fit([tf_idf_title_x.shape, tf_idf_title_y.shape, tf_idf_keywords.shape, tf_idf_excerpt.shape],
                  self.df['excerpt'])

    def set_cosine_sim(self):
        cosine_sim = cosine_similarity(self.tfidf_tuples)
        cosine_sim_df = pd.DataFrame(cosine_sim, index=self.df['slug_x'], columns=self.df['slug_x'])
        self.cosine_sim_df = cosine_sim_df

    # # @profile

    # @profile
    def get_cleaned_text(self, df, row):
        return row

    def get_recommended_posts_for_keywords(self, keywords, data_frame, items, k=10):

        keywords_list = []
        keywords_list.append(keywords)
        txt_cleaned = self.get_cleaned_text(self.df,
                                            self.df['title_x'] + self.df['title_y'] + self.df['keywords'] + self.df[
                                                'excerpt'])
        tfidf = self.tfidf_vectorizer.fit_transform(txt_cleaned)
        tfidf_keywords_input = self.tfidf_vectorizer.transform(keywords_list)
        cosine_similarities = cosine_similarity(tfidf_keywords_input, tfidf).flatten()
        # cosine_similarities = linear_kernel(tfidf_keywords_input, tfidf).flatten()

        data_frame['coefficient'] = cosine_similarities

        # related_docs_indices = cosine_similarities.argsort()[:-(number+1):-1]
        related_docs_indices = cosine_similarities.argsort()[::-1][:k]

        # print('index:', related_docs_indices)

        # print('similarity:', cosine_similarities[related_docs_indices])

        # print('\n--- related_docs_indices ---\n')

        # print(data_frame.iloc[related_docs_indices])

        # print('\n--- sort_values ---\n')
        # print(data_frame.sort_values('coefficient', ascending=False)[:k])

        closest = data_frame.sort_values('coefficient', ascending=False)[:k]

        closest.reset_index(inplace=True)
        # closest = closest.set_index('index1')
        closest['index1'] = closest.index
        # closest.index.name = 'index1'
        closest.columns.name = 'index'
        # # print("closest.columns.tolist()")
        # # print(closest.columns.tolist())
        # print("index name:")
        # print(closest.index.name)
        # print("""closest["slug_x","coefficient"]""")
        # print(closest[["slug_x","coefficient"]])

        return closest[["slug_x", "coefficient"]]
        # return pd.DataFrame(closest).merge(items).head(k)

    # @profile
    def recommend_posts_by_all_features_preprocessed(self, slug):

        recommenderMethods = RecommenderMethods()

        print("Loading posts.")

        recommenderMethods.get_posts_dataframe()  # load posts to dataframe
        recommenderMethods.get_categories_dataframe()  # load categories to dataframe
        recommenderMethods.join_posts_ratings_categories()  # joining posts and categories into one table

        fit_by_all_features_matrix = recommenderMethods.get_fit_by_feature('all_features_preprocessed')
        fit_by_title = recommenderMethods.get_fit_by_feature('title_y')

        tuple_of_fitted_matrices = (fit_by_all_features_matrix, fit_by_title) # join feature tuples into one matrix

        gc.collect()

        post_recommendations = recommenderMethods.recommend_by_more_features(slug, tuple_of_fitted_matrices)

        del recommenderMethods
        return post_recommendations

    # @profile
    def recommend_posts_by_all_features_preprocessed_with_full_text(self, slug):

        recommenderMethods = RecommenderMethods()
        print("Loading posts")
        recommenderMethods.get_posts_dataframe()  # load posts to dataframe
        gc.collect()
        recommenderMethods.get_categories_dataframe()  # load categories to dataframe
        recommenderMethods.join_posts_ratings_categories()  # joining posts and categories into one table

        # replacing None values with empty strings
        recommenderMethods.df['full_text'] = recommenderMethods.df['full_text'].replace([None], '')

        fit_by_all_features_matrix = recommenderMethods.get_fit_by_feature('all_features_preprocessed')
        fit_by_title = recommenderMethods.get_fit_by_feature('title_y')
        fit_by_full_text = recommenderMethods.get_fit_by_feature('full_text')

        # join feature tuples into one matrix
        tuple_of_fitted_matrices = (fit_by_title, fit_by_all_features_matrix, fit_by_full_text)
        del fit_by_title
        del fit_by_all_features_matrix
        del fit_by_full_text
        gc.collect()

        post_recommendations = recommenderMethods.recommend_by_more_features_with_full_text(slug,
                                                                                            tuple_of_fitted_matrices)
        del recommenderMethods
        return post_recommendations

    # # @profile
    def recommend_posts_by_all_features(self, slug):

        recommenderMethods = RecommenderMethods()

        recommenderMethods.get_posts_dataframe()  # load posts to dataframe
        # print("posts dataframe:")
        # print(recommenderMethods.get_posts_dataframe())
        # print("posts categories:")
        # print(recommenderMethods.get_categories_dataframe())
        recommenderMethods.get_categories_dataframe()  # load categories to dataframe
        # tfidf.get_ratings_dataframe() # load post rating to dataframe

        recommenderMethods.join_posts_ratings_categories()  # joining posts and categories into one table
        print("posts ratings categories dataframe:")
        print(recommenderMethods.join_posts_ratings_categories())

        # preprocessing

        # feature tuples of (document_id, token_id) and coefficient
        fit_by_post_title_matrix = recommenderMethods.get_fit_by_feature('title_x', 'title_y')
        print("fit_by_post_title_matrix")
        print(fit_by_post_title_matrix)
        # fit_by_category_matrix = recommenderMethods.get_fit_by_feature('title_y')
        fit_by_excerpt_matrix = recommenderMethods.get_fit_by_feature('excerpt')
        print("fit_by_excerpt_matrix")
        print(fit_by_excerpt_matrix)
        fit_by_keywords_matrix = recommenderMethods.get_fit_by_feature('keywords')
        print("fit_by_keywords_matrix")
        print(fit_by_keywords_matrix)

        # join feature tuples into one matrix
        tuple_of_fitted_matrices = (fit_by_post_title_matrix, fit_by_excerpt_matrix, fit_by_keywords_matrix)
        post_recommendations = recommenderMethods.recommend_by_more_features(slug, tuple_of_fitted_matrices)

        del recommenderMethods
        return post_recommendations

    def convert_to_json_one_row(self, key, value):
        list_for_json = []
        dict_for_json = {key: value}
        list_for_json.append(dict_for_json)
        # print("------------------------------------")
        # print("JSON:")
        # print("------------------------------------")
        # print(list_for_json)
        return list_for_json

    def convert_to_json_keyword_based(self, post_recommendations):

        list_of_article_slugs = []
        dict = post_recommendations.to_dict('records')
        list_of_article_slugs.append(dict.copy())
        return list_of_article_slugs[0]

    def fill_recommended_for_all_posts(self, skip_already_filled):

        database = Database()
        database.connect()
        all_posts = database.get_all_posts()

        number_of_inserted_rows = 0

        for post in all_posts:

            post_id = post[0]
            slug = post[3]
            current_recommended = post[15]

            if skip_already_filled is True:
                if current_recommended is None:
                    actual_recommended_json = self.recommend_posts_by_all_features(slug)
                    actual_recommended_json = json.dumps(actual_recommended_json)
                    database.insert_recommended_json(articles_recommended_json=actual_recommended_json,
                                                     article_id=post_id)
                    number_of_inserted_rows += 1
                    # print(str(number_of_inserted_rows) + " rows insertd.")
                else:
                    print("Skipping.")
            else:
                actual_recommended_json = self.recommend_posts_by_all_features(slug)
                actual_recommended_json = json.dumps(actual_recommended_json)
                database.insert_recommended_json(articles_recommended_json=actual_recommended_json, article_id=post_id)
                number_of_inserted_rows += 1
                # print(str(number_of_inserted_rows) + " rows insertd.")

    def preprocess_dataframe(self):

        # preprocessing

        self.df["title_y"] = self.df["title_y"].map(lambda s: self.preprocess(s, stemming=False))
        self.df["title_x"] = self.df["title_x"].map(lambda s: self.preprocess(s, stemming=False))
        self.df["excerpt"] = self.df["excerpt"].map(lambda s: self.preprocess(s, stemming=False))
        self.df["keywords"] = self.df["keywords"].map(lambda s: self.preprocess(s, stemming=False))

        print("datframe preprocessing:")
        print(self.df)

    def preprocess(self, sentence, stemming=False):
        # print(sentence)
        sentence = str(sentence)
        sentence = sentence.lower()
        sentence = sentence.replace('{html}', "")
        cleanr = re.compile('<.*?>')
        cleantext = re.sub(cleanr, '', sentence)
        cleantext.translate(str.maketrans('', '', string.punctuation))  # removing punctuation
        rem_url = re.sub(r'http\S+', '', cleantext)
        rem_num = re.sub('[0-9]+', '', rem_url)
        # print("rem_num")
        # print(rem_num)
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(rem_num)
        print("tokens:")
        print(tokens)
        filename = "cz_stemmer/czech_stopwords.txt"

        with open(filename, encoding="utf-8") as file:
            cz_stopwords = file.readlines()
            cz_stopwords = [line.rstrip() for line in cz_stopwords]
        # print(cz_stopwords)

        tokens = [
            [word for word in document.lower().split() if word not in cz_stopwords]
            for document in tokens
        ]

        tokens = [item for sublist in tokens for item in sublist if len(sublist) > 0]

        print("tokens after lemma:")
        print(tokens)

        if stemming is True:
            print("STEMMING...")
            edited_words = [cz_stem(w, True) for w in tokens]  # aggresive
            edited_words = list(filter(None, edited_words))  # empty strings removal
        else:
            print("LEMMA...")
            edited_words = []
            for w in tokens:
                lemmatized_word = self.cz_lemma(str(w))
                if lemmatized_word is w:
                    lemmatized_word = self.cz_lemma(lemmatized_word.capitalize())
                    lemmatized_word = lemmatized_word.lower()
                edited_words.append(lemmatized_word)
            print(edited_words)
            edited_words = list(filter(None, edited_words))  # empty strings removal
            print(edited_words)
        # print(lemma_words)
        print("edited_words:")
        print(edited_words)
        return " ".join(edited_words)

    def preprocess_single_post(self, slug, json=False, stemming=False):
        post_dataframe = self.find_post_by_slug(slug)
        post_dataframe_joined = self.join_post_ratings_categories(post_dataframe)
        print("post_dataframe_joined:")
        print(post_dataframe_joined)
        post_dataframe["title_x"] = self.preprocess(post_dataframe_joined["title_x"], stemming)
        post_dataframe["excerpt"] = self.preprocess(post_dataframe_joined["excerpt"], stemming)
        post_dataframe["title_y"] = post_dataframe_joined["title_y"]
        print("preprocessing single post:")
        print(post_dataframe)
        if json is False:
            return post_dataframe
        else:
            return self.convert_df_to_json(post_dataframe)

    def cz_stem(self, string, aggresive="False", json=False):
        if aggresive == "False":
            aggressive_bool = False
        else:
            aggressive_bool = True

        if json is True:
            return self.convert_to_json_one_row("stem", cz_stem(string, aggressive_bool))
        else:
            return cz_stem(string, aggressive_bool)

    @DeprecationWarning
    def cz_lemma(self, string, json=False):
        morph = majka.Majka('morphological_database/majka.w-lt')

        morph.flags |= majka.ADD_DIACRITICS  # find word forms with diacritics
        morph.flags |= majka.DISALLOW_LOWERCASE  # do not enable to find lowercase variants
        morph.flags |= majka.IGNORE_CASE  # ignore the word case whatsoever
        morph.flags = 0  # unset all flags

        morph.tags = False  # return just the lemma, do not process the tags
        morph.tags = True  # turn tag processing back on (default)

        morph.compact_tag = True  # return tag in compact form (as returned by Majka)
        morph.compact_tag = False  # do not return compact tag (default)

        morph.first_only = True  # return only the first entry
        morph.first_only = False  # return all entries (default)

        morph.tags = False
        morph.first_only = True
        morph.negative = "ne"

        ls = morph.find(string)

        if json is not True:
            if not ls:
                return string
            else:
                # # print(ls[0]['lemma'])
                return str(ls[0]['lemma'])
        else:
            return ls

    def get_document_scores(self):
        search_terms = 'Domácí. Zemřel poslední krkonošský nosič Helmut Hofer, ikona Velké Úpy. Ve věku 88 let zemřel potomek slavného rodu vysokohorských nosičů Helmut Hofer z Velké Úpy. Byl posledním žijícím nosičem v Krkonoších, starodávným řemeslem se po staletí živili generace jeho předků. Jako nosič pracoval pro Českou boudu na Sněžce mezi lety 1948 až 1953.'

        self.df["title_y"] = self.df["title_y"]
        self.df["title_x"] = self.df["title_x"].map(lambda s: self.preprocess(s, stemming=False))
        self.df["excerpt"] = self.df["excerpt"].map(lambda s: self.preprocess(s, stemming=False))
        self.df["keywords"] = self.df["keywords"]

        cols = ["keywords", "title_y", "title_x", "excerpt", "slug_x"]
        documents_df = pd.DataFrame()
        documents_df['features_combined'] = self.df[cols].apply(lambda row: '. '.join(row.values.astype(str)), axis=1)
        documents = list(map(' '.join, documents_df[['features_combined']].values.tolist()))

        filename = "cz_stemmer/czech_stopwords.txt"
        with open(filename, encoding="utf-8") as file:
            cz_stopwords = file.readlines()
            cz_stopwords = [line.rstrip() for line in cz_stopwords]
        # print(cz_stopwords)

        doc_vectors = TfidfVectorizer(dtype=np.float32, stop_words=cz_stopwords).fit_transform(
            [search_terms] + documents)

        cosine_similarities = linear_kernel(doc_vectors[0:1], doc_vectors).flatten()
        document_scores = [item.item() for item in cosine_similarities[1:]]

        return document_scores