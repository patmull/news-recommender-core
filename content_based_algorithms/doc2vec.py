import csv
import gc
import logging
import os
import pickle
import random
import warnings

import gensim
import pandas as pd
import smart_open
import tqdm
from gensim import corpora
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from pymongo import MongoClient
from sklearn.model_selection import train_test_split

from content_based_algorithms.tfidf import TfIdf
from data_handling.data_queries import RecommenderMethods
from preprocessing.cz_preprocessing import CzPreprocess
from data_connection import Database

DEFAULT_MODEL_LOCATION = "models/d2v_limited.model"


class Doc2VecClass:

    def __init__(self):
        self.documents = None
        self.df = None
        self.posts_df = None
        self.categories_df = None
        self.database = Database()
        self.doc2vec_model = None

    def prepare_posts_df(self):
        # self.database.insert_post_dataframe_to_cache() # uncomment for UPDATE of DB records
        self.posts_df = self.database.get_posts_dataframe_from_cache()
        self.posts_df.drop_duplicates(subset=['title'], inplace=True)
        self.posts_df = self.posts_df.rename({'title': 'post_title'})
        return self.posts_df

    def prepare_categories_df(self):
        self.categories_df = self.database.get_categories_dataframe(pd)
        self.posts_df = self.posts_df.rename({'title': 'category_title'})
        return self.categories_df

    """
    def join_posts_ratings_categories(self, include_prefilled=False):
        self.posts_df = self.prepare_posts_df()
        self.categories_df = self.prepare_categories_df()
        print("self.posts_df:")
        print(self.posts_df)
        print("self.categories_df:")
        print(self.categories_df)
        self.posts_df = self.posts_df.rename(columns={'title':'post_title'})
        self.posts_df = self.posts_df.rename(columns={'slug':'post_slug'})
        self.categories_df = self.categories_df.rename(columns={'title':'category_title'})
        self.categories_df = self.categories_df.rename(columns={'slug':'category_slug'})

        if include_prefilled is False:
            self.df = self.posts_df.merge(self.categories_df, left_on='category_id', right_on='id')
            # clean up from unnecessary columns
            self.df = self.df.rename(columns={'post_slug': 'slug'})
            self.df = self.df[
                ['id_x', 'post_title', 'slug', 'excerpt', 'body', 'views', 'keywords', 'category_title', 'description',
                 'all_features_preprocessed', 'body_preprocessed']]
        else:
            self.df = self.posts_df.merge(self.categories_df, left_on='category_id', right_on='id')
            # clean up from unnecessary columns
            self.df = self.df.rename(columns={'post_slug': 'slug'})
            try:
                self.df = self.df[
                    ['id_x', 'post_title', 'slug', 'excerpt', 'body', 'views', 'keywords', 'category_title', 'description',
                     'all_features_preprocessed', 'body_preprocessed',
                     'recommended_tfidf_full_text']]
            except KeyError:
                self.df = self.database.insert_posts_dataframe_to_cache()
                self.posts_df.drop_duplicates(subset=['title'], inplace=True)
                print(self.df.columns.values)
                self.df = self.df[
                    ['id_x', 'post_title', 'slug', 'excerpt', 'body', 'views', 'keywords', 'category_title', 'description',
                     'all_features_preprocessed', 'body_preprocessed',
                     'recommended_tfidf_full_text']]

        return self.df
    """

    def train_doc2vec(self, documents_all_features_preprocessed, body_text, limited=True, create_csv=False):
        print("documents_all_features_preprocessed")
        print(documents_all_features_preprocessed)

        filename = "preprocessing/stopwords/czech_stopwords.txt"
        with open(filename, encoding="utf-8") as file:
            cz_stopwords = file.readlines()
            cz_stopwords = [line.rstrip() for line in cz_stopwords]

        filename = "preprocessing/stopwords/general_stopwords.txt"
        with open(filename, encoding="utf-8") as file:
            general_stopwords = file.readlines()
            general_stopwords = [line.rstrip() for line in general_stopwords]
        all_stopwords = cz_stopwords + general_stopwords

        tagged_data = []
        cz_lemma = CzPreprocess()
        for i, doc in enumerate(documents_all_features_preprocessed):
            selected_list = []
            for word in doc.split(", "):
                # if not word in all_stopwords:
                words_preprocessed = gensim.utils.simple_preprocess(cz_lemma.preprocess(word))
                for sublist in words_preprocessed:
                    if len(sublist) > 0:
                        selected_list.append(sublist)

            print("Preprocessing doc. num. " + str(i))
            print("Selected list:")
            print(selected_list)
            tagged_data.append(TaggedDocument(words=selected_list, tags=[str(i)]))
            if create_csv is True:
                # Will append to exising file! CSV needs to be removed first if needs to be up updated as a whole
                with open("datasets/idnes_preprocessed.txt", "a+", encoding="utf-8") as fp:
                    wr = csv.writer(fp, dialect='excel')
                    wr.writerow(selected_list)

        print("tagged_data:")
        print(tagged_data)

        self.train(tagged_data)

    def get_similar_doc2vec(self, slug, source=None, doc2vec_model=None, train=False, limited=True, number_of_recommended_posts=21, from_db=False):
        recommender_methods = RecommenderMethods()
        recommender_methods.database.connect()
        self.df = recommender_methods.get_posts_categories_dataframe()

        recommender_methods.database.disconnect()
        print("self.df")
        print(self.df.columns.values)

        cols = ['keywords', 'all_features_preprocessed']
        documents_df = pd.DataFrame()
        self.df['all_features_preprocessed'] = self.df['all_features_preprocessed'].apply(
            lambda x: x.replace(' ', ', '))
        documents_df['all_features_preprocessed'] = self.df[cols].apply(lambda row: ' '.join(row.values.astype(str)),
                                                                        axis=1)
        documents_df['all_features_preprocessed'] = self.df['category_title'] + ', ' + documents_df[
            'all_features_preprocessed']

        documents_all_features_preprocessed = list(
            map(' '.join, documents_df[['all_features_preprocessed']].values.tolist()))

        del documents_df
        gc.collect()

        if 'post_slug' in self.df:
            self.df = self.df.rename(columns={'post_slug':'slug'})
        documents_slugs = self.df['slug'].tolist()

        if train is True:
            self.train_doc2vec(documents_all_features_preprocessed, body_text=False, limited=False, create_csv=False)

        del documents_all_features_preprocessed
        gc.collect()

        if limited is True:
            doc2vec_model = Doc2Vec.load("models/d2v_limited.model")
        else:
            doc2vec_model = Doc2Vec.load("models/d2v.model") # or download from Dropbox / AWS bucket

        recommender_methods = RecommenderMethods()
        # not necessary
        post_found = recommender_methods.find_post_by_slug(slug)
        keywords_preprocessed = post_found.iloc[0]['keywords'].split(", ")
        all_features_preprocessed = post_found.iloc[0]['all_features_preprocessed'].split(" ")

        tokens = keywords_preprocessed + all_features_preprocessed

        vector_source = doc2vec_model.infer_vector(tokens)
        most_similar = doc2vec_model.dv.most_similar([vector_source], topn=number_of_recommended_posts)

        return self.get_similar_by_posts_slug(most_similar, documents_slugs, number_of_recommended_posts)

    def get_similar_doc2vec_with_full_text(self, slug, train=False, number_of_recommended_posts=21):
        recommender_methods = RecommenderMethods()
        recommender_methods.database.connect()
        self.df = recommender_methods.get_posts_categories_dataframe()
        recommender_methods.database.disconnect()

        cols = ['keywords', 'all_features_preprocessed', 'body_preprocessed']
        documents_df = pd.DataFrame()
        self.df['all_features_preprocessed'] = self.df['all_features_preprocessed'].apply(
            lambda x: x.replace(' ', ', '))

        self.df.fillna("", inplace=True)

        self.df['body_preprocessed'] = self.df['body_preprocessed'].apply(
            lambda x: x.replace(' ', ', '))
        documents_df['all_features_preprocessed'] = self.df[cols].apply(lambda row: ' '.join(row.values.astype(str)),
                                                                        axis=1)

        documents_df['all_features_preprocessed'] = self.df['category_title'] + ', ' + documents_df[
            'all_features_preprocessed'] + ", " + self.df['body_preprocessed']

        documents_all_features_preprocessed = list(
            map(' '.join, documents_df[['all_features_preprocessed']].values.tolist()))

        del documents_df
        gc.collect()

        print("self.df.columns")
        print(self.df.columns)

        documents_slugs = self.df['slug'].tolist()

        if train is True:
            self.train_doc2vec(documents_all_features_preprocessed, body_text=True)
        del documents_all_features_preprocessed
        gc.collect()

        doc2vec_model = Doc2Vec.load("models/d2v_full_text_limited.model_variant")

        recommendMethods = RecommenderMethods()

        # not necessary
        post_found = recommendMethods.find_post_by_slug(slug)
        keywords_preprocessed = post_found.iloc[0]['keywords'].split(", ")
        all_features_preprocessed = post_found.iloc[0]['all_features_preprocessed'].split(" ")
        full_text = post_found.iloc[0]['body_preprocessed'].split(" ")
        tokens = keywords_preprocessed + all_features_preprocessed + full_text
        vector_source = doc2vec_model.infer_vector(tokens)

        most_similar = doc2vec_model.dv.most_similar([vector_source], topn=number_of_recommended_posts)

        return self.get_similar_by_posts_slug(most_similar, documents_slugs, number_of_recommended_posts)

    def get_similar_doc2vec_by_keywords(self, slug, number_of_recommended_posts=21):
        warnings.warn("Method not tested properly yet...", PendingDeprecationWarning)

        recommenderMethods = RecommenderMethods()
        recommenderMethods.get_posts_dataframe()
        recommenderMethods.get_categories_dataframe()
        self.df = recommenderMethods.join_posts_ratings_categories()

        cols = ["keywords"]
        documents_df = pd.DataFrame()
        documents_df['keywords'] = self.df[cols].apply(lambda row: '. '.join(row.values.astype(str)), axis=1)
        documents_slugs = self.df['slug'].tolist()

        filename = "preprocessing/czech_stopwords.txt"
        with open(filename, encoding="utf-8") as file:
            cz_stopwords = file.readlines()
            cz_stopwords = [line.rstrip() for line in cz_stopwords]

        # to find the vector of a document which is not in training data
        tfidf = TfIdf()
        # not necessary
        post_preprocessed = tfidf.preprocess_single_post(slug)
        post_features_to_find = post_preprocessed.iloc[0]['keywords']

        tokens = post_features_to_find.split()

        global doc2vec_model

        doc2vec_model = Doc2Vec.load("models/d2v.models")
        vector = doc2vec_model.infer_vector(tokens)

        most_similar = doc2vec_model.docvecs.most_similar([vector], topn=number_of_recommended_posts)
        return self.get_similar_by_posts_slug(most_similar, documents_slugs, number_of_recommended_posts)

    def get_similar_by_posts_slug(self, most_similar, documents_slugs, number_of_recommended_posts):
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

    def load_model(self, path_to_model=None):
        print("Loading Doc2Vec vectors...")
        if path_to_model is None:
            self.doc2vec_model = Doc2Vec.load(DEFAULT_MODEL_LOCATION)
        else:
            self.doc2vec_model = Doc2Vec.load(path_to_model)

    def get_vector_representation(self, slug):
        """
        For Learn-to-Rank
        """
        recommender_methods = RecommenderMethods()
        return self.doc2vec_model.infer_vector(recommender_methods.find_post_by_slug(slug))

    def train(self, tagged_data):

        max_epochs = 20
        vec_size = 8
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

        model.save("models/d2v_mini_vectors.model")
        print("Doc2Vec model Saved")

    def train_full_text(self, tagged_data, full_body, limited):

        max_epochs = 20
        vec_size = 150

        if limited is True:
            model = Doc2Vec(vector_size=vec_size,
                            min_count=1,
                            dm=0, max_vocab_size=87000)
        else:
            model = Doc2Vec(vector_size=vec_size,
                            min_count=1,
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
        print("Doc2Vec model Saved")

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

    def create_tagged_document(self, list_of_list_of_words):
        for i, list_of_words in enumerate(list_of_list_of_words):
            yield gensim.models.doc2vec.TaggedDocument(list_of_words, [i])

    def find_best_doc2vec_model(self, source, number_of_trials=512, path_to_corpus="precalc_vectors/corpus_idnes.mm"):

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        # Enabling Word2Vec logging
        logging.basicConfig(format="%(asctime)s:%(levelname)s:%(message)s",
                            level=logging.NOTSET)
        logger = logging.getLogger()  # get the root logger
        logger.info("Testing file write")

        print("Building sentences...")
        # TODO: Replace with iterator when is fixed: sentences = MyCorpus(dictionary)
        sentences = []
        client = MongoClient("localhost", 27017, maxPoolSize=50)
        global db
        if source == "idnes":
            db = client.idnes
        elif source == "cswiki":
            db = client.cswiki
        else:
            ValueError("No source selected from available options.")
        collection = db.preprocessed_articles_trigrams

        cursor = collection.find({})
        for document in cursor:
            # joined_string = ' '.join(document['text'])
            # sentences.append([joined_string])
            sentences.append(document['text'])
        # TODO:
        train_corpus, test_corpus = train_test_split(sentences, test_size=0.2, shuffle=True)
        train_corpus = list(self.create_tagged_document(sentences))

        print("print(train_corpus[:2])")
        print(print(train_corpus[:2]))
        print("print(test_corpus[:2])")
        print(print(test_corpus[:2]))

        model_variants = [0, 1]  # sg parameter: 0 = CBOW; 1 = Skip-Gram
        hs_softmax_variants = [0, 1]  # 1 = Hierarchical SoftMax
        negative_sampling_variants = range(5, 20, 5)  # 0 = no negative sampling
        no_negative_sampling = 0  # use with hs_soft_max
        # vector_size_range = [50, 100, 158, 200, 250, 300, 450]
        vector_size_range = [50, 100, 158, 200, 250, 300, 450]
        # window_range = [1, 2, 4, 5, 8, 12, 16, 20]
        window_range = [1, 2, 4, 5, 8, 12, 16, 20]
        min_count_range = [0, 1, 2, 3, 5, 8, 12]
        epochs_range = [20, 25, 30]
        sample_range = [0.0, 1.0 * (10.0 ** -1.0), 1.0 * (10.0 ** -2.0), 1.0 * (10.0 ** -3.0), 1.0 * (10.0 ** -4.0),
                        1.0 * (10.0 ** -5.0)]
        corpus_title = ['100% Corpus']
        model_results = {'Validation_Set': [],
                         'Model_Variant': [],
                         'Negative': [],
                         'Vector_size': [],
                         'Window': [],
                         'Min_count': [],
                         'Epochs': [],
                         'Sample': [],
                         'Softmax': [],
                         'Word_pairs_test_Pearson_coeff': [],
                         'Word_pairs_test_Pearson_p-val': [],
                         'Word_pairs_test_Spearman_coeff': [],
                         'Word_pairs_test_Spearman_p-val': [],
                         'Word_pairs_test_Out-of-vocab_ratio': [],
                         'Analogies_test': []
                         }
        pbar = tqdm.tqdm(total=540)

        for i in range(0, number_of_trials):

            hs_softmax = 1
            # hs_softmax = random.choice(hs_softmax_variants)
            # TODO: Get Back random choice!!!
            # This is temporary due to adding softmax one values to results after fixed tab indent in else.
            model_variant = random.choice(model_variants)
            vector_size = random.choice(vector_size_range)
            window = random.choice(window_range)
            min_count = random.choice(min_count_range)
            epochs = random.choice(epochs_range)
            sample = random.choice(sample_range)
            negative_sampling_variant = random.choice(negative_sampling_variants)

            if hs_softmax == 1:
                word_pairs_eval, analogies_eval = self.compute_eval_values(train_corpus=train_corpus,
                                                                           test_corpus=test_corpus,
                                                                           model_variant=model_variant,
                                                                           negative_sampling_variant=no_negative_sampling,
                                                                           vector_size=vector_size,
                                                                           window=window,
                                                                           min_count=min_count,
                                                                           epochs=epochs,
                                                                           sample=sample,
                                                                           force_update_model=True, source=source)
            else:
                word_pairs_eval, analogies_eval = self.compute_eval_values(train_corpus=train_corpus,
                                                                           test_corpus=test_corpus,
                                                                           model_variant=model_variant,
                                                                           negative_sampling_variant=no_negative_sampling,
                                                                           vector_size=vector_size,
                                                                           window=window,
                                                                           min_count=min_count,
                                                                           epochs=epochs,
                                                                           sample=sample,
                                                                           force_update_model=True,
                                                                           source=source)

            print(word_pairs_eval[0][0])
            if source == "idnes":
                model_results['Validation_Set'].append("iDnes.cz " + corpus_title[0])
            elif source == "cswiki":
                model_results['Validation_Set'].append("cs.Wikipedia.org " + corpus_title[0])
            else:
                ValueError("No source from available options selected")

            model_results['Model_Variant'].append(model_variant)
            model_results['Negative'].append(negative_sampling_variant)
            model_results['Vector_size'].append(vector_size)
            model_results['Window'].append(window)
            model_results['Min_count'].append(min_count)
            model_results['Epochs'].append(epochs)
            model_results['Sample'].append(sample)
            model_results['Softmax'].append(hs_softmax)
            model_results['Word_pairs_test_Pearson_coeff'].append(word_pairs_eval[0][0])
            model_results['Word_pairs_test_Pearson_p-val'].append(word_pairs_eval[0][1])
            model_results['Word_pairs_test_Spearman_coeff'].append(word_pairs_eval[1][0])
            model_results['Word_pairs_test_Spearman_p-val'].append(word_pairs_eval[1][1])
            model_results['Word_pairs_test_Out-of-vocab_ratio'].append(word_pairs_eval[2])
            model_results['Analogies_test'].append(analogies_eval)

            pbar.update(1)
            if source == "idnes":
                saved_file_name = 'doc2vec_tuning_results_random_search_idnes.csv'
            elif source == "cswiki":
                saved_file_name = 'doc2vec_tuning_results_random_search_cswiki.csv'
            else:
                ValueError("Source does not matech available options.")

            pd.DataFrame(model_results).to_csv(saved_file_name, index=False,
                                               mode="w")

            print("Saved training results...")

    def create_or_update_corpus_and_dict_from_mongo_idnes(self):
        dict = self.create_dictionary_from_mongo_idnes(force_update=True)
        self.create_corpus_from_mongo_idnes(dict, force_update=True)

    def compute_eval_values(self, source, train_corpus=None, test_corpus=None, model_variant=None,
                            negative_sampling_variant=None,
                            vector_size=None, window=None, min_count=None,
                            epochs=None, sample=None, force_update_model=True,
                            default_parameters=False):
        global model_path
        if source == "idnes":
            model_path = "models/d2v_idnes.model"
        elif source == "cswiki":
            model_path = "models/d2v_cswiki.model"
        else:
            ValueError("No sourc matches available options.")

        if os.path.isfile(model_path) is False or force_update_model is True:
            print("Started training on iDNES.cz dataset...")

            if default_parameters is True:
                # DEFAULT:
                d2v_model = Doc2Vec()
            else:
                # CUSTOM:
                d2v_model = Doc2Vec(dm=model_variant, negative=negative_sampling_variant,
                                    vector_size=vector_size, window=window, min_count=min_count, epochs=epochs,
                                    sample=sample, workers=7)

            print("Sample of train corpus:")
            print(train_corpus[:2])
            d2v_model.build_vocab(train_corpus)
            d2v_model.train(train_corpus, total_examples=d2v_model.corpus_count, epochs=d2v_model.epochs)
            d2v_model.save(model_path)

        else:
            print("Loading Doc2Vec iDNES.cz doc2vec_model from saved doc2vec_model file")
            d2v_model = Doc2Vec.load(model_path)

        path_to_cropped_wordsim_file = 'research/word2vec/similarities/WordSim353-cs-cropped.tsv'
        if os.path.exists(path_to_cropped_wordsim_file):
            word_pairs_eval = d2v_model.wv.evaluate_word_pairs(
                path_to_cropped_wordsim_file)
        else:
            df = pd.read_csv('research/word2vec/similarities/WordSim353-cs.csv',
                             usecols=['cs_word_1', 'cs_word_2', 'cs mean'])
            cz_preprocess = CzPreprocess()
            df['cs_word_1'] = df['cs_word_1'].apply(lambda x: gensim.utils.deaccent(cz_preprocess.preprocess(x)))
            df['cs_word_2'] = df['cs_word_2'].apply(lambda x: gensim.utils.deaccent(cz_preprocess.preprocess(x)))

            df.to_csv(path_to_cropped_wordsim_file, sep='\t', encoding='utf-8', index=False)
            word_pairs_eval = d2v_model.wv.evaluate_word_pairs(path_to_cropped_wordsim_file)

        overall_score, _ = d2v_model.wv.evaluate_word_analogies('research/word2vec/analogies/questions-words-cs.txt')
        print("Analogies evaluation of doc2vec_model:")
        print(overall_score)

        doc_id = random.randint(0, len(test_corpus) - 1)
        print("print(test_corpus[:2])")
        print(print(train_corpus[:2]))
        print("print(test_corpus[:2])")
        print(print(test_corpus[:2]))
        inferred_vector = d2v_model.infer_vector(test_corpus[doc_id])
        sims = d2v_model.dv.most_similar([inferred_vector], topn=len(d2v_model.dv))
        # Compare and print the most/median/least similar documents from the train train_corpus
        print('Test Document ({}): «{}»\n'.format(doc_id, ' '.join(test_corpus[doc_id])))
        print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % d2v_model)
        for label, index in [('MOST', 0), ('MEDIAN', len(sims) // 2), ('LEAST', len(sims) - 1)]:
            print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))

        return word_pairs_eval, overall_score

    def create_dictionary_from_mongo_idnes(self, sentences=None, force_update=False, filter_extremes=False):
        # a memory-friendly iterator
        path_to_train_dict = 'precalc_vectors/dictionary_train_idnes.gensim'
        if os.path.isfile(path_to_train_dict) is False or force_update is True:
            if sentences is None:
                print("Building sentences...")
                sentences = []
                client = MongoClient("localhost", 27017, maxPoolSize=50)
                db = client.idnes
                collection = db.preprocessed_articles_bigrams
                cursor = collection.find({})
                for document in cursor:
                    # joined_string = ' '.join(document['text'])
                    # sentences.append([joined_string])
                    sentences.append(document['text'])
            sentences_train, sentences_test = train_test_split(train_size=0.2, shuffle=True)
            print("Creating dictionary...")
            preprocessed_dictionary_train = gensim.corpora.Dictionary(line for line in sentences_train)
            del sentences
            gc.collect()
            if filter_extremes is True:
                preprocessed_dictionary_train.filter_extremes()
            print("Saving dictionary...")
            preprocessed_dictionary_train.save(path_to_train_dict)
            print("Dictionary saved to: " + path_to_train_dict)
            return preprocessed_dictionary_train
        else:
            print("Dictionary already exists. Loading...")
            loaded_dict = corpora.Dictionary.load(path_to_train_dict)
            return loaded_dict

    def create_corpus_from_mongo_idnes(self, dict, force_update):
        pass

    def train_final_doc2vec_model(self, source):

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        # Enabling Word2Vec logging
        logging.basicConfig(format="%(asctime)s:%(levelname)s:%(message)s",
                            level=logging.NOTSET)
        logger = logging.getLogger()  # get the root logger
        logger.info("Testing file write")

        print("Building sentences...")
        # TODO: Replace with iterator when is fixed: sentences = MyCorpus(dictionary)
        sentences = []
        client = MongoClient("localhost", 27017, maxPoolSize=50)
        global db
        if source == "idnes":
            db = client.idnes
        elif source == "cswiki":
            db = client.cswiki
        else:
            ValueError("No source matches the selected options.")
        collection = db.preprocessed_articles_trigrams
        cursor = collection.find({})
        for document in cursor:
            sentences.append(document['text'])
        # TODO:
        train_corpus, test_corpus = train_test_split(sentences, test_size=0.2, shuffle=True)
        train_corpus = list(self.create_tagged_document(sentences))

        print("print(train_corpus[:2])")
        print(print(train_corpus[:2]))
        print("print(test_corpus[:2])")
        print(print(test_corpus[:2]))
        # corpus = list(self.read_corpus(path_to_corpus))

        model_variant = 1  # sg parameter: 0 = CBOW; 1 = Skip-Gram
        hs_softmax_variants = [0, 1]  # 1 = Hierarchical SoftMax
        negative_sampling_variant = 10  # 0 = no negative sampling
        no_negative_sampling = 0  # use with hs_soft_max
        # vector_size_range = [50, 100, 158, 200, 250, 300, 450]
        vector_size = 200
        # window_range = [1, 2, 4, 5, 8, 12, 16, 20]
        window = 16
        min_count = 3
        epoch = 25
        sample = 0.0
        hs_softmax = 0

        corpus_title = ['100% Corpus']
        model_results = {'Validation_Set': [],
                         'Model_Variant': [],
                         'Negative': [],
                         'Vector_size': [],
                         'Window': [],
                         'Min_count': [],
                         'Epochs': [],
                         'Sample': [],
                         'Softmax': [],
                         'Word_pairs_test_Pearson_coeff': [],
                         'Word_pairs_test_Pearson_p-val': [],
                         'Word_pairs_test_Spearman_coeff': [],
                         'Word_pairs_test_Spearman_p-val': [],
                         'Word_pairs_test_Out-of-vocab_ratio': [],
                         'Analogies_test': []
                         }
        pbar = tqdm.tqdm(total=540)

        # hs_softmax = random.choice(hs_softmax_variants)
        # TODO: Get Back random choice!!!
        # This is temporary due to adding softmax one values to results after fixed tab indent in else.
        model_variant = model_variant
        vector_size = vector_size
        window = window
        min_count = min_count
        epochs = epoch
        sample = sample
        negative_sampling_variant = negative_sampling_variant

        if hs_softmax == 1:
            word_pairs_eval, analogies_eval = self.compute_eval_values(train_corpus=train_corpus,
                                                                       test_corpus=test_corpus,
                                                                       model_variant=model_variant,
                                                                       negative_sampling_variant=no_negative_sampling,
                                                                       vector_size=vector_size,
                                                                       window=window,
                                                                       min_count=min_count,
                                                                       epochs=epochs,
                                                                       sample=sample,
                                                                       force_update_model=True,
                                                                       source=source)
        else:
            word_pairs_eval, analogies_eval = self.compute_eval_values(train_corpus=train_corpus,
                                                                       test_corpus=test_corpus,
                                                                       model_variant=model_variant,
                                                                       negative_sampling_variant=negative_sampling_variant,
                                                                       vector_size=vector_size,
                                                                       window=window,
                                                                       min_count=min_count,
                                                                       epochs=epochs,
                                                                       sample=sample,
                                                                       source=source)

        print(word_pairs_eval[0][0])
        model_results['Validation_Set'].append(source + " " + corpus_title[0])
        model_results['Model_Variant'].append(model_variant)
        model_results['Negative'].append(negative_sampling_variant)
        model_results['Vector_size'].append(vector_size)
        model_results['Window'].append(window)
        model_results['Min_count'].append(min_count)
        model_results['Epochs'].append(epochs)
        model_results['Sample'].append(sample)
        model_results['Softmax'].append(hs_softmax)
        model_results['Word_pairs_test_Pearson_coeff'].append(word_pairs_eval[0][0])
        model_results['Word_pairs_test_Pearson_p-val'].append(word_pairs_eval[0][1])
        model_results['Word_pairs_test_Spearman_coeff'].append(word_pairs_eval[1][0])
        model_results['Word_pairs_test_Spearman_p-val'].append(word_pairs_eval[1][1])
        model_results['Word_pairs_test_Out-of-vocab_ratio'].append(word_pairs_eval[2])
        model_results['Analogies_test'].append(analogies_eval)

        pbar.update(1)
        final_results_file_name = 'doc2vec_tuning_results_random_search_final_' + source + '.csv'
        pd.DataFrame(model_results).to_csv(final_results_file_name, index=False,
                                           mode="w")
        print("Saved training results...")
