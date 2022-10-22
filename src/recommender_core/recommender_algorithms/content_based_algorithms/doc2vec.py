import csv
import gc
import logging
import os
import random

import gensim
import pandas as pd
import tqdm
from gensim import corpora
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from scipy import spatial
from sklearn.model_selection import train_test_split

from src.recommender_core.recommender_algorithms.content_based_algorithms.models_manipulation.models_loaders import \
    load_doc2vec_model
from src.recommender_core.data_handling.data_handlers import flatten
from src.recommender_core.recommender_algorithms.content_based_algorithms.helper import verify_searched_slug_sanity, \
    preprocess_columns
from src.recommender_core.checks.data_types import check_empty_string, accepts_first_argument
from src.recommender_core.data_handling.reader import build_sentences
from src.recommender_core.data_handling.data_queries import RecommenderMethods, save_wordsim, append_training_results, \
    get_eval_results_header, prepare_hyperparameters_grid, random_hyperparameter_choice
from src.prefillers.preprocessing.cz_preprocessing import preprocess
from src.recommender_core.data_handling.data_manipulation import DatabaseMethods

DEFAULT_MODEL_LOCATION = "models/d2v_limited.model"


def create_tagged_document(list_of_list_of_words):
    for i, list_of_words in enumerate(list_of_list_of_words):
        yield gensim.models.doc2vec.TaggedDocument(list_of_words, [i])


def compute_eval_values(source, train_corpus=None, test_corpus=None, model_variant=None,
                        negative_sampling_variant=None,
                        vector_size=None, window=None, min_count=None,
                        epochs=None, sample=None, force_update_model=True,
                        default_parameters=False):
    if source == "idnes":
        model_path = "models/d2v_idnes.model"
    elif source == "cswiki":
        model_path = "models/d2v_cswiki.model"
    else:
        raise ValueError("No source matches available options.")

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

        print("Sample of train_enabled corpus:")
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
        save_wordsim(path_to_cropped_wordsim_file)
        word_pairs_eval = d2v_model.wv.evaluate_word_pairs(path_to_cropped_wordsim_file)

    overall_score, _ = d2v_model.wv.evaluate_word_analogies('research/word2vec/analogies/questions-words-cs.txt')
    print("Analogies evaluation of doc2vec_model:")
    print(overall_score)

    doc_id = random.randint(0, len(test_corpus) - 1)
    logging.debug("print(test_corpus[:2])")
    logging.debug(train_corpus[:2])
    logging.debug("print(test_corpus[:2])")
    logging.debug(test_corpus[:2])
    inferred_vector = d2v_model.infer_vector(test_corpus[doc_id])
    sims = d2v_model.dv.most_similar([inferred_vector], topn=len(d2v_model.dv))
    # Compare and print the most/median/least similar documents from the train_enabled train_corpus
    print('Test Document ({}): «{}»\n'.format(doc_id, ' '.join(test_corpus[doc_id])))
    print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % d2v_model)
    for label, index in [('MOST', 0), ('MEDIAN', len(sims) // 2), ('LEAST', len(sims) - 1)]:
        print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))

    return word_pairs_eval, overall_score


def create_dictionary_from_mongo_idnes(sentences=None, force_update=False, filter_extremes=False):
    # a memory-friendly iterator
    path_to_train_dict = 'precalc_vectors/dictionary_train_idnes.gensim'
    if os.path.isfile(path_to_train_dict) is False or force_update is True:
        if sentences is None:
            sentences, db = build_sentences()

        sentences_train, sentences_test = train_test_split(sentences, train_size=0.2, shuffle=True)
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


def init_and_start_training(model, tagged_data, max_epochs):
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
    return model


def most_similar(model, search_term):
    inferred_vector = model.infer_vector(search_term)
    sims = model.docvecs.most_similar([inferred_vector], topn=20)

    res = []
    for elem in sims:
        inner = {'index': elem[0], 'distance': elem[1]}
        res.append(inner)

    return res[:20]


def prepare_train_test_corpus():
    sentences, db = build_sentences()
    collection = db.preprocessed_articles_trigrams

    cursor = collection.find({})
    for document in cursor:
        # joined_string = ' '.join(document['text'])
        # sentences.append([joined_string])
        sentences.append(document['text'])
    # TODO:
    train_corpus, test_corpus = train_test_split(sentences, test_size=0.2, shuffle=True)
    train_corpus = list(create_tagged_document(sentences))

    print("print(train_corpus[:2])")
    print(print(train_corpus[:2]))
    print("print(test_corpus[:2])")
    print(print(test_corpus[:2]))
    return train_corpus, test_corpus


def train(tagged_data):
    max_epochs = 20
    vec_size = 8
    alpha = 0.025
    """
    minimum_alpha = 0.0025
    reduce_alpha = 0.0002
    """

    model = Doc2Vec(vector_size=vec_size,
                    alpha=alpha,
                    min_count=0,
                    dm=0)

    model = init_and_start_training(model, tagged_data, max_epochs)

    model.save("models/d2v_mini_vectors.model")
    print("Doc2Vec model Saved")


def train_full_text(tagged_data, full_body, limited):
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

    model = init_and_start_training(model, tagged_data, max_epochs)

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


def find_best_doc2vec_model(source, number_of_trials=512, file_name='doc2vec_final_evaluation_results'):
    print("Building sentences...")
    # TODO: Replace with iterator when is fixed: sentences = MyCorpus(dictionary)

    train_corpus, test_corpus = prepare_train_test_corpus()

    model_variants = [0, 1]  # sg parameter: 0 = CBOW; 1 = Skip-Gram
    """
    hs_softmax_variants = [0, 1]  # 1 = Hierarchical SoftMax
    """
    # noinspection PyPep8
    negative_sampling_variants, no_negative_sampling, vector_size_range, window_range, min_count_range, epochs_range, \
    sample_range, corpus_title, model_results = prepare_hyperparameters_grid()

    model_variant, vector_size, window, min_count, epochs, sample, negative_sampling_variant \
        = random_hyperparameter_choice(model_variants=model_variants,
                                       negative_sampling_variants=negative_sampling_variants,
                                       vector_size_range=vector_size_range,
                                       sample_range=sample_range,
                                       epochs_range=epochs_range, window_range=window_range,
                                       min_count_range=min_count_range)

    pbar = tqdm.tqdm(total=540)

    for i in range(0, number_of_trials):

        hs_softmax = 1
        # hs_softmax = random_order.choice(hs_softmax_variants)
        # TODO: Get Back random_order choice!!!
        # This is temporary due to adding softmax one values to results after fixed tab indent in else.

        if hs_softmax == 1:
            word_pairs_eval, analogies_eval = compute_eval_values(train_corpus=train_corpus,
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
            word_pairs_eval, analogies_eval = compute_eval_values(train_corpus=train_corpus,
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

        # noinspection DuplicatedCode
        append_training_results(source=source, corpus_title=corpus_title[0],
                                model_variant=model_variant,
                                negative_sampling_variant=negative_sampling_variant,
                                vector_size=vector_size,
                                window=window, min_count=min_count, epochs=epochs,
                                sample=sample,
                                hs_softmax=hs_softmax,
                                pearson_coeff_word_pairs_eval=word_pairs_eval[0][0],
                                pearson_p_val_word_pairs_eval=word_pairs_eval[0][1],
                                spearman_p_val_word_pairs_eval=word_pairs_eval[1][1],
                                spearman_coeff_word_pairs_eval=word_pairs_eval[1][0],
                                out_of_vocab_ratio=word_pairs_eval[2],
                                analogies_eval=analogies_eval,
                                model_results=model_results)
        pd.DataFrame(model_results).to_csv(file_name + source + '.csv', index=False,
                                           mode="a")
        pbar.update(1)

        print("Saved training results...")


def train_final_doc2vec_model(source):

    print("Building sentences...")
    # TODO: Replace with iterator when is fixed: sentences = MyCorpus(dictionary)
    train_corpus, test_corpus = prepare_train_test_corpus()
    # corpus = list(self.read_corpus(path_to_corpus))

    model_variant = 1  # sg parameter: 0 = CBOW; 1 = Skip-Gram
    """
    hs_softmax_variants = [0, 1]  # 1 = Hierarchical SoftMax
    """
    # noinspection DuplicatedCode
    negative_sampling_variant = 10  # 0 = no negative sampling
    no_negative_sampling = 0  # use with hs_soft_max
    vector_size = 200
    window = 16
    min_count = 3
    epoch = 25
    sample = 0.0
    hs_softmax = 0

    corpus_title, model_results = get_eval_results_header()
    pbar = tqdm.tqdm(total=540)

    # hs_softmax = random_order.choice(hs_softmax_variants)
    # TODO: Get Back random_order choice!!!
    # This is temporary due to adding softmax one values to results after fixed tab indent in else.
    model_variant = model_variant
    vector_size = vector_size
    window = window
    min_count = min_count
    epochs = epoch
    sample = sample
    negative_sampling_variant = negative_sampling_variant

    if hs_softmax == 1:
        # TODO: Unit test computing of evaluations
        word_pairs_eval, analogies_eval = compute_eval_values(train_corpus=train_corpus,
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
        word_pairs_eval, analogies_eval = compute_eval_values(train_corpus=train_corpus,
                                                              test_corpus=test_corpus,
                                                              model_variant=model_variant,
                                                              negative_sampling_variant=negative_sampling_variant,
                                                              vector_size=vector_size,
                                                              window=window,
                                                              min_count=min_count,
                                                              epochs=epochs,
                                                              sample=sample,
                                                              source=source)
    # noinspection DuplicatedCode
    append_training_results(source=source, corpus_title=corpus_title[0],
                            model_variant=model_variant,
                            negative_sampling_variant=negative_sampling_variant,
                            vector_size=vector_size,
                            window=window, min_count=min_count, epochs=epochs, sample=sample,
                            hs_softmax=hs_softmax,
                            pearson_coeff_word_pairs_eval=word_pairs_eval[0][0],
                            pearson_p_val_word_pairs_eval=word_pairs_eval[0][1],
                            spearman_p_val_word_pairs_eval=word_pairs_eval[1][1],
                            spearman_coeff_word_pairs_eval=word_pairs_eval[1][0],
                            out_of_vocab_ratio=word_pairs_eval[2],
                            analogies_eval=analogies_eval,
                            model_results=model_results)

    pbar.update(1)
    final_results_file_name = 'doc2vec_tuning_results_random_search_final_' + source + '.csv'
    pd.DataFrame(model_results).to_csv(final_results_file_name, index=False,
                                       mode="w")
    print("Saved training results...")

    if source == "idnes":
        file_name_for_save = 'doc2vec_tuning_results_random_search_idnes.csv'
    elif source == "cswiki":
        file_name_for_save = 'doc2vec_tuning_results_random_search_cswiki.csv'
    else:
        raise ValueError("Source does not matech available options.")

    # noinspection PyTypeChecker
    pd.DataFrame(model_results).to_csv(file_name_for_save, index=False,
                                       mode="w")


def train_doc2vec(documents_all_features_preprocessed, create_csv=False):
    print("documents_all_features_preprocessed")
    print(documents_all_features_preprocessed)

    tagged_data = []
    for i, doc in enumerate(documents_all_features_preprocessed):
        selected_list = []
        for word in doc.split(", "):
            # if not word in all_stopwords:
            words_preprocessed = gensim.utils.simple_preprocess(preprocess(word))
            for sublist in words_preprocessed:
                if len(sublist) > 0:
                    selected_list.append(sublist)

        print("Preprocessing doc. num. " + str(i))
        print("Selected list:")
        print(selected_list)
        tagged_data.append(TaggedDocument(words=selected_list, tags=[str(i)]))
        if create_csv is True:
            # Will append to exising file! CSV needs to be removed first if needs to be up updated as a whole
            with open("testing_datasets/idnes_preprocessed.txt", "a+", encoding="utf-8") as fp:
                wr = csv.writer(fp, dialect='excel')
                wr.writerow(selected_list)

    print("tagged_data:")
    print(tagged_data)

    train(tagged_data)


def get_similar_by_posts_slug(most_similar_items, documents_slugs, number_of_recommended_posts):
    print('\n')

    post_recommendations = pd.DataFrame()
    list_of_article_slugs = []
    list_of_coefficients = []

    most_similar_items = most_similar_items[1:number_of_recommended_posts]

    # for label, index in [('MOST', 0), ('SECOND-MOST', 1), ('THIRD-MOST', 2), ('FOURTH-MOST', 3),
    # ('FIFTH-MOST', 4), ('MEDIAN', len(most_similar_items) // 2), ('LEAST', len(most_similar_items) - 1)]:
    for index in range(0, len(most_similar_items)):
        print(u'%s: %s\n' % (most_similar_items[index][1], documents_slugs[int(most_similar_items[index][0])]))
        list_of_article_slugs.append(documents_slugs[int(most_similar_items[index][0])])
        list_of_coefficients.append(most_similar_items[index][1])
    print('=====================\n')

    post_recommendations['slug'] = list_of_article_slugs
    post_recommendations['coefficient'] = list_of_coefficients

    posts_dict = post_recommendations.to_dict('records')

    list_of_articles = [posts_dict.copy()]
    # print("------------------------------------")
    # print("JSON:")
    # print("------------------------------------")
    # print(list_of_article_slugs[0])
    return flatten(list_of_articles)


class Doc2VecClass:

    def __init__(self):
        self.documents = None
        self.df = None
        self.posts_df = None
        self.categories_df = None
        self.database = DatabaseMethods()
        self.doc2vec_model = None

    def prepare_posts_df(self):
        # self.database.insert_post_dataframe_to_cache() # uncomment for UPDATE of DB records
        self.posts_df = self.database.get_posts_dataframe_from_cache()
        self.posts_df.drop_duplicates(subset=['title'], inplace=True)
        self.posts_df = self.posts_df.rename({'title': 'post_title'})
        return self.posts_df

    def prepare_categories_df(self):
        self.categories_df = self.database.get_categories_dataframe()
        self.posts_df = self.posts_df.rename({'title': 'category_title'})
        return self.categories_df

    @check_empty_string
    def get_similar_doc2vec(self, searched_slug, train_enabled=False, limited=True, number_of_recommended_posts=21,
                            full_text=False, posts_from_cache=True):

        if type(searched_slug) is not str:
            raise ValueError("Bad searched_slug parameter inserted.")

        # TODO: Replace other duplicated code like this:
        verify_searched_slug_sanity(searched_slug)
        recommender_methods = RecommenderMethods()
        self.df = recommender_methods.get_posts_categories_dataframe(from_cache=posts_from_cache)
        print("self.df")
        print(self.df.columns.values)

        if 'post_slug' in self.df.columns:
            self.df = self.df.rename(columns={'post_slug': 'slug'})
        if 'slug_x' in self.df.columns:
            self.df = self.df.rename(columns={'slug_x': 'slug'})
        if searched_slug not in self.df['slug'].to_list():
            print('Slug does not appear in dataframe. Refreshing datafreme of posts.')
            recommender_methods = RecommenderMethods()
            recommender_methods.get_posts_dataframe(force_update=True)
            self.df = recommender_methods.get_posts_categories_dataframe(from_cache=True)

        if full_text is False:
            cols = ['keywords', 'all_features_preprocessed']
            documents_all_features_preprocessed = preprocess_columns(self.df, cols)
        else:
            cols = ['keywords', 'all_features_preprocessed', 'body_preprocessed']
            documents_all_features_preprocessed = preprocess_columns(self.df, cols)

        gc.collect()

        if 'post_slug' in self.df:
            self.df = self.df.rename(columns={'post_slug': 'slug'})
        documents_slugs = self.df['slug'].tolist()

        if train_enabled is True:
            train_doc2vec(documents_all_features_preprocessed, create_csv=False)

        del documents_all_features_preprocessed
        gc.collect()

        if limited is True:
            if full_text is False:
                doc2vec_loaded_model = Doc2Vec.load("models/d2v_limited.model")
            else:
                doc2vec_loaded_model = Doc2Vec.load("models/d2v_full_text_limited.model")
        else:
            if full_text is False:
                doc2vec_loaded_model = Doc2Vec.load("models/d2v.model")  # or download from Dropbox / AWS bucket
            else:
                doc2vec_loaded_model = Doc2Vec.load("models/d2v_full_text.model")

        recommender_methods = RecommenderMethods()
        post_found = recommender_methods.find_post_by_slug(searched_slug)
        # TODO: REPAIR
        # IndexError: single positional indexer is out-of-bounds
        keywords_preprocessed = post_found.iloc[0]['keywords'].split(", ")
        all_features_preprocessed = post_found.iloc[0]['all_features_preprocessed'].split(" ")

        if full_text is False:
            tokens = keywords_preprocessed + all_features_preprocessed
        else:
            full_text = post_found.iloc[0]['body_preprocessed'].split(" ")
            tokens = keywords_preprocessed + all_features_preprocessed + full_text

        vector_source = doc2vec_loaded_model.infer_vector(tokens)
        most_similar_posts = doc2vec_loaded_model.dv.most_similar([vector_source], topn=number_of_recommended_posts)

        return get_similar_by_posts_slug(most_similar_posts, documents_slugs, number_of_recommended_posts)

    @accepts_first_argument(str)
    @check_empty_string
    def get_similar_doc2vec_with_full_text(self, searched_slug, train_enabled=False, number_of_recommended_posts=21,
                                           posts_from_cache=True):

        recommender_methods = RecommenderMethods()
        self.df = recommender_methods.get_posts_categories_dataframe(from_cache=posts_from_cache)

        if 'post_slug' in self.df.columns:
            self.df = self.df.rename(columns={'post_slug': 'slug'})
        if 'slug_x' in self.df.columns:
            self.df = self.df.rename(columns={'slug_x': 'slug'})

        # TODO: REPAIR
        # ValueError: Slug does not appear in dataframe.
        if searched_slug not in self.df['slug'].to_list():
            print('Slug does not appear in dataframe. Refreshing datafreme of posts.')
            recommender_methods = RecommenderMethods()
            recommender_methods.get_posts_dataframe(force_update=True)
            self.df = recommender_methods.get_posts_categories_dataframe(from_cache=True)

        cols = ['keywords', 'all_features_preprocessed', 'body_preprocessed']
        documents_all_features_preprocessed = preprocess_columns(self.df, cols)

        gc.collect()

        print("self.df.columns")
        print(self.df.columns)

        documents_slugs = self.df['slug'].tolist()

        if train_enabled is True:
            train_doc2vec(documents_all_features_preprocessed)
        del documents_all_features_preprocessed
        gc.collect()

        doc2vec_loaded_model = Doc2Vec.load("models/d2v_full_text_limited.model")

        recommend_methods = RecommenderMethods()

        # not necessary
        post_found = recommend_methods.find_post_by_slug(searched_slug)
        # TODO: REPAIR
        # IndexError: single positional indexer is out-of-bounds
        keywords_preprocessed = post_found.iloc[0]['keywords'].split(", ")
        all_features_preprocessed = post_found.iloc[0]['all_features_preprocessed'].split(" ")
        full_text = post_found.iloc[0]['body_preprocessed'].split(" ")
        tokens = keywords_preprocessed + all_features_preprocessed + full_text
        vector_source = doc2vec_loaded_model.infer_vector(tokens)

        most_similar_items = doc2vec_loaded_model.dv.most_similar([vector_source], topn=number_of_recommended_posts)

        return get_similar_by_posts_slug(most_similar_items, documents_slugs, number_of_recommended_posts)

    def get_vector_representation(self, searched_slug):
        """
        For Learn-to-Rank
        """
        if type(searched_slug) is not str:
            raise ValueError("Entered slug must be a input_string.")
        else:
            if searched_slug == "":
                raise ValueError("Entered input_string is empty.")
            else:
                pass

        recommender_methods = RecommenderMethods()
        return self.doc2vec_model.infer_vector(recommender_methods.find_post_by_slug(searched_slug))

    @DeprecationWarning
    def create_or_update_corpus_and_dict_from_mongo_idnes(self):
        dict_idnes = create_dictionary_from_mongo_idnes(force_update=True)
        return dict_idnes

    @staticmethod
    def get_prefilled_full_text(slug, variant):
        recommender_methods = RecommenderMethods()
        return recommender_methods.get_prefilled_full_text(slug, variant)

    def get_pair_similarity_doc2vec(self, slug_1, slug_2, d2v_model=None):
        if d2v_model is None:
            d2v_model = self.load_model('models/d2v_full_text_limited.model')

        recommend_methods = RecommenderMethods()
        post_1 = recommend_methods.find_post_by_slug(slug_1)
        post_2 = recommend_methods.find_post_by_slug(slug_2)

        feature_1 = 'all_features_preprocessed'
        feature_2 = 'title'

        first_text = post_1[feature_2].iloc[0] + ' ' + post_1[feature_1].iloc[0]
        second_text = post_2[feature_2].iloc[0] + ' ' + post_2[feature_1].iloc[0]

        print(first_text)
        print(second_text)

        vec1 = d2v_model.infer_vector(first_text.split())
        vec2 = d2v_model.infer_vector(second_text.split())

        cos_distance = spatial.distance.cosine(vec1, vec2)
        print("post_1:")
        print(post_1)
        print("post_2:")
        print(post_2)
        print("cos_distance:")
        print(cos_distance)

        return cos_distance

    def load_model(self, path=None):
        self.doc2vec_model = load_doc2vec_model(path_to_model=path)
        return self.doc2vec_model


