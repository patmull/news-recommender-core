import pandas as pd
from sklearn.model_selection import train_test_split
import spacy_sentence_bert
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from recommender_core.data_handling.data_manipulation import Database
from recommender_core.data_handling.data_queries import RecommenderMethods


class SVM:

    def train(self):
        recommender_methods = RecommenderMethods()
        df = recommender_methods.get_posts_users_categories_thumbs_ratings_df()
        nlp = spacy_sentence_bert.load_model('en_stsb_distilbert_base')
        df_predicted = pd.DataFrame()
        df_original = df
        df_predicted['original_context'] = df['original_context']

        df['topic_feature_wiki_summary_1'] = df_original['topic_feature_wiki_summary_1']
        df['author_feature_wiki_summary_1'] = df_original['author_feature_wiki_summary_1']
        df['name_feature_wiki_summary_1'] = df_original['name_feature_wiki_summary_1']

        # TOODO: Somehow combine multiple columns
        df = df.fillna('')
        columns = list(df.columns.values)
        df['combined'] = df[columns].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
        print("df['combined']")
        print(df['combined'].iloc[0])

        # leaving out 20% for validation set
        X_train, X_validation, y_train, y_validation = train_test_split(df['combined'].tolist(),
                                                                        df_predicted['original_context']
                                                                        .tolist(), test_size=0.2)
        # TODO: Preprocessing
        df['vector'] = df['combined'].apply(lambda x: nlp(x).vector)
        print()
        X_train, X_test, y_train, y_test = train_test_split(df['vector'].tolist(), df_predicted['original_context']
                                                            .tolist(), test_size=0.2)

        print("X_train:")
        print(X_train)

        print("y_train")
        print(y_train)

        clf = SVC(gamma='auto')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print("SVC results:")
        print(accuracy_score(y_test, y_pred))

        clf = RandomForestClassifier(max_depth=9, random_state=0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print("Random Forest Classifier:")
        print(accuracy_score(y_test, y_pred))

        features_list = X_validation
        contexts_list = y_validation

        print("features_list")
        print(features_list)

        print("contexts_list")
        print(contexts_list)

        for features_combined, context in zip(features_list, contexts_list):
            print(f"True Label: {context}, Predicted Label: {clf.predict(nlp(features_combined).vector.reshape(1, -1))[0]} \n")
