import pandas as pd
from sklearn.model_selection import train_test_split
import spacy_sentence_bert
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from src.recommender_core.data_handling.data_queries import RecommenderMethods


class SVM:

    def train_thumbs(self):
        recommender_methods = RecommenderMethods()
        df = recommender_methods.get_posts_users_categories_thumbs_df()
        # TODO: Use multilingual model!
        # https://metatext.io/models/distilbert-base-multilingual-cased
        bert_model = spacy_sentence_bert.load_model('cs_paraphrase_xlm_r_multilingual_v1')
        df_predicted = pd.DataFrame()
        df_predicted['thumbs_value'] = df['thumbs_value']

        # TOODO: Somehow combine multiple columns
        df = df.fillna('')
        columns = ['category_title', 'trigrams_full_text']
        df['combined'] = df[columns].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
        print("df['combined']")
        print(df['combined'].iloc[0])

        # leaving out 20% for validation set
        X_train, X_validation, y_train, y_validation = train_test_split(df['combined'].tolist(),
                                                                        df_predicted['thumbs_value']
                                                                        .tolist(), test_size=0.2)
        # TODO: Preprocessing
        df['vector'] = df['combined'].apply(lambda x: bert_model(x).vector)
        print()
        X_train, X_test, y_train, y_test = train_test_split(df['vector'].tolist(), df_predicted['thumbs_value']
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
            print(f"True Label: {context}, Predicted Label: {clf.predict(bert_model(features_combined).vector.reshape(1, -1))[0]} \n")
