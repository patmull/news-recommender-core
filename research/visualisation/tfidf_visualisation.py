import gensim
import numpy as np
import altair as alt
import pandas as pd


class TfIdfVisualizer:

    def __init__(self):
        self.top_tfidf = None

    def prepare_for_heatmap(self, tfidf_vectors, text_titles, tfidf_vectorizer):
        tfidf_df = pd.DataFrame(tfidf_vectors.toarray(), index=text_titles, columns=tfidf_vectorizer.get_feature_names())
        tfidf_df = self.calculate_frequencies(tfidf_df)
        tfidf_df = tfidf_df.drop('Document Frequency', errors='ignore')
        tfidf_df = tfidf_df.stack().reset_index()
        tfidf_df = tfidf_df.rename(columns={0: 'tfidf', 'level_0': 'document', 'level_1': 'term', 'level_2': 'term'})
        print("tfidf_df.columns:")
        print(tfidf_df.columns)
        tfidf_df.sort_values(by=['document', 'tfidf'], ascending=[True, False]).groupby(['document']).head(10)
        self.top_tfidf = tfidf_df.sort_values(by=['document', 'tfidf'], ascending=[True, False]).groupby(['document']).head(10)
        test_word_1 = "žena"
        print("Top occurences for word " + test_word_1 + " in terms:")
        print(self.top_tfidf[self.top_tfidf['term'].str.lower().str.contains(gensim.utils.deaccent(test_word_1.lower()))])
        print("Top occurences for word " + test_word_1 + " in documents:")
        print(self.top_tfidf[self.top_tfidf['document'].str.lower().str.contains(gensim.utils.deaccent(test_word_1.lower()))])
        test_word_2 = "fotbal"
        print("Top occurences for word " + test_word_2 + " in terms:")
        print(self.top_tfidf[self.top_tfidf['term'].str.lower().str.contains(gensim.utils.deaccent(test_word_2).lower())])
        print("Top occurences for word " + test_word_2 + "in documents:")
        print(self.top_tfidf[self.top_tfidf['document'].str.lower().str.contains(gensim.utils.deaccent(test_word_2).lower())])
        test_word_3 = "bitcoin"
        print("Top occurences for word " + test_word_3 + " in terms:")
        print(self.top_tfidf[self.top_tfidf['term'].str.lower().str.contains(gensim.utils.deaccent(test_word_3.lower()))])
        print("Top occurences for word " + test_word_3 + " in documents:")
        print(self.top_tfidf[self.top_tfidf['document'].str.lower().str.contains(gensim.utils.deaccent(test_word_3.lower()))])
        test_word_4 = "zeman"
        print("Top occurences for word " + test_word_3 + " in terms:")
        print(self.top_tfidf[self.top_tfidf['term'].str.lower().str.contains(gensim.utils.deaccent(test_word_4.lower()))])
        print("Top occurences for word " + test_word_3 + " in documents:")
        print(self.top_tfidf[self.top_tfidf['document'].str.lower().str.contains(gensim.utils.deaccent(test_word_4.lower()))])

    def calculate_frequencies(self, tfidf_df, print_sample_frequencies=True, number_of_printed=50):
        tfidf_df.loc['Document Frequency'] = (tfidf_df > 0).sum()
        printed_features = ['hokej', 'fotbal', 'zápas', 'gól', 'výhra', 'prohra', 'měna', 'půjčka', 'účet']
        tfidf_df = tfidf_df.head(number_of_printed)
        if print_sample_frequencies:
            tfidf_slice = tfidf_df[printed_features]
            print("Frequencies:")
            print(tfidf_slice.sort_index().round(decimals=2))
        return tfidf_df


    def plot_tfidf_heatmap(self):
        # Terms in this list will get a red dot in the visualization
        term_list = ['válka', 'mír']

        # adding a little randomness to break ties in term ranking
        top_tfidf_plusRand = self.top_tfidf.copy()
        top_tfidf_plusRand['tfidf'] = top_tfidf_plusRand['tfidf'] + np.random.rand(self.top_tfidf.shape[0]) * 0.0001

        # base for all visualizations, with rank calculation
        base = alt.Chart(top_tfidf_plusRand).encode(
            x='rank:O',
            y='document:N'
        ).transform_window(
            rank="rank()",
            sort=[alt.SortField("tfidf", order="descending")],
            groupby=["document"],
        )

        # heatmap specification
        heatmap = base.mark_rect().encode(
            color='tfidf:Q'
        )

        # red circle over terms in above list
        circle = base.mark_circle(size=100).encode(
            color=alt.condition(
                alt.FieldOneOfPredicate(field='term', oneOf=term_list),
                alt.value('red'),
                alt.value('#FFFFFF00')
            )
        )

        # text labels, white for darker heatmap colors
        text = base.mark_text(baseline='middle').encode(
            text='term:N',
            color=alt.condition(alt.datum.tfidf >= 0.23, alt.value('white'), alt.value('black'))
        )

        # display the three superimposed visualizations
        superimposed_vis = (heatmap + circle + text).properties(width=600)
        print("Preparing Altair Viewer...")
        alt.renderers.enable('altair_viewer')
        superimposed_vis.show()