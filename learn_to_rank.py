class LearnToRank:

    def linear_regression(self, user_id):
        # predictions(tfidf,doc2vec,lda,wor2vec,user_rating,thumbs) = c0 + c1 * tfidf + c2 * doc2vec + c3 * lda + c4 * wor2vec + c5 * user_rating + c6 * thumbs

def main():
    user_id = 411
    learn_to_rank = LearnToRank()
    learn_to_rank.linear_regression(user_id)

if __name__ == "__main__": main()