SELECT COUNT(*) FROM posts WHERE recommended_tfidf IS NULL;
SELECT COUNT(*) FROM posts WHERE recommended_doc2vec IS NULL;
SELECT COUNT(*) FROM posts WHERE posts.recommended_tfidf_full_text IS NULL;
SELECT COUNT(*) FROM posts WHERE posts.recommended_word2vec_eval_3 IS NULL;
SELECT COUNT(*) FROM posts WHERE posts.recommended_lda_full_text IS NULL;