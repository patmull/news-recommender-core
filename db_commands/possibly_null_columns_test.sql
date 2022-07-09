SELECT keywords FROM posts WHERE slug = 'za-tyden-jsme-s-elektromobilem-ujeli-3-381-km-porad-vam-to-prijde-malo';

SELECT body_preprocessed FROM posts WHERE body_preprocessed IS NULL;
SELECT keywords FROM posts WHERE keywords IS NULL;
SELECT all_feature_preprocessed FROM posts WHERE body_preprocessed IS NULL;

SELECT * FROM posts WHERE NOT (posts IS NOT NULL);

SELECT COUNT (*) AS "recommended_tfidf" FROM posts WHERE posts.recommended_tfidf IS NULL;
SELECT COUNT (*) AS "recommended_tfidf_full_text" FROM posts WHERE posts.recommended_tfidf_full_text IS NULL;
SELECT COUNT (*) AS "recommended_word2vec" FROM posts WHERE posts.recommended_word2vec IS NULL;
SELECT COUNT (*) AS "recommended_word2vec_full_text" FROM posts WHERE posts.recommended_word2vec_full_text IS NULL;
SELECT COUNT (*) AS "recommended_doc2vec" FROM posts WHERE posts.recommended_doc2vec IS NULL;
SELECT COUNT (*) AS "recommended_doc2vec_full_text" FROM posts WHERE posts.recommended_doc2vec_full_text IS NULL;
SELECT COUNT (*) AS "recommended_lda" FROM posts WHERE posts.recommended_lda IS NULL;
SELECT COUNT (*) AS "recommended_lda_full_text" FROM posts WHERE posts.recommended_lda_full_text IS NULL;
