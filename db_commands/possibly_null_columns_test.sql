SELECT keywords FROM posts WHERE slug = 'za-tyden-jsme-s-elektromobilem-ujeli-3-381-km-porad-vam-to-prijde-malo';

SELECT body_preprocessed FROM posts WHERE body_preprocessed IS NULL;
SELECT keywords FROM posts WHERE keywords IS NULL;
SELECT all_feature_preprocessed FROM posts WHERE body_preprocessed IS NULL;

SELECT * FROM posts WHERE NOT (posts IS NOT NULL);