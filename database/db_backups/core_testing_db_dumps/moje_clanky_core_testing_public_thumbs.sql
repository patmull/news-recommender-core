create table thumbs
(
    id         bigserial,
    value      boolean,
    user_id    bigint,
    post_id    bigint,
    created_at timestamp,
    updated_at timestamp
);

alter table thumbs
    owner to postgres;

INSERT INTO public.thumbs (id, value, user_id, post_id, created_at, updated_at) VALUES (7, true, 431, 1618, '2022-09-15 13:39:11.000000', '2022-09-15 13:39:11.000000');
INSERT INTO public.thumbs (id, value, user_id, post_id, created_at, updated_at) VALUES (8, true, 431, 729161, '2022-09-15 13:41:21.000000', '2022-09-15 13:41:21.000000');
INSERT INTO public.thumbs (id, value, user_id, post_id, created_at, updated_at) VALUES (2, false, 431, 729261, '2022-09-05 14:19:36.000000', '2022-09-05 14:19:36.000000');
INSERT INTO public.thumbs (id, value, user_id, post_id, created_at, updated_at) VALUES (3, true, 431, 729641, '2022-09-05 15:25:52.000000', '2022-09-05 15:25:52.000000');
INSERT INTO public.thumbs (id, value, user_id, post_id, created_at, updated_at) VALUES (4, true, 431, 729341, '2022-09-05 16:36:44.000000', '2022-09-05 16:36:44.000000');
INSERT INTO public.thumbs (id, value, user_id, post_id, created_at, updated_at) VALUES (5, false, 431, 729571, '2022-09-05 16:37:18.000000', '2022-09-05 16:37:18.000000');
INSERT INTO public.thumbs (id, value, user_id, post_id, created_at, updated_at) VALUES (14, true, 431, 729631, '2022-09-15 18:06:18.000000', '2022-09-15 18:06:18.000000');
INSERT INTO public.thumbs (id, value, user_id, post_id, created_at, updated_at) VALUES (9, true, 431, 3090, '2022-09-15 18:06:18.000000', '2022-09-15 18:06:18.000000');
INSERT INTO public.thumbs (id, value, user_id, post_id, created_at, updated_at) VALUES (8, false, 431, 2752, '2022-09-15 18:06:18.000000', '2022-09-15 18:06:18.000000');
INSERT INTO public.thumbs (id, value, user_id, post_id, created_at, updated_at) VALUES (1, true, 431, 1618, '2022-09-15 18:06:18.000000', '2022-09-15 18:06:18.000000');
INSERT INTO public.thumbs (id, value, user_id, post_id, created_at, updated_at) VALUES (11, false, 431, 1662, '2022-09-15 13:41:21.000000', '2022-09-15 13:41:21.000000');
INSERT INTO public.thumbs (id, value, user_id, post_id, created_at, updated_at) VALUES (13, true, 431, 734917, '2022-09-05 15:25:52.000000', '2022-09-05 15:25:52.000000');
INSERT INTO public.thumbs (id, value, user_id, post_id, created_at, updated_at) VALUES (10, true, 431, 734677, '2022-09-15 13:39:11.000000', '2022-09-15 13:39:11.000000');
INSERT INTO public.thumbs (id, value, user_id, post_id, created_at, updated_at) VALUES (6, true, 431, 2182, '2022-09-05 16:36:44.000000', '2022-09-05 16:36:44.000000');
INSERT INTO public.thumbs (id, value, user_id, post_id, created_at, updated_at) VALUES (7, true, 431, 3087, '2022-09-05 16:37:18.000000', '2022-09-05 16:37:18.000000');
INSERT INTO public.thumbs (id, value, user_id, post_id, created_at, updated_at) VALUES (3, true, 431, 1792, '2022-09-15 13:41:21.000000', '2022-09-15 13:41:21.000000');
INSERT INTO public.thumbs (id, value, user_id, post_id, created_at, updated_at) VALUES (5, false, 431, 729421, '2022-09-05 15:25:52.000000', '2022-09-05 15:25:52.000000');
INSERT INTO public.thumbs (id, value, user_id, post_id, created_at, updated_at) VALUES (12, true, 431, 1612, '2022-09-05 14:19:36.000000', '2022-09-05 14:19:36.000000');
INSERT INTO public.thumbs (id, value, user_id, post_id, created_at, updated_at) VALUES (2, false, 431, 1503, '2022-09-15 13:39:11.000000', '2022-09-15 13:39:11.000000');
INSERT INTO public.thumbs (id, value, user_id, post_id, created_at, updated_at) VALUES (4, true, 431, 1682, '2022-09-05 14:19:36.000000', '2022-09-05 14:19:36.000000');