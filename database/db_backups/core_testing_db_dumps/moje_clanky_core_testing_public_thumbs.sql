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