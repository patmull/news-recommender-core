create table user_history
(
    id         bigint,
    user_id    bigint,
    post_id    bigint,
    created_at timestamp,
    updated_at timestamp
);

alter table user_history
    owner to postgres;

INSERT INTO public.user_history (id, user_id, post_id, created_at, updated_at) VALUES (1, 431, 3176, '2022-09-17 15:09:13.000000', '2022-09-17 15:09:16.000000');
INSERT INTO public.user_history (id, user_id, post_id, created_at, updated_at) VALUES (2, 431, 3083, '2022-09-17 15:14:06.000000', '2022-09-17 15:14:09.000000');
INSERT INTO public.user_history (id, user_id, post_id, created_at, updated_at) VALUES (null, 431, 1618, '2022-09-17 15:09:13.000000', '2022-09-17 15:09:16.000000');
INSERT INTO public.user_history (id, user_id, post_id, created_at, updated_at) VALUES (null, 431, 1503, '2022-09-17 15:14:06.000000', '2022-09-17 15:14:09.000000');