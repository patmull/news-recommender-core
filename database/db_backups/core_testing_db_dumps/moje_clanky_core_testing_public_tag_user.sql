create table tag_user
(
    id           bigserial,
    user_id      bigint,
    tag_id       bigint,
    created_at   timestamp,
    published_at timestamp
);

alter table tag_user
    owner to postgres;

INSERT INTO public.tag_user (id, user_id, tag_id, created_at, published_at) VALUES (258, 371, 231, null, null);
INSERT INTO public.tag_user (id, user_id, tag_id, created_at, published_at) VALUES (251, 371, 229, null, null);
INSERT INTO public.tag_user (id, user_id, tag_id, created_at, published_at) VALUES (247, 371, 224, null, null);
INSERT INTO public.tag_user (id, user_id, tag_id, created_at, published_at) VALUES (30, 381, 27, null, null);
INSERT INTO public.tag_user (id, user_id, tag_id, created_at, published_at) VALUES (220, 421, 216, null, null);
INSERT INTO public.tag_user (id, user_id, tag_id, created_at, published_at) VALUES (335, 421, 292, null, null);
INSERT INTO public.tag_user (id, user_id, tag_id, created_at, published_at) VALUES (340, 421, 297, null, null);
INSERT INTO public.tag_user (id, user_id, tag_id, created_at, published_at) VALUES (289, 421, 240, null, null);
INSERT INTO public.tag_user (id, user_id, tag_id, created_at, published_at) VALUES (28, 431, 25, null, null);
INSERT INTO public.tag_user (id, user_id, tag_id, created_at, published_at) VALUES (26, 431, 23, null, null);
INSERT INTO public.tag_user (id, user_id, tag_id, created_at, published_at) VALUES (25, 431, 22, null, null);
INSERT INTO public.tag_user (id, user_id, tag_id, created_at, published_at) VALUES (214, 479, 210, null, null);
INSERT INTO public.tag_user (id, user_id, tag_id, created_at, published_at) VALUES (215, 479, 211, null, null);
INSERT INTO public.tag_user (id, user_id, tag_id, created_at, published_at) VALUES (216, 479, 212, null, null);