create table categories
(
    id          bigserial,
    title       varchar(191),
    description text,
    slug        varchar(191),
    created_at  timestamp,
    updated_at  timestamp
);

alter table categories
    owner to postgres;

INSERT INTO public.categories (id, title, description, slug, created_at, updated_at) VALUES (1, 'Ekonomika', 'Zprávy - Ekonomika', 'ekonomika', null, null);
INSERT INTO public.categories (id, title, description, slug, created_at, updated_at) VALUES (2, 'Regionální', 'Regionální', 'regionalni', null, null);
INSERT INTO public.categories (id, title, description, slug, created_at, updated_at) VALUES (3, 'Zahraničí', 'Zprávy - Zahraničí', 'zahranici', null, null);
INSERT INTO public.categories (id, title, description, slug, created_at, updated_at) VALUES (4, 'Domácí', 'Zprávy - Domácí', 'domaci', null, null);
INSERT INTO public.categories (id, title, description, slug, created_at, updated_at) VALUES (5, 'Kultura', 'Zprávy z kultury', 'kultura', null, null);
INSERT INTO public.categories (id, title, description, slug, created_at, updated_at) VALUES (6, 'Finance', 'Finance', 'finance', null, null);
INSERT INTO public.categories (id, title, description, slug, created_at, updated_at) VALUES (7, 'Móda', 'Móda', 'moda', null, null);
INSERT INTO public.categories (id, title, description, slug, created_at, updated_at) VALUES (8, 'Zdraví', 'Zdraví', 'zdravi', null, null);
INSERT INTO public.categories (id, title, description, slug, created_at, updated_at) VALUES (9, 'Vztahy', 'Vztahy', 'vztahy', null, null);
INSERT INTO public.categories (id, title, description, slug, created_at, updated_at) VALUES (11, 'Celebrity', 'Zprávy o celebritách', 'celebrity', null, null);
INSERT INTO public.categories (id, title, description, slug, created_at, updated_at) VALUES (12, 'Technologie', 'Technologie', 'technologie', null, null);
INSERT INTO public.categories (id, title, description, slug, created_at, updated_at) VALUES (13, 'Věda', 'Věda', 'veda', null, null);
INSERT INTO public.categories (id, title, description, slug, created_at, updated_at) VALUES (14, 'Auto-moto', 'Zpravodajství ze světa aut', 'auto-moto', null, null);
INSERT INTO public.categories (id, title, description, slug, created_at, updated_at) VALUES (15, 'Hry', 'Videoherní články', 'video-hry', null, null);
INSERT INTO public.categories (id, title, description, slug, created_at, updated_at) VALUES (16, 'Ostatní', 'Ostatní', 'ostatni', null, null);
INSERT INTO public.categories (id, title, description, slug, created_at, updated_at) VALUES (17, 'Sport', 'Sport', 'sport', null, null);