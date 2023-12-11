#!/bin/bash
source ~/.profile
source ~/.bashrc
echo "Exporting from $DB_RECOMMENDER_NAME"
PGPASSWORD=$DB_RECOMMENDER_PASSWORD pg_dump --no-owner --no-acl -U $DB_RECOMMENDER_USER -h $DB_RECOMMENDER_HOST -p 5432 $DB_RECOMMENDER_NAME > ~/Documents/Codes/moje-clanky-core/news-recommender-core/database/db_backups/production_dumps/db_production.sql
echo "Importing to $DB_RECOMMENDER_NAME_LOCAL"
PGPASSWORD=$DB_RECOMMENDER_PASSWORD_LOCAL psql -U $DB_RECOMMENDER_USER_LOCAL -h $DB_RECOMMENDER_HOST_LOCAL -p 5432 $DB_RECOMMENDER_NAME_LOCAL < ~/Documents/Codes/moje-clanky-core/news-recommender-core/database/db_backups/production_dumps/db_production.sql
# TODO:
# _source /home/patri/.virtualenvs/venv_deploy/bin/activate
# python3 ~/Documents/Codes/news-parser/news-parser/rss-scrapper.py
# python3 ~/Documents/Codes/moje-clanky-core/news-recommender-core/run_prefillers.py
echo "Exporting from $DB_RECOMMENDER_NAME_LOCAL"
PGPASSWORD=$DB_RECOMMENDER_PASSWORD_LOCAL pg_dump --no-owner --no-acl -U $DB_RECOMMENDER_USER_LOCAL -h $DB_RECOMMENDER_HOST_LOCAL -p 5432 $DB_RECOMMENDER_NAME_LOCAL > ~/Documents/Codes/moje-clanky-core/news-recommender-core/database/db_backups/local_dumps/db_preliminary.sql
echo "Importing to $DB_RECOMMENDER_NAME (WIP)"
# TODO:
# psql $DB_RECOMMENDER_NAME < ~/Documents/Codes/moje-clanky-core/news-recommender-core/database/db_backups/local_dumps/db_preliminary.sql

# For OSU server:
PGPASSWORD=$DB_MC_PRODUCTION_COPY_PASSWORD psql -U $DB_MC_PRODUCTION_COPY_USER -h $DB_MC_PRODUCTION_COPY_HOST -p 5432 $DB_MC_PRODUCTION_COPY_NAME < /home/muller/Dokumenty/Codes/moje-clanky-api/database/db_backups/production_dumps/db_production.sql
