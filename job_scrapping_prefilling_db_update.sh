#!/bin/bash
source ~/.profile
source ~/.bashrc
echo "Exporting from $DB_RECOMMENDER_NAME"
PGPASSWORD=$DB_RECOMMENDER_PASSWORD pg_dump -U $DB_RECOMMENDER_USER -h $DB_RECOMMENDER_HOST -p 5432 $DB_RECOMMENDER_NAME > ~/Documents/Codes/moje-clanky-core/news-recommender-core/database/db_backups/production_dumps/db_production.sql
echo "Importing to $DB_RECOMMENDER_NAME_LOCAL"
PGPASSWORD=$DB_RECOMMENDER_PASSWORD_LOCAL psql -h $DB_RECOMMENDER_HOST_LOCAL -p 5432 -U $DB_RECOMMENDER_USER_LOCAL -d $DB_RECOMMENDER_NAME_LOCAL
psql $DB_RECOMMENDER_NAME_LOCAL < ~/Documents/Codes/moje-clanky-core/news-recommender-core/database/db_backups/production_dumps/db_production.sql
# TODO:
# source /home/patri/.virtualenvs/venv_deploy/bin/activate
# python3 ~/Documents/Codes/news-parser/news-parser/rss-scrapper.py
# python3 ~/Documents/Codes/moje-clanky-core/news-recommender-core/run_prefillers.py
echo "Exporting from $DB_RECOMMENDER_NAME_LOCAL"
PGPASSWORD=$DB_RECOMMENDER_PASSWORD_LOCAL pg_dump -U $DB_RECOMMENDER_USER_LOCAL -h $DB_RECOMMENDER_HOST_LOCAL -p 5432 $DB_RECOMMENDER_NAME_LOCAL > ~/Documents/Codes/moje-clanky-core/news-recommender-core/database/db_backups/local_dumps/db_preliminary.sql
echo "Importing to $DB_RECOMMENDER_NAME (WIP)"
PGPASSWORD=$DB_RECOMMENDER_PASSWORD psql -h $DB_RECOMMENDER_HOST -p 5432 -U $DB_RECOMMENDER_USER -d $DB_RECOMMENDER_NAME
# TODO:
# psql $DB_RECOMMENDER_NAME < ~/Documents/Codes/moje-clanky-core/news-recommender-core/database/db_backups/local_dumps/db_preliminary.sql
