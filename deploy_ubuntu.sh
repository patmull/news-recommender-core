#!/bin/bash
# TODO: Priority: MEDIUM
echo "Assuming python-dev and GIT installed"
git fetch
git pull
sudo apt install -y python3-venv
python3 -m venv testing_env
source testing_env/bin/activate
python3 -m pip install -r requirements.txt
sudo apt update --yes
sudo apt install postgresql --yes
sudo service postgresql start
sudo -u postgres psql -c "DROP DATABASE moje_clanky_core_testing WITH (FORCE);"
# cd ~postgres/
sudo -u postgres psql -c "ALTER USER postgres WITH PASSWORD 'braf';"
export PGPASSWORD='braf'
sudo -u postgres createdb moje_clanky_core_testing
for i in database/db_backups/core_testing_db_dumps/*.sql; do psql -h localhost -d moje_clanky_core_testing -U postgres -p 5432 -a -w -f $i; done
python3 -m pip install -U pytest
python3 -m pip install pytest
python3 -m pip install pytest-cov
pytest
deactivate