#!/bin/sh

while true; do
# Local (laptop DELL):
# nohup python3 ~/Documents/Codes/moje-clanky-core/news-recommender-core/rabbitmq_multithreaded_thumbs.py >> thumbs.out
  nohup python3 /home/muller/Dokumenty/Codes/moje-clanky-api/rabbitmq_multithreaded_thumbs.py >> thumbs.out
done &
