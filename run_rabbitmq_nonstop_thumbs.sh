#!/bin/sh

while true; do
  nohup python3 ~/Documents/Codes/moje-clanky-core/news-recommender-core/rabbitmq_multithreaded_thumbs.py >> thumbs.out
done &
