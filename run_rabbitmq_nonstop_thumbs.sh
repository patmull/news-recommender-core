#!/bin/sh

while true; do
  # Local (laptop DELL):
  # nohup python3 ~/Documents/Codes/moje-clanky-core/news-recommender-core/rabbitmq_multithreaded_thumbs.py >> thumbs.out
  # Previous version:
  # nohup python3 /home/muller/Dokumenty/Codes/moje-clanky-api/rabbitmq_multithreaded_thumbs.py >> thumbs.out
  # Experimental:
  exec >  >(awk '{ print $0; fflush();}')
  exec 2>  >(awk '{ print $0; fflush();}')
  nohup python3 /home/muller/Dokumenty/Codes/moje-clanky-api/rabbitmq_multithreaded_thumbs.py >> thumbs.out &
done &
