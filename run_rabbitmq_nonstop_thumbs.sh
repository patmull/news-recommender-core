#!/bin/sh

while true; do
  nohup python3 /home/muller/Dokumenty/Codes/moje-clanky-api/rabbitmq_multithreaded_thumbs.py >> thumbs.out
done &
