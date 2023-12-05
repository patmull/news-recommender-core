#!/bin/sh

while true; do
# Local (laptop DELL):
  nohup python3 /home/muller/Dokumenty/Codes/moje-clanky-api/rabbitmq_multithreaded_keywords.py >> keywords.out
done &
