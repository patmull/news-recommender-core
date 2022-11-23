#!/bin/bash
echo -n "WARNING: Run this with sudo!"
journalctl --vacuum-time=1d
journalctl --vacuum-size=300M
sudo apt-get clean
sudo apt-get autoclean
sudo apt-get autoremove