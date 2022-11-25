#!/bin/bash

systemctl --user daemon-reload
systemctl --user restart python_categories
systemctl --user restart python_keywords
systemctl --user restart python_scraper
systemctl --user restart python_stars
systemctl --user restart python_thumbs
systemctl --user restart python_hybrid_prefilling
