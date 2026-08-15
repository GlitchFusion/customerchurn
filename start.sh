#!/bin/bash
pip install -r requirements.txt
gunicorn --chdir development app:app