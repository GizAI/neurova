#!/bin/bash
rsync -avz --exclude '.git' --exclude '__pycache__' --exclude 'neurova_state' --exclude '.venv' ./ ml-dmc8:~/workspace/neurova/
