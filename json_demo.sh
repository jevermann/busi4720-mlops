#!/usr/bin/bash
curl -X POST -H "Content-Type: application/json" --data '[6, 250, 66.5]' http://localhost:5000/predict_json