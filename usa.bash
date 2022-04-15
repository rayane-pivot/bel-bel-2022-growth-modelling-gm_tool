#!/bin/bash
country="USA"
geo="US"
python main.py \
       --country $country \
       --path "/Users/ahmed/Projects/gm_tool/data/${country}/" \
       --weeks \
       --periods 157 \
       --geo $geo \
       --competition \
       --scenarios -ss -smn "prophet" \
       -saap 0 5 10 20 \
       -sppv 0 14.5 16.5 17 \
       -spc  5 10 15  \
       -sdist 0 5 20 \
       --verbose
