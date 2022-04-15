#!/bin/bash
country="JPN"
geo="JP"
python main.py \
       --country $country \
       --path "/Users/ahmed/Projects/gm_tool/data/${country}/" \
       --weeks \
       --periods 157 \
       --geo $geo \
       -fbwr 'A&P' 'Price per volume' 'Distribution' 'Rate of Innovation' -sfbwr -spfbwr "WithRegressors" \
       -fbnr -sfbnr -spfbnr "NoRegressors" \
       --competition \
       --verbose

# -gdp 'Date' 'Brand' 'A&P' 'Price per volume' 'Distribution' 'Rate of Innovation' 'Sales in volume' \
    # --list-controllable-features 'A&P' 'Price per volume' 'Distribution' 'Rate of Innovation' \
    # --gd-years 2019 2020 2021 \
    # -sgdp \

# -fc -sfc -spfc "Categories" \
    # --scenarios --cagr -ss \
    # -saap -50 -45 -40 -35 -30 -25 -20 -15 -10 -5 0 5 10 15 20 25 30 35 40 45 50 \
    #       -sppv -50 -45 -40 -35 -30 -25 -20 -15 -10 -5 0 5 10 15 20 25 30 35 40 45 50 \
    #       -spc  -50 -45 -40 -35 -30 -25 -20 -15 -10 -5 0 5 10 15 20 25 30 35 40 45 50 \
    #       -sdist -50 -45 -40 -35 -30 -25 -20 -15 -10 -5 0 5 10 15 20 25 30 35 40 45 50 \
    #       --verbose
