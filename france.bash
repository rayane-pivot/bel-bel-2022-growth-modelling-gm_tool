#!/bin/bash

country="FR"
geo="FR"

python main.py \
       --country $country \
       --path "/Users/ahmed/Projects/gm_tool/data/${country}/${1}/" \
       --weeks \
       --periods 157 \
       --geo $geo \
       --compute-trends \
       --competition \
       --scenarios -ss -smn "prophet" \
       -saap -2 0 2 5 10 \
       -sppv 10 11 15 \
       -spc  0 3 7 10 \
       -sdist  0 \
       --verbose

# -gdp 'Date' 'Brand' 'A&P' 'Price per volume' 'Rate of Innovation' 'Promo Cost' 'Distribution' 'Trends' 'Sales in volume' \
    #      --list-controllable-features 'A&P' 'Price per volume' 'Rate of Innovation' 'Promo Cost' 'Distribution' \
    # --gd-years 2019 2020 2021
# -sgdp
# -fc -sfc -spfc "Categories" \
    #     -fbwr 'A&P' 'Price per volume' 'Distribution' 'Promo Cost' 'Rate of Innovation' -sfbwr -spfbwr "WithRegressors" \
    #     -fbnr -sfbnr -spfbnr "NoRegressors" \
