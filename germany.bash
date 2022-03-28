#!/bin/bash

country="GER"
geo="DE"

python main.py \
       --country $country \
       --path "/Users/augustecousin/Documents/bel_gm_tool/gm_tool/data/${country}/${1}/" \
       --weeks \
       --periods 157 \
       --geo $geo \
       --compute-trends \
       --competition --markets \
       --scenarios --cagr -ss \
       -saap -20 -10 0 10 20 30 \
       -sppv -5 0 5 10 15 \
       -spc  -10 -5 0 5 10 \
       -sdist  -10 -5 0 5 10 \
       --verbose


# -fc -sfc -spfc "Categories" \
    #        -fbwr 'A&P' 'Price per volume' 'Distribution' 'Promo Cost' 'Rate of Innovation' -sfbwr -spfbwr "WithRegressors" --markets --competition \
# -fbnr -sfbnr -spfbnr "NoRegressors" \
# -gdp 'Date' 'Brand' 'A&P' 'Price per volume' 'Rate of Innovation' 'Promo Cost' 'Distribution' 'Trends' 'Sales in volume' \
#      --list-controllable-features 'A&P' 'Price per volume' 'Rate of Innovation' 'Promo Cost' 'Distribution' \
#      --gd-years 2019 2020 2021 \
#      -sgdp \
