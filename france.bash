#!/bin/bash

country="FR"
geo="FR"

python main.py \
       --country $country \
       --path "/Users/augustecousin/Documents/bel_gm_tool/gm_tool/view/data/${country}/${1}/" \
       --weeks \
       --periods 157 \
       --geo $geo \
       --logistic \
       --compute-trends \
       --scenarios -ss \
       -saap -5 -2 0 2 5 \
       -sppv 0 1 3 5 7 \
       -spc  -6 -3 0 3 10 \
       -sdist  -10 -5 0 5 10 \
       --verbose

# -gdp 'Date' 'Brand' 'A&P' 'Price per volume' 'Rate of Innovation' 'Promo Cost' 'Distribution' 'Trends' 'Sales in volume' \
    #      --list-controllable-features 'A&P' 'Price per volume' 'Rate of Innovation' 'Promo Cost' 'Distribution' \
    # --gd-years 2019 2020 2021
# -sgdp
# -fc -sfc -spfc "Categories" \
    #     -fbwr 'A&P' 'Price per volume' 'Distribution' 'Promo Cost' 'Rate of Innovation' -sfbwr -spfbwr "WithRegressors" \
    #     -fbnr -sfbnr -spfbnr "NoRegressors" \
