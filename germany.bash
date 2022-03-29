#!/bin/bash

country="GER"
geo="DE"

python main.py \
       --country $country \
       --path "/Users/ahmed/Projects/gm_tool/data/${country}/${1}/" \
       --weeks \
       --periods 157 \
       --geo $geo \
       --compute-trends \
       --competition \
       --scenarios --cagr -ss \
       -saap -20 -10 -5 0 5 10 20 \
       -sppv -5 0 5 10 \
       -spc  -5 0 5 \
       -sdist -5 0 5 \
       --verbose


# -fc -sfc -spfc "Categories" \
    #        -fbwr 'A&P' 'Price per volume' 'Distribution' 'Promo Cost' 'Rate of Innovation' -sfbwr -spfbwr "WithRegressors" --markets --competition \
    # -fbnr -sfbnr -spfbnr "NoRegressors" \
    # -gdp 'Date' 'Brand' 'A&P' 'Price per volume' 'Rate of Innovation' 'Promo Cost' 'Distribution' 'Trends' 'Sales in volume' \
    #      --list-controllable-features 'A&P' 'Price per volume' 'Rate of Innovation' 'Promo Cost' 'Distribution' \
    #      --gd-years 2019 2020 2021 \
    #      -sgdp \
