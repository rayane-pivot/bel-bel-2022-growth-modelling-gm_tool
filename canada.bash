#!/bin/bash

country="CAN"
geo="CA"
python main.py \
       --country $country \
       --path "/Users/ahmed/Projects/gm_tool/data/${country}/" \
       --weeks \
       --periods 157 \
       --geo $geo \
       --logistic \
       --compute-trends \
       --verbose \
       -fc -sfc -spfc "Categories"
# -fbwr 'A&P' 'Price per volume' 'Distribution' 'Promo Cost' 'Rate of Innovation' -sfbwr -spfbwr "WithCompetRegressors"
# -fc -sfc \
    # -fbnr -sfbnr \
    # -gdp 'Date' 'Brand' 'A&P' 'Price per volume' 'Rate of Innovation' 'Promo Cost' 'Distribution' 'Trends' 'Sales in volume' \
    # --list-controllable-features 'A&P' 'Price per volume' 'Rate of Innovation' 'Promo Cost' 'Distribution' \
    # --gd-years 2019 2020 2021 \
    # -sgdp \
