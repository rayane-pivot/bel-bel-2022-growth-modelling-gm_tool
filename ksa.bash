#!/bin/bash
country="KSA"
geo="SA"
python main.py \
       --country $country \
       --path "/Users/ahmed/Projects/gm_tool/data/${country}/Trad/" \
       --months \
       --periods 36 \
       --geo $geo \
       --compute-trends \
       -gdp 'Date' 'Brand' 'A&P' 'Price per volume' 'Rate of Innovation' 'Promo Cost' 'Distribution' 'Trends' 'Sales in volume' \
       --list-controllable-features 'A&P' 'Price per volume' 'Rate of Innovation' 'Promo Cost' 'Distribution' \
       --gd-years 2019 2020 2021 \
       -sgdp \
       --verbose
# --logistic \
    # -fc -sfc \
    # -fbwr 'A&P' 'Price per volume' 'Distribution' 'Promo Cost' 'Rate of Innovation' -sfbwr \
    # -fbnr -sfbnr \
