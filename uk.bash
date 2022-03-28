#!/bin/bash
country="UK"
geo="GB"
python main.py \
       --country $country \
       --path "/Users/ahmed/Projects/gm_tool/data/${country}/" \
       --weeks \
       --periods 157 \
       --geo $geo \
       --compute-trends \
       -fc -sfc -spfc "Categories" \
       -fbwr 'A&P' 'Price per volume' 'Distribution' 'Promo Cost' 'Rate of Innovation' -sfbwr -spfbwr "WithRegressors" \
       -fbnr -sfbnr -spfbnr "NoRegressors" \
       -gdp 'Date' 'Brand' 'A&P' 'Price per volume' 'Rate of Innovation' 'Promo Cost' 'Distribution' 'Trends' 'Sales in volume' \
       --list-controllable-features 'A&P' 'Price per volume' 'Rate of Innovation' 'Promo Cost' 'Distribution' \
       --gd-years 2019 2020 2021 \
       -sgdp --markets --competition \
       --verbose
