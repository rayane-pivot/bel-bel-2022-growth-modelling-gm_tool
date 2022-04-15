#!/bin/bash
country="BEL"
geo="BE"
python main.py \
       --country $country \
       --path "/Users/ahmed/Projects/gm_tool/data/${country}/" \
       --weeks \
       --periods 157 \
       --geo $geo \
       --compute-trends \
       --competition \
       --scenarios --cagr -smn "orbit" \
       -saap 0 5 10 20 \
       -sppv 0 14.5 16.5 17 \
       -spc  0 5 20 \
       -sdist 5 10 15 \
       --verbose

# -fc -sfc -spfc "Categories" \
    #     -fbwr 'A&P' 'Price per volume' 'Distribution' 'Promo Cost' 'Rate of Innovation' -sfbwr -spfbwr "WithRegressors" \
    #     -fbnr -sfbnr -spfbnr "NoRegressors" \
    # -gdp 'Date' 'Brand' 'A&P' 'Price per volume' 'Rate of Innovation' 'Promo Cost' 'Distribution' 'Trends' 'Sales in volume' \
    #      --list-controllable-features 'A&P' 'Price per volume' 'Rate of Innovation' 'Promo Cost' 'Distribution' \
    #      --gd-years 2019 2020 2021 \
    #      -sgdp
