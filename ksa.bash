#!/bin/bash
country="KSA"
geo="SA"
python main.py \
       --country $country \
       --path "/Users/ahmed/Projects/gm_tool/data/${country}/${1}/" \
       --months \
       --periods 36 \
       --geo $geo \
       --compute-trends \
       --competition \
       --scenarios --cagr -ss \
       -saap -20 -10 -5 0 5 10 20 30 \
       -sppv -5 0 5 10 15 \
       -spc  -10 -5 0 5 10 \
       -sdist -10 -5 0 5 10 \
       --verbose

# --logistic \
    #     -fc -sfc -spfc "Categories" \
    #     -fbwr 'A&P' 'Price per volume' 'Distribution' 'Promo Cost' 'Rate of Innovation' -sfbwr -spfbwr "WithRegressors" --markets --competition \
    #     -fbnr -sfbnr -spfbnr "NoRegressors" \
    #     -gdp 'Date' 'Brand' 'A&P' 'Price per volume' 'Rate of Innovation' 'Promo Cost' 'Distribution' 'Trends' 'Sales in volume' \
    #     --list-controllable-features 'A&P' 'Price per volume' 'Rate of Innovation' 'Promo Cost' 'Distribution' \
    #     --gd-years 2019 2020 2021 \
    #     -sgdp \
    #     --verbose
