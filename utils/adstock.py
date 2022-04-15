######################################################
# This script is used to run ad stock transformation #
######################################################

import numpy as np
import pandas as pd
from tqdm import tqdm


def compute_adstock(x, L, P, D):
    """
    Function that applies ad stock tranformation to an array of feature

    Parameters
    ----------

    x : array
        initial feature array

    L : int
        length of the feature effect

    P : int
        peak/delay of the feature effect

    D : int
        decay/retention rate, concentration of the effect


    Returns
    -------
    adstock_x : array
        transformed feature array

    """

    x = np.append(np.zeros(L - 1), x)
    weights = np.zeros(L)

    for l in range(L):
        weight = D ** ((l - P) ** 2)
        weights[L - 1 - l] = weight

    adstocked_x = []

    for i in range(L - 1, len(x)):
        x_array = x[i - L + 1 : i + 1]
        xi = sum(x_array * weights) / sum(weights)
        adstocked_x.append(xi)

    adstocked_x = np.array(adstocked_x)

    return adstocked_x


def run_adstock_loop(df, features, target, adstock_params):
    """
    Function that runs ad stock loop to find best L, P, D parameters for each feature and return the df with adstocked features

    Parameters
    ----------

    df : dataframe
        initial dataframe with features and target (must also have a "Date" column for plotting purposes)

    features : list of str
        list of features to apply ad stock transformation

    target : str
        target variable

    adstock_params : dict
        dict with range (list) to consider for each ad stock parameter, L, P, D and each feature
        e.g {'Feature1': {'L': np.arange(2, 5), 'P': [0, 2, 6], 'D': [i for i in range(1, 6)]},
             'Feature2': {'L': np.arange(1, 20), 'P': [0, 2, 8], 'D': [i for i in range(1, 6)]}}


    Returns
    -------
    df_adstocked : dataframe
        adstocked df

    """

    df_adstocked = pd.DataFrame()
    df_final_adstock_params = pd.DataFrame()

    pbar = tqdm(features, ascii=True)
    for feature in pbar:
        pbar.set_description(f"Features|{feature}")
        # print(black(feature, ["bold"]))

        # For each feature, this dataframe saves params of each ad stock run (L, P, D, Correlation)
        df_adstock_params = pd.DataFrame()

        # For each feature, this dataframe saves results of each ad stock run (L, P, D, Time series of feature before and after transfo)
        df_adstock_results = pd.DataFrame()

        for L in adstock_params[feature]["L"]:
            for P in adstock_params[feature]["P"]:
                for D in adstock_params[feature]["D"]:
                    X = df[[feature]]
                    X_adstocked = compute_adstock(X, L, P, D)

                    df_temp = df.assign(Feature_adstocked=X_adstocked).assign(
                        L=L, P=P, D=D
                    )
                    distance = abs(df_temp[target].corr(df_temp["Feature_adstocked"]))

                    df_adstock_params = pd.concat(
                        [
                            df_adstock_params,
                            pd.DataFrame(
                                {
                                    "Feature": [feature],
                                    "L": [L],
                                    "P": [P],
                                    "D": [D],
                                    "Correlation": [distance],
                                }
                            ),
                        ]
                    )

                    df_adstock_results = pd.concat([df_adstock_results, df_temp])

        # Check the best correlation
        df_adstock_params_sorted = df_adstock_params.sort_values(
            "Correlation", ascending=False
        )
        L = df_adstock_params_sorted["L"].iloc[0]
        P = df_adstock_params_sorted["P"].iloc[0]
        D = df_adstock_params_sorted["D"].iloc[0]
        Correlation = round(df_adstock_params_sorted["Correlation"].iloc[0], 2)

        print("L:" + str(L))
        print("P:" + str(P))
        print("D:" + str(D))
        print("Correlation:" + str(Correlation))
        print("\n")

        # Save final parameters for each feature
        df_final_adstock_params = pd.concat(
            [
                df_final_adstock_params,
                pd.DataFrame(
                    {
                        "Feature": [feature],
                        "L": [L],
                        "P": [P],
                        "D": [D],
                        "Correlation": [Correlation],
                    }
                ),
            ]
        )

        # Keep the transformation matching with the highest correlation
        df_adstocked_temp = df_adstock_results.loc[
            lambda x: (x["L"] == L) & (x["P"] == P) & (x["D"] == D)
        ][["ds", "Feature_adstocked"]]
        df_adstocked = pd.concat(
            [df_adstocked, df_adstocked_temp.assign(Feature=feature)]
        )

    # Final adstocked df
    df_adstocked = (
        pd.pivot_table(
            df_adstocked.fillna(0),
            index=["ds"],
            columns="Feature",
            values="Feature_adstocked",
        )
        .rename_axis(None, axis=1)
        .reset_index()
    )

    return df_adstocked
