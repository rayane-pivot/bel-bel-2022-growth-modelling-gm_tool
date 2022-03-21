import calendar
import datetime as dt
import json
import re
from pydoc import locate

import pandas as pd


class DataManager:
    """ABSTRACT class for sales data"""

    """" Class that handles data processing from csv to usable dataframes """

    _df = pd.DataFrame()
    _df_bel = pd.DataFrame()

    def open_excel(self, json_sell_out_params, country):
        """read Sale out data excel file

        :param json_sell_out_params: params dict
        :param country: country code for json params dict
        :returns: dict of data for each file

        """
        headers = json_sell_out_params.get(country).get("header_lines")
        sheet_name = json_sell_out_params.get(country).get("sheet_name")
        dict_distrib = dict()
        dict_path = json_sell_out_params.get(country).get("dict_path").get("PATH_SALES")
        for distrib, PATH in dict_path.items():
            print(f"<open_excel> Loading data from file: {PATH}")
            dict_temp = pd.read_excel(PATH, header=headers, sheet_name=sheet_name)
            dict_distrib[distrib] = dict_temp
        return dict_distrib

    def fill_df(self, json_sell_out_params, country):
        """Format extracted data to DataFrame, rename columns, check types

        :param json_sell_out_params: params dict
        :param country: country code for json params dict
        :returns: df

        """
        dict_distrib = self.open_excel(json_sell_out_params, country)
        df_concat = pd.DataFrame()
        for distrib, dict_df in dict_distrib.items():
            if json_sell_out_params.get(country).get("source") == "Nielsen":
                for sheet, df_sheet in dict_df.items():
                    df_sheet['Feature'] = sheet
            df = pd.concat(dict_df.values(), axis=0)
            print(f"<fill_df> Size of {distrib} dataframe : {df.shape}")
            if json_sell_out_params.get(country).get("levels"):
                df.columns = df.columns.droplevel(0)
            if json_sell_out_params.get(country).get("source") == "IRI":
                df.columns = json_sell_out_params.get(country).get("sales_renaming_columns")
            df["Channel"] = distrib
            df_concat = pd.concat([df_concat, df])
        return df_concat.reset_index(drop=True)

    def fill_Finance(
        self,
        json_sell_out_params,
        country:str
    ):
        """Read Finance file, rename columns

        :param path: path to Finance file
        :param finance_cols: original finance columns
        :param finance_renaming_columns: new finance columns
        :param header: line in excel at which the headers are 
        :returns: df_finance

        """
        path = (
            json_sell_out_params.get(country)
            .get("dict_path")
            .get("PATH_FINANCE")
            .get("Total Country")
        )
        
        finance_cols = json_sell_out_params.get(country).get("A&P_columns")
        finance_renaming_columns = json_sell_out_params.get(country).get(
            "finance_renaming_columns"
        )
        
        header = (
            json_sell_out_params.get(country).get("Finance").get("header")
        )
        print(f"<fill_Finance> Loading data from file {path}")
        # Load finance file and some formating
        df_finance = pd.read_excel(path, header=header)
        df_finance = df_finance[finance_cols]
        # Rename columns
        df_finance.columns = finance_renaming_columns
        # Handle dates
        df_finance["Month"] = df_finance["Year"].apply(lambda x: int(str(x)[5:8]))
        df_finance["Year"] = df_finance["Year"].apply(lambda x: int(str(x)[:4]))
        return df_finance
    
    def fill_Inno(
        self,
        path: str,
        header: list,
        brand_column_name: str,
        week_name: str,
        columns_to_remove: list,
        date_format: str,
    ):
        """Read Innovation file, rename columns

        :param path: path to Innovation file
        :param header: line in excel at which the headers are 
        :param brand_column_name: column for brands in Innovation file
        :param week_name: name of weeks in columns of Innovation file
        :param columns_to_remove: useless columns in innovation file
        :param date_format: date format in innovation columns
        :returns: df_inno

        """
        print(f"<fill_Inno> Loading data from file {path}")
        # Load innovation file and some formating
        df_ino = pd.read_excel(path, header=header)
        # rename Brands
        df_ino = df_ino.rename(columns={brand_column_name: "Brand"})
        # Remove 'all categories'
        try:
            df_ino = df_ino[~df_ino["Product"].str.contains("ALL CATEGORIES")]
        except KeyError:
            pass
        # Convert columns names to date format
        cols = [x for x in df_ino.columns if week_name in x]
        df_ino = df_ino.rename(
            columns={
                x: dt.datetime.strftime(
                    dt.datetime.strptime(self.date_to_re(x, date_format), date_format), "%Y-%m-%d"
                )
                for x in cols
            }
        )
        # remove unwanted columns
        df_ino = df_ino.drop(
            columns=[x for x in df_ino.columns if x in columns_to_remove]
        )
        return df_ino
    
    def date_to_re(self, string:str, date_format:str):
        look_up_table = {
            "d":"[0-9]{2}",
            "y":"[0-9]{2}",
            "Y":"[0-9]{4}",
            "m":"[0-9]{2}",
            "b":"[A-Za-z]{3}",
            "W":"[0-9]{2}",
            "U":"[0-9]{2}"
        }
        reg = r".* ("
        char = date_format[2]
        for i in range(0, len(date_format), 3):
            reg+=look_up_table.get(date_format[i+1])
            reg+=char
        reg+='*)'
        return re.findall(reg, string)[0]

    def load(self, path):
        """load df excel file

        :param path: path to file
        :returns: None

        """
        self._df = pd.read_excel(path)

    def load_df_bel(self, path):
        """load df_bel from excel file

        :param path: path to df_bel excel file
        :returns: None

        """
        self._df_bel = pd.read_excel(path)

    def count_num_sundays_in_month(self, year, month):
        """count number of sundays per month for year

        :param year: year
        :param month: month
        :returns: number of sundays

        """
        day_to_count = calendar.SUNDAY
        matrix = calendar.monthcalendar(year, month)
        num_days = sum(1 for x in matrix if x[day_to_count] != 0)
        return num_days

    def assert_dataframe(self):
        """HERE ASSERT DF COLUMNS NAMES AND TYPES
        READ IN JSON FILE, colmuns, types, etc
        assert shape for Category-Sub Category-Brand
        """
        with open("assets/data_params.json", "r") as f:
            params = json.load(f)
        for col in params["columns"]:
            assert (
                col["column"] in self.df.columns
            ), f'Column not found: {col["column"]}'
            var_type = locate(col["type"])
            assert isinstance(
                self.df[col["column"]][0], var_type
            ), f'Column {col["column"]} of type {type(self.df[col["column"]][0])} is not of type {var_type}'
        print("columns and types are correct")

    def get_df(self):
        """get df

        :returns: df

        """
        assert (
            not self._df.empty
        ), "df is empty, call ad_hoc_COUNTRY() or load_df() first"
        return self._df

    def get_df_bel(self):
        """get df_bel

        :returns: df_bel

        """
        assert (
            not self._df_bel.empty
        ), "df_bel is empty, call fill_df_bel() or load_df_bel() first"
        return self._df_bel

    def get_df_competition_brands(self, df, features, brands_name, bel_markets):
        """Get the dataframe of all the other brands except the given ones with

        :param df:
        :param features:
        :param brands_name:
        :param bel_markets:

        """

        df_res = df[~(df.Brand.isin(brands_name)) & (df.Category.isin(bel_markets))][
            features
        ]
        df_res = df_res.fillna(0)

        return df_res

    def get_df_markets(self, df):
        """Get Markets dataframe

        :param df:

        """
        df_res = (
            df.groupby(["Category", "Date"]).agg("sum")[["Sales in volume"]].unstack(0)
        )
        df_res = df_res.droplevel(level=0, axis=1).reset_index()
        df_res["TOTAL CHEESE"] = df_res.iloc[:, 1:].sum(axis=1)
        df_res = df_res.fillna(0)
        df_res.columns.name = ""

        return df_res
