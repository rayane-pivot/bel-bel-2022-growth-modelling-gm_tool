import calendar
import datetime as dt
import json
from pydoc import locate

import pandas as pd


class DataManager:
    """ABSTRACT class for sales data"""

    """" Class that handles data processing from csv to usable dataframes """

    _df = pd.DataFrame()
    _df_bel = pd.DataFrame()

    def open_excel(self, json_sell_out_params, country):
        """TODO describe function

        :param json_sell_out_params:
        :param country:
        :returns:

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
        """TODO describe function

        :param json_sell_out_params:
        :param country:
        :returns:

        """
        dict_distrib = self.open_excel(json_sell_out_params, country)
        df_concat = pd.DataFrame()
        for distrib, dict_df in dict_distrib.items():
            df = pd.concat(dict_df.values(), axis=0)
            print(f"<fill_df> Size of {distrib} dataframe : {df.shape}")
            if json_sell_out_params.get(country).get("levels"):
                df.columns = df.columns.droplevel(0)
            df.columns = json_sell_out_params.get(country).get("sales_renaming_columns")
            df["Channel"] = distrib
            df_concat = pd.concat([df_concat, df])
        return df_concat.reset_index(drop=True)

    def fill_Finance(
        self,
        path: str,
        finance_cols: list,
        finance_renaming_columns: list,
        header: list,
    ):
        """TODO describe function

        :param path:
        :param finance_cols:
        :param finance_renaming_columns:
        :param header:
        :returns:

        """
        print(f"<fill_Finance> Loading data from file {path}")
        # Load finance file and some formating
        df_finance = pd.read_excel(path, header=header)
        df_finance = df_finance[finance_cols]
        # Rename columns
        df_finance.columns = finance_renaming_columns
        # Handle dates
        df_finance["Month"] = df_finance["Year"].apply(lambda x: int(x[5:8]))
        df_finance["Year"] = df_finance["Year"].apply(lambda x: int(x[:4]))
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
        """TODO describe function

        :param path:
        :param header:
        :param brand_column_name:
        :param week_name:
        :param columns_to_remove:
        :param date_format:
        :returns:

        """
        print(f"<fill_Inno> Loading data from file {path}")
        # Load innovation file and some formating
        df_ino = pd.read_excel(path, header=header)
        # rename Brands
        df_ino = df_ino.rename(columns={brand_column_name: "Brand"})
        # Remove 'all categories'
        df_ino = df_ino[~df_ino["Product"].str.contains("ALL CATEGORIES")]
        # Convert columns names to date format
        cols = [x for x in df_ino.columns if week_name in x]
        df_ino = df_ino.rename(
            columns={
                x: dt.datetime.strftime(
                    dt.datetime.strptime(x.split()[-1], date_format), "%Y-%m-%d"
                )
                for x in cols
            }
        )
        # remove unwanted columns
        df_ino = df_ino.drop(
            columns=[x for x in df_ino.columns if x in columns_to_remove]
        )
        return df_ino

    def load(self, path):
        """TODO describe function

        :param path:
        :returns:

        """
        self._df = pd.read_excel(path)

    def load_df_bel(self, path):
        """TODO describe function

        :param path:
        :returns:

        """
        self._df_bel = pd.read_excel(path)

    def fill_df_bel_old(self):
        """TODO describe function

        :returns:

        """
        df_bel = self.df[self.df["Brand"].isin(self.bel_brands)]
        self.df_bel = df_bel

    def count_num_sundays_in_month(self, year, month):
        """TODO describe function

        :param year:
        :param month:
        :returns:

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
        """TODO describe function

        :returns:

        """
        assert (
            not self._df.empty
        ), "df is empty, call ad_hoc_COUNTRY() or load_df() first"
        return self._df

    def get_df_bel(self):
        """TODO describe function

        :returns:

        """
        assert (
            not self._df_bel.empty
        ), "df_bel is empty, call fill_df_bel() or load_df_bel() first"
        return self._df_bel
