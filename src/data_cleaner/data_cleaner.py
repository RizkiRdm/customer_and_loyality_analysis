import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)


class DataCleaner:
    def __init__(self, transactions_df, loyalty_df, customer_df):
        """
        Init DataCleaner with raw dataframe
        Args:
            transactions_df (pd.DataFrame): DataFrame raw transaction data.
            loyalty_df (pd.DataFrame): DataFrame raw loyalty data.
            customer_df (pd.DataFrame): DataFrame raw customer data.
        """
        self.transactions_df = transactions_df.copy()
        self.loyalty_df = loyalty_df.copy()
        self.customer_df = customer_df.copy()

        # init data for cleaned data
        self.cleaned_transactions_df = None
        self.cleaned_loyalty_df = None
        self.cleaned_customer_df = None

        logging.info("DataCleaner initialized")

    def _handle_missing_values(
        self,
        df,
        columns_to_fill_mode=None,
        columns_to_fill_unknown=None,
        columns_to_drop_rows_if_na=None,
    ):
        """
        internal method to handle missing values.
        Args:
            df (pd.DataFrame): dataframe to be processed.
            columns_to_fill_mode (list): column to be fill with fill mode.
            columns_to_fill_unknown (list): column to be fill with fill 'Unknown'.
            columns_to_drop_rows_if_na (list): column if NaN in data, rows will be deleted.
        Returns:
            pd.DataFrame: DataFrame after handle missing values.
        """
        if columns_to_fill_mode:
            for col in columns_to_fill_mode:
                if col in df.columns:
                    mode_val = df[col].mode()
                    if not mode_val.empty:
                        df[col] = df[col].fillna(mode_val[0])
                    else:
                        # if mode is still null, fill with 'unknown'
                        df[col] = df[col].fillna("unknown")
                        logging.warning(
                            f"Mode for column '{col}' is empty. Filling with 'unknown'."
                        )

        if columns_to_fill_unknown:
            for col in columns_to_fill_unknown:
                if col in df.columns:
                    df[col] = df[col].fillna("Unknown")

        if columns_to_drop_rows_if_na:
            original_rows = len(df)
            df.dropna(subset=columns_to_drop_rows_if_na, inplace=True)
            logging.info(
                f"Dropped {original_rows - len(df)} rows due to NaN in {columns_to_drop_rows_if_na}."
            )

        return df

    def _correct_datatypes(self, df, date_columns=None, numeric_columns=None):
        """
        Metode internal untuk mengoreksi tipe data kolom.
        Args:
            df (pd.DataFrame): DataFrame yang akan diproses.
            date_columns (list): Kolom yang akan dikonversi ke datetime.
            numeric_columns (list): Kolom yang akan dikonversi ke numerik.
        Returns:
            pd.DataFrame: DataFrame setelah tipe data dikoreksi.
        """
        if date_columns:
            for col in date_columns:
                if col in df.columns:
                    original_nan_count = df[col].isnull().sum()
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                    new_nan_count = df[col].isnull().sum()

                    if new_nan_count > original_nan_count:
                        logging.warning(
                            f"Column '{col}': {new_nan_count - original_nan_count} values failed datetime conversion."
                        )

                    future_dates_mask = df[col] > pd.Timestamp.now()
                    if future_dates_mask.any():
                        logging.warning(
                            f"Column '{col}': {future_dates_mask.sum()} future dates found. Setting to NaT."
                        )
                        df.loc[future_dates_mask, col] = pd.NaT

        if numeric_columns:
            for col in numeric_columns:
                if col in df.columns:
                    original_nan_count = df[col].isnull().sum()
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                    new_nan_count = df[col].isnull().sum()

                    # Record how many values failed to be converted to numeric
                    if new_nan_count > original_nan_count:
                        logging.warning(
                            f"Column '{col}': {new_nan_count - original_nan_count} values failed numeric conversion."
                        )
        return df

    def _handle_duplicates(self, df, subset_cols):
        """
        Method to handel duplicate data
        Args:
            df (pd.DataFrame): DataFrame tobe processed.
            subset_cols (list): list as identifier duplicate column.
        Returns:
            pd.DataFrame: Dataframe after duplicate data is dropped.
        """
        if subset_cols and all(col in df.columns for col in subset_cols):
            original_rows = len(df)
            df.drop_duplicates(subset=subset_cols, inplace=True)
            # records how many duplicate columns dropped
            logging.info(
                f"Dropped {original_rows - len(df)} duplicate rows based on {subset_cols}."
            )
        return df

    def _handle_outliers(
        self, df, col_name, lower_bound=None, upper_quantile=None, strategy="cap"
    ):
        """
        Method to handle outliers
        Args:
            df (pd.DataFrame): Dataframe to be processed.
            col_name (str): numeric columns.
            lower_bound (float): minimum value is valid.
            upper_quantile (float): upper percentile value for outlier threshold.
            strategy (str): 'cap' (limit to bound), 'nan' (replace with NaN), 'remove' (remove row).
        Returns:
            pd.DataFrame: Dataframe after handle outliers.
        """
        # !TODO: IMPROVEMENT 9: Fleksibilitas metode outlier handling
        # REKOMENDASI: Berikan opsi strategi yang berbeda (cap, replace with NaN, remove row).
        # Saat ini defaultnya 'cap'. Implementasikan opsi 'nan' dan 'remove'.

        if col_name in df.columns and pd.api.types.is_numeric_dtype(df[col_name]):
            initial_outlier_count = 0

            # hanlde negative value or zero value
            if lower_bound is not None:
                invalid_lower_mask = df[col_name] < lower_bound
                initial_outlier_count += invalid_lower_mask.sum()
                if strategy == "cap":
                    df.loc[invalid_lower_mask, col_name] = lower_bound
                elif strategy == "nan":
                    df.loc[invalid_lower_mask, col_name] = np.nan
                elif strategy == "remove":
                    df = df[~invalid_lower_mask]

            # handle outlier at upper threshold
            if upper_quantile is not None:
                upper_limit = df[col_name].quantile(upper_quantile)
                invalid_upper_mask = df[col_name] > upper_limit
                initial_outlier_count += invalid_upper_mask.sum()
                if strategy == "cap":
                    df.loc[invalid_upper_mask, col_name] = upper_limit
                elif strategy == "nan":
                    df.loc[invalid_upper_mask, col_name] = np.nan
                elif strategy == "remove":
                    df = df[~invalid_upper_mask]

            logging.info(
                f"Column '{col_name}': Handled {initial_outlier_count} outliers using '{strategy}' strategy."
            )
        return df

    def _standardize_text_columns(self, df, columns):
        """
        method to standarize text format in some column.
        Args:
            df (pd.DataFrame): DataFrame to be processed.
            columns (list): list column text for standarized.
        Returns:
            pd.DataFrame: DataFrame after standarized.
        """
        for col in columns:
            if col in df.columns and pd.api.types.is_string_dtype(df[col]):
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].str.title()
        return df

    def clean_transactions_data(self):
        """
        Membersihkan DataFrame transaksi.
        """
        print("Cleaning transactions data")
        # !TODO: IMPROVEMENT 12: Tambahkan validasi kolom
        # REKOMENDASI: Pastikan kolom-kolom yang diharapkan ada sebelum diproses untuk menghindari KeyError.
        # Misal: assert all(col in df.columns for col in ['transaction_id', ])
        df = self.transactions_df

        # Handle Duplicates
        df = self._handle_duplicates(
            df,
            subset_cols=[
                "transaction_id",
                "customer_id",
                "transaction_timestamps",
                "order_total",
                "item_name",
            ],
        )

        # Correct Data Types
        df = self._correct_datatypes(
            df,
            date_columns=["transaction_timestamps"],
            numeric_columns=["order_total", "quantity", "item_price"],
        )

        # Handle Missing Values
        df = self._handle_missing_values(
            df,
            columns_to_fill_mode=["payment_method", "order_source"],
            columns_to_fill_unknown=["item_name", "item_category"],
            columns_to_drop_rows_if_na=["order_total", "quantity"],
        )  # Drop rows where these critical values are NaN after type conversion

        # Handle Outliers (order_total, quantity)
        df = self._handle_outliers(
            df, "order_total", lower_bound=0.1, upper_quantile=0.99, strategy="cap"
        )
        df = self._handle_outliers(
            df, "quantity", lower_bound=1, upper_quantile=0.99, strategy="cap"
        )

        # Standardize Text Columns
        df = self._standardize_text_columns(
            df, ["item_name", "item_category", "payment_method", "order_source"]
        )

        self.cleaned_transactions_df = df
        logging.info("Transactions data cleaned.")
        return self.cleaned_transactions_df

    def clean_loyalty_data(self):
        """
        clean loyalty Dataframe.
        """
        print("Cleaning loyalty data")
        df = self.loyalty_df

        # Handle Duplicates
        df = self._handle_duplicates(
            df, subset_cols=["loyalty_member_id", "customer_id"]
        )

        # Correct Data Types
        df = self._correct_datatypes(df, numeric_columns=["total_loyalty_points"])

        # Handle Missing Values
        df = self._handle_missing_values(
            df, columns_to_fill_unknown=["assigned_discount_group(s)"]
        )

        df.dropna(subset=["customer_id"], inplace=True)

        # Handle Outliers (total_loyalty_points)
        df = self._handle_outliers(
            df, "total_loyalty_points", lower_bound=0, strategy="cap"
        )  # Loyalty points cannot be negative, set to 0 if found negative

        # Standardize Text Columns
        df = self._standardize_text_columns(df, ["assigned_discount_group(s)"])

        self.cleaned_loyalty_df = df
        print("Loyalty data cleaned.")
        return self.cleaned_loyalty_df

    def clean_customer_data(self):
        """
        clean customer Dataframe.
        """
        print("Cleaning customer data")
        df = self.customer_df

        # Handle Duplicates
        df = self._handle_duplicates(df, subset_cols=["customer_id", "email"])

        # Correct Data Types
        df = self._correct_datatypes(df, date_columns=["DOB"])

        # Handle Missing Values
        df = self._handle_missing_values(
            df,
            columns_to_fill_unknown=["email", "phone", "address", "gender"],
            columns_to_drop_rows_if_na=["customer_id"],
        )  # customer_id is critical, drop if NaN

        # Handle Outliers (age from DOB can be checked, but for now assuming DOB handling is enough)
        df = self._handle_outliers(
            df, col_name="age", upper_quantile=0.999, lower_bound=0, strategy="cap"
        )

        # Standardize Text Columns
        df = self._standardize_text_columns(df, ["gender"])

        self.cleaned_customer_df = df
        print("Customer data cleaned.")
        return self.cleaned_customer_df

    def get_cleaned_data(self):
        """
        Returning cleaned DataFrames.
        Returns:
            tuple: (cleaned_transactions_df, cleaned_loyalty_df, cleaned_customer_df)
        """
        if self.cleaned_transactions_df is None:
            self.cleaned_transactions_df = self.clean_transactions_data()
        if self.cleaned_loyalty_df is None:
            self.cleaned_loyalty_df = self.clean_loyalty_data()
        if self.cleaned_customer_df is None:
            self.cleaned_customer_df = self.clean_customer_data()

        return (
            self.cleaned_transactions_df,
            self.cleaned_loyalty_df,
            self.cleaned_customer_df,
        )
