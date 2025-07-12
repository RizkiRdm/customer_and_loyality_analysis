import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from faker import Faker
import uuid
import json


def make_data_dirty(
    df_transactions, df_customer_profiles, df_loyalty, corruption_rate=0.03
):
    print(
        f"\nIntroducing data dirtiness with a corruption rate of {corruption_rate*100:.0f}%..."
    )

    # --- Transactions Data Dirtiness ---
    # 1. Missing Values (e.g., payment method not recorded)
    for col in ["payment_method", "order_source"]:
        df_transactions.loc[df_transactions.sample(frac=corruption_rate).index, col] = (
            np.nan
        )

    # 2. Duplicate Transactions (some transactions recorded twice)
    num_duplicates_transactions = int(
        len(df_transactions) * (corruption_rate / 2)
    )  # Fewer duplicates, usually
    if num_duplicates_transactions > 0:
        duplicate_rows = df_transactions.sample(
            n=num_duplicates_transactions, replace=True
        )
        df_transactions = pd.concat(
            [df_transactions, duplicate_rows], ignore_index=True
        )
        print(f"- Added {num_duplicates_transactions} duplicate transactions.")

    # 3. Outliers / Invalid Values (Order Total, Quantity)
    # Order total can be zero, negative, or extremely high due to errors
    outlier_indices_total = df_transactions.sample(frac=corruption_rate).index
    df_transactions.loc[outlier_indices_total, "order_total"] = np.random.choice(
        [0, -100, 999999]
    )

    # Quantity can be zero, negative, or extremely high
    outlier_indices_qty = df_transactions.sample(frac=corruption_rate).index
    df_transactions.loc[outlier_indices_qty, "quantity"] = np.random.choice(
        [0, -1, 999]
    )

    # 4. Inconsistent Formatting / Typos for Item Name/Category
    corrupt_indices_item = df_transactions.sample(frac=corruption_rate).index
    for idx in corrupt_indices_item:
        original_name = df_transactions.loc[idx, "item_name"]
        original_category = df_transactions.loc[idx, "item_category"]

        choice = random.randint(0, 3)  # Randomly apply one type of corruption
        if choice == 0 and isinstance(original_name, str):  # UPPERCASE
            df_transactions.loc[idx, "item_name"] = original_name.upper()
        elif choice == 1 and isinstance(original_name, str):  # Extra spaces
            df_transactions.loc[idx, "item_name"] = "  " + original_name + " "
        elif choice == 2 and isinstance(
            original_name, str
        ):  # Simple typo (replace a char)
            if len(original_name) > 3:
                idx_typo = random.randint(0, len(original_name) - 2)
                df_transactions.loc[idx, "item_name"] = (
                    original_name[:idx_typo]
                    + random.choice(["x", "z", "y"])
                    + original_name[idx_typo + 1 :]
                )
        elif choice == 3 and isinstance(original_category, str):  # lowercase category
            df_transactions.loc[idx, "item_category"] = original_category.lower()

    # 5. Invalid Dates for Transaction Timestamps (e.g., future dates)
    corrupt_indices_date = df_transactions.sample(frac=corruption_rate).index
    for idx in corrupt_indices_date:
        df_transactions.loc[idx, "transaction_timestamps"] = datetime(
            2030, 1, 1
        )  # Set to a fixed future date

    print("- Applied various types of dirtiness to df_transactions.")

    # --- Customer Data Dirtiness ---
    # 1. Missing Values (e.g., contact info not available)
    for col in ["email", "phone", "address"]:
        df_customer_profiles.loc[
            df_customer_profiles.sample(frac=corruption_rate).index, col
        ] = np.nan

    # 2. Duplicate Customer Profiles (same customer entered twice, maybe with different IDs)
    num_duplicates_customers = int(len(df_customer_profiles) * (corruption_rate / 2))
    if num_duplicates_customers > 0:
        duplicate_customers = df_customer_profiles.sample(
            n=num_duplicates_customers, replace=True
        )
        # For some duplicates, generate a new customer_id to make them harder to spot
        duplicate_customers["customer_id"] = [
            str(uuid.uuid4()) if random.random() > 0.5 else cid
            for cid in duplicate_customers["customer_id"]
        ]
        df_customer_profiles = pd.concat(
            [df_customer_profiles, duplicate_customers], ignore_index=True
        )
        print(f"- Added {num_duplicates_customers} duplicate customer profiles.")

    # 3. Invalid DOB (e.g., future birth dates)
    corrupt_indices_dob = df_customer_profiles.sample(frac=corruption_rate).index
    for idx in corrupt_indices_dob:
        df_customer_profiles.loc[idx, "DOB"] = datetime(
            2030, 1, 1
        ).date()  # Set to future date

    print("- Applied various types of dirtiness to df_customer_profiles.")

    # --- Loyalty Data Dirtiness ---
    # 1. Outliers in total_loyalty_points
    outlier_indices_points = df_loyalty.sample(frac=corruption_rate).index
    df_loyalty.loc[outlier_indices_points, "total_loyalty_points"] = np.random.choice(
        [0, -10, 99999]
    )

    # 2. Inconsistent discount group names
    corrupt_indices_group = df_loyalty.sample(frac=corruption_rate).index
    for idx in corrupt_indices_group:
        original_group = df_loyalty.loc[idx, "assigned_discount_group(s)"]
        if isinstance(original_group, str):
            df_loyalty.loc[idx, "assigned_discount_group(s)"] = (
                original_group.upper()
            )  # Or add other types of inconsistencies

    print("- Applied various types of dirtiness to df_loyalty.")

    # Reshuffle index to mix duplicates and NAs
    df_transactions = df_transactions.sample(frac=1, random_state=42).reset_index(
        drop=True
    )
    df_customer_profiles = df_customer_profiles.sample(
        frac=1, random_state=42
    ).reset_index(drop=True)
    df_loyalty = df_loyalty.sample(frac=1, random_state=42).reset_index(drop=True)

    print("Data dirtiness introduction complete!")
    return df_transactions, df_customer_profiles, df_loyalty
