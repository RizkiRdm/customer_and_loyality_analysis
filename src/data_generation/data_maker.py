import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from faker import Faker
import uuid
import json

from data_dirty import make_data_dirty

# --- Configuration ---
data_product_path = "data/raw/jp_grocery_products.json"
fake = Faker("ja_JP")

# Configure data counts
num_unique_customers = 5000
num_total_transactions = random.randint(50000, 70000)
start_date = datetime(2024, 1, 1)
end_date = datetime(2025, 7, 6)

# --- Generate Customer Data ---
customer_profiles = []
for i in range(num_unique_customers):
    customer_id = str(uuid.uuid4())
    email = fake.unique.email()
    phone = fake.unique.phone_number()
    dob = fake.date_of_birth(minimum_age=18, maximum_age=60)
    name = fake.name()
    address = fake.address()

    customer_profiles.append(
        {
            "customer_id": customer_id,
            "email": email,
            "phone": phone,
            "DOB": dob,
            "full_name": name,
            "address": address,
        }
    )
df_customer_profiles = pd.DataFrame(customer_profiles)

# --- Load Japanese Data Products ---
try:
    with open(data_product_path, "r", encoding="utf-8") as f:
        base_jp_grocery_products = json.load(f)
    print(f"Product data loaded from {data_product_path}")
except FileNotFoundError:
    print(
        f"Error: {data_product_path} not found. Please run the JSON creation script first."
    )
    base_jp_grocery_products = []

# Handle case where product data might be empty
if not base_jp_grocery_products:
    print("Warning: No product data found. Generating dummy products.")
    base_jp_grocery_products = [
        {
            "base_name": "Default Item",
            "category": "Misc",
            "base_unit_price": 100,
            "variants": [{"size": "1pc", "multiplier": 1.0}],
        }
    ]

# Flatten the base products into a final list of all unique products with specific prices
jp_grocery_products_flat = []
for product in base_jp_grocery_products:
    for variant in product["variants"]:
        item_name_full = f"{product['base_name']} {variant['size']}"
        calculated_price = round(
            product["base_unit_price"] * float(variant["multiplier"])
        )
        jp_grocery_products_flat.append(
            {
                "item_name": item_name_full,
                "category": product["category"],
                "base_price": calculated_price,
            }
        )

print(f"Total unique product variants generated: {len(jp_grocery_products_flat)}")

# --- 3. Generate Transaction Data ---
transactions_list = []
for _ in range(num_total_transactions):  # Use num_total_transactions
    customer = df_customer_profiles.sample(1).iloc[0]
    random_date = start_date + timedelta(
        days=random.randint(0, (end_date - start_date).days)
    )

    selected_product_variant = random.choice(jp_grocery_products_flat)
    item_name = selected_product_variant["item_name"]
    item_category = selected_product_variant["category"]
    unit_price = selected_product_variant["base_price"]

    # Refined quantity logic for realism (FIX 4)
    if (
        "Single" in item_name
        or "Pack" in item_name
        or "Bottle" in item_name
        or "Loaf" in item_name
        or "pcs" in item_name
        or "Jar" in item_name
        or "Can" in item_name
        or "Tray" in item_name
    ):
        quantity = random.choices(
            [1, 2, 3, 4, 7], weights=[0.75, 0.55, 0.35, 0.20, 0.05], k=1
        )[0]
    else:  # For items like 'Rice 5kg', 'Miso Paste 1kg' where the unit implies one purchase
        quantity = 1  # Usually 1 unit for these larger packages

    order_total = unit_price * quantity
    order_total = order_total * random.uniform(0.98, 1.05)
    order_total = round(order_total)

    transactions_list.append(
        {
            "customer_id": customer["customer_id"],
            "email": customer["email"],
            "phone": customer["phone"],
            "DOB": customer["DOB"],
            "transaction_timestamps": random_date,
            "order_type": np.random.choice(
                [
                    "店舗 (In-Store)",
                    "オンライン配達 (Online Delivery)",
                    "ピックアップ (Pick-up)",
                ]
            ),
            "order_total": order_total,
            "item_name": item_name,
            "item_category": item_category,
            "quantity": quantity,
            "unit_price_at_purchase": unit_price,
            "order_source": np.random.choice(
                [
                    fake.city() + "店",
                    fake.city() + "店",
                    "ウェブサイト (Website)",
                    "モバイルアプリ (Mobile App)",
                ]
            ),
            "payment_method": np.random.choice(
                [
                    "現金 (Cash)",
                    "クレジットカード (Credit Card)",
                    "電子マネー (E-Money)",
                    "QRコード決済 (QR Pay)",
                ]
            ),
        }
    )

df_transactions = pd.DataFrame(transactions_list)

# --- Generate Loyalty Data ---
num_loyalty_members = int(num_unique_customers * 0.4)  # Use num_unique_customers
loyalty_members_profiles = df_customer_profiles.sample(num_loyalty_members)

loyalty_list = []
for index, member in loyalty_members_profiles.iterrows():
    signup_date = start_date + timedelta(
        days=random.randint(0, (end_date - start_date).days // 2)
    )

    loyalty_list.append(
        {
            "customer_identifiers": member["customer_id"],
            "email": member["email"],
            "loyalty_signup_date": signup_date,
            "total_loyalty_points": np.random.randint(100, 10000),
            "number_of_transactions": np.random.randint(5, 120),
            "total_spend": np.round(np.random.uniform(5000, 200000)),
            "contact_information": member["phone"],
            "assigned_discount_group(s)": np.random.choice(
                [
                    "常連客 (Regular)",
                    "シルバー会員 (Silver Member)",
                    "ゴールド会員 (Gold Member)",
                    "学生割引 (Student Discount)",
                ]
            ),
        }
    )

df_loyalty = pd.DataFrame(loyalty_list)

# --- Save to Excel Files ---
df_transactions, df_customer_profiles, df_loyalty = make_data_dirty(
    df_transactions, df_customer_profiles, df_loyalty, corruption_rate=0.04
)

df_transactions.to_excel(r"data/raw/transaction_data_variants.xlsx", index=False)
df_loyalty.to_excel(r"data/raw/loyalty_data_variants.xlsx", index=False)
df_customer_profiles.to_excel(r"data/raw/customer_data.xlsx", index=False)

print("data for 'Sakura Mart' with product variants successfully generated!")
print(f"Total Unique Product Variants: {len(jp_grocery_products_flat)}")
print(f"Total Transactions (after dirtiness): {len(df_transactions)} rows")
print(f"Total Loyalty Members (after dirtiness): {len(df_loyalty)} rows")
print(f"Total Customer Profiles (after dirtiness): {len(df_customer_profiles)} rows")
