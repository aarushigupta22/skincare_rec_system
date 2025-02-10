import streamlit as st
import pandas as pd
from PIL import Image

# Load dataset (replace with your actual dataset)
df = pd.read_csv("nykaa_products.csv")  # Ensure dataset has columns: name, price, url, skin_type

# Sidebar user inputs
st.sidebar.header("Product Filters")
skin_type = st.sidebar.selectbox("Select Skin Type", df["skin_type"].unique())
price_range = st.sidebar.slider("Select Price Range", int(df["price"].min()), int(df["price"].max()), (500, 2000))

# Filter dataset based on user inputs
filtered_df = df[(df["skin_type"] == skin_type) & (df["price"].between(price_range[0], price_range[1]))]

st.title("Nykaa Product Recommendation")
st.write("Select a product to get personalized recommendations!")

# Display products as selectable images
selected_product = None
cols = st.columns(4)
for idx, row in filtered_df.iterrows():
    with cols[idx % 4]:
        st.image(row["url"], caption=row["name"], use_column_width=True)
        if st.button(f"Select {row['name']}"):
            selected_product = row

# Display recommendations
if selected_product:
    st.subheader("Recommended Products")
    recommended_products = filtered_df.sample(5)  # Placeholder for hybrid recommendation logic
    for _, rec in recommended_products.iterrows():
        st.image(rec["url"], caption=rec["name"], use_column_width=True)
