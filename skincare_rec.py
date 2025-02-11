import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import random

def load_data():
    df = pd.read_csv("processed_nykaa_products.csv")
    return df

def filter_products(df, min_price, max_price, selected_skin_types, excluded_brands):
    # Filter by price range
    df = df[(df['Price'] >= min_price) & (df['Price'] <= max_price)]
    # Filter by skin type
    if selected_skin_types:
        df = df[df[selected_skin_types].sum(axis=1) > 0]
    # Remove unwanted brands
    if excluded_brands:
        df = df[~df['Brand'].isin(excluded_brands)]
    return df

def get_recommendation(accepted_products, rejected_products, filtered_df, cosine_sim_df, top_n=5):
    recommended_products = []
    final_recommendations = []
    seen_products = set(accepted_products + rejected_products)  
    for product_name in accepted_products:
        if product_name not in filtered_df["Name"].values:
            continue  # Skip if product not found
        idx = filtered_df[filtered_df["Name"] == product_name].index[0]
        sim_scores = list(enumerate(cosine_sim_df.iloc[idx].values))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        top_indices = [i[0] for i in sim_scores if filtered_df.iloc[i[0]]["Name"] not in seen_products][:top_n]
        recommended_products.extend(filtered_df.iloc[top_indices][["Name", "Price", "URL", "Image"]].to_dict(orient="records"))
        seen_products.update(filtered_df.iloc[top_indices]["Name"].tolist())
    for product in recommended_products:
        if product["Name"] in rejected_products:
            continue 
        final_recommendations.append(product)
    final_recommendations = final_recommendations[:top_n]
    return pd.DataFrame(final_recommendations)

def main():
    st.title("Nykaa Product Recommender")
    df = load_data()
    min_price, max_price = st.slider("Select Price Range", 0, int(df['Price'].max()), (100, 1000))
    
    # User selects skin type
    skin_types = ['normal', 'dry', 'oily', 'sensitive', 'combination']
    selected_skin_types = st.multiselect("What is your skin type?", skin_types)
    
    # User selects brands to exclude
    brands = df['Brand'].unique()
    excluded_brands = st.multiselect("Any Brand you don't like? Exclude them from the recommendation system!", brands)
    
    filtered_df = filter_products(df, min_price, max_price, selected_skin_types, excluded_brands)
    # Compute cosine similarity between products based on TF-IDF features
    tfidf_features = filtered_df.iloc[:, 11:]  # Columns after 'Brand' (TF-IDF keywords)
    cosine_sim = cosine_similarity(tfidf_features)
    # Convert to DataFrame for easy lookup
    cosine_sim_df = pd.DataFrame(cosine_sim, index=filtered_df["Cleaned_Name"], columns=filtered_df["Cleaned_Name"])
    image_filtered_df = filtered_df[filtered_df["Image"].notna() & (filtered_df["Image"] != "N/A")]

    st.title("Select Products You Like And Reject the ones you Don't!")
    if "current_index" not in st.session_state:
        st.session_state.current_index = random.randint(0,10)
    if "accepted" not in st.session_state:
        st.session_state.accepted = []
    if "rejected" not in st.session_state:
        st.session_state.rejected = []
    if st.session_state.current_index < len(image_filtered_df):
        row = image_filtered_df.iloc[st.session_state.current_index]
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(row['Image'], width=100)
        with col2:
            st.write(f"**{row['Name']}**")
            st.write(f"₹ {row['Price']}")
            
        colA, colB = st.columns(2)
        with colA:
            if st.button("✅ Accept"):
                st.session_state.accepted.append(row["Name"])
                st.session_state.current_index += 1
        with colB:
            if st.button("❌ Reject"):
                st.session_state.rejected.append(row["Name"])
                st.session_state.current_index += 1
    else:
        st.success("You have selected enough products! Generating recommendations...")


    if len(st.session_state.accepted) >= 5:
        st.success("You have selected enough products! Products recommended for you are...")
        recommendations = get_recommendation(
            st.session_state.accepted,
            st.session_state.rejected,
            filtered_df,
            cosine_sim_df
        )
        if recommendations.empty:  # ✅ Proper way to check if DataFrame is empty
            st.write("⚠️ No recommendations available.")
        for _, row in recommendations.iterrows():
            col1, col2 = st.columns([1, 3])
            with col1:
                image_url = row["Image"] if isinstance(row["Image"], str) and row["Image"] != "N/A" else None
                if image_url:
                    st.image(image_url, width=100)
                else:
                    st.write("No Image Available*")
            with col2:
                st.write(f"**{row['Name']}**")
                st.write(f"₹{row['Price']}  |  [ View Product on the website!]({row['URL']})")
    if st.button("🔄 Retry"):
        st.session_state.clear()  # Reset all selections
        st.rerun()

if __name__ == "__main__":
    main()