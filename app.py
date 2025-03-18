import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load product data (replace with your dataset path)
products = pd.read_csv('C:\\Users\\Akhil\\Desktop\\NTCC\\Project\\Dataset\\amazon.csv')

# Handle missing values
products['about_product'] = products['about_product'].fillna('')

# Remove non-numeric characters from 'discounted_price' and 'actual_price' columns
products['discounted_price'] = products['discounted_price'].replace('[\₹,]', '', regex=True).astype(float)
products['actual_price'] = products['actual_price'].replace('[\₹,]', '', regex=True).astype(float)

# Initialize a scaler
scaler = MinMaxScaler()

# Scale numerical columns like 'discounted_price' and 'actual_price'
products[['discounted_price', 'actual_price']] = scaler.fit_transform(products[['discounted_price', 'actual_price']])

# Function to preprocess product data
def preprocess_data(products):
    products['features'] = products['about_product'] + ' ' + products['category'] + ' ' + products['discounted_price'].astype(str)
    return products

# Preprocess the data
products = preprocess_data(products)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(products['features'])

# Calculate cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get product recommendations based on content similarity
def content_based_recommendations(product_name, num_recommendations=5):
    idx = products.index[products['product_name'] == product_name][0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]
    
    product_indices = [i[0] for i in sim_scores]
    return products['product_name'].iloc[product_indices].tolist()

# Hybrid recommendation function (combining content-based and user-based methods)
def hybrid_recommendations(user_id, product_name, num_recommendations=5):
    # Implement user-based collaborative filtering or any other method as needed
    
    # Get content-based recommendations
    content_recs = content_based_recommendations(product_name, num_recommendations*2)

    # Further processing and normalization as per your requirement

    return content_recs[:num_recommendations]

# Streamlit UI
def main():
    st.title('Product Recommendation System')
    
    user_id = st.number_input('Enter user ID:', min_value=1)
    product_name = st.text_input('Enter product name:')
    
    if st.button('Get Recommendations'):
        recommendations = hybrid_recommendations(user_id, product_name)
        st.subheader(f'Recommended Products for User {user_id} and Product "{product_name}":')
        for i, recommendation in enumerate(recommendations, 1):
            st.write(f"{i}. {recommendation}")

if __name__ == "__main__":
    main()
