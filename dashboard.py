import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import CountVectorizer
import re
import plotFunctions as pf
# Set page config
st.set_page_config(page_title="ModCloth Dashboard", layout="wide")

def load_dataset():
    # Load the dataset from JSON file (JSON Lines format)
    modcloth_final_data = pd.read_json('dataset/modcloth_final_data/modcloth_final_data.json', lines=True)
    return modcloth_final_data

def to_snake_case(col):
        col = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', col)
        col = re.sub('([a-z0-9])([A-Z])', r'\1_\2', col)
        col = col.replace('__', '_')
        col = col.replace(' ', '_')
        col = col.replace('.', '_')
        col = col.replace('-', '_')
        return col.lower()

def clean_datset():
    modcloth_final_data = load_dataset()

    modcloth_final_data.columns = [to_snake_case(col) for col in modcloth_final_data.columns]

    # Data cleaning for modcloth_final_data

    # 1. Remove duplicates
    modcloth_final_data = modcloth_final_data.drop_duplicates()

    # 2. Drop rows with missing essential values (e.g., item_id, review_text, quality)
    modcloth_final_data = modcloth_final_data.dropna(subset=['item_id', 'review_text', 'quality'])

    # 3. Fill missing values in less critical columns with sensible defaults
    modcloth_final_data['category'] = modcloth_final_data['category'].fillna('unknown')
    modcloth_final_data['bust'] = modcloth_final_data['bust'].fillna('unknown')
    modcloth_final_data['cup_size'] = modcloth_final_data['cup_size'].fillna('unknown')
    modcloth_final_data['shoe_width'] = modcloth_final_data['cup_size'].fillna('unknown')
    modcloth_final_data['cup_size'] = 'cup_size_' + modcloth_final_data['cup_size'].astype(str)
    modcloth_final_data['height'] = modcloth_final_data['height'].fillna('unknown')
    modcloth_final_data['waist'] = modcloth_final_data['waist'].fillna(modcloth_final_data['waist'].median())
    modcloth_final_data['size'] = modcloth_final_data['size'].fillna(modcloth_final_data['size'].median())
    modcloth_final_data['shoe_size'] = modcloth_final_data['shoe_size'].fillna(modcloth_final_data['shoe_size'].median())
    modcloth_final_data['hips'] = modcloth_final_data['hips'].fillna(modcloth_final_data['hips'].median())


    # 4. Strip whitespace from string columns
    for col in ['review_text', 'category']:
        modcloth_final_data[col] = modcloth_final_data[col].astype(str).str.strip()

    # 5. Reset index after cleaning
    modcloth_final_data = modcloth_final_data.reset_index(drop=True)

    modcloth_final_data = modcloth_final_data.rename(columns={'quality': 'rating'})

    return modcloth_final_data

df = clean_datset()

if df is not None:
    # --- Sidebar ---
    st.sidebar.title("📊 ModCloth Analytics")
    st.sidebar.markdown("---")
    
    # Grouped navigation
    menu_options = {
        "📈 Overview": [
            "KPI Cards"
        ],
        "👗 Fit & Sizing": [
            "Fit Distribution by Category",
            "Body Measurement vs. Size Chosen",
            "Length Analysis by Height",
            "Does It Fit? Category Heatmap",
            "Height vs. Length Boxplot",
            "Fit Distribution (Donut Chart)",
            "Fit by Category (100% Stacked Horizontal Bar)",
            "Sizing Consistency (Strip Plot)"
        ],
        "📏 Body Measurements": [
            "Bra Size Distribution",
            "Shoe Size Distribution",
            "Body Shape Clustering",
            "Hips vs. Waist Scatter Plot",
            "Body Shape Analyzer (Interactive Histogram)"
        ],
        "⭐ Ratings & Reviews": [
            "Rating vs. Category",
            "Rating Distribution Histogram",
            "Rating Distribution (Bar Chart)",
            "Quality Violin Plot",
            "Review Length vs. Rating",
            "Sentiment Polarity vs. Rating"
        ],
        "🏷️ Product & Category": [
            "Category Breakdown",
            "Top Categories (Horizontal Bar Chart)",
            "Category Treemap",
            "Popularity Head/Tail",
            "Quality vs. Popularity Matrix (Bubble Chart)"
        ],
        "🔍 Advanced Analytics": [
            "Correlation Heatmap",
            "Parallel Categories Diagram",
            "Treemap Adjectives (Voice of Customer)",
            "Sparsity Heatmap (Collaborative Filtering)"
        ]
    }
    
    # Section selector
    section = st.sidebar.radio("Select Section", list(menu_options.keys()))
    
    st.sidebar.markdown("---")
    
    # Page selector within section
    page = st.sidebar.radio("Select Visualization", menu_options[section])
    
    # Display section info
    st.sidebar.markdown("---")
    st.sidebar.info(f"**Current Section:** {section}")

    # 1 KPI Cards Page
    if page == "KPI Cards":
        st.title("📈 Overview & KPIs")
        
        # Category Filter
        category_options = ['all'] + sorted(df['category'].unique())
        selected_category = st.selectbox("Filter by Category", category_options)
        
        if selected_category == 'all':
            filtered_df = df
        else:
            filtered_df = df[df['category'] == selected_category]

        # KPI Cards
        total_reviews = len(filtered_df)
        total_products = filtered_df['item_id'].nunique()
        total_users = filtered_df['user_id'].nunique()
        avg_rating = filtered_df['rating'].mean()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Reviews", total_reviews)
        col2.metric("Unique Products", total_products)
        col3.metric("Unique Users", total_users)
        col4.metric("Average Rating", f"{avg_rating:.2f}")

    # 2 Fit Analysis Page 
    elif page == "Fit Distribution by Category":
        st.title("👗 Fit Distribution by Category")
        fig_fit = pf.plot_fit_distribution_by_category(df)
        fig_fit.update_layout(barmode='stack')
        st.plotly_chart(fig_fit, use_container_width=True)

    # 3 Body Measurement vs. Size Chosen Page
    elif page == "Body Measurement vs. Size Chosen":
        st.title("👗 Body Measurement vs. Size Chosen")
        fig_body_size = pf.plot_body_measurement_vs_size(df)
        st.plotly_chart(fig_body_size, use_container_width=True)

    # 4 Product Ratings Page
    elif page == "Length Analysis by Height":
        st.title("👗 Length vs. Height Analysis") 
        fig_body_size = pf.plot_length_vs_height(df)
        st.plotly_chart(fig_body_size, use_container_width=True)

    # 5 Voice of Customer Page
    elif page == "Bra Size Distribution":
        st.title("📏 Bra Size Distribution") 
        fig_body_size = pf.plot_bra_size_heatmap(df)
        st.plotly_chart(fig_body_size, use_container_width=True)

    # 6 Shoe Size Distribution Page
    elif page == "Shoe Size Distribution":
        st.title("📏 Shoe Size Distribution") 
        fig_body_size = pf.plot_shoe_size_distribution(df)
        st.plotly_chart(fig_body_size, use_container_width=True)
    # 7 Body Shape Clustering Page
    elif page == "Body Shape Clustering":   
        st.title("📏 Body Shape Clustering") 
        fig_body_size = pf.plot_body_shape_clustering(df)
        st.plotly_chart(fig_body_size, use_container_width=True)
    #8 Rating vs. Category Page
    elif page == "Rating vs. Category":
        st.title("⭐ Rating vs. Category") 
        fig_body_size = pf.plot_rating_vs_category(df)
        st.plotly_chart(fig_body_size, use_container_width=True)
    #9 Popularity Head/Tail Page
    elif page == "Popularity Head/Tail":
        st.title("🏷️ Popularity Head/Tail") 
        fig_body_size = pf.plot_popularity_head_tail(df)
        st.plotly_chart(fig_body_size, use_container_width=True)
    #10 Category Breakdown Page
    elif page == "Category Breakdown":
        st.title("🏷️ Category Breakdown") 
        fig_body_size = pf.plot_category_breakdown(df)
        st.plotly_chart(fig_body_size, use_container_width=True)
    #12 Review Length vs. Rating Page
    elif page == "Review Length vs. Rating":
        st.title("⭐ Review Length vs. Rating") 
        fig_body_size = pf.plot_review_length_vs_rating(df)
        st.plotly_chart(fig_body_size, use_container_width=True)
    #13 Sentiment Polarity vs. Rating Page
    elif page == "Sentiment Polarity vs. Rating":
        st.title("⭐ Sentiment Polarity vs. Rating") 
        fig_body_size = pf.plot_sentiment_polarity(df)
        st.plotly_chart(fig_body_size, use_container_width=True)
    #14 Correlation Heatmap Page
    elif page == "Correlation Heatmap":
        st.title("🔍 Correlation Heatmap") 
        fig_body_size = pf.plot_correlation_heatmap(df)
        st.plotly_chart(fig_body_size, use_container_width=True)
    #15 Does It Fit? Category Heatmap Page
    elif page == "Does It Fit? Category Heatmap":
        st.title("👗 Does It Fit? Category Heatmap") 
        fig_body_size = pf.plot_fit_category_heatmap(df)
        st.plotly_chart(fig_body_size, use_container_width=True)
    #16 Height vs. Length Boxplot Page
    elif page == "Height vs. Length Boxplot":
        st.title("👗 Height vs. Length Boxplot") 
        fig_body_size = pf.plot_height_length_boxplot(df)
        st.plotly_chart(fig_body_size, use_container_width=True)
    #17 Rating Distribution Histogram Page
    elif page == "Rating Distribution Histogram":
        st.title("⭐ Rating Distribution Histogram") 
        fig_body_size = pf.plot_rating_distribution_histogram(df)
        st.plotly_chart(fig_body_size, use_container_width=True)
    #18 Rating Distribution (Bar Chart) Page
    elif page == "Rating Distribution (Bar Chart)":
        st.title("⭐ Rating Distribution (Bar Chart)") 
        fig_body_size = pf.plot_rating_distribution(df)
        st.plotly_chart(fig_body_size, use_container_width=True)
    #19 Top Categories (Horizontal Bar Chart) Page
    elif page == "Top Categories (Horizontal Bar Chart)":
        st.title("🏷️ Top Categories (Horizontal Bar Chart)") 
        fig_body_size = pf.plot_top_categories(df)
        st.plotly_chart(fig_body_size, use_container_width=True)
    #20 Fit Distribution (Donut Chart) Page
    elif page == "Fit Distribution (Donut Chart)":
        st.title("👗 Fit Distribution (Donut Chart)") 
        fig_body_size = pf.plot_fit_distribution(df)
        st.plotly_chart(fig_body_size, use_container_width=True)
    #21 Hips vs. Waist Scatter Plot Page
    elif page == "Hips vs. Waist Scatter Plot":
        st.title("📏 Hips vs. Waist Scatter Plot") 
        fig_body_size = pf.plot_body_measurements(df)
        st.plotly_chart(fig_body_size, use_container_width=True)    
    #22 Parallel Categories Diagram Page
    elif page == "Parallel Categories Diagram":
        st.title("🔍 Parallel Categories Diagram") 
        fig_body_size = pf.plot_parallel_categories(df)
        st.plotly_chart(fig_body_size, use_container_width=True)
    #23 Category Treemap Page
    elif page == "Category Treemap":
        st.title("🏷️ Category Treemap") 
        fig_body_size = pf.plot_category_treemap(df)
        st.plotly_chart(fig_body_size, use_container_width=True)
    #24 Quality Violin Plot Page
    elif page == "Quality Violin Plot":
        st.title("⭐ Quality Violin Plot") 
        fig_body_size = pf.plot_quality_violin(df)
        st.plotly_chart(fig_body_size, use_container_width=True)
    #25 Body Shape Analyzer (Interactive Histogram) Page
    elif page == "Body Shape Analyzer (Interactive Histogram)":
        st.title("📏 Body Shape Analyzer (Interactive Histogram)") 
        fig_body_size = pf.plot_body_shape_analyzer(df)
        st.plotly_chart(fig_body_size, use_container_width=True)
    #26 Fit by Category (100% Stacked Horizontal Bar) Page
    elif page == "Fit by Category (100% Stacked Horizontal Bar)":
        st.title("👗 Fit by Category (100% Stacked Horizontal Bar)") 
        fig_body_size = pf.plot_quality_popularity_matrix(df)
        st.plotly_chart(fig_body_size, use_container_width=True)
    #27 Quality vs. Popularity Matrix (Bubble Chart) Page
    elif page == "Quality vs. Popularity Matrix (Bubble Chart)":
        st.title("🏷️ Quality vs. Popularity Matrix (Bubble Chart)") 
        fig_body_size = pf.plot_sizing_consistency(df)
        st.plotly_chart(fig_body_size, use_container_width=True)
    #28 Sizing Consistency (Strip Plot) Page
    elif page == "Sizing Consistency (Strip Plot)":
        st.title("👗 Sizing Consistency (Strip Plot)") 
        fig_body_size = pf.plot_treemap_review(df)
        st.plotly_chart(fig_body_size, use_container_width=True)
    #29 Treemap Adjectives (Voice of Customer) Page
    elif page == "Treemap Adjectives (Voice of Customer)":  
        st.title("🔍 Treemap Adjectives (Voice of Customer)") 
        fig_body_size = pf.plot_treemap_review(df)
        st.plotly_chart(fig_body_size, use_container_width=True)
    #30 Sparsity Heatmap (Collaborative Filtering) Page
    elif page == "Sparsity Heatmap (Collaborative Filtering)":
        st.title("🔍 Sparsity Heatmap (Collaborative Filtering)") 
        fig_body_size = pf.plot_sparsity_heatmap(df)
        st.plotly_chart(fig_body_size, use_container_width=True)
else:
    st.error("Failed to load the dataset. Please check the data source.")
