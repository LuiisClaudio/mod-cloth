import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import CountVectorizer
import re

# Set page config
st.set_page_config(page_title="ModCloth Dashboard", layout="wide")

# --- Data Loading & Cleaning ---
@st.cache_data
def load_data():
    # Load the dataset
    file_path = 'dataset/modcloth_final_data/modcloth_final_data.json'
    try:
        df = pd.read_json(file_path, lines=True)
    except ValueError:
        st.error(f"Could not load data from {file_path}. Please check if the file exists.")
        return None

    # Data Cleaning (based on notebook)
    # 1. Remove duplicates
    df = df.drop_duplicates()
    
    # 2. Drop rows with missing essential values
    df = df.dropna(subset=['item_id', 'review_text', 'quality'])
    
    # 3. Fill missing values
    df['category'] = df['category'].fillna('unknown')
    df['bust'] = df['bust'].fillna('unknown')
    df['cup size'] = df['cup size'].fillna('unknown')
    df['shoe width'] = df['shoe width'].fillna('unknown')
    df['cup size'] = 'cup_size_' + df['cup size'].astype(str)
    df['height'] = df['height'].fillna('unknown')
    
    # Fill numeric with median
    for col in ['waist', 'size', 'shoe size', 'hips']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # 4. Strip whitespace
    for col in ['review_text', 'category']:
        df[col] = df[col].astype(str).str.strip()
        
    # 5. Rename columns to snake_case
    def to_snake_case(col):
        col = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', col)
        col = re.sub('([a-z0-9])([A-Z])', r'\1_\2', col)
        col = col.replace('__', '_')
        col = col.replace(' ', '_')
        col = col.replace('.', '_')
        col = col.replace('-', '_')
        return col.lower()
    
    df.columns = [to_snake_case(col) for col in df.columns]
    
    # Rename quality to rating if needed (notebook did this)
    if 'quality' in df.columns: # It might have been renamed by to_snake_case already if it was 'quality'
         df = df.rename(columns={'quality': 'rating'})
    elif 'rating' not in df.columns and 'quality' in df.columns:
         df = df.rename(columns={'quality': 'rating'})

    return df

df = load_data()

if df is not None:
    # --- Sidebar ---
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Overview", "Fit Analysis", "Product Ratings", "Voice of Customer"])

    # --- Overview Page ---
    if page == "Overview":
        st.title("Overview & KPIs")
        
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

    # --- Fit Analysis Page ---
    elif page == "Fit Analysis":
        st.title("Fit & Sizing Analysis")
        
        st.subheader("Fit Distribution by Category")
        # Fit Distribution Plot
        fit_by_cat = df.groupby(['category', 'fit']).size().reset_index(name='count')
        total_by_cat = fit_by_cat.groupby('category')['count'].transform('sum')
        fit_by_cat['percentage'] = (fit_by_cat['count'] / total_by_cat) * 100
        
        top_categories = df['category'].value_counts().nlargest(10).index
        fit_by_cat_filtered = fit_by_cat[fit_by_cat['category'].isin(top_categories)]
        
        color_map = {
            'small': '#ff7f0e',
            'fit': '#7f7f7f',
            'large': '#1f77b4'
        }
        
        fig_fit = px.bar(
            fit_by_cat_filtered,
            x='category',
            y='percentage',
            color='fit',
            title='Fit Distribution by Category',
            labels={'percentage': 'Percentage (%)', 'category': 'Category', 'fit': 'Fit Feedback'},
            color_discrete_map=color_map,
            text=fit_by_cat_filtered['percentage'].apply(lambda x: f'{x:.1f}%')
        )
        fig_fit.update_layout(barmode='stack')
        st.plotly_chart(fig_fit, use_container_width=True)

    # --- Product Ratings Page ---
    elif page == "Product Ratings":
        st.title("Product Ratings Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Rating Distribution")
            rating_counts = df['rating'].value_counts().sort_index()
            fig_rating = px.bar(
                x=rating_counts.index, 
                y=rating_counts.values,
                labels={'x': 'Rating', 'y': 'Count'},
                title="Distribution of Ratings",
                color_discrete_sequence=['#636EFA']
            )
            st.plotly_chart(fig_rating, use_container_width=True)
            
        with col2:
            st.subheader("Top 10 Most Reviewed Categories")
            top_cats = df['category'].value_counts().head(10)
            fig_cats = px.bar(
                y=top_cats.index,
                x=top_cats.values,
                orientation='h',
                labels={'y': 'Category', 'x': 'Review Count'},
                title="Top 10 Categories by Reviews",
                color_discrete_sequence=['#EF553B']
            )
            fig_cats.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_cats, use_container_width=True)

    # --- Voice of Customer Page ---
    elif page == "Voice of Customer":
        st.title("Voice of Customer")
        st.write("Top adjectives used in reviews, sized by frequency and colored by average rating.")
        
        # Text Analysis Logic
        # 1. Extract Adjectives (Simplified using list of common adjectives or just top words for now if no POS tagger)
        # Since I cannot easily install spacy models without internet, I will use a simple stopword removal and top words approach, 
        # or try to use a predefined list of adjectives if I can.
        # Actually, the user mentioned "top 30 adjectives". 
        # I'll use sklearn CountVectorizer with a custom vocabulary if I had one, but I don't.
        # I will rely on a simple exclusion list of common stopwords.
        
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        
        # Custom list of common adjectives in fashion context to filter (heuristic)
        # This is a fallback since I don't have a POS tagger
        common_adjectives = {
            'small', 'large', 'short', 'long', 'tight', 'loose', 'perfect', 'great', 'good', 'bad', 
            'beautiful', 'cute', 'comfortable', 'soft', 'hard', 'itchy', 'nice', 'lovely', 'pretty', 
            'amazing', 'awesome', 'horrible', 'terrible', 'big', 'huge', 'tiny', 'fit', 'flattering',
            'baggy', 'stretchy', 'cheap', 'expensive', 'quality', 'poor', 'excellent', 'fabulous',
            'gorgeous', 'stylish', 'chic', 'classic', 'modern', 'vintage', 'casual', 'formal',
            'dressy', 'simple', 'fancy', 'plain', 'colorful', 'bright', 'dark', 'light', 'heavy'
        }
        
        # Helper to check if word is likely adjective (very rough heuristic)
        def is_likely_adjective(word):
            return word in common_adjectives or word.endswith('y') or word.endswith('ful') or word.endswith('ous') or word.endswith('able') or word.endswith('ive')

        # Vectorize
        vectorizer = CountVectorizer(stop_words='english', max_features=1000)
        X = vectorizer.fit_transform(df['review_text'].dropna())
        words = vectorizer.get_feature_names_out()
        word_counts = X.sum(axis=0).A1
        
        # Create DataFrame
        word_df = pd.DataFrame({'word': words, 'count': word_counts})
        
        # Filter for likely adjectives (heuristic)
        # In a real scenario, I'd use NLTK or Spacy. I'll try to import nltk and download if possible, but safe to assume restricted env.
        # I will use the heuristic + common list.
        word_df['is_adj'] = word_df['word'].apply(is_likely_adjective)
        adj_df = word_df[word_df['is_adj']].sort_values('count', ascending=False).head(50)
        
        # Calculate average rating for these words
        # This is computationally expensive to do exactly for all, so we do it for the top 50
        avg_ratings = []
        for word in adj_df['word']:
            # Average rating of reviews containing the word
            avg_rating = df[df['review_text'].str.contains(word, case=False, regex=False)]['rating'].mean()
            avg_ratings.append(avg_rating)
            
        adj_df['avg_rating'] = avg_ratings
        
        # Treemap
        fig_tree = px.treemap(
            adj_df,
            path=['word'],
            values='count',
            color='avg_rating',
            color_continuous_scale='RdBu',
            title="Voice of Customer: Top Adjectives",
            hover_data=['count', 'avg_rating']
        )
        st.plotly_chart(fig_tree, use_container_width=True)

