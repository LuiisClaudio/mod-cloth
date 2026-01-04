import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import CountVectorizer
import re
import plotFunctions as pf
# Set page config
st.set_page_config(page_title="ModCloth Dashboard", layout="wide")
st._config.set_option('theme.base', 'light')
st._config.set_option('theme.primaryColor', '#008080')
st._config.set_option('theme.backgroundColor', '#FFFFFF')
st._config.set_option('theme.secondaryBackgroundColor', '#F0F2F6')
st._config.set_option('theme.textColor', '#262730')


# Custom CSS for bright/clear interface
st.markdown("""
<style>
    /* Main Background force Light */
    .stApp {
        background-color: #FFFFFF;
        color: #262730;
    }
    
    /* Sidebar Background force Light */
    [data-testid="stSidebar"] {
        background-color: #F8F9FA;
        border-right: 1px solid #E6E6E6;
    }

    /* Force all base text in sidebar to dark grey (Fixes Dark Mode White Text) */
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] span, 
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] .stMarkdown {
        color: #262730 !important;
    }
    
    /* Headers (h1, h2, h3) - Teal */
    h1, h2, h3, h4 {
        color: #008080 !important;
        font-family: 'Arial', sans-serif;
        font-weight: 700;
    }
    
    /* Metric Cards */
    [data-testid="stMetricValue"] {
        color: #008080 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #555555 !important;
    }
    
    /* Custom containers */
    .info-box {
        background-color: #F0F2F6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #008080;
        margin-top: 20px;
    }
    
    /* Radio Button Group Label (The Title) */
    .stRadio > label {
        color: #008080 !important;
        font-weight: bold;
    }
    
    /* Radio Button Options (The Items) - Specific fix for navigation visibility */
    .stRadio [data-testid="stMarkdownContainer"] > p {
        color: #262730 !important; 
    }
    
    /* Selectbox Label */
    .stSelectbox label {
        color: #008080 !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

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
    # Sidebar 
    st.sidebar.title("📊 ModCloth Analytics")
    st.sidebar.markdown("---")
    
    # Grouped navigation
    menu_options = {
        "📈 Overview": [
            "Dashboard"
        ],
        "👗 Fit & Sizing": [
            "General Fit Analysis",
            "Sizing Accuracy & Consistency",
            "Length & Height Analysis"
        ],
        "📏 Body Measurements": [
            "Measurement Distributions (Bra & Shoe)",
            "Body Shape Analysis"
        ],
        "⭐ Ratings & Reviews": [
            "Rating Overview & Distributions",
            "Sentiment & Review Text Analysis"
        ],
        "🏷️ Product & Category": [
            "Category Breakdown",
            "Category Treemap",
            "Popularity Head/Tail",
            "Quality vs. Popularity Matrix (Bubble Chart)"
        ],
        "🔍 Advanced Analytics": [
            "Parallel Categories Diagram",
            "Treemap Adjectives (Voice of Customer)"
        ]
    }
    
    # Section selector
    section = st.sidebar.radio("Select Section", list(menu_options.keys()))
    
    st.sidebar.markdown("---")
    
    # Page selector within section
    # Page selector within section
    if len(menu_options[section]) > 1:
        page = st.sidebar.radio("Select Visualization", menu_options[section])
    else:
        page = menu_options[section][0]
    
    # Display section info
    st.sidebar.markdown("---")
    st.sidebar.info(f"**Current Section:** {section}")

    # 1 Overview Page (Combined)
    if section == "📈 Overview":
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
        
        st.markdown("""
        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-top: 20px;">
            <h4 style="color: #1f77b4; margin-top: 0;">💡 What's this about?</h4>
            <p>Think of these cards like the dashboard in your car they give you the most important info at a quick glance. Total reviews, products, users, and average rating. That's it. No clutter.</p>
            <p>We show exactly 4 numbers because that's what your brain can handle quickly. Any more and you'd get overwhelmed. Any less and you wouldn't get the full picture. The average rating shows 2 decimals (like 4.23) because "about 4 stars" isn't specific enough to actually be useful.</p>
            <p>This is perfect when someone asks "How's the business doing?" you can literally answer in 5 seconds just by looking at these numbers.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        
        # Statistical Summary
        st.header("📈 Statistical Summary")
        
        fig_stats = pf.plot_statistical_summary(df)
        st.plotly_chart(fig_stats, use_container_width=True)
        
        st.markdown("""
        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-top: 20px;">
            <h4 style="color: #1f77b4; margin-top: 0;">💡 What's this about?</h4>
            <p>This is basically a health checkup for your data. It shows averages, minimums, maximums, and how spread out everything is all in one place.</p>
            <p>Why a table instead of a fancy chart? Because sometimes you just need the actual numbers. Like, if you want to know the exact median bust size or how much waist measurements vary, a table gives you that faster than any visualization.</p>
            <p>The real value? Spotting weird stuff. If someone entered 999 for their height (obviously wrong), or if certain measurements are all over the place (might mean your sizing system is broken), you'll catch it here.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        
        # Correlation Heatmap
        st.header("📈 Correlation Heatmap")
        
        fig_corr = pf.plot_correlation_heatmap(df)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        st.markdown("""
        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-top: 20px;">
            <h4 style="color: #1f77b4; margin-top: 0;">💡 What's this about?</h4>
            <p>This chart answers: "Which things tend to move together?" Like, do taller people have bigger shoe sizes? Do certain body measurements correlate with better ratings?</p>
            <p>The colors are key red means things go together, blue means they move opposite, white means no relationship. Your brain gets this instantly, way faster than reading a bunch of correlation numbers.</p>
            <p>Why care? If bust and waist are super correlated, maybe you can simplify your sizing. If ratings correlate with certain measurements, you know which body types love (or hate) your products.</p>
        </div>
        """, unsafe_allow_html=True)

    # 2 Fit Analysis - General Fit Analysis
    elif page == "General Fit Analysis":
        st.title("👗 General Fit Analysis")
        
        st.header("Overall Fit Health")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            fig_donut = pf.plot_fit_distribution(df)
            st.plotly_chart(fig_donut, use_container_width=True)
            
        with col2:
            st.markdown("""
            <div class="info-box">
                <h4>Fit Health Score</h4>
                <p><strong>The Donut Chart</strong> gives you the instant high-level picture. If the green slice ("Fit") isn't the majority, you have a systemic issue.</p>
                <p>This is your baseline. Track this number over time. If you change your sizing chart, does the green slice grow?</p>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")
        
        st.header("Fit by Category")
        fig_bar = pf.plot_fit_by_category(df)
        st.plotly_chart(fig_bar, use_container_width=True)
        
        st.markdown("""
        <div class="info-box">
            <h4>Category Breakdown</h4>
            <p>We normalize everything to 100% so you can compare categories fairly. A category with tons of reviews might look "noisy" in a regular chart, but here we just check the <strong>rate</strong> of fit issues.</p>
            <p>Look for the <strong>Red</strong> (Small) or <strong>Orange</strong> (Large) bars dominating. Those are your priority fix lists.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.header("Deep Dive: The 'Emergency Room'")
        fig_heatmap = pf.plot_fit_category_heatmap(df)
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        st.markdown("""
        <div class="info-box">
            <h4>Heatmap Triage</h4>
            <p>This heatmap isolates specific problems. Instead of just "Dresses don't fit," you see "Dresses are specifically too small."</p>
            <p><strong>Actionable Insight:</strong> Use this to instruct your design team. "Widen the hips on Tops" (if Tops are 'Small') or "Shorten the hem on Dresses" (if Dresses are 'Large').</p>
        </div>
        """, unsafe_allow_html=True)

    # 3 Fit Analysis - Sizing Accuracy
    elif page == "Sizing Accuracy & Consistency":
        st.title("👗 Sizing Accuracy & Consistency")
        
        st.header("Sizing Consistency (Strip Plot)")
        fig_strip = pf.plot_sizing_consistency(df)
        st.plotly_chart(fig_strip, use_container_width=True)
        
        st.markdown("""
        <div class="info-box">
            <h4>Are your sizes real?</h4>
            <p>This chart shows individual data points. <strong>Tight clusters</strong> mean your sizing is consistent. <strong>Wide spreads</strong> mean a "Size 8" isn't always a "Size 8".</p>
            <p>Inconsistent categories confuse customers and cause returns. Fix the categories that look like scattered buckshot first.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.header("Customer Choice Analysis")
        fig_scatter = pf.plot_body_measurement_vs_size(df)
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        st.markdown("""
        <div class="info-box">
            <h4>Body Measurement vs. Size Chosen</h4>
            <p>Are customers picking the right size for their body? If you see many disjointed dots (e.g., small measurements picking large sizes), your <strong>Size Chart</strong> might be misleading.</p>
            <p>This helps differentiate between "Bad Product Fit" (production issue) and "Bad User Choice" (education/UI issue).</p>
        </div>
        """, unsafe_allow_html=True)

    # 4 Fit Analysis - Length & Height
    elif page == "Length & Height Analysis":
        st.title("👗 Length & Height Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Feedback by Height Group")
            fig_length = pf.plot_length_vs_height(df)
            st.plotly_chart(fig_length, use_container_width=True)
            
        with col2:
            st.subheader("Height Distribution vs. Opinion")
            fig_box = pf.plot_height_length_boxplot(df)
            st.plotly_chart(fig_box, use_container_width=True)
            
        st.markdown("""
        <div class="info-box">
            <h4>The Petite/Tall Dilemma</h4>
            <p>Do you need specific Petite or Tall lines? These charts answer that.</p>
            <p>If your "Too Long" feedback (Red) comes almost exclusively from users under 5'3", you have a <strong>Petite Market Opportunity</strong>. If everyone complains about length regardless of height, you just have a bad product design.</p>
        </div>
        """, unsafe_allow_html=True)

    # 5 Body Measurements - Distributions
    elif page == "Measurement Distributions (Bra & Shoe)":
        st.title("📏 Measurement Distributions")

        col1, col2 = st.columns(2)
        
        with col1:
            st.header("Bra Size Heatmap")
            fig_bra = pf.plot_bra_size_heatmap(df)
            st.plotly_chart(fig_bra, use_container_width=True)
            
            st.markdown("""
            <div class="info-box">
                <h4>Inventory Planning: Bra Sizes</h4>
                <p>This heatmap identifies your highest volume sizes immediately. The dark spots are where you must never stock out.</p>
                <p>It also reveals "Ghost Sizes" combinations that exist in theory but have zero customers. Don't waste warehouse space on them.</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.header("Shoe Size & Width")
            fig_shoe = pf.plot_shoe_size_distribution(df)
            st.plotly_chart(fig_shoe, use_container_width=True)
            
            st.markdown("""
            <div class="info-box">
                <h4>Inventory Planning: Shoes</h4>
                <p>Shoe sizes follow an almost perfect bell curve. This chart validates if your inventory ratio matches the population bell curve.</p>
                <p>If your sales don't look like this curve, you are either under-stocking the popular sizes (middle) or over-stocking the outliers (ends).</p>
            </div>
            """, unsafe_allow_html=True)

    # 6 Body Measurements - Shape Analysis
    elif page == "Body Shape Analysis":
        st.title("📏 Body Shape Analysis")
        
        st.header("Interactive Measurements Analyzer")
        fig_interact = pf.plot_body_shape_analyzer(df)
        st.plotly_chart(fig_interact, use_container_width=True)
        
        st.markdown("""
        <div class="info-box">
            <h4>Explore the Population</h4>
            <p>Use the dropdown to explore any body dimension (Hips, Waist, Bust, etc.). This allows you to check for skewness or outliers in any specific measurement you are designing for.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Hips vs. Waist Clusters")
            fig_scatter = pf.plot_body_measurements(df)
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        with col2:
            st.subheader("Body Shape Segmentation")
            fig_clusters = pf.plot_body_shape_clustering(df)
            st.plotly_chart(fig_clusters, use_container_width=True)

        st.markdown("""
        <div class="info-box">
            <h4>Beyond Just "Size"</h4>
            <p>Two women can both be "Size 10" but have totally different shapes. These charts cluster users into <strong>shape profiles</strong> (Apple, Pear, Hourglass, Rectangle).</p>
            <p><strong>Strategic Value:</strong> If 40% of your customers are "Pear Shaped" (wider hips), you need to ensure your pants are cut with more curve, regardless of the waist size number.</p>
        </div>
        """, unsafe_allow_html=True)

    # 8 Ratings - Overview
    elif page == "Rating Overview & Distributions":
        st.title("⭐ Rating Overview & Distributions")

        st.header("Rating Distribution")
        fig_hist = pf.plot_rating_distribution(df)
        st.plotly_chart(fig_hist, use_container_width=True)
        
        st.markdown("""
        <div class="info-box">
            <h4>Customer Sentiment Snapshot</h4>
            <p>This checks your "Brand Health". A healthy brand has a "J-curve" (mostly 5 stars, some 4s). If you see a U-shape (lots of 1s and 5s), your product is polarizing.</p>
        </div>
        """, unsafe_allow_html=True)
            
        st.markdown("---")
        
        st.header("Category Performance Comparison")
        fig_box = pf.plot_rating_vs_category(df)
        st.plotly_chart(fig_box, use_container_width=True)
        
        st.markdown("""
        <div class="info-box">
            <h4>Winners & Losers</h4>
            <p>Box plots allow fairness. You can compare the <strong>Quality</strong> of categories, not just their popularity. High medians = High Quality.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # st.markdown("---")
        
        # st.header("Deep Dive: Violin Plot (Top 10)")
        # fig_violin = pf.plot_quality_violin(df)
        # st.plotly_chart(fig_violin, use_container_width=True) 
        
        # st.markdown("""
        # <div class="info-box">
        #     <h4>Consistency Check</h4>
        #     <p>Violin plots show the <strong>shape</strong> of satisfaction. A fat bottom means many unhappy customers. A fat top means distinct excellence.</p>
        # </div>
        # """, unsafe_allow_html=True)



    # 9 Ratings - Sentiment
    elif page == "Sentiment & Review Text Analysis":
        st.title("⭐ Sentiment & Review Text Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("Review Length vs. Rating")
            fig_len = pf.plot_review_length_vs_rating(df)
            st.plotly_chart(fig_len, use_container_width=True)
            
            st.markdown("""
            <div class="info-box">
                <h4>The "Rant vs. Rave" Curve</h4>
                <p>Long reviews usually happen at extremes (1 Star Rants or 5 Star Love Letters). Short reviews in the middle often mean indifference.</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.header("Sentiment Polarity")
            fig_sent = pf.plot_sentiment_polarity(df)
            st.plotly_chart(fig_sent, use_container_width=True)
            
            st.markdown("""
            <div class="info-box">
                <h4>Text vs. Score Alignment</h4>
                <p>Do the words match the stars? This helps identify:</p>
                <ul>
                    <li><strong>Sarcasm:</strong> 5 stars but negative words.</li>
                    <li><strong>User Error:</strong> "Loved it!" but 1 star.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)















    #22 Parallel Categories Diagram Page
    elif page == "Parallel Categories Diagram":
        st.title("🔍 Parallel Categories Diagram")
        
        fig_body_size = pf.plot_parallel_categories(df)
        st.plotly_chart(fig_body_size, use_container_width=True)
        
        st.markdown("""
        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-top: 20px;">
            <h4 style="color: #1f77b4; margin-top: 0;">💡 What's this about?</h4>
            <p>This shows how things flow between categories like category → fit → rating. It's like watching a river split into streams. Thicker flows = more people taking that path.</p>
            <p>The flow design with proportional widths shows volume, while consistent colors help you track specific segments across different dimensions.</p>
            <p>You can spot patterns like "dresses that fit well get 5 stars" or "swimwear that's too small gets 1 star." This tells you exactly where problems are and what causes them.</p>
        </div>
        """, unsafe_allow_html=True)

    #23 Category Treemap Page
    elif page == "Category Treemap":
        st.title("🏷️ Category Treemap")
        
        fig_body_size = pf.plot_category_treemap(df)
        st.plotly_chart(fig_body_size, use_container_width=True)
        
        st.markdown("""
        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-top: 20px;">
            <h4 style="color: #1f77b4; margin-top: 0;">💡 What's this about?</h4>
            <p>Treemap uses space efficiently bigger rectangles = more reviews/products. Color can show a second thing (like average rating), so you get two pieces of info at once.</p>
            <p>You can see your whole portfolio at a glance which categories are biggest AND which deliver best customer satisfaction through the colors.</p>
            <p>This helps you assess your product mix quickly. Not just "what's big" but "what's big AND good quality" (large + good color) vs "what's big but problematic" (large + bad color).</p>
        </div>
        """, unsafe_allow_html=True)

    # Popularity & Categories Re-inserted (Lost in overwrite)
    elif page == "Popularity Head/Tail":
        st.title("🏷️ Popularity Head/Tail")
        fig_body_size = pf.plot_popularity_head_tail(df)
        st.plotly_chart(fig_body_size, use_container_width=True)
        st.markdown('<div class="info-box"><h4>The 80/20 Rule</h4><p>Identify your "Star" products vs. the "Tail".</p></div>', unsafe_allow_html=True)

    elif page == "Category Breakdown":
        st.title("🏷️ Category Breakdown")
        fig_body_size = pf.plot_category_breakdown(df)
        st.plotly_chart(fig_body_size, use_container_width=True)
        st.markdown('<div class="info-box"><h4>Business Mix</h4><p>Understand your portfolio composition.</p></div>', unsafe_allow_html=True)





    #27 Quality vs. Popularity Matrix (Bubble Chart) Page
    elif page == "Quality vs. Popularity Matrix (Bubble Chart)":
        st.title("🏷️ Quality vs. Popularity Matrix (Bubble Chart)")
        
        fig_body_size = pf.plot_quality_popularity_matrix(df)
        st.plotly_chart(fig_body_size, use_container_width=True)
        
        st.markdown("""
        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-top: 20px;">
            <h4 style="color: #1f77b4; margin-top: 0;">💡 What's this about?</h4>
            <p>Shows three things at once: quality (Y axis), popularity (X axis), and category size (bubble size). Creates quadrants like "stars" (high quality + high popularity) or "dogs" (low on both).</p>
            <p>Four quadrant layout is like the famous BCG matrix instantly shows which categories to invest in, maintain, or phase out.</p>
            <p>Strategic view of your portfolio. High quality + growing popularity? Invest there. Established performers? Maintain. Low quality + declining? Maybe time to cut.</p>
        </div>
        """, unsafe_allow_html=True)



    #29 Treemap Adjectives (Voice of Customer) Page
    elif page == "Treemap Adjectives (Voice of Customer)":
        st.title("🔍 Treemap Adjectives (Voice of Customer)")
        
        fig_body_size = pf.plot_treemap_review(df)
        st.plotly_chart(fig_body_size, use_container_width=True)
        
        st.markdown("""
        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-top: 20px;">
            <h4 style="color: #1f77b4; margin-top: 0;">💡 What's this about?</h4>
            <p>Pulls out the most common adjectives from reviews. Word size = how often it's used. Instantly shows you what customers are actually saying.</p>
            <p>Color coding by sentiment (blue = positive words, red = negative) shows both frequency AND emotional tone at once.</p>
            <p>This captures authentic customer language for marketing, reveals what attributes customers actually care about, and shows gaps between what you say about products and what customers think.</p>
        </div>
        """, unsafe_allow_html=True)
