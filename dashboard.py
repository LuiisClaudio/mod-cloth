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
    # Sidebar 
    st.sidebar.title("📊 ModCloth Analytics")
    st.sidebar.markdown("---")
    
    # Grouped navigation
    menu_options = {
        "📈 Overview": [
            "KPI Cards",
            "Statistical Summary",
            "Correlation Heatmap"
        ],
        "👗 Fit & Sizing": [
            "Fit Distribution by Category",
            "Body Measurement vs. Size Chosen",
            "Length Analysis by Height",
            "Does It Fit? Category Heatmap",
            "Height vs. Length Boxplot",
            "Fit Distribution (Donut Chart)",
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
            "Rating Distribution (Bar Chart)",
            "Review Length vs. Rating",
            "Sentiment Polarity vs. Rating"
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
        
        st.markdown("""
        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-top: 20px;">
            <h4 style="color: #1f77b4; margin-top: 0;">💡 What's this about?</h4>
            <p>Think of these cards like the dashboard in your car they give you the most important info at a quick glance. Total reviews, products, users, and average rating. That's it. No clutter.</p>
            <p>We show exactly 4 numbers because that's what your brain can handle quickly. Any more and you'd get overwhelmed. Any less and you wouldn't get the full picture. The average rating shows 2 decimals (like 4.23) because "about 4 stars" isn't specific enough to actually be useful.</p>
            <p>This is perfect when someone asks "How's the business doing?" you can literally answer in 5 seconds just by looking at these numbers.</p>
        </div>
        """, unsafe_allow_html=True)

    # Statistical Summary Page
    elif page == "Statistical Summary":
        st.title("📈 Statistical Summary")
        
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

    # Correlation Heatmap Page (moved to Overview)
    elif page == "Correlation Heatmap":
        st.title("📈 Correlation Heatmap")
        
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

    # 2 Fit Analysis Page 
    elif page == "Fit Distribution by Category":
        st.title("👗 Fit Distribution by Category")
        
        fig_fit = pf.plot_fit_distribution_by_category(df)
        fig_fit.update_layout(barmode='stack')
        st.plotly_chart(fig_fit, use_container_width=True)
        
        st.markdown("""
        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-top: 20px;">
            <h4 style="color: #1f77b4; margin-top: 0;">💡 What's this about?</h4>
            <p>Each bar is a clothing category (dresses, tops, etc.), split into colored sections showing how many people said "too small," "perfect," or "too big."</p>
            <p>The colors aren't random red for "too small" (problem), green for "perfect" (good), yellow for "too big" (also a problem). Your brain already knows red = bad and green = good, so it feels natural.</p>
            <p>This tells you which categories have sizing problems. If dresses are mostly red, size them up. If jeans are mostly green, you nailed it don't change a thing!</p>
        </div>
        """, unsafe_allow_html=True)

    # 3 Body Measurement vs. Size Chosen Page
    elif page == "Body Measurement vs. Size Chosen":
        st.title("👗 Body Measurement vs. Size Chosen")
        
        fig_body_size = pf.plot_body_measurement_vs_size(df)
        st.plotly_chart(fig_body_size, use_container_width=True)
        
        st.markdown("""
        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-top: 20px;">
            <h4 style="color: #1f77b4; margin-top: 0;">💡 What's this about?</h4>
            <p>Simple question: Are people picking the right size for their body? Each dot is a customer their actual measurement vs. what size they bought.</p>
            <p>If everyone's making good choices, you'd see many grey colors. Dots all over the place? People are confused about sizing.</p>
            <p>The fix: If you see people with certain measurements consistently picking the wrong size, your size chart is probably confusing. Fix the chart = fewer returns = more money.</p>
        </div>
        """, unsafe_allow_html=True)

    # 4 Product Ratings Page
    elif page == "Length Analysis by Height":
        st.title("👗 Length vs. Height Analysis")
        
        fig_body_size = pf.plot_length_vs_height(df)
        st.plotly_chart(fig_body_size, use_container_width=True)
        
        st.markdown("""
        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-top: 20px;">
            <h4 style="color: #1f77b4; margin-top: 0;">💡 What's this about?</h4>
            <p>Does your clothing length work for both short and tall people? We group customers by height and see what they think.</p>
            <p>If short people say "too long" and tall people say "too short," you've got a problem. The visualization shows you if opinions are consistent (good) or all over the place (bad).</p>
            <p>Real talk: Petite and tall versions cost money. This chart tells you if it's worth it. If everyone's happy, save your money. If there's clear frustration at the height extremes, time to expand your offerings.</p>
        </div>
        """, unsafe_allow_html=True)

    # 5 Voice of Customer Page
    elif page == "Bra Size Distribution":
        st.title("📏 Bra Size Distribution")
        
        fig_body_size = pf.plot_bra_size_heatmap(df)
        st.plotly_chart(fig_body_size, use_container_width=True)
        
        st.markdown("""
        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-top: 20px;">
            <h4 style="color: #1f77b4; margin-top: 0;">💡 What's this about?</h4>
            <p>Bra sizes are tricky they have TWO parts (band AND cup). A heatmap lets you see both at once. Darker colors = more customers with that combo.</p>
            <p>Your eye naturally goes to the dark spots, which are your most common sizes. No math needed just look and you know what to stock up on.</p>
            <p>Business wise? This is literally your inventory shopping list. Dark spot at 34B but you're low on stock? You're losing sales. Light spot at 38DD but you're fully stocked? You're wasting money and warehouse space.</p>
        </div>
        """, unsafe_allow_html=True)

    # 6 Shoe Size Distribution Page
    elif page == "Shoe Size Distribution":
        st.title("📏 Shoe Size Distribution")
        
        fig_body_size = pf.plot_shoe_size_distribution(df)
        st.plotly_chart(fig_body_size, use_container_width=True)
        
        st.markdown("""
        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-top: 20px;">
            <h4 style="color: #1f77b4; margin-top: 0;">💡 What's this about?</h4>
            <p>Shoe sizes follow a bell curve most people are in the middle (like 7 8), fewer at the extremes (size 5 or 11). The histogram literally shows this as a bell shape.</p>
            <p>We use continuous bars because sizes are sequential (7 comes before 8). This makes it super easy to see where the "hump" is that's your average customer.</p>
            <p>Stock based on this! If the peak is at size 8, order tons of 8s. Don't over buy size 5 just because "we should have all sizes" the data shows you won't sell many. Stock what people actually want.</p>
        </div>
        """, unsafe_allow_html=True)

    # 7 Body Shape Clustering Page
    elif page == "Body Shape Clustering":
        st.title("📏 Body Shape Clustering")
        
        fig_body_size = pf.plot_body_shape_clustering(df)
        st.plotly_chart(fig_body_size, use_container_width=True)
        
        st.markdown("""
        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-top: 20px;">
            <h4 style="color: #1f77b4; margin-top: 0;">💡 What's this about?</h4>
            <p>Not all size 10s are the same! This uses math to group people by actual body shape pear, apple, hourglass, etc. Each colored blob is a different body type.</p>
            <p>The center of each blob is the "typical" person in that group super useful for designing clothes that actually fit.</p>
            <p>Why bother? If you design everything for one body type, you're alienating tons of customers. This shows you how many different shapes you're dealing with, so you can create better fits for everyone. Plus, you can target marketing like "perfect for pear shapes" and actually mean it!</p>
        </div>
        """, unsafe_allow_html=True)

    #8 Rating vs. Category Page
    elif page == "Rating vs. Category":
        st.title("⭐ Rating vs. Category")
        
        fig_body_size = pf.plot_rating_vs_category(df)
        st.plotly_chart(fig_body_size, use_container_width=True)
        
        st.markdown("""
        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-top: 20px;">
            <h4 style="color: #1f77b4; margin-top: 0;">💡 What's this about?</h4>
            <p>Box plots show way more than just averages. For each category, you see the middle rating (median line), where most ratings fall (the box), and weird outliers (the dots).</p>
            <p>Side by side layout lets you compare instantly. Dresses have a high box? Customers love them. Swimwear has a low box? Needs work.</p>
            <p>This tells you where to focus. Low rated categories need improvement maybe it's quality, maybe it's sizing. High rated categories? Keep doing whatever you're doing there, maybe even expand those lines!</p>
        </div>
        """, unsafe_allow_html=True)

    #9 Popularity Head/Tail Page
    elif page == "Popularity Head/Tail":
        st.title("🏷️ Popularity Head/Tail")
        
        fig_body_size = pf.plot_popularity_head_tail(df)
        st.plotly_chart(fig_body_size, use_container_width=True)
        
        st.markdown("""
        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-top: 20px;">
            <h4 style="color: #1f77b4; margin-top: 0;">💡 What's this about?</h4>
            <p>The 80/20 rule in action usually 20% of your products get 80% of the attention. This chart shows you exactly which products are your stars (the "head") and which are just... there (the "tail").</p>
            <p>Products are ranked from most to least popular, with a line showing cumulative impact. Different colors help you see the cutoff instantly.</p>
            <p>Limited marketing budget? Focus on the head those are your money makers. The tail products might be worth cutting to free up resources, or maybe they're niche items worth keeping for specific customers. Now you actually know which is which.</p>
        </div>
        """, unsafe_allow_html=True)

    #10 Category Breakdown Page
    elif page == "Category Breakdown":
        st.title("🏷️ Category Breakdown")
        
        fig_body_size = pf.plot_category_breakdown(df)
        st.plotly_chart(fig_body_size, use_container_width=True)
        
        st.markdown("""
        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-top: 20px;">
            <h4 style="color: #1f77b4; margin-top: 0;">💡 What's this about?</h4>
            <p>Bar charts answer "what chunk of my business is each category?" Your brain gets bars better than numbers, so seeing is more intuitive"</p>
            <p>This shows your business mix on reviews. Are you mostly dresses reviews? Pretty balanced? If you say you're diversified but 60% is one category, that's risky if trends shift.</p>
        </div>
        """, unsafe_allow_html=True)

    #12 Review Length vs. Rating Page
    elif page == "Review Length vs. Rating":
        st.title("⭐ Review Length vs. Rating")
        
        fig_body_size = pf.plot_review_length_vs_rating(df)
        st.plotly_chart(fig_body_size, use_container_width=True)
        
        st.markdown("""
        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-top: 20px;">
            <h4 style="color: #1f77b4; margin-top: 0;">💡 What's this about?</h4>
            <p>Do people write longer reviews when they're super happy or super mad? Each dot is a review length vs. star rating.</p>
            <p>Colors show the rating, and trend lines help spot patterns. Usually both extremes (1 star and 5 star) inspire longer reviews people either want to rave or rant. The "meh" 3 star reviews? Usually short.</p>
            <p>Why care? Longer reviews help future customers decide. If you're featuring reviews on product pages, prioritize the detailed ones. Also, super short 5 star reviews might be fake, so this helps catch suspicious patterns.</p>
        </div>
        """, unsafe_allow_html=True)

    #13 Sentiment Polarity vs. Rating Page
    elif page == "Sentiment Polarity vs. Rating":
        st.title("⭐ Sentiment Polarity vs. Rating")
        
        fig_body_size = pf.plot_sentiment_polarity(df)
        st.plotly_chart(fig_body_size, use_container_width=True)
        
        st.markdown("""
        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-top: 20px;">
            <h4 style="color: #1f77b4; margin-top: 0;">💡 What's this about?</h4>
            <p>Does what people write match what they rate? We analyze if the review text is positive or negative, then compare to the star rating.</p>
            <p>Dots near the diagonal line = perfect match (positive text + high stars). Dots far away = something weird (like "this is terrible" but 5 stars sarcasm? mistake?).</p>
            <p>This shows you what's really going on beyond the stars. Sometimes people give 3 stars but their text is super positive they're just tough graders. Or you catch fake reviews that don't match up (paid review that says "great!" but accidentally clicked 1 star).</p>
        </div>
        """, unsafe_allow_html=True)

    #15 Does It Fit? Category Heatmap Page
    elif page == "Does It Fit? Category Heatmap":
        st.title("👗 Does It Fit? Category Heatmap")
        
        fig_body_size = pf.plot_fit_category_heatmap(df)
        st.plotly_chart(fig_body_size, use_container_width=True)
        
        st.markdown("""
        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-top: 20px;">
            <h4 style="color: #1f77b4; margin-top: 0;">💡 What's this about?</h4>
            <p>This is your "sizing emergency room" shows which categories are bleeding customers due to fit problems. Each row is a category, columns are fit outcomes (too small/perfect/too big).</p>
            <p>Orange or Yellow spots = danger zones (lots of poor fits). Green spots = healthy (lots of good fits). Your eye naturally jumps to red, which is exactly where you need to focus.</p>
            <p>If swimwear is all yellow in "too small," fix those sizes NOW. If jackets are green in "perfect fit," you nailed it don't mess with success. This helps you triage fix the broken stuff first.</p>
        </div>
        """, unsafe_allow_html=True)

    #16 Height vs. Length Boxplot Page
    elif page == "Height vs. Length Boxplot":
        st.title("👗 Height vs. Length Boxplot")
        
        fig_body_size = pf.plot_height_length_boxplot(df)
        st.plotly_chart(fig_body_size, use_container_width=True)
        
        st.markdown("""
        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-top: 20px;">
            <h4 style="color: #1f77b4; margin-top: 0;">💡 What's this about?</h4>
            <p>Do shorter and taller people feel differently about your garment lengths? We group by height (like under 160 cm, 160–170 cm, over 170 cm) and see what they say.</p>
            <p>Boxes overlapping a lot? Your standard lengths work for everyone great! Boxes totally separate? Short people say "too long," tall people say "too short" you need petite/tall versions.</p>
            <p>The business decision: Petite and tall versions cost money. If this shows a real problem AND you have enough customers at those extremes, it's worth it. If everyone's basically happy, save your money.</p>
        </div>
        """, unsafe_allow_html=True)

    #17 Rating Distribution Histogram Page
    elif page == "Rating Distribution Histogram":
        st.title("⭐ Rating Distribution Histogram")
        
        fig_body_size = pf.plot_rating_distribution_histogram(df)
        st.plotly_chart(fig_body_size, use_container_width=True)
        
        st.markdown("""
        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-top: 20px;">
            <h4 style="color: #1f77b4; margin-top: 0;">💡 What's this about?</h4>
            <p>This shows the overall "mood" of your customers. Are most giving you 5 stars? 3 stars? 1 star? Lines for mean and median help you see the real picture.</p>
            <p>The shape tells a story bell curve means most people are middle of the road. Skewed positive? Mostly happy customers. Two humps? People either love it or hate it, no in between.</p>
            <p>Watch for weird patterns. Way too many 5 stars and nothing else? Might be fake reviews. Mostly 2 3 stars? You've got serious quality problems to fix.</p>
        </div>
        """, unsafe_allow_html=True)

    #18 Rating Distribution (Bar Chart) Page
    elif page == "Rating Distribution (Bar Chart)":
        st.title("⭐ Rating Distribution (Bar Chart)")
        
        fig_body_size = pf.plot_rating_distribution(df)
        st.plotly_chart(fig_body_size, use_container_width=True)
        
        st.markdown("""
        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-top: 20px;">
            <h4 style="color: #1f77b4; margin-top: 0;">💡 What's this about?</h4>
            <p>Instead of grouping, this shows each star rating (1, 2, 3, 4, 5) as its own bar. You can see exactly how many 4 star vs 5 star reviews you have.</p>
            <p>Colors go from red (1 star) to green (5 star) because nobody needs to be told stronger color = good and weaker color = bad. Your brain already knows.</p>
            <p>Quick checks made easy: "Do we have more 5 stars or 1 stars?" "Are people mostly giving 3 stars?" Just look at the bar heights and you know instantly.</p>
        </div>
        """, unsafe_allow_html=True)

    #19 Top Categories (Horizontal Bar Chart) Page
    elif page == "Top Categories (Horizontal Bar Chart)":
        st.title("🏷️ Top Categories (Horizontal Bar Chart)")
        
        fig_body_size = pf.plot_top_categories(df)
        st.plotly_chart(fig_body_size, use_container_width=True)
        
        st.markdown("""
        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-top: 20px;">
            <h4 style="color: #1f77b4; margin-top: 0;">💡 What's this about?</h4>
            <p>Horizontal bars work better with long category names (vertical would be cramped and diagonal). Ranking top to bottom makes it obvious which categories are your rockstars.</p>
            <p>We only show the top ones because all 50+ categories would be overwhelming, and the tiny ones don't matter much anyway.</p>
            <p>This tells you where to spend marketing money. Your top 3 5 categories are where customers are engaging, so that's where ads and new launches should focus. Don't spread yourself too thin.</p>
        </div>
        """, unsafe_allow_html=True)

    #20 Fit Distribution (Donut Chart) Page
    elif page == "Fit Distribution (Donut Chart)":
        st.title("👗 Fit Distribution (Donut Chart)")
        
        fig_body_size = pf.plot_fit_distribution(df)
        st.plotly_chart(fig_body_size, use_container_width=True)
        
        st.markdown("""
        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-top: 20px;">
            <h4 style="color: #1f77b4; margin-top: 0;">💡 What's this about?</h4>
            <p>A donut chart is a pie chart with a hole and that hole is useful! You can put your main number there (like "67% Perfect Fit") in big bold text, while the ring shows the breakdown.</p>
            <p>This is your overall sizing health score. Big green slice? You're killing it. Red or blue dominating? You have systematic sizing problems. Track this over time to see if changes actually help.</p>
        </div>
        """, unsafe_allow_html=True)

    #21 Hips vs. Waist Scatter Plot Page
    elif page == "Hips vs. Waist Scatter Plot":
        st.title("📏 Hips vs. Waist Scatter Plot")
        
        fig_body_size = pf.plot_body_measurements(df)
        st.plotly_chart(fig_body_size, use_container_width=True)
        
        st.markdown("""
        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-top: 20px;">
            <h4 style="color: #1f77b4; margin-top: 0;">💡 What's this about?</h4>
            <p>Shows hip vs. waist measurements. Each dot is a person. The pattern tells you about body shapes diagonal line means hourglass, clusters above = pear shaped, below = more rectangular.</p>
            <p>Colors can show which body shapes get better fits super useful for understanding who your clothes work well for.</p>
            <p>Why designers care: If you just scale a size 4 up to make a size 10, you're assuming everyone has the same proportions they don't! This shows the variety of shapes you need to accommodate. Larger sizes might need proportionally more room in hips, not just bigger everywhere.</p>
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

    #24 Quality Violin Plot Page
    elif page == "Quality Violin Plot":
        st.title("⭐ Quality Violin Plot (TOP 10)")
        
        fig_body_size = pf.plot_quality_violin(df)
        st.plotly_chart(fig_body_size, use_container_width=True)
        
        st.markdown("""
        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-top: 20px;">
            <h4 style="color: #1f77b4; margin-top: 0;">💡 What's this about?</h4>
            <p>Violin plots are like box plots on steroids they show the full shape of your data distribution, not just the summary stats. You can see if ratings are clustered or spread out.</p>
            <p>The symmetrical shape with embedded box plot gives you both detailed distribution AND quick summaries (median, quartiles).</p>
            <p>This reveals if ratings are polarized (two humps people either love or hate) or consensus driven (one hump everyone kinda agrees). Totally different situations that need different fixes.</p>
        </div>
        """, unsafe_allow_html=True)

    #25 Body Shape Analyzer (Interactive Histogram) Page
    elif page == "Body Shape Analyzer (Interactive Histogram)":
        st.title("📏 Body Shape Analyzer (Interactive Histogram)")
        
        fig_body_size = pf.plot_body_shape_analyzer(df)
        st.plotly_chart(fig_body_size, use_container_width=True)
        
        st.markdown("""
        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-top: 20px;">
            <h4 style="color: #1f77b4; margin-top: 0;">💡 What's this about?</h4>
            <p>CHOOOOOOSE YOUR OPTION IN THE FILTER: You can interact with this to explore measurement distributions. Like "show me customers with bust 34 36 AND waist 26 28" the chart updates live.</p>
            <p>Interactive filtering lets you dig into specific segments without creating a million separate charts.</p>
            <p>This helps merchandisers define precise customer segments, find underserved size combinations, and test ideas about sizing by exploring the actual data instead of guessing.</p>
        </div>
        """, unsafe_allow_html=True)

    #26 Fit by Category (100% Stacked Horizontal Bar) Page
    elif page == "Fit by Category (100% Stacked Horizontal Bar)":
        st.title("👗 Fit by Category (100% Stacked Horizontal Bar)")
        
        fig_body_size = pf.plot_fit_by_category(df)
        st.plotly_chart(fig_body_size, use_container_width=True)
        
        st.markdown("""
        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-top: 20px;">
            <h4 style="color: #1f77b4; margin-top: 0;">💡 What's this about?</h4>
            <p>Everything's normalized to 100% so you can compare fit success rates fairly across categories, regardless of how popular each is.</p>
            <p>Horizontal bars fit long category names, and consistent colors (green for perfect fit) let you scan quickly for winners.</p>
            <p>This shows which categories have best/worst fit rates independent of volume. Small categories with great fit won't get overlooked just because they're not popular.</p>
        </div>
        """, unsafe_allow_html=True)

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

    #28 Sizing Consistency (Strip Plot) Page
    elif page == "Sizing Consistency (Strip Plot)":
        st.title("👗 Sizing Consistency (Strip Plot)")
        
        fig_body_size = pf.plot_sizing_consistency(df)
        st.plotly_chart(fig_body_size, use_container_width=True)
        
        st.markdown("""
        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-top: 20px;">
            <h4 style="color: #1f77b4; margin-top: 0;">💡 What's this about?</h4>
            <p>Shows individual data points across categories. Tight clustering = consistent sizing. Wide spread = people can't agree on how it fits.</p>
            <p>Jittered points (slightly offset) prevent dots from stacking on top of each other, so you can see density without losing individual observations.</p>
            <p>This identifies categories with inconsistent sizing where customers are confused. Prioritize fixing those standardization matters most where there's the biggest mess.</p>
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
