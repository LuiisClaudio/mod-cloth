import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import CountVectorizer
import re
import ipywidgets as widgets
from IPython.display import display


# 1. KPI Cards
def show_kpi_cards(df, category_filter='all'):
    if category_filter == 'all':
        df = df
    else:
        df = df[df['category'] == category_filter]
    total_reviews = len(df)
    total_products = df['item_id'].nunique()
    total_users = df['user_id'].nunique()
    avg_rating = df['rating'].mean()

    fig = go.Figure()

    fig.add_trace(go.Indicator(
        mode="number",
        value=total_reviews,
        title={"text": "Total Reviews"},
        domain={'row': 0, 'column': 0}
    ))

    fig.add_trace(go.Indicator(
        mode="number",
        value=total_products,
        title={"text": "Unique Products"},
        domain={'row': 0, 'column': 1}
    ))

    fig.add_trace(go.Indicator(
        mode="number",
        value=total_users,
        title={"text": "Unique Users"},
        domain={'row': 0, 'column': 2}
    ))

    fig.add_trace(go.Indicator(
        mode="number",
        value=avg_rating,
        number={"valueformat": ".2f"},
        title={"text": "Average Rating"},
        domain={'row': 0, 'column': 3}
    ))

    fig.update_layout(
        grid={'rows': 1, 'columns': 4, 'pattern': "independent"},
        template="plotly_white",
        height=200
    )
    return fig.show()
# 2. Fit Distribution by Category (Stacked Bar Chart - Percentage)
def plot_fit_distribution_by_category(df):
    """
    Creates a 100% stacked bar chart showing fit feedback distribution by category.
    """
    # Aggregate data
    fit_by_cat = df.groupby(['category', 'fit']).size().reset_index(name='count')
    
    # Calculate percentages
    total_by_cat = fit_by_cat.groupby('category')['count'].transform('sum')
    fit_by_cat['percentage'] = (fit_by_cat['count'] / total_by_cat) * 100
    
    # Get top categories by volume
    top_categories = df['category'].value_counts().nlargest(10).index
    fit_by_cat_filtered = fit_by_cat[fit_by_cat['category'].isin(top_categories)]
    
    # Color mapping (consistent with previous designs)
    color_map = {
        'small': '#ff7f0e',      # Orange
        'fit': '#7f7f7f',        # Grey
        'large': '#1f77b4'       # Blue
    }
    
    fig = px.bar(
        fit_by_cat_filtered,
        x='category',
        y='percentage',
        color='fit',
        title='Fit Distribution by Category: Does Sizing Vary Across Product Types?',
        labels={'percentage': 'Percentage (%)', 'category': 'Category', 'fit': 'Fit Feedback'},
        color_discrete_map=color_map,
        text=fit_by_cat_filtered['percentage'].apply(lambda x: f'{x:.1f}%')
    )
    
    fig.update_layout(
        barmode='stack',
        xaxis=dict(title_font=dict(size=18), tickfont=dict(size=14)),
        yaxis=dict(title_font=dict(size=18), tickfont=dict(size=14), range=[0, 100]),
        title_font=dict(size=24),
        legend_title=dict(font=dict(size=16)),
        legend=dict(font=dict(size=14)),
        hoverlabel=dict(font_size=16),
        height=500
    )
    
    return fig

# 3. Body Measurement vs. Size Chosen (Colored by Fit)
def plot_body_measurement_vs_size(df):
    """
    Creates scatter plots showing body measurements vs. size chosen, colored by fit feedback.
    """
    # Filter valid data
    plot_df = df[(df['hips'] > 0) & (df['size'] > 0)].copy()
    
    # Color mapping
    color_map = {
        'small': '#ff7f0e',
        'fit': '#7f7f7f',
        'large': '#1f77b4'
    }
    
    fig = px.scatter(
        plot_df,
        x='size',
        y='hips',
        color='fit',
        title='Body Measurements vs. Size Chosen: Finding the Right Fit',
        labels={'size': 'Size Chosen', 'hips': 'Hip Measurement (inches)', 'fit': 'Fit Feedback'},
        color_discrete_map=color_map,
        opacity=0.6
    )
    
    fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color='white')))
    
    fig.update_layout(
        title_font=dict(size=24),
        xaxis=dict(title_font=dict(size=18), tickfont=dict(size=14)),
        yaxis=dict(title_font=dict(size=18), tickfont=dict(size=14)),
        legend_title=dict(font=dict(size=16)),
        legend=dict(font=dict(size=14)),
        hoverlabel=dict(font_size=16),
        height=600
    )
    
    return fig

# 4. Length Analysis by Height
def plot_length_vs_height(df):
    """
    Analyzes length feedback against user height to find the ideal height range.
    """
    # Filter valid length data
    plot_df = df[df['length'].notna()].copy()
    
    # Group and count
    length_height = plot_df.groupby(['height', 'length']).size().reset_index(name='count')
    
    # Calculate percentages
    total_by_height = length_height.groupby('height')['count'].transform('sum')
    length_height['percentage'] = (length_height['count'] / total_by_height) * 100
    
    # Focus on "just right" vs others
    length_height['length_category'] = length_height['length'].apply(
        lambda x: 'Just Right' if x == 'just right' else 'Too Long/Short'
    )
    
    # Aggregate by category
    summary = length_height.groupby(['height', 'length_category'])['percentage'].sum().reset_index()
    
    fig = px.bar(
        summary,
        x='height',
        y='percentage',
        color='length_category',
        title='Length Feedback by Height: Finding the Perfect Fit',
        labels={'percentage': 'Percentage (%)', 'height': 'Customer Height', 'length_category': 'Length Feedback'},
        color_discrete_map={'Just Right': '#2ca02c', 'Too Long/Short': '#d62728'},
        barmode='stack'
    )
    
    fig.update_layout(
        xaxis=dict(title_font=dict(size=18), tickfont=dict(size=12), tickangle=-45),
        yaxis=dict(title_font=dict(size=18), tickfont=dict(size=14), range=[0, 100]),
        title_font=dict(size=24),
        legend_title=dict(font=dict(size=16)),
        legend=dict(font=dict(size=14)),
        hoverlabel=dict(font_size=16),
        height=600
    )
    
    return fig

# 5. Bra Size Distribution
def plot_bra_size_heatmap(df):
    """
    Creates a heatmap showing the distribution of cup_size vs. bra_size.
    Identifies the most common body types for inventory planning.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'cup_size' and 'bra_size' columns.
        
    Returns:
        plotly.graph_objects.Figure: The heatmap.
    """
    # Filter out 'unknown' and missing values
    plot_df = df[(df['bra_size'].notna()) & (df['bra_size'] != '')].copy()
    
    # Clean cup_size - remove 'cup_size_' prefix if present
    plot_df['cup_size_clean'] = plot_df['cup_size'].str.replace('cup_size_', '', regex=False).str.upper()
    
    # Convert bra_size to numeric, filter valid ranges (typically 28-48)
    plot_df['bra_size_num'] = pd.to_numeric(plot_df['bra_size'], errors='coerce')
    plot_df = plot_df[(plot_df['bra_size_num'] >= 28) & (plot_df['bra_size_num'] <= 48)]
    
    # Create cross-tabulation
    heatmap_data = pd.crosstab(plot_df['cup_size_clean'], plot_df['bra_size_num'])
    
    # Keep only top cup sizes for clarity
    top_cups = plot_df['cup_size_clean'].value_counts().nlargest(10).index
    heatmap_data = heatmap_data.loc[heatmap_data.index.isin(top_cups)]
    
    # Sort cup sizes logically (A, B, C, D, DD, etc.)
    cup_order = ['A', 'B', 'C', 'D', 'DD', 'DDD/F', 'DDDD/G', 'E', 'F', 'G']
    heatmap_data = heatmap_data.reindex([c for c in cup_order if c in heatmap_data.index])
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns.astype(str),
        y=heatmap_data.index,
        colorscale='YlOrRd',
        hovertemplate='<b>Cup Size:</b> %{y}<br><b>Band Size:</b> %{x}<br><b>Count:</b> %{z}<extra></extra>',
        colorbar=dict(title="Count")
    ))
    
    fig.update_layout(
        title='Bra Size Distribution: Cup Size vs. Band Size',
        title_font=dict(size=24),
        xaxis=dict(
            title='Band Size (inches)',
            title_font=dict(size=18),
            tickfont=dict(size=14)
        ),
        yaxis=dict(
            title='Cup Size',
            title_font=dict(size=18),
            tickfont=dict(size=14)
        ),
        hoverlabel=dict(font_size=16),
        height=600
    )
    
    return fig

# 6. Shoe Size Distribution
def plot_shoe_size_distribution(df):
    """
    Creates histograms showing the distribution of shoe_size and shoe_width.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'shoe_size' and 'shoe_width' columns.
        
    Returns:
        plotly.graph_objects.Figure: Combined subplot figure.
    """
    
    # Filter valid shoe sizes
    plot_df = df[(df['shoe_size'].notna()) & (df['shoe_size'] != '')].copy()
    plot_df['shoe_size_num'] = pd.to_numeric(plot_df['shoe_size'], errors='coerce')
    plot_df = plot_df[(plot_df['shoe_size_num'] >= 4) & (plot_df['shoe_size_num'] <= 13)]
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Shoe Size Distribution', 'Shoe Width Distribution'),
        horizontal_spacing=0.15
    )
    
    # Shoe Size Histogram
    fig.add_trace(
        go.Histogram(
            x=plot_df['shoe_size_num'],
            marker_color='#3366CC',
            opacity=0.75,
            nbinsx=20,
            name='Shoe Size',
            hovertemplate='<b>Size:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Shoe Width Distribution
    width_counts = df['shoe_width'].value_counts().nlargest(10)
    fig.add_trace(
        go.Bar(
            x=width_counts.index,
            y=width_counts.values,
            marker_color='#FF6B6B',
            opacity=0.75,
            name='Shoe Width',
            hovertemplate='<b>Width:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>'
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Shoe Size", title_font=dict(size=16), tickfont=dict(size=12), row=1, col=1)
    fig.update_xaxes(title_text="Shoe Width", title_font=dict(size=16), tickfont=dict(size=12), row=1, col=2)
    fig.update_yaxes(title_text="Count", title_font=dict(size=16), tickfont=dict(size=12), row=1, col=1)
    fig.update_yaxes(title_text="Count", title_font=dict(size=16), tickfont=dict(size=12), row=1, col=2)
    
    fig.update_layout(
        title_text='Shoe Size and Width Distribution',
        title_font=dict(size=24),
        showlegend=False,
        hoverlabel=dict(font_size=16),
        height=500
    )
    
    return fig

#7 Body Shape Clustering
def plot_body_shape_clustering(df):
    """
    Creates a scatter plot using waist and hips as proxy for body shape clustering.
    Attempts to identify body shapes like Hourglass, Pear, Rectangle.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'waist', 'hips', and 'height' columns.
        
    Returns:
        plotly.graph_objects.Figure: The scatter plot with body shape clusters.
    """
    # Filter valid measurements
    plot_df = df[(df['hips'] > 0) & (df['waist'] > 0)].copy()
    
    # Calculate waist-to-hip ratio
    plot_df['waist_hip_ratio'] = plot_df['waist'] / plot_df['hips']
    
    # Simple body shape classification based on waist-to-hip ratio
    def classify_body_shape(ratio):
        if ratio < 0.75:
            return 'Hourglass/Pear'
        elif ratio < 0.85:
            return 'Balanced'
        else:
            return 'Rectangle/Apple'
    
    plot_df['body_shape'] = plot_df['waist_hip_ratio'].apply(classify_body_shape)
    
    # Sample data if too large for performance
    if len(plot_df) > 10000:
        plot_df = plot_df.sample(10000, random_state=42)
    
    fig = px.scatter(
        plot_df,
        x='hips',
        y='waist',
        color='body_shape',
        title='Body Shape Analysis: Waist vs. Hips Measurements',
        labels={'hips': 'Hip Measurement (inches)', 'waist': 'Waist Measurement (inches)', 'body_shape': 'Body Shape'},
        color_discrete_map={
            'Hourglass/Pear': '#E74C3C',
            'Balanced': '#F39C12',
            'Rectangle/Apple': '#3498DB'
        },
        opacity=0.6
    )
    
    fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color='white')))
    
    fig.update_layout(
        title_font=dict(size=24),
        xaxis=dict(title_font=dict(size=18), tickfont=dict(size=14)),
        yaxis=dict(title_font=dict(size=18), tickfont=dict(size=14)),
        legend_title=dict(font=dict(size=16)),
        legend=dict(font=dict(size=14)),
        hoverlabel=dict(font_size=16),
        height=600
    )
    
    return fig

#8 Rating vs. Category (Violin Plot)
def plot_rating_vs_category(df):
    """
    Creates a violin plot showing rating distribution by category.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'category' and 'rating' columns.
        
    Returns:
        plotly.graph_objects.Figure: The violin plot.
    """
    # Filter to top categories for clarity
    top_cats = df['category'].value_counts().nlargest(10).index
    plot_df = df[df['category'].isin(top_cats)]
    
    fig = px.violin(
        plot_df,
        x='category',
        y='rating',
        color='category',
        box=True,  # Show box plot inside
        points='outliers',  # Show outlier points
        title='Rating Distribution by Category',
        labels={'rating': 'Rating', 'category': 'Category'}
    )
    
    fig.update_layout(
        title_font=dict(size=24),
        xaxis=dict(
            title_font=dict(size=18),
            tickfont=dict(size=14),
            tickangle=-45
        ),
        yaxis=dict(
            title_font=dict(size=18),
            tickfont=dict(size=14),
            range=[0, 5.5]
        ),
        showlegend=False,
        hoverlabel=dict(font_size=16),
        height=600
    )
    
    return fig

#9 Popularity Head/Tail
def plot_popularity_head_tail(df):
    """
    Creates a plot showing the count of reviews per item_id to visualize the Pareto Principle.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'item_id' column.
        
    Returns:
        plotly.graph_objects.Figure: The popularity distribution plot.
    """
    # Count reviews per item
    item_counts = df['item_id'].value_counts().reset_index()
    item_counts.columns = ['item_id', 'review_count']
    item_counts = item_counts.sort_values('review_count', ascending=False).reset_index(drop=True)
    item_counts['rank'] = range(1, len(item_counts) + 1)
    
    # Calculate cumulative percentage
    item_counts['cumulative_reviews'] = item_counts['review_count'].cumsum()
    total_reviews = item_counts['review_count'].sum()
    item_counts['cumulative_percentage'] = (item_counts['cumulative_reviews'] / total_reviews) * 100
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add bar chart for review counts
    fig.add_trace(
        go.Bar(
            x=item_counts['rank'],
            y=item_counts['review_count'],
            name='Review Count',
            marker_color='#3366CC',
            opacity=0.7
        ),
        secondary_y=False
    )
    
    # Add line chart for cumulative percentage
    fig.add_trace(
        go.Scatter(
            x=item_counts['rank'],
            y=item_counts['cumulative_percentage'],
            name='Cumulative %',
            line=dict(color='#FF6B6B', width=3),
            mode='lines'
        ),
        secondary_y=True
    )
    
    # Add reference line at 80%
    fig.add_hline(
        y=80,
        line_dash="dash",
        line_color="gray",
        secondary_y=True,
        annotation_text="80% of Reviews",
        annotation_position="right"
    )
    
    fig.update_xaxes(
        title_text="Product Rank",
        title_font=dict(size=18),
        tickfont=dict(size=14)
    )
    
    fig.update_yaxes(
        title_text="Number of Reviews",
        title_font=dict(size=18),
        tickfont=dict(size=14),
        secondary_y=False
    )
    
    fig.update_yaxes(
        title_text="Cumulative Percentage (%)",
        title_font=dict(size=18),
        tickfont=dict(size=14),
        range=[0, 105],
        secondary_y=True
    )
    
    fig.update_layout(
        title='Popularity Head/Tail: Pareto Principle in Product Reviews',
        title_font=dict(size=24),
        hovermode='x unified',
        hoverlabel=dict(font_size=16),
        height=600
    )
    
    return fig

#10 Category Breakdown
def plot_category_breakdown(df):
    """
    Creates a bar chart showing the volume of items per category.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'category' column.
        
    Returns:
        plotly.graph_objects.Figure: The category breakdown chart.
    """
    # Count items per category
    category_counts = df['category'].value_counts().reset_index()
    category_counts.columns = ['category', 'count']
    
    fig = px.bar(
        category_counts,
        x='category',
        y='count',
        title='Category Breakdown: Volume of Reviews by Category',
        labels={'count': 'Number of Reviews', 'category': 'Category'},
        text='count',
        color='count',
        color_continuous_scale='Viridis'
    )
    
    fig.update_traces(
        textposition='outside',
        textfont_size=14,
        marker_line_width=1.5,
        marker_line_color='black'
    )
    
    fig.update_layout(
        title_font=dict(size=24),
        xaxis=dict(
            title_font=dict(size=18),
            tickfont=dict(size=14),
            tickangle=-45
        ),
        yaxis=dict(
            title_font=dict(size=18),
            tickfont=dict(size=14)
        ),
        showlegend=False,
        hoverlabel=dict(font_size=16),
        height=600
    )
    
    return fig

#11 Word Cloud for Reviews
from wordcloud import WordCloud
from textblob import TextBlob
from io import BytesIO
import base64
from plotly.subplots import make_subplots
def generate_wordcloud_image(text, colormap='viridis'):
    """
    Generate a word cloud image and return it as a base64 encoded string.
    
    Args:
        text (str): Combined text for word cloud generation.
        colormap (str): Matplotlib colormap to use.
        
    Returns:
        str: Base64 encoded image string.
    """
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        colormap=colormap,
        max_words=100,
        relative_scaling=0.5,
        min_font_size=10
    ).generate(text)
    
    # Save to bytes buffer
    buffer = BytesIO()
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
    plt.close()
    
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode()
    return f"data:image/png;base64,{img_str}"

def plot_wordclouds(df):
    """
    Creates two word clouds: one for 5-star reviews and one for 1-2 star reviews.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'rating' and 'review_text' columns.
        
    Returns:
        plotly.graph_objects.Figure: Figure with two word cloud images.
    """
    # Filter reviews
    high_rated = df[df['rating'] == 5]['review_text'].str.cat(sep=' ')
    low_rated = df[df['rating'].isin([1, 2])]['review_text'].str.cat(sep=' ')
    
    # Generate word clouds
    img_high = generate_wordcloud_image(high_rated, colormap='Greens')
    img_low = generate_wordcloud_image(low_rated, colormap='Reds')
    
    # Create subplot with images
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('5-Star Reviews: What Customers Love', '1-2 Star Reviews: What Customers Dislike'),
        horizontal_spacing=0.1
    )
    
    fig.add_layout_image(
        dict(
            source=img_high,
            xref="x", yref="y",
            x=0, y=1,
            sizex=1, sizey=1,
            sizing="stretch",
            layer="below"
        ),
        row=1, col=1
    )
    
    fig.add_layout_image(
        dict(
            source=img_low,
            xref="x2", yref="y2",
            x=0, y=1,
            sizex=1, sizey=1,
            sizing="stretch",
            layer="below"
        ),
        row=1, col=2
    )
    
    # Update axes to hide ticks
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, range=[0, 1])
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, range=[0, 1])
    
    fig.update_layout(
        title_text='Word Clouds: Sentiment Analysis of Reviews',
        title_font=dict(size=24),
        height=500,
        showlegend=False
    )
    
    return fig

#12 Review Length vs. Rating
def plot_review_length_vs_rating(df):
    """
    Creates a scatter plot of review character count vs. rating.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'rating' and 'review_text' columns.
        
    Returns:
        plotly.graph_objects.Figure: The scatter plot.
    """
    # Calculate review length
    plot_df = df.copy()
    plot_df['review_length'] = plot_df['review_text'].str.len()
    
    # Sample if too large for performance
    if len(plot_df) > 10000:
        plot_df = plot_df.sample(10000, random_state=42)
    
    fig = px.scatter(
        plot_df,
        x='rating',
        y='review_length',
        color='rating',
        title='Review Length vs. Rating: Do Emotions Drive Verbosity?',
        labels={'rating': 'Rating (Stars)', 'review_length': 'Review Length (Characters)'},
        color_continuous_scale='RdYlGn',
        opacity=0.6
    )
    
    fig.update_traces(
        marker=dict(size=6, line=dict(width=0.5, color='white'))
    )
    
    fig.update_layout(
        title_font=dict(size=24),
        xaxis=dict(
            title_font=dict(size=18),
            tickfont=dict(size=14),
            dtick=1
        ),
        yaxis=dict(
            title_font=dict(size=18),
            tickfont=dict(size=14)
        ),
        hoverlabel=dict(font_size=16),
        height=600
    )
    
    return fig

#13. Sentiment Polarity vs. Rating
def plot_sentiment_polarity(df):
    """
    Calculates sentiment polarity using TextBlob and correlates it with rating.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'rating' and 'review_text' columns.
        
    Returns:
        plotly.graph_objects.Figure: Scatter plot of sentiment vs. rating.
    """
    
    df['sentiment'] = df['review_text'].apply(
        lambda x: TextBlob(str(x)).sentiment.polarity
    )
    
    fig = px.scatter(
        df,
        x='sentiment',
        y='rating',
        color='rating',
        title='Sentiment Polarity vs. Rating: Does Text Match Stars?',
        labels={'sentiment': 'Sentiment Polarity (-1 = Negative, +1 = Positive)', 'rating': 'Rating (Stars)'},
        color_continuous_scale='RdYlGn',
        opacity=0.6,
        trendline='ols'  # Add trend line
    )
    
    fig.update_traces(
        marker=dict(size=8, line=dict(width=0.5, color='white'))
    )
    
    fig.update_layout(
        title_font=dict(size=24),
        xaxis=dict(
            title_font=dict(size=18),
            tickfont=dict(size=14),
            range=[-1.1, 1.1]
        ),
        yaxis=dict(
            title_font=dict(size=18),
            tickfont=dict(size=14),
            dtick=1
        ),
        hoverlabel=dict(font_size=16),
        height=600
    )
    
    return fig

#14 Correlation Heatmap
def plot_correlation_heatmap_numerical(df):
    """
    Creates a correlation heatmap for numerical features.
    
    Args:
        df (pd.DataFrame): DataFrame containing numerical columns.
        
    Returns:
        plotly.graph_objects.Figure: The correlation heatmap.
    """
    # Convert height to inches for correlation analysis
    height_mapping_cm = {
    "4ft 11in": 149.86, "5ft": 152.4, "5ft 1in": 154.94, "5ft 2in": 157.48,
    "5ft 3in": 160.02, "5ft 4in": 162.56, "5ft 5in": 165.1, "5ft 6in": 167.64,
    "5ft 7in": 170.18, "5ft 8in": 172.72, "5ft 9in": 175.26, "5ft 10in": 177.8,
    "5ft 11in": 180.34, "6ft": 182.88, "6ft 1in": 185.42, "6ft 2in": 187.96,
    "6ft 3in": 190.5
    }

    
    df_corr = df.copy()
    df_corr['height_cm'] = df_corr['height'].map(height_mapping_cm)
    
    # Select numerical columns for correlation
    numerical_cols = ['waist', 'hips', 'bra_size', 'size', 'rating', 'height_cm', 'shoe_size']
    
    # Filter only available numerical columns
    available_cols = [col for col in numerical_cols if col in df_corr.columns]
    
    # Compute correlation matrix
    corr_matrix = df_corr[available_cols].corr()
    
    # Create heatmap
    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1,
        title='Multivariate Correlation Heatmap',
        labels={'color': 'Correlation'},
        aspect='auto'
    )
    
    fig.update_layout(
        title_font=dict(size=24),
        xaxis=dict(
            title_font=dict(size=18),
            tickfont=dict(size=14),
            tickangle=-45
        ),
        yaxis=dict(
            title_font=dict(size=18),
            tickfont=dict(size=14)
        ),
        hoverlabel=dict(font_size=16),
        height=600
    )
    
    return fig

#15. "Does It Fit?" Category Heatmap
def plot_fit_category_heatmap(df):
    """
    Creates a heatmap showing fit feedback distribution by category.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'category' and 'fit' columns.
        
    Returns:
        plotly.graph_objects.Figure: The heatmap.
    """
    # Get top categories
    top_cats = df['category'].value_counts().nlargest(10).index
    filtered_df = df[df['category'].isin(top_cats)]
    
    # Create cross-tabulation
    fit_category_crosstab = pd.crosstab(filtered_df['category'], filtered_df['fit'])
    
    fig = px.imshow(
        fit_category_crosstab,
        text_auto=True,
        color_continuous_scale='YlOrRd',
        title='"Does It Fit?" Category Breakdown',
        labels={'x': 'Fit Feedback', 'y': 'Category', 'color': 'Count'},
        aspect='auto'
    )
    
    fig.update_layout(
        title_font=dict(size=24),
        xaxis=dict(
            title_font=dict(size=18),
            tickfont=dict(size=14)
        ),
        yaxis=dict(
            title_font=dict(size=18),
            tickfont=dict(size=14)
        ),
        hoverlabel=dict(font_size=16),
        height=600
    )
    
    return fig

#16. Height vs. Length Boxplot
def plot_height_length_boxplot(df):
    """
    Creates a boxplot showing height distribution by length feedback.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'height' and 'length' columns.
        
    Returns:
        plotly.graph_objects.Figure: The boxplot.
    """
    # Convert height to numerical for boxplot
    height_mapping_cm = {
    "4ft 11in": 149.86, "5ft": 152.4, "5ft 1in": 154.94, "5ft 2in": 157.48,
    "5ft 3in": 160.02, "5ft 4in": 162.56, "5ft 5in": 165.1, "5ft 6in": 167.64,
    "5ft 7in": 170.18, "5ft 8in": 172.72, "5ft 9in": 175.26, "5ft 10in": 177.8,
    "5ft 11in": 180.34, "6ft": 182.88, "6ft 1in": 185.42, "6ft 2in": 187.96,
    "6ft 3in": 190.5
    }
    
    plot_df = df[df['length'].notna()].copy()
    plot_df['height_cm'] = plot_df['height'].map(height_mapping_cm)
    plot_df = plot_df[plot_df['height_cm'].notna()]
    
    fig = px.box(
        plot_df,
        x='length',
        y='height_cm',
        color='length',
        title='Height vs. Length Feedback: Finding the Right Length',
        labels={'height_cm': 'Height (cm)', 'length': 'Length Feedback'},
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    
    fig.update_layout(
        title_font=dict(size=24),
        xaxis=dict(
            title_font=dict(size=18),
            tickfont=dict(size=14),
            tickangle=-45
        ),
        yaxis=dict(
            title_font=dict(size=18),
            tickfont=dict(size=14)
        ),
        showlegend=False,
        hoverlabel=dict(font_size=16),
        height=600
    )
    
    return fig

#17. Rating Distribution Histogram
def plot_rating_distribution_histogram(df):
    """
    Creates a histogram showing the distribution of ratings.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'rating' column.
        
    Returns:
        plotly.graph_objects.Figure: The histogram.
    """
    fig = px.histogram(
        df,
        x='rating',
        nbins=5,
        title='Rating Distribution: Class Imbalance Check',
        labels={'rating': 'Rating (Stars)', 'count': 'Number of Reviews'},
        color='rating',  # Color bars by rating value
        color_discrete_map={
            1: '#d62728',  # Red
            2: '#ff7f0e',  # Orange
            3: '#bcbd22',  # Olive
            4: '#1f77b4',  # Blue
            5: '#2ca02c'   # Green
        }
    )
    
    fig.update_traces(
        marker_line_width=1.5,
        marker_line_color='black',
        hovertemplate='<b>Rating:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>'
    )
    
    fig.update_layout(
        title_font=dict(size=24),
        xaxis=dict(
            title_font=dict(size=18),
            tickfont=dict(size=14),
            dtick=1
        ),
        yaxis=dict(
            title_font=dict(size=18),
            tickfont=dict(size=14)
        ),
        hoverlabel=dict(font_size=16),
        height=600,
        bargap=0.1
    )
    
    return fig

#18. Interactive Rating Distribution Bar Chart
def plot_rating_distribution(df):
    """
    Creates an interactive bar chart for the distribution of ratings.
    
    Args:
        df (pd.DataFrame): DataFrame containing a 'rating' column.
        
    Returns:
        plotly.graph_objects.Figure: The interactive bar chart.
    """
    # Aggregate data
    rating_counts = df['rating'].value_counts().sort_index().reset_index()
    rating_counts.columns = ['rating', 'count']
    
    # Create bar chart
    fig = px.bar(
        rating_counts, 
        x='rating', 
        y='count',
        color='rating', # Distinct colors for 1-5 stars
        title='Distribution of Ratings',
        labels={'rating': 'Rating (1-5 Stars)', 'count': 'Number of Reviews'},
        text='count' # Show count on bars
    )
    
    # Design updates: Large bars, large tooltips
    fig.update_traces(
        hovertemplate='<b>Rating: %{x}</b><br>Count: %{y}<extra></extra>', # Custom tooltip
        textfont_size=14,
        marker_line_width=1.5, 
        marker_line_color='black'
    )
    
    fig.update_layout(
        hoverlabel=dict(
            font_size=16, # Large tooltips
            font_family="Arial"
        ),
        xaxis=dict(
            tickmode='linear',
            dtick=1,
            title_font=dict(size=18),
            tickfont=dict(size=14)
        ),
        yaxis=dict(
            title_font=dict(size=18),
            tickfont=dict(size=14)
        ),
        title_font=dict(size=24),
        showlegend=False # Legend not strictly necessary if color encodes rating
    )
    
    return fig

#19. Top Categories Horizontal Bar Chart
def plot_top_categories(df, color_palette='Viridis'):
    """
    Creates a horizontal bar chart for the top 10 most reviewed categories.

    Args:
        df (pd.DataFrame): DataFrame containing a 'category' column.
        color_palette (str): Name of the Plotly color palette to use.

    Returns:
        plotly.graph_objects.Figure: The horizontal bar chart.
    """
    # Aggregate data
    category_counts = df['category'].value_counts().nlargest(10).reset_index()
    category_counts.columns = ['category', 'count']

    # Create horizontal bar chart
    fig = px.bar(
        category_counts,
        x='count',
        y='category',
        orientation='h',
        title='Top 10 Most Reviewed Categories',
        labels={'count': 'Number of Reviews', 'category': 'Category'},
        text='count',
        color='count',
        color_continuous_scale=color_palette
    )

    # Design updates: Large font for category names
    fig.update_layout(
        yaxis=dict(
            autorange="reversed",
            title_font=dict(size=18),
            tickfont=dict(size=16)
        ),
        xaxis=dict(
            title_font=dict(size=18),
            tickfont=dict(size=14)
        ),
        title_font=dict(size=24),
        hoverlabel=dict(
            font_size=16
        )
    )

    fig.update_traces(
        textposition='outside'
    )

    return fig
#20. "Does It Fit?" Donut Chart
def plot_fit_distribution(df):
    """
    Creates a donut chart for the distribution of 'fit' feedback.
    
    Args:
        df (pd.DataFrame): DataFrame containing a 'fit' column.
        
    Returns:
        plotly.graph_objects.Figure: The donut chart.
    """
    # Aggregate data
    fit_counts = df['fit'].value_counts().reset_index()
    fit_counts.columns = ['fit', 'count']
    
    # Create donut chart
    fig = px.pie(
        fit_counts, 
        values='count', 
        names='fit',
        title='Distribution of "Fit" Feedback',
        hole=0.5, # Donut shape
        color_discrete_sequence=px.colors.qualitative.Bold # Distinct colors
    )
    
    # Design updates: Direct slice labels
    fig.update_traces(
        textposition='inside', 
        textinfo='percent+label',
        textfont_size=16,
        marker=dict(line=dict(color='#000000', width=2))
    )
    
    fig.update_layout(
        title_font=dict(size=24),
        hoverlabel=dict(
            font_size=16
        ),
        showlegend=True
    )
    
    return fig

#21. Hips vs. Waist Scatter Plot
def plot_body_measurements(df):
    """
    Creates a scatter plot of Hips vs. Waist measurements, colored by Size.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'hips', 'waist', and 'size' columns.
        
    Returns:
        plotly.graph_objects.Figure: The scatter plot.
    """
    # Create scatter plot
    fig = px.scatter(
        df, 
        x='waist', 
        y='hips', 
        color='size',
        title='Hips vs. Waist Measurements (Colored by Size)',
        labels={'waist': 'Waist Measurement', 'hips': 'Hips Measurement', 'size': 'Size'},
        color_continuous_scale=px.colors.sequential.Viridis # High contrast if size is numeric, or qualitative if categorical
    )
    
    # Design updates: Large markers
    fig.update_traces(
        marker=dict(
            size=12, # Large markers
            line=dict(
                width=1,
                color='DarkSlateGrey'
            )
        ),
        selector=dict(mode='markers')
    )
    
    fig.update_layout(
        title_font=dict(size=24),
        xaxis=dict(
            title_font=dict(size=18),
            tickfont=dict(size=14)
        ),
        yaxis=dict(
            title_font=dict(size=18),
            tickfont=dict(size=14)
        ),
        hoverlabel=dict(
            font_size=16
        ),
        legend_title=dict(font=dict(size=16)),
        legend=dict(font=dict(size=14))
    )
    
    return fig

#22. Parallel Categories Diagram
def plot_parallel_categories(df):
    """
    Creates a Parallel Categories Diagram to visualize the flow between attributes.
    Focuses on 'cup size' (if available), 'body type' (if available), and 'fit'.
    Falls back to other columns if specific ones are missing.
    
    Args:
        df (pd.DataFrame): DataFrame containing categorical columns.
        
    Returns:
        plotly.graph_objects.Figure: The Parcats diagram.
    """
    # Select relevant columns. Adjust these based on actual column names in the dataset.
    # Common ModCloth columns: 'cup size', 'body type', 'fit', 'category'
    cols_to_try = ['cup size', 'body type', 'fit']
    available_cols = [c for c in cols_to_try if c in df.columns]
    
    if len(available_cols) < 2:
        # Fallback if we don't have enough specific columns
        available_cols = [c for c in ['category', 'fit', 'rating'] if c in df.columns]
        
    if len(available_cols) < 2:
        print("Not enough categorical columns found for Parallel Categories.")
        return go.Figure()

    # Preprocessing: Fill NaNs to avoid dropping data
    plot_df = df[available_cols].fillna('Unknown').copy()
    
    # Create Parcats
    # Map 'fit' to integers for coloring if we want to color by it, or just let Plotly handle it.
    # The error "received value of type <class 'str'>" suggests it tried to interpret 'fit' values as colors directly.
    # We will map fit to numbers for the color scale.
    
    if 'fit' in plot_df.columns:
        fit_map = {val: i for i, val in enumerate(plot_df['fit'].unique())}
        plot_df['fit_id'] = plot_df['fit'].map(fit_map)
        color_col = 'fit_id'
    else:
        color_col = None

    fig = px.parallel_categories(
        plot_df, 
        dimensions=available_cols,
        color=color_col, 
        title='Parallel Categories: Flow of User Attributes to Fit Feedback',
        labels={col: col.title() for col in available_cols},
        color_continuous_scale=px.colors.sequential.Inferno
    )
    
    fig.update_layout(
        title_font=dict(size=24),
        hovermode='closest',
        margin=dict(l=50, r=50, t=50, b=20)
    )
    
    return fig

#23. Category Treemap
def plot_category_treemap(df):
    """
    Creates a Treemap of Category popularity, colored by Average Rating.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'category' and 'rating'.
        
    Returns:
        plotly.graph_objects.Figure: The Treemap.
    """
    # Aggregate data
    # Group by category to get count and mean rating
    cat_stats = df.groupby('category').agg(
        count=('rating', 'count'),
        avg_rating=('rating', 'mean')
    ).reset_index()
    
    # Create Treemap
    fig = px.treemap(
        cat_stats, 
        path=['category'], # Hierarchy (add subcategory if available)
        values='count', 
        color='avg_rating',
        color_continuous_scale='RdBu', # Red (low) to Blue (high)
        color_continuous_midpoint=3.5, # Center divergence at neutral rating
        title='Category Popularity (Size) & Sentiment (Color)',
        hover_data=['avg_rating', 'count']
    )
    
    fig.update_traces(
        textinfo="label+value",
        textfont_size=18,
        hovertemplate='<b>%{label}</b><br>Reviews: %{value}<br>Avg Rating: %{color:.2f}<extra></extra>'
    )
    
    fig.update_layout(
        title_font=dict(size=24),
        margin=dict(t=50, l=10, r=10, b=10)
    )
    
    return fig

#24. Quality Violin Plot
def plot_quality_violin(df):
    """
    Creates a Violin Plot to compare the distribution of 'quality' ratings across categories.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'category' and 'quality'.
        
    Returns:
        plotly.graph_objects.Figure: The Violin plot.
    """
    # Check if 'quality' column exists, otherwise use 'rating'
    y_col = 'quality' if 'quality' in df.columns else 'rating'
    
    # Filter top categories to avoid overcrowding
    top_cats = df['category'].value_counts().nlargest(10).index
    plot_df = df[df['category'].isin(top_cats)]
    
    fig = px.violin(
        plot_df, 
        y=y_col, 
        x='category', 
        color='category',
        box=True, # Overlay box plot
        points="all", # Show all points (or "outliers" if too many)
        hover_data=df.columns,
        title=f'Distribution of {y_col.title()} by Category (Top 10)',
    )
    
    fig.update_layout(
        title_font=dict(size=24),
        xaxis=dict(title_font=dict(size=18), tickfont=dict(size=14)),
        yaxis=dict(title_font=dict(size=18), tickfont=dict(size=14)),
        showlegend=False # Redundant since x-axis labels are clear
    )
    
    return fig

#25. Body Shape Analyzer with Dropdown
def plot_body_shape_analyzer(df):
    """
    Creates an interactive histogram with a dropdown to switch between body measurements.
    
    Args:
        df (pd.DataFrame): DataFrame containing body measurement columns.
        
    Returns:
        plotly.graph_objects.Figure: The interactive histogram.
    """
    # Define columns to analyze
    measurements = ['waist', 'hips', 'bust', 'bra_size']
    measurements = ['bra_size', 'hips', 'bust', 'waist']
    available_measurements = [col for col in measurements if col in df.columns]
    
    if not available_measurements:
        print("No body measurement columns found (waist, hips, bust, bra_size).")
        return go.Figure()

    fig = go.Figure()

    # Add a trace for each measurement (initially only the first one is visible)
    for col in available_measurements:
        fig.add_trace(
            go.Histogram(
                x=df[col],
                name=col.replace('_', ' ').title(),
                visible=(col == available_measurements[0]), # Only first visible initially
                marker_color='#3366CC',
                opacity=0.75,
                hovertemplate=f'<b>{col.title()}:</b> %{{x}}<br><b>Count:</b> %{{y}}<extra></extra>'
            )
        )

    # Create Dropdown Buttons
    buttons = []
    for i, col in enumerate(available_measurements):
        # Create a visibility list: [False, False, ..., True, ..., False]
        visibility = [False] * len(available_measurements)
        visibility[i] = True
        
        button = dict(
            label=col.replace('_', ' ').title(),
            method="update",
            args=[{"visible": visibility},
                  {"title": f"Distribution of {col.replace('_', ' ').title()} Measurements",
                   "xaxis.title": f"{col.replace('_', ' ').title()} (inches)"}]
        )
        buttons.append(button)

    # Layout Updates
    fig.update_layout(
        title=f"Distribution of {available_measurements[0].replace('_', ' ').title()} Measurements",
        title_font=dict(size=24),
        xaxis=dict(
            title=f"{available_measurements[0].replace('_', ' ').title()} (inches)",
            title_font=dict(size=18),
            tickfont=dict(size=14)
        ),
        yaxis=dict(
            title="Count",
            title_font=dict(size=18),
            tickfont=dict(size=14)
        ),
        updatemenus=[
            dict(
                active=0,
                buttons=buttons,
                x=1.15, # Position to the right
                y=1.15,
                xanchor='right',
                yanchor='top',
                font=dict(size=14),
                bgcolor="white",
                bordercolor="black",
                borderwidth=1
            )
        ],
        bargap=0.1, # Gap between bars
        hoverlabel=dict(
            font_size=16
        )
    )
    
    return fig

#26. "Does It Fit?" Stacked Bar Chart by Category
def plot_fit_by_category(df):
    """
    Creates a 100% Stacked Horizontal Bar Chart showing 'fit' feedback by category.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'category' and 'fit' columns.
        
    Returns:
        plotly.graph_objects.Figure: The stacked bar chart.
    """
    # Aggregate data
    # Group by category and fit, then count
    fit_counts = df.groupby(['category', 'fit']).size().reset_index(name='count')
    
    # Calculate percentages per category
    total_counts = fit_counts.groupby('category')['count'].transform('sum')
    fit_counts['percentage'] = fit_counts['count'] / total_counts
    
    # Sort categories by total count or some other metric if desired, 
    # but usually sorting by one of the fit percentages (e.g. % fit) looks best.
    # For now, let's sort by total volume of reviews to keep popular ones at top.
    category_volume = df['category'].value_counts().reset_index()
    category_volume.columns = ['category', 'total']
    top_categories = category_volume.nlargest(10, 'total')['category'].tolist() # Limit to top 10 for readability
    
    filtered_data = fit_counts[fit_counts['category'].isin(top_categories)]
    
    # Define Color Palette (Blue=Large, Grey=Fit, Orange=Small)
    # Note: Ensure the mapping matches the actual values in 'fit' column.
    # Assuming values are 'Small', 'Fit', 'Large' (case sensitive check needed usually)
    color_map = {
        'Small': '#ff7f0e',  # Orange
        'Fit': '#7f7f7f',    # Grey
        'Large': '#1f77b4',  # Blue
        'slightly small': '#ffbb78', # Lighter Orange fallback
        'slightly large': '#aec7e8'  # Lighter Blue fallback
    }
    
    fig = px.bar(
        filtered_data,
        y='category',
        x='percentage',
        color='fit',
        orientation='h',
        title='Does It Fit? Breakdown by Category (Top 10)',
        labels={'percentage': 'Percentage', 'category': 'Category', 'fit': 'Fit Feedback'},
        color_discrete_map=color_map,
        category_orders={'category': top_categories} # Keep order
    )
    
    # Design Updates
    fig.update_layout(
        barmode='stack',
        xaxis=dict(
            tickformat='.0%', # Show as percentage
            title_font=dict(size=18),
            tickfont=dict(size=14)
        ),
        yaxis=dict(
            title_font=dict(size=18),
            tickfont=dict(size=14),
            autorange="reversed" # Top category at top
        ),
        title_font=dict(size=24),
        legend_title=dict(font=dict(size=16)),
        legend=dict(font=dict(size=14)),
        hoverlabel=dict(font_size=16)
    )
    
    fig.update_traces(
        hovertemplate='<b>%{y}</b><br>Fit: %{data.name}<br>Percentage: %{x:.1%}<extra></extra>'
    )
    
    return fig

#27. Quality vs. Popularity Matrix (Bubble Chart)
def plot_quality_popularity_matrix(df):
    """
    Creates a Bubble Chart plotting Categories based on Average Rating vs. Number of Reviews.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'category' and 'rating' columns.
        
    Returns:
        plotly.graph_objects.Figure: The bubble chart.
    """
    # Aggregate data
    cat_stats = df.groupby('category').agg(
        avg_rating=('rating', 'mean'),
        review_count=('rating', 'count')
    ).reset_index()
    
    # Create Bubble Chart
    fig = px.scatter(
        cat_stats,
        x='avg_rating',
        y='review_count',
        size='review_count', # Bubble size proportional to popularity
        color='category', # Color by category for distinction
        hover_name='category',
        title='Quality vs. Popularity Matrix (Bubble Chart)',
        labels={'avg_rating': 'Average Rating (Quality)', 'review_count': 'Number of Reviews (Popularity)'},
        size_max=60 # Ensure bubbles are large enough
    )
    
    # Design Updates
    fig.update_traces(
        marker=dict(
            line=dict(width=2, color='DarkSlateGrey'),
            opacity=0.8
        ),
        selector=dict(mode='markers')
    )
    
    fig.update_layout(
        title_font=dict(size=24),
        xaxis=dict(
            title_font=dict(size=18),
            tickfont=dict(size=14),
            range=[1, 5.2] # Focus on valid rating range, slightly extended
        ),
        yaxis=dict(
            title_font=dict(size=18),
            tickfont=dict(size=14)
        ),
        hoverlabel=dict(
            font_size=16
        ),
        showlegend=False # Legend might be too long if many categories
    )
    
    # Add annotations for quadrants (optional but helpful for "Matrix" concept)
    # We can add lines for global averages
    avg_rating_global = df['rating'].mean()
    avg_count_global = cat_stats['review_count'].mean()
    
    fig.add_hline(y=avg_count_global, line_dash="dash", line_color="gray", annotation_text="Avg Popularity")
    fig.add_vline(x=avg_rating_global, line_dash="dash", line_color="gray", annotation_text="Avg Quality")
    
    return fig

#28. Sizing Consistency Strip Plot
def plot_sizing_consistency(df):
    """
    Creates a Strip Plot showing individual purchases grouped by Category and Size.
    Visualizes data density and sparsity.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'category' and 'size' columns.
        
    Returns:
        plotly.graph_objects.Figure: The strip plot.
    """
    # Filter to top categories to avoid overcrowding if too many
    top_cats = df['category'].value_counts().nlargest(10).index
    plot_df = df[df['category'].isin(top_cats)].copy()
    
    # Ensure size is numeric if possible for better Y-axis scaling, 
    # or keep as categorical if mixed. ModCloth sizes are often numeric (4, 6, 8...).
    # We'll try to convert to numeric for sorting, but keep as is if it fails.
    try:
        plot_df['size_num'] = pd.to_numeric(plot_df['size'])
        y_col = 'size_num'
    except:
        y_col = 'size'
    
    fig = px.strip(
        plot_df, 
        x='category', 
        y=y_col, 
        title='Sizing Consistency: Data Density by Category & Size',
        labels={'category': 'Category', y_col: 'Size'},
        color_discrete_sequence=['#000000'] # High contrast dark dots
    )
    
    # Design Updates
    fig.update_traces(
        marker=dict(
            size=4, 
            opacity=0.5 # Slight transparency to show density overlap
        ),
        jitter=0.7 # Spread points out
    )
    
    fig.update_layout(
        title_font=dict(size=24),
        xaxis=dict(
            title_font=dict(size=18),
            tickfont=dict(size=14),
            showgrid=False # Minimalist: No gridlines
        ),
        yaxis=dict(
            title_font=dict(size=18),
            tickfont=dict(size=14),
            showgrid=False,
            zeroline=False
        ),
        plot_bgcolor='white', # White background
        hovermode='closest'
    )
    
    return fig

#29. Treemap of Adjective Frequency and Sentiment
import spacy
def plot_treemap_review(df):

    # Adjective Extraction 

    nlp = spacy.load("en_core_web_sm")
    nlp = None

    adjective_data = []

    if nlp:
        # Process in batches if data is large, but for this example we iterate
        # Using nlp.pipe is faster for large datasets
        #TOO SLOW - So we will use a "mock" extraction -> Im forcing the ESLE to happen
        for doc, rating in zip(nlp.pipe(df['review_text']), df['rating']):
            for token in doc:
                if token.pos_ == "ADJ":
                    adjective_data.append({
                        'word': token.text.lower(), # Normalize to lowercase
                        'rating': rating
                    })
    else:
        # Using this way once it is faster
        known_adjectives = ["perfect", "soft", "tight", "cheap", "comfortable","big",
    "good",
    "small",
    "hot",
    "cold",
    "expensive",
    "difficult",
    "easy",
    "new",
    "old",
    "beautiful",
    "funny",
    "friendly",
    "happy",
    "sad",
    "strong",
    "weak",
    "fast",
    "slow",
    "bright"]
        for index, row in df.iterrows():
            for word in row['review_text'].lower().replace('.', '').split():
                if word in known_adjectives:
                    adjective_data.append({'word': word, 'rating': row['rating']})

    if not adjective_data:
        print("No adjectives found. Exiting.")
        return

    adj_df = pd.DataFrame(adjective_data)

    #3. Aggregation
    print("Aggregating data")
    # Group by word to get Frequency and Average Rating
    summary = adj_df.groupby('word').agg(
        Frequency=('word', 'count'),
        Average_Rating=('rating', 'mean')
    ).reset_index()

    # Filter for top 30 adjectives by frequency
    top_30_summary = summary.sort_values(by='Frequency', ascending=False).head(30)

    #4. Visualization (Treemap)
    print("Creating visualization")
    fig = px.treemap(
        top_30_summary,
        path=['word'], # Hierarchy (just words here)
        values='Frequency', # Size of the tile
        color='Average_Rating', # Color of the tile
        color_continuous_scale='RdBu', # Red to Blue (Low to High)
        range_color=[1, 5], # Fixed range for ratings (1 to 5 stars)
        title='Voice of the Customer: Top Adjectives by Frequency and Rating',
        hover_data={
            'word': False,
            'Frequency': True,
            'Average_Rating': ':.2f' # Format to 2 decimal places
        }
    )

    # Design Aesthetics for 65+ 
    fig.update_layout(
        font=dict(size=16, family="Arial"), # Larger font, readable family
        margin=dict(t=50, l=25, r=25, b=25),
        coloraxis_colorbar=dict(
            title="Avg Rating",
            tickvals=[1, 2, 3, 4, 5],
            ticktext=["1 Star", "2 Stars", "3 Stars", "4 Stars", "5 Stars"]
        )
    )

    # Ensure text is horizontal and informative
    fig.update_traces(
        textinfo="label+value", # Show Label and Frequency
        textfont=dict(size=20),
        hovertemplate='<b>%{label}</b><br>Used in %{value} reviews<br>Average Rating: %{color:.1f} Stars<extra></extra>'
    )

    return fig

#30. Statistical Summary Table
def plot_statistical_summary(df):
    """
    Visualizes df.describe() as a polished Plotly table.
    """
    # 1. Calculate describe() and reset index to make 'count', 'mean', etc., a column
    desc_df = df.describe().reset_index()
    
    # 2. Round numeric columns for readability
    for col in desc_df.columns:
        if col != 'index':
            desc_df[col] = desc_df[col].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)

    # 3. Create the Table
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['<b>Statistic</b>'] + [f'<b>{col.replace("_", " ").title()}</b>' for col in desc_df.columns if col != 'index'],
            fill_color='teal',
            align='left',
            font=dict(color='white', size=14),
            height=40
        ),
        cells=dict(
            values=[desc_df['index']] + [desc_df[col] for col in desc_df.columns if col != 'index'],
            fill_color='white',
            align='left',
            font=dict(color='black', size=12),
            height=30,
            line_color='lightgrey'
        )
    )])

    fig.update_layout(
        title_text="Dataset Statistical Summary (df.describe)",
        title_font=dict(size=24),
        height=400,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    return fig

#31. Correlation Heatmap
def plot_correlation_heatmap(df):
    """
    Visualizes the correlation matrix of numerical features.
    """
    # Select numerical columns of interest
    cols = ['waist', 'hips', 'bra_size', 'size', 'rating', 'shoe_size']
    
    # Calculate correlation matrix
    corr_matrix = df[cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto='.2f', # Show values with 2 decimal places
        aspect="auto",
        color_continuous_scale='RdBu_r', # Red-Blue diverging scale
        origin='lower',
        title='Correlation Matrix: Relationships between Features'
    )
    
    fig.update_layout(
        title_font=dict(size=24),
        width=700,
        height=700
    )
    
    return fig

