import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import CountVectorizer
import re

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
def plot_correlation_heatmap(df):
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