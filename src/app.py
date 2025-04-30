import os
import sys

# Add project root directory to Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime, timedelta

# Import custom modules
from src.reddit_api import RedditDataFetcher
from src.sentiment import SentimentAnalyzer

# Try to check if modules can be found
try:
    import src.reddit_api
    print("Successfully imported reddit_api")
except ImportError as e:
    print(f"Failed to import reddit_api: {e}")

try:
    import src.sentiment
    print("Successfully imported sentiment")
except ImportError as e:
    print(f"Failed to import sentiment: {e}")

# Page configuration
st.set_page_config(
    page_title="Web3 Social Media Sentiment Analyzer",
    page_icon="📊",
    layout="wide",
)

# Add custom CSS styles
st.markdown("""
<style>
    .title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #333;
        margin-bottom: 2rem;
    }
    .stProgress > div > div > div > div {
        background-color: #1E88E5;
    }
    .card {
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_sentiment_analyzer():
    """Load sentiment analysis model (cached to avoid reloading)"""
    return SentimentAnalyzer()

@st.cache_data(ttl=3600)  # Cache for one hour
def fetch_reddit_data(keywords, subreddits, post_limit, include_comments):
    """Fetch Reddit data with caching"""
    fetcher = RedditDataFetcher()
    
    # Get posts
    posts_df = fetcher.fetch_posts(keywords, subreddits, limit=post_limit)
    
    if include_comments and not posts_df.empty:
        # Only get comments for the first 10 posts to avoid API limits
        post_ids = posts_df["id"].tolist()[:10]
        comments_df = fetcher.fetch_comments(post_ids, limit=10)
        return posts_df, comments_df
    
    return posts_df, pd.DataFrame()

def create_sentiment_charts(df, title):
    """Create sentiment analysis visualization charts"""
    if df.empty:
        return None
    
    # Calculate sentiment category distribution
    sentiment_counts = df["sentiment_category"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]
    
    # Create pie chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Pie chart: sentiment distribution
    colors = ["#2ECC71", "#E74C3C", "#3498DB"]
    sentiment_counts["Color"] = colors[:len(sentiment_counts)]
    ax1.pie(
        sentiment_counts["Count"], 
        labels=sentiment_counts["Sentiment"], 
        autopct='%1.1f%%',
        colors=sentiment_counts["Color"].tolist(),
        startangle=90
    )
    ax1.set_title(f"Sentiment Distribution of {title}")
    
    # Histogram: sentiment score distribution
    sns.histplot(df["sentiment_score"], bins=20, kde=True, ax=ax2)
    ax2.set_title(f"Sentiment Score Distribution of {title}")
    ax2.set_xlabel("Sentiment Score")
    ax2.set_ylabel("Frequency")
    
    plt.tight_layout()
    return fig

def main():
    """Main function"""
    # Display title
    st.markdown('<div class="title">Web3 Social Media Sentiment Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Analyze sentiment of Web3 discussions on Reddit</div>', unsafe_allow_html=True)
    
    # Sidebar: user input options
    st.sidebar.header("Analysis Options")
    
    # Keywords input
    keywords_input = st.sidebar.text_area(
        "Enter keywords (one per line)",
        "web3\nethereum\nblockchain\nnft\ndefi",
        help="Enter Web3-related keywords you want to analyze, one per line"
    )
    keywords = [k.strip() for k in keywords_input.split("\n") if k.strip()]
    
    # Preset subreddits list
    default_subreddits = ["web3", "ethereum", "cryptocurrency", "CryptoTechnology", "ethdev"]
    custom_subreddits = st.sidebar.text_area(
        "Subreddits to search (one per line)",
        "\n".join(default_subreddits),
        help="Enter Reddit communities you want to search, one per line"
    )
    subreddits = [s.strip() for s in custom_subreddits.split("\n") if s.strip()]
    
    # Post limit
    post_limit = st.sidebar.slider("Max posts per keyword/subreddit", 1, 25, 5)
    
    # Include comments
    include_comments = st.sidebar.checkbox("Include comment analysis", True)
    
    # Analysis button
    if st.sidebar.button("Start Analysis", type="primary"):
        if not keywords:
            st.error("Please enter at least one keyword")
            return
            
        if not subreddits:
            st.error("Please enter at least one subreddit")
            return
            
        # Display loading progress
        progress_text = "Fetching data and analyzing sentiment..."
        progress_bar = st.progress(0)
        
        # Fetch Reddit data
        with st.spinner("Fetching data from Reddit..."):
            progress_bar.progress(10)
            posts_df, comments_df = fetch_reddit_data(keywords, subreddits, post_limit, include_comments)
            
        if posts_df.empty:
            st.error("Could not fetch any posts. Try different keywords or subreddits")
            return
            
        # Load sentiment analysis model
        progress_bar.progress(30)
        with st.spinner("Loading sentiment analysis model..."):
            analyzer = load_sentiment_analyzer()
            
        # Analyze post sentiment
        progress_bar.progress(50)
        with st.spinner("Analyzing post sentiment..."):
            posts_with_sentiment = analyzer.analyze_dataframe(posts_df, "title")
            
        # Analyze comment sentiment (if available)
        if not comments_df.empty and include_comments:
            progress_bar.progress(70)
            with st.spinner("Analyzing comment sentiment..."):
                comments_with_sentiment = analyzer.analyze_dataframe(comments_df, "text")
        else:
            comments_with_sentiment = pd.DataFrame()
            
        progress_bar.progress(90)
            
        # Display results
        st.header("Analysis Results")
        
        # Display data statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Posts Analyzed", len(posts_with_sentiment))
        with col2:
            st.metric("Comments Analyzed", len(comments_with_sentiment) if not comments_with_sentiment.empty else 0)
        with col3:
            overall_sentiment = "Positive" if posts_with_sentiment["sentiment_category"].value_counts().idxmax() == "positive" else "Negative"
            st.metric("Overall Sentiment", overall_sentiment)
            
        # Create post sentiment charts
        st.subheader("Post Sentiment Analysis")
        posts_chart = create_sentiment_charts(posts_with_sentiment, "Posts")
        if posts_chart:
            st.pyplot(posts_chart)
            
        # Create comment sentiment charts (if available)
        if not comments_with_sentiment.empty:
            st.subheader("Comment Sentiment Analysis")
            comments_chart = create_sentiment_charts(comments_with_sentiment, "Comments")
            if comments_chart:
                st.pyplot(comments_chart)
                
        # Sentiment analysis by subreddit
        st.subheader("Sentiment Analysis by Subreddit")
        subreddit_sentiment = posts_with_sentiment.groupby("subreddit")["sentiment_category"].value_counts().unstack().fillna(0)
        st.bar_chart(subreddit_sentiment)
            
        # Display detailed data tables
        with st.expander("View Detailed Post Data"):
            # Display important columns
            display_columns = ["title", "subreddit", "score", "created_utc", "sentiment_category", "sentiment_score", "permalink"]
            st.dataframe(posts_with_sentiment[display_columns])
            
        if not comments_with_sentiment.empty:
            with st.expander("View Detailed Comment Data"):
                # Display important columns
                display_columns = ["text", "score", "created_utc", "sentiment_category", "sentiment_score", "permalink"]
                st.dataframe(comments_with_sentiment[display_columns])
                
        # Complete progress bar
        progress_bar.progress(100)
        time.sleep(0.5)  # Short delay to show completion effect
        progress_bar.empty()
            
        st.success("Analysis complete! Results shown above.")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info(
        """
        **About this app**
        
        This app uses Reddit API to fetch data and performs sentiment analysis using Hugging Face pre-trained models.
        
        Application: Web3 Social Media Sentiment Analyzer
        """
    )
    
if __name__ == "__main__":
    main()