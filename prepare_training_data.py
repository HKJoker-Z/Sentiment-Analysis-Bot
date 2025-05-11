import pandas as pd
from src.reddit_api import RedditDataFetcher

def prepare_training_data():
    # Fetch Web3-specific posts and comments
    fetcher = RedditDataFetcher()
    keywords = ["web3", "ethereum", "blockchain", "defi", "nft"]
    subreddits = ["web3", "ethereum", "cryptocurrency", "CryptoTechnology"]
    
    # Get posts and comments
    posts_df = fetcher.fetch_posts(keywords, subreddits, limit=100)
    
    if not posts_df.empty:
        post_ids = posts_df["id"].tolist()[:50]
        comments_df = fetcher.fetch_comments(post_ids, limit=20)
        
        # Combine posts and comments text
        posts_text = posts_df[["title", "text"]].apply(lambda x: f"{x['title']} {x['text']}", axis=1)
        training_texts = list(posts_text) + list(comments_df["text"])
        
        # Save raw data for manual labeling
        output_df = pd.DataFrame({"text": training_texts})
        output_df.to_csv("training_data_unlabeled.csv", index=False)
        print(f"Saved {len(output_df)} samples for labeling")
        
        return output_df

if __name__ == "__main__":
    prepare_training_data()