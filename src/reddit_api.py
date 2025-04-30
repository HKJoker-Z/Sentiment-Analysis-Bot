import os
import praw
import time
import urllib.parse
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd

# 加载环境变量
load_dotenv()

class RedditDataFetcher:
    def __init__(self):
        """初始化Reddit API连接"""
        self.client_id = os.getenv("REDDIT_CLIENT_ID")
        self.client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        
        # 使用硬编码的用户代理，确保只包含ASCII字符
        self.user_agent = "Web3SentimentAnalyzer/0.1 by Researcher"
        
        if not all([self.client_id, self.client_secret]):
            raise ValueError("缺少Reddit API凭证，请检查.env文件")
            
        # 初始化Reddit API客户端
        self.reddit = praw.Reddit(
            client_id=self.client_id,
            client_secret=self.client_secret,
            user_agent=self.user_agent
        )
        
    def fetch_posts(self, keywords, subreddits=None, limit=50):
        """
        获取与关键词相关的Reddit帖子
        
        参数:
            keywords (list): 关键词列表
            subreddits (list): 要搜索的子版块列表，如果为None则在所有版块搜索
            limit (int): 每个子版块/关键词组合要获取的最大帖子数
            
        返回:
            pandas.DataFrame: 包含帖子数据的DataFrame
        """
        if subreddits is None:
            # 默认web3相关子版块
            subreddits = ["web3", "ethereum", "cryptocurrency"]  # 移除了有问题的子版块
        
        all_posts = []
        
        for subreddit_name in subreddits:
            try:
                # 添加延迟以避免API速率限制
                time.sleep(1)
                
                subreddit = self.reddit.subreddit(subreddit_name)
                
                for keyword in keywords:
                    try:
                        # 确保关键词只包含ASCII字符
                        safe_keyword = keyword.encode('ascii', errors='ignore').decode()
                        
                        # 关键词不能为空
                        if not safe_keyword:
                            safe_keyword = "web3"
                        
                        # 添加小延迟在每个请求之间
                        time.sleep(0.5)
                        
                        # 使用新方式获取帖子，避免搜索
                        # 而是直接获取新帖子并过滤包含关键词的
                        for post in subreddit.new(limit=limit*2):  # 获取更多帖子然后过滤
                            title_lower = post.title.lower()
                            selftext_lower = post.selftext.lower() if post.selftext else ""
                            
                            # 检查帖子标题或内容是否包含关键词
                            if safe_keyword.lower() in title_lower or safe_keyword.lower() in selftext_lower:
                                # 提取帖子数据
                                post_data = {
                                    "id": post.id,
                                    "title": post.title,
                                    "text": post.selftext[:1000] if post.selftext else "",  # 限制文本长度
                                    "score": post.score,
                                    "created_utc": datetime.fromtimestamp(post.created_utc),
                                    "permalink": f"https://www.reddit.com{post.permalink}",
                                    "subreddit": post.subreddit.display_name,
                                    "keyword": keyword
                                }
                                all_posts.append(post_data)
                                
                                # 如果已找到足够的帖子，停止搜索
                                if sum(1 for p in all_posts if p["keyword"] == keyword and p["subreddit"] == subreddit_name) >= limit:
                                    break

                    except Exception as e:
                        print(f"搜索关键词 '{keyword}' 在子版块 'r/{subreddit_name}' 时出错: {e}")
                        # 尝试备用方法
                        try:
                            print(f"尝试备用方法获取 'r/{subreddit_name}' 的热门帖子...")
                            for post in subreddit.hot(limit=limit):
                                post_data = {
                                    "id": post.id,
                                    "title": post.title,
                                    "text": post.selftext[:1000] if post.selftext else "",
                                    "score": post.score,
                                    "created_utc": datetime.fromtimestamp(post.created_utc),
                                    "permalink": f"https://www.reddit.com{post.permalink}",
                                    "subreddit": post.subreddit.display_name,
                                    "keyword": "general"  # 标记为一般帖子
                                }
                                all_posts.append(post_data)
                        except Exception as inner_e:
                            print(f"备用方法也失败: {inner_e}")
                            
            except Exception as e:
                print(f"访问子版块 'r/{subreddit_name}' 时出错: {e}")
                
        # 将数据转换为DataFrame
        df = pd.DataFrame(all_posts) if all_posts else pd.DataFrame()
        
        # 删除重复的帖子（基于帖子ID）
        if not df.empty and 'id' in df.columns:
            df = df.drop_duplicates(subset='id')
            
        return df
    
    def fetch_comments(self, post_ids, limit=100):
        """
        获取指定帖子的评论
        
        参数:
            post_ids (list): 帖子ID列表
            limit (int): 每个帖子要获取的最大评论数
            
        返回:
            pandas.DataFrame: 包含评论数据的DataFrame
        """
        all_comments = []
        
        for post_id in post_ids:
            try:
                # 添加延迟以避免API速率限制
                time.sleep(1)
                
                submission = self.reddit.submission(id=post_id)
                submission.comments.replace_more(limit=0)  # 仅获取顶层评论，不获取更多评论
                
                for comment in submission.comments[:limit]:
                    comment_data = {
                        "id": comment.id,
                        "text": comment.body[:1000] if comment.body else "",  # 限制文本长度
                        "score": comment.score,
                        "created_utc": datetime.fromtimestamp(comment.created_utc),
                        "permalink": f"https://www.reddit.com{comment.permalink}",
                        "post_id": post_id,
                        "subreddit": comment.subreddit.display_name
                    }
                    all_comments.append(comment_data)
            except Exception as e:
                print(f"获取帖子 '{post_id}' 的评论时出错: {e}")
                
        # 将数据转换为DataFrame
        return pd.DataFrame(all_comments) if all_comments else pd.DataFrame()

# 测试代码
if __name__ == "__main__":
    fetcher = RedditDataFetcher()
    
    # 测试获取帖子
    keywords = ["web3", "ethereum", "blockchain"]
    posts_df = fetcher.fetch_posts(keywords, limit=5)
    
    if not posts_df.empty:
        print(f"成功获取 {len(posts_df)} 个帖子!")
        print("\n帖子示例:")
        print(posts_df[["title", "subreddit", "score", "created_utc"]].head())
        
        # 测试获取评论
        if len(posts_df) > 0:
            post_ids = posts_df["id"].tolist()[:2]  # 获取前两个帖子的评论
            comments_df = fetcher.fetch_comments(post_ids, limit=5)
            
            if not comments_df.empty:
                print(f"\n成功获取 {len(comments_df)} 条评论!")
                print("\n评论示例:")
                print(comments_df[["text", "score", "created_utc"]].head())
    else:
        print("未能获取任何帖子，请检查关键词或Reddit API凭证")