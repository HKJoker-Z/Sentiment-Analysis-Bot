import os
import praw
from dotenv import load_dotenv
from datetime import datetime
import re
import sys

# 设置UTF-8编码
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# 加载.env文件中的环境变量
load_dotenv()

def is_english(text):
    """简单检查文本是否主要为英文"""
    # 检查是否包含非ASCII字符
    if bool(re.search(r'[^\x00-\x7F]', text)):
        # 如果非ASCII字符占比较高，可能不是英文
        non_ascii_count = len(re.findall(r'[^\x00-\x7F]', text))
        if non_ascii_count / len(text) > 0.15:  # 如果非ASCII字符超过15%
            return False
    return True

def fetch_web3_posts():
    # 从环境变量中获取Reddit API凭据
    reddit_client_id = os.getenv("REDDIT_CLIENT_ID")
    reddit_client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    # 确保用户代理只包含ASCII字符
    reddit_user_agent = "Web3SentimentAnalyzer/0.1 by GlumExtension2622"
    
    # 检查凭据是否已加载
    if not all([reddit_client_id, reddit_client_secret]):
        print("错误: 无法加载Reddit API凭据，请检查.env文件")
        return
    
    try:
        # 初始化Reddit实例
        reddit = praw.Reddit(
            client_id=reddit_client_id,
            client_secret=reddit_client_secret,
            user_agent=reddit_user_agent
        )
        
        print("成功连接到Reddit API!")
        
        # 改用主要使用英语的web3相关子版块
        print("获取web3相关英文子版块中的10条最新帖子...\n")
        
        # 这些是主要使用英语的web3相关子版块
        subreddits = ["web3", "ethereum", "cryptocurrency", "blockchain", "CryptoTechnology", "ethdev"]
        
        posts_found = 0
        for subreddit_name in subreddits:
            try:
                for post in reddit.subreddit(subreddit_name).new(limit=5):
                    # 只处理标题主要为英文的帖子
                    if is_english(post.title):
                        posts_found += 1
                        post_time = datetime.fromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S')
                        print(f"{posts_found}. 标题: {post.title}")
                        print(f"   子版块: r/{post.subreddit}")
                        print(f"   发布时间: {post_time}")
                        print(f"   链接: https://www.reddit.com{post.permalink}")
                        print(f"   赞同数: {post.score}")
                        print("-" * 80)
                    
                        if posts_found >= 10:
                            break
            except Exception as e:
                print(f"获取r/{subreddit_name}内容时出错: {e}")
                continue
                
            if posts_found >= 10:
                break
            
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    fetch_web3_posts()