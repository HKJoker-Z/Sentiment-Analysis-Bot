项目名称: Web3 社交媒体情绪分析器 (基于 Reddit 数据)

项目目标: 开发一个 Web 应用，允许用户输入 Web3 相关的关键词（如加密货币名称、NFT 项目、技术概念等），应用会从 Reddit 的相关社区 (Subreddits) 获取讨论数据（帖子标题、内容、评论），利用自然语言处理 (NLP) 技术分析这些文本的情绪倾向（积极、消极、中性），并将分析结果以可视化和统计摘要的形式展示给用户。该项目旨在展示在数据获取、AI/NLP 应用、后端逻辑和 Web UI 开发方面的综合能力，特别适用于展示给潜在雇主。

整体架构 (Data Flow):

用户交互 (Frontend - Streamlit UI):

用户通过浏览器访问 Streamlit 应用。
用户在界面上输入一个或多个 Web3 相关的关键词。
(可选) 用户可能可以选择要分析的目标 Subreddits（例如，从预设列表中选择或手动输入）。
(可选) 用户可能可以选择分析的内容类型（例如，仅帖子、仅评论、或两者都分析）和数量限制。
用户点击“分析”按钮，触发后端处理流程。
后端逻辑 (Backend - Python Script running via Streamlit):

Streamlit 应用接收到用户的输入（关键词、目标 Subreddits 等）。
调用数据获取模块 (reddit_api.py 或类似模块)。
数据获取 (Data Acquisition - Reddit API via PRAW):

数据获取模块使用 PRAW 库。
从 .env 文件安全地加载 Reddit API 凭证 (Client ID, Client Secret, User Agent)。
根据用户输入的关键词和选择的 Subreddits，向 Reddit API 发送搜索请求 (例如 subreddit.search() 或获取热门帖子的评论)。
获取指定数量的公开帖子标题、帖子正文（如果适用）和/或评论文本。
提取相关元数据，如永久链接 (permalink)、得分 (score)、创建时间 (created_utc)、来源 Subreddit 等。
数据处理与准备 (Data Processing - Pandas):

将从 PRAW 获取的原始数据（通常是 PRAW 对象）转换并清洗。
将文本数据和元数据整理成结构化的格式，通常是 Pandas DataFrame，方便后续处理和分析。列可能包括 text, id, permalink, score, subreddit, created_utc 等。
情绪分析 (Core Logic - Hugging Face Transformers):

调用情绪分析模块 (sentiment.py 或类似模块)。
加载预训练的 NLP 情绪分析模型（例如，来自 Hugging Face Hub 的 Transformer 模型，如 distilbert-base-uncased-finetuned-sst-2-english, cardiffnlp/twitter-roberta-base-sentiment, 或其他适合社交媒体文本的模型）。
将 DataFrame 中的文本数据批量输入到模型中进行预测。
模型输出每段文本的情绪标签（例如，'POSITIVE', 'NEGATIVE', 'NEUTRAL' 或 'LABEL_1', 'LABEL_0' 等）和相应的置信度分数。
结果整合与统计 (Results Aggregation - Pandas):

将情绪分析结果（标签和分数）添加回原始的 Pandas DataFrame 中，与对应的文本和元数据关联起来。
计算整体的情绪分布（例如，积极、消极、中性的帖子/评论数量及百分比）。
计算其他相关统计数据（例如，分析的总文本数量）。
结果展示 (Frontend - Streamlit UI):

后端逻辑将处理和分析后的 DataFrame 及统计结果传递给 Streamlit 的 UI 组件。
使用 Streamlit 的图表组件（如 st.bar_chart, st.pyplot 配合 Matplotlib/Seaborn，或 st.plotly_chart）可视化情绪分布（例如，饼图或条形图）。
使用 st.metric 或 st.write 显示关键统计数据。
使用 st.dataframe 或 st.table 展示部分原始文本、对应的分析情绪、得分以及指向原始 Reddit 内容的链接 (permalink)，让用户可以查看上下文。
使用的工具与技术:

编程语言: Python 3.x
Web 应用框架 (UI): Streamlit - 用于快速构建交互式数据科学 Web 应用。
数据源 API: Reddit API - 获取公开的帖子和评论数据。
Reddit API 交互库: PRAW (Python Reddit API Wrapper) - 简化与 Reddit API 的交互。
数据处理与分析: Pandas - 用于数据清洗、转换、存储和基本分析。
AI / 自然语言处理 (NLP):
Hugging Face Transformers - 用于加载和使用预训练的 Transformer 模型进行情绪分析。
(可能) NLTK 或 spaCy - 用于可选的文本预处理（如分词、去除停用词等，但对于现代 Transformer 模型通常不是必需的）。
配置管理: python-dotenv - 用于管理环境变量，特别是安全地处理 API 密钥（从 .env 文件加载）。
依赖管理: pip 和 requirements.txt - 用于管理和安装项目所需的 Python 包。
开发环境: 虚拟环境 (如 venv, conda) - 隔离项目依赖。
版本控制: Git 和 GitHub - 用于代码管理、协作和托管。
实现的功能:

关键词输入: 允许用户指定感兴趣的 Web3 相关话题或实体。
(可选) Subreddit 选择: 允许用户聚焦于特定的 Reddit 社区进行分析。
(可选) 内容类型与数量控制: 允许用户定义分析的范围（帖子/评论）和深度（数据量）。
Reddit 数据获取: 从指定的 Reddit 社区自动获取相关的公开文本数据。
情绪分析: 对获取的文本数据进行自动化的情绪分类（积极/消极/中性）。
结果可视化: 以图表形式（如饼图、条形图）直观展示整体的情绪分布。
统计摘要: 提供分析文本总数等关键统计信息。
样本数据展示: 显示部分原始文本、其分析出的情绪以及其他元数据（如得分、来源链接），方便用户验证和理解上下文。
Web 界面: 提供一个简单易用的 Web 界面供用户操作和查看结果。
这个架构和技术栈构成了一个功能完整、技术栈现代且适合作为作品集展示的项目。它清楚地展示了从数据获取、处理、AI 分析到结果呈现的全流程能力。