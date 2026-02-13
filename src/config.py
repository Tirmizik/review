import os


# Google Play Store App IDs Mapping
APP_IDS = {
   'beU Delivery': 'com.xmt.beu.applite',
    'CakeFizz': 'com.cakefizz.cakefizz',
    'Kekmart': 'com.sr.kekmart',
    'How to Bake': 'com.howtobake.primex'
}

# App Names Mapping
APP_NAMES = {
    'beU Delivery': 'beU Delivery',
    'CakeFizz': 'CakeFizz',
    'Kekmart': 'Kekmart',
    'How to Bake': 'How to Bake',
}


# Scraping Configuration
SCRAPING_CONFIG = {
    'reviews_per_app': int(os.getenv('REVIEWS_PER_APP', 1500)),
    'max_retries': int(os.getenv('MAX_RETRIES', 4)),
    'lang': 'en',
    'country': 'us'
}

# File Paths
DATA_PATHS = {
    'raw': r'C:\Users\hakima\Desktop\Project\review\data\raw',
    'processed': r'C:\Users\hakima\Desktop\Project\review\data\processed',
    'raw_reviews': r'C:\Users\hakima\Desktop\Project\review\data\raw\reviews_raw.csv',
    'processed_reviews': r'C:\Users\hakima\Desktop\Project\review\data\processed\reviews_processed.csv',
    'sentiment_results': r'C:\Users\hakima\Desktop\Project\review\data\processed\reviews_with_sentiment.csv',
    'final_results': r'C:\Users\hakima\Desktop\Project\review\data\processed\reviews_final.csv'
}

# Sentiment Analysis Configuration
SENTIMENT_CONFIG = {
    'model': os.getenv('SENTIMENT_MODEL', 'distilbert-base-uncased-finetuned-sst-2-english'),
    'use_vader': os.getenv('USE_VADER', 'false').lower() == 'true',
    'batch_size': 16
}

# Thematic Analysis Configuration
THEME_CONFIG = {
    'num_themes': int(os.getenv('NUM_THEMES', 5)),
    'max_features': 100,
    'ngram_range': (1, 3),  # unigrams, bigrams, trigrams
    'min_df': 2
}



# Visualization Configuration
VIZ_CONFIG = {
    'style': 'seaborn-v0_8-darkgrid',
    'figure_size': (12, 8),
    'dpi': 300,
    'color_palette': 'Set2'
}




# Predefined Theme Keywords (for manual clustering)
THEME_KEYWORDS = {
    'Account Access Issues': [
        'login', 'password', 'authentication', 'forgot', 'reset',
        'sign in', 'sign up', 'register', 'account', 'access'
    ],
    'Transaction Performance': [
        'transfer', 'slow', 'fast', 'speed', 'loading', 'payment',
        'transaction', 'delay', 'timeout', 'processing'
    ],
    'User Interface & Experience': [
        'ui', 'interface', 'design', 'layout', 'navigation', 'menu',
        'button', 'screen', 'display', 'easy', 'difficult', 'confusing'
    ],
    'Technical Issues': [
        'crash', 'bug', 'error', 'freeze', 'hang', 'broken',
        'not working', 'problem', 'issue', 'glitch', 'fail'
    ],
    'Customer Support': [
        'support', 'help', 'service', 'response', 'contact',
        'customer care', 'assistance', 'call center'
    ],
    'Feature Requests': [
        'feature', 'add', 'need', 'want', 'wish', 'should',
        'missing', 'would like', 'suggestion', 'improve'
    ],
    'Security & Privacy': [
        'security', 'safe', 'secure', 'privacy', 'fingerprint',
        'biometric', 'otp', 'verification', 'fraud'
    ]
}