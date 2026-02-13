


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm
import re

# NLP imports
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy

from config import THEME_KEYWORDS, THEME_CONFIG, DATA_PATHS


class ThemeAnalyzer:

    def __init__(self):
        """
        Initialize the theme analyzer with TF-IDF and spaCy.
        """
        self.theme_keywords = THEME_KEYWORDS
        self.tfidf_vectorizer = None
        self.nlp = None
        
        # Initialize components
        self._init_tfidf()
        self._init_spacy()
    
    def _init_tfidf(self):
        """
        Initialize TF-IDF Vectorizer.
        
        What is TF-IDF?
        - TF (Term Frequency): How often a word appears in a document
        - IDF (Inverse Document Frequency): log(total docs / docs containing word)
        - TF-IDF = TF × IDF
        
        Why use it?
        - Common words like "the", "is", "app" get LOW scores (appear everywhere)
        - Unique meaningful words like "crash", "login" get HIGH scores
        - Helps identify what makes each review unique/important
        
        Parameters from config:
        - max_features: Maximum number of keywords to extract (100)
        - ngram_range: (1,3) means we look at single words, pairs, and triplets
          e.g., "login", "login error", "login error message"
        - min_df: Minimum document frequency (word must appear in at least 2 reviews)
        """
        print("Initializing TF-IDF vectorizer...")
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=THEME_CONFIG['max_features'],
            ngram_range=THEME_CONFIG['ngram_range'],
            min_df=THEME_CONFIG['min_df'],
            stop_words='english',  # Remove common English words
            lowercase=True,
            token_pattern=r'[a-zA-Z]{2,}'  # Only words with 2+ letters
        )
        print("TF-IDF ready!")
    
    def _init_spacy(self):
        """
        Initialize spaCy for advanced NLP.
        
        What is spaCy?
        - Industrial-strength NLP library
        - Provides: tokenization, lemmatization, POS tagging, NER
        
        Why use it?
        - Lemmatization: "crashes" → "crash", "logging" → "log"
        - This helps match keywords more accurately
        - POS tagging: Identify nouns, verbs, adjectives
        """
        print("Loading spaCy model...")
        try:
            self.nlp = spacy.load('en_core_web_sm')
            print("spaCy ready!")
        except OSError:
            print("WARNING: spaCy model not found. Run: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def preprocess_text(self, text):
        """
        Preprocess text for theme extraction.
        
        Steps:
        1. Convert to lowercase
        2. Remove special characters (keep only letters and spaces)
        3. Lemmatize words (if spaCy available)
        
        Args:
            text (str): Raw review text
            
        Returns:
            str: Cleaned and lemmatized text
        """
        if not text or pd.isna(text):
            return ""
        
        text = str(text).lower()
        
        # Remove special characters, keep only letters and spaces
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Lemmatize if spaCy is available
        if self.nlp:
            doc = self.nlp(text)
            # Get lemmas, excluding stop words and punctuation
            lemmas = [token.lemma_ for token in doc 
                     if not token.is_stop and not token.is_punct and len(token.text) > 2]
            text = ' '.join(lemmas)
        
        return text
    
    def extract_keywords_tfidf(self, texts, top_n=10):
        """
        Extract top keywords from a collection of texts using TF-IDF.
        
        How it works:
        1. Fit TF-IDF on all texts to learn word importance
        2. For each text, get the words with highest TF-IDF scores
        
        Args:
            texts (list): List of review texts
            top_n (int): Number of top keywords to extract per review
            
        Returns:
            list: List of keyword lists, one per review
        """
        # Preprocess all texts
        processed_texts = [self.preprocess_text(t) for t in texts]
        
        # Fit and transform
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(processed_texts)
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        # Extract top keywords for each document
        all_keywords = []
        for i in range(tfidf_matrix.shape[0]):
            # Get TF-IDF scores for this document
            row = tfidf_matrix[i].toarray().flatten()
            
            # Get indices of top N scores
            top_indices = row.argsort()[-top_n:][::-1]
            
            # Get the actual words
            keywords = [feature_names[idx] for idx in top_indices if row[idx] > 0]
            all_keywords.append(keywords)
        
        return all_keywords
    
    def identify_themes(self, text, keywords=None):
        """
        Identify themes in a review based on keyword matching.
        
        How it works:
        1. Check if any predefined theme keywords appear in the text
        2. A theme is assigned if ANY of its keywords are found
        3. Multiple themes can be assigned to one review
        
        Args:
            text (str): Review text
            keywords (list): Optional pre-extracted keywords
            
        Returns:
            dict: Contains 'themes' (list) and 'primary_theme' (str)
        """
        if not text or pd.isna(text):
            return {'themes': [], 'primary_theme': 'Unknown', 'matched_keywords': []}
        
        text_lower = str(text).lower()
        matched_themes = []
        all_matched_keywords = []
        
        # Check each theme's keywords against the text
        for theme, keywords_list in self.theme_keywords.items():
            matched_keywords = []
            for keyword in keywords_list:
                # Check if keyword appears in text (as whole word or phrase)
                if keyword.lower() in text_lower:
                    matched_keywords.append(keyword)
            
            if matched_keywords:
                matched_themes.append({
                    'theme': theme,
                    'matches': len(matched_keywords),
                    'keywords': matched_keywords
                })
                all_matched_keywords.extend(matched_keywords)
        
        # Sort by number of matches (most relevant theme first)
        matched_themes.sort(key=lambda x: x['matches'], reverse=True)
        
        # Extract theme names
        themes = [t['theme'] for t in matched_themes]
        primary_theme = themes[0] if themes else 'Other'
        
        return {
            'themes': themes,
            'primary_theme': primary_theme,
            'matched_keywords': list(set(all_matched_keywords))
        }
    
    def analyze_dataframe(self, df, text_column='review_text'):
        """
        Analyze themes for all reviews in a DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame containing reviews
            text_column (str): Name of the column containing review text
            
        Returns:
            pd.DataFrame: Original DataFrame with added theme columns
        """
        print(f"\nAnalyzing themes for {len(df)} reviews...")
        print("=" * 60)
        
        # Step 1: Extract TF-IDF keywords for the corpus
        print("\n[1/2] Extracting keywords using TF-IDF...")
        keywords_list = self.extract_keywords_tfidf(df[text_column].tolist())
        
        # Step 2: Identify themes for each review
        print("[2/2] Mapping keywords to themes...")
        themes_data = []
        
        for idx, text in tqdm(enumerate(df[text_column]), total=len(df), desc="Identifying themes"):
            result = self.identify_themes(text, keywords_list[idx] if idx < len(keywords_list) else None)
            themes_data.append(result)
        
        # Add results to DataFrame
        result_df = df.copy()
        result_df['themes'] = [d['themes'] for d in themes_data]
        result_df['primary_theme'] = [d['primary_theme'] for d in themes_data]
        result_df['matched_keywords'] = [d['matched_keywords'] for d in themes_data]
        result_df['tfidf_keywords'] = keywords_list
        
        # Print summary
        self._print_summary(result_df)
        
        return result_df
    
    def _print_summary(self, df):
        """Print thematic analysis summary statistics."""
        print("\n" + "=" * 60)
        print("THEMATIC ANALYSIS SUMMARY")
        print("=" * 60)
        
        # Overall theme distribution
        print("\nOverall Theme Distribution:")
        all_themes = []
        for themes in df['themes']:
            all_themes.extend(themes)
        
        theme_counts = Counter(all_themes)
        total_assignments = len(all_themes)
        
        for theme, count in theme_counts.most_common():
            pct = (count / len(df)) * 100  # Percentage of reviews with this theme
            print(f"  {theme}: {count} reviews ({pct:.1f}%)")
        
        # Reviews with no theme
        no_theme = len(df[df['primary_theme'] == 'Other'])
        print(f"\n  Reviews with no identified theme: {no_theme} ({(no_theme/len(df))*100:.1f}%)")

        # By app
        if 'app_name' in df.columns:
            print("\nTop Theme by App:")
            for app in df['app_name'].unique():
                app_df = df[df['app_name'] == app]
                app_themes = []
                for themes in app_df['themes']:
                    app_themes.extend(themes)
                
                if app_themes:
                    top_theme = Counter(app_themes).most_common(1)[0]
                    print(f"  {app}: {top_theme[0]} ({top_theme[1]} mentions)")
                else:
                    print(f"  {app}: No themes identified")

    def get_theme_sentiment_correlation(self, df):
        """
        Analyze correlation between themes and sentiment.
        
        Purpose: Find which themes are associated with positive/negative sentiment
        
        Args:
            df (pd.DataFrame): DataFrame with themes and sentiment columns
            
        Returns:
            pd.DataFrame: Theme-sentiment correlation matrix
        """
        # Check for sentiment column
        sentiment_col = None
        for col in ['sentiment_label_vader', 'sentiment_label_distilbert', 'sentiment_label']:
            if col in df.columns:
                sentiment_col = col
                break
        
        if not sentiment_col:
            print("No sentiment column found. Run sentiment analysis first.")
            return None
        
        print("\nTheme-Sentiment Correlation:")
        print("-" * 60)
        
        results = []
        for theme in self.theme_keywords.keys():
            # Get reviews with this theme
            theme_mask = df['themes'].apply(lambda x: theme in x if isinstance(x, list) else False)
            theme_df = df[theme_mask]
            
            if len(theme_df) > 0:
                pos_count = len(theme_df[theme_df[sentiment_col] == 'POSITIVE'])
                neg_count = len(theme_df[theme_df[sentiment_col] == 'NEGATIVE'])
                total = len(theme_df)
                
                pos_pct = (pos_count / total) * 100
                neg_pct = (neg_count / total) * 100
                
                results.append({
                    'theme': theme,
                    'total_reviews': total,
                    'positive_pct': pos_pct,
                    'negative_pct': neg_pct,
                    'sentiment_ratio': pos_pct / neg_pct if neg_pct > 0 else float('inf')
                })
                
                sentiment_indicator = "👍" if pos_pct > neg_pct else "👎"
                print(f"  {sentiment_indicator} {theme}:")
                print(f"      Positive: {pos_pct:.1f}% | Negative: {neg_pct:.1f}%")
        
        return pd.DataFrame(results)


def analyze_themes(input_path=None, output_path=None):
    """
    Convenience function to run thematic analysis on reviews.
    
    Args:
        input_path (str): Path to reviews CSV (with sentiment results)
        output_path (str): Path to save results
        
    Returns:
        pd.DataFrame: Reviews with theme assignments
    """
    input_path = input_path or DATA_PATHS['sentiment_results']
    output_path = output_path or DATA_PATHS['final_results']
    
    # Load data
    print(f"Loading reviews from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Analyze
    analyzer = ThemeAnalyzer()
    result_df = analyzer.analyze_dataframe(df)
    
    # Get theme-sentiment correlation if sentiment data exists
    correlation_df = analyzer.get_theme_sentiment_correlation(result_df)
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    
    return result_df


def main():
    """
    Main function to run thematic analysis.
    """
    print("=" * 60)
    print("THEMATIC ANALYSIS PIPELINE")
    print("=" * 60)
    
    result_df = analyze_themes()
    
    print("\n✓ Thematic analysis complete!")
    return result_df


if __name__ == "__main__":
    result = main()