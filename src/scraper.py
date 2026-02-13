
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google_play_scraper import app, Sort, reviews_all, reviews
import pandas as pd
from datetime import datetime
import time
from tqdm import tqdm
from config import APP_IDS, APP_NAMES, SCRAPING_CONFIG, DATA_PATHS


class PlayStoreScraper:
    """Scraper class for Google Play Store reviews"""

    def __init__(self):
        self.app_ids = APP_IDS
        self.app_names = APP_NAMES
        self.reviews_per_app = SCRAPING_CONFIG['reviews_per_app']
        self.lang = SCRAPING_CONFIG['lang']
        self.country = SCRAPING_CONFIG['country']
        self.max_retries = SCRAPING_CONFIG['max_retries']

    def get_app_info(self, app_id):
        try:
            result = app(app_id, lang=self.lang, country=self.country)
            return {
                'app_id': app_id,
                'title': result.get('title', 'N/A'),
                'score': result.get('score', 0),
                'ratings': result.get('ratings', 0),
                'reviews': result.get('reviews', 0),
                'installs': result.get('installs', 'N/A')
            }
        except Exception as e:
            print(f"Error getting app info for {app_id}: {str(e)}")
            return None

    def scrape_reviews(self, app_id, count=1500):
    
        print(f"\nScraping reviews for {app_id}...")
        print(f"Target: {count} reviews (will scrape all available)")

        for attempt in range(self.max_retries):
            try:
                # Use reviews_all to get ALL available reviews (no limit)
                result = reviews_all(
                    app_id,
                    sleep_milliseconds=100,  # Small delay to avoid rate limiting
                    lang=self.lang,
                    country=self.country,
                    sort=Sort.NEWEST,
                )

                print(f"Successfully scraped {len(result)} reviews")
                return result

            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    print(f"Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    print(f"Failed to scrape reviews after {self.max_retries} attempts")
                    return []

        return []

    def process_reviews(self, reviews_data, app_code):
    
        processed = []

        for review in reviews_data:
            processed.append({
                'review_id': review.get('reviewId', ''),
                'review_text': review.get('content', ''),
                'rating': review.get('score', 0),
                'review_date': review.get('at', datetime.now()),
                'user_name': review.get('userName', 'Anonymous'),
                'thumbs_up': review.get('thumbsUpCount', 0),
                'reply_content': review.get('replyContent', None),
                'app_code': app_code,
                'app_name': self.app_names[app_code],
                'app_id': review.get('reviewCreatedVersion', 'N/A'),
                'source': 'Google Play'
            })

        return processed

    def scrape_all_apps(self):
        """
        Scrape reviews for all banks

        Returns:
            pd.DataFrame: Combined dataframe with all reviews
        """
        all_reviews = []
        app_info_list = []

        print("=" * 60)
        print("Starting Google Play Store Review Scraper")
        print("=" * 60)

        # Get app information
        print("\n[1/2] Fetching app information...")
        for app_code, app_id in self.app_ids.items():
            print(f"\n{app_code}: {self.app_names[app_code]}")
            print(f"App ID: {app_id}")

            info = self.get_app_info(app_id)
            if info:
                info['app_code'] = app_code
                info['app_name'] = self.app_names[app_code]
                app_info_list.append(info)
                print(f"Current Rating: {info['score']}")
                print(f"Total Ratings: {info['ratings']}")
                print(f"Total Reviews: {info['reviews']}")

        # Save app info
        if app_info_list:
            app_info_df = pd.DataFrame(app_info_list)
            os.makedirs(DATA_PATHS['raw'], exist_ok=True)
            app_info_df.to_csv(f"{DATA_PATHS['raw']}/app_info.csv", index=False)
            print(f"\nApp information saved to {DATA_PATHS['raw']}/app_info.csv")

        # Scrape reviews
        print("\n[2/2] Scraping reviews...")
        for app_code, app_id in tqdm(self.app_ids.items(), desc="apps"):
            reviews_data = self.scrape_reviews(app_id, self.reviews_per_app)

            if reviews_data:
                processed = self.process_reviews(reviews_data, app_code)
                all_reviews.extend(processed)
                print(f"Collected {len(processed)} reviews for {self.app_names[app_code]}")
            else:
                print(f"WARNING: No reviews collected for {self.app_names[app_code]}")
            # Small delay between banks
            time.sleep(2)

        if all_reviews:
            df = pd.DataFrame(all_reviews)

            # Save raw data
            os.makedirs(DATA_PATHS['raw'], exist_ok=True)
            df.to_csv(DATA_PATHS['raw_reviews'], index=False)

            print("\n" + "=" * 60)
            print("Scraping Complete!")
            print("=" * 60)
            print(f"\nTotal reviews collected: {len(df)}")
            print(f"Reviews per app:")
            for app_code in self.app_names.keys():
                count = len(df[df['app_code'] == app_code])
                print(f"  {self.app_names[app_code]}: {count}")

            print(f"\nData saved to: {DATA_PATHS['raw_reviews']}")

            return df
        else:
            print("\nERROR: No reviews were collected!")
            return pd.DataFrame()

    def display_sample_reviews(self, df, n=3):
        """
        Display sample reviews from each bank

        Args:
            df (pd.DataFrame): Reviews dataframe
            n (int): Number of samples per bank
        """
        print("\n" + "=" * 60)
        print("Sample Reviews")
        print("=" * 60)

        for app_code in self.app_names.keys():
            app_df = df[df['app_code'] == app_code]
            if not app_df.empty:
                print(f"\n{self.app_names[app_code]}:")
                print("-" * 60)
                samples = app_df.head(n)
                for idx, row in samples.iterrows():
                    print(f"\nRating: {'⭐' * row['rating']}")
                    print(f"Review: {row['review_text'][:200]}...")
                    print(f"Date: {row['review_date']}")


def main():
    """Main execution function"""

    # Initialize scraper
    scraper = PlayStoreScraper()

    # Scrape all reviews
    df = scraper.scrape_all_apps()

    # Display samples
    if not df.empty:
        scraper.display_sample_reviews(df)

    return df


if __name__ == "__main__":
    reviews_df = main()