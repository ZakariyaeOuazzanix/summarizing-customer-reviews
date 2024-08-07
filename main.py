from textblob import TextBlob
from newspaper import Article
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer 
import nltk
from collections import Counter

nltk.download('punkt', quiet=True)

def summarize_text(text, sentences_count=3):
    """
    Summarize the given text using LSA (Latent Semantic Analysis) algorithm.
    
    Args:
    text (str): The input text to summarize.
    sentences_count (int): The number of sentences in the summary (default: 3).
    
    Returns:
    str: A summary of the input text.
    """
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentences_count)
    return ' '.join(str(sentence) for sentence in summary)

def summarize_reviews(reviews, sentences=3):
    """
    Summarize a list of reviews by extracting key themes and sentiments.
    
    Args:
    reviews (list): A list of review strings.
    sentences (int): The number of sentences in the summary (default: 3).
    
    Returns:
    str: A summary of the reviews.
    """
    # Combine all reviews into a single text
    combined_reviews = ' '.join(reviews)
    
    # Perform sentiment analysis
    blob = TextBlob(combined_reviews)
    overall_sentiment = 'positive' if blob.sentiment.polarity > 0 else 'negative' if blob.sentiment.polarity < 0 else 'neutral'
    
    # Extract common words (excluding stop words)
    words = [word.lower() for word in blob.words if len(word) > 3]
    common_words = Counter(words).most_common(5)
    
    # Generate summary
    summary = f"The overall sentiment of the reviews is {overall_sentiment}. "
    summary += f"Common themes include: {', '.join(word for word, _ in common_words)}. "
    summary += summarize_text(combined_reviews, sentences - 1)  # Use LSA for additional summary
    
    return summary

# ... [rest of the code remains the same] ...

# Sample reviews for analysis
reviews = [
    "This phone is amazing! The camera quality is outstanding, and the battery life is impressive.",
    "I'm disappointed with the build quality. It feels cheap compared to previous models.",
    "The new features are great, but the learning curve is steep. It took me a while to get used to it.",
    "Absolutely love this phone! It's fast, sleek, and the display is gorgeous.",
    "The price is too high for what you get. There are better options out there for less money.",
    "I've had issues with the touchscreen responsiveness. Sometimes it lags or doesn't register my taps.",
    "The camera's night mode is a game-changer. I can finally take great photos in low light!",
    "Battery life is decent, but not as long-lasting as advertised. Heavy users might struggle.",
    "The water resistance feature saved my phone during a recent accident. Definitely a plus!",
    "I'm not a fan of the new operating system update. It feels cluttered and less intuitive.",
    "The fast charging feature is fantastic. I can get a full charge in no time.",
    "The phone tends to overheat during gaming sessions, which is concerning.",
    "Customer support was unhelpful when I had issues. It took weeks to resolve a simple problem.",
    "The facial recognition is lightning fast and works even in dim lighting.",
    "I wish it had a headphone jack. Having to use an adapter is inconvenient.",
    "The amount of bloatware pre-installed on the phone is frustrating.",
    "Call quality is crystal clear, even in noisy environments.",
    "The new AI assistant is hit or miss. Sometimes it's helpful, other times it's just annoying.",
    "I love how customizable the interface is. I can make the phone truly my own.",
    "The phone is a bit too large for comfortable one-handed use. A smaller option would be nice."
]

# Perform sentiment analysis on reviews
review_sentiments = [TextBlob(review).sentiment.polarity for review in reviews]
print(review_sentiments)
avg_polarity = sum(review_sentiments) / len(reviews)
print(f'Average review polarity: {avg_polarity:.2f}')

# Summarize the reviews
review_summary = summarize_reviews(reviews, 4)
print(f'Summary of the reviews: {review_summary}')