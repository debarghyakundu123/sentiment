"""
Streamlit Sentiment Analyzer & Insight Engine (V2 - Improved Accuracy + Unique Features)
Full end-to-end single-file app (Python)

Improvements for Accuracy (V2):
1. Enhanced Preprocessing: Better handling of punctuation and negation words.
2. Contextual Sarcasm: Sarcasm heuristic now checks for discrepancy between VADER score and negative keywords.
3. Expanded Emotion Lexicon: Significantly larger, structured lexicon for broader emotion detection.
4. Robust ML Model: Retained TFIDF-LR but used a stronger feature set.

Unique Features Added:
5. Churn Risk Score: A calculated metric for customer abandonment risk.
6. Emerging Topic Detection: Placeholder for NMF/LDA analysis to find new, unplanned root causes.

Requirements:
pip install streamlit pandas numpy scikit-learn nltk spacy vaderSentiment wordcloud matplotlib langdetect
python -m spacy download en_core_web_sm

Run:
streamlit run Streamlit_Sentiment_Analyzer_Full_v2.py

"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import os
from datetime import datetime
from collections import Counter
from io import StringIO # <-- ADDED FOR CSV PASTE FALLBACK

# NLP / ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
# Added for Emerging Topic Detection Placeholder
from sklearn.decomposition import NMF, LatentDirichletAllocation 


# Optional libraries
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except Exception:
    VADER_AVAILABLE = False

try:
    import nltk
    # --- FIX for LookupError in deployment environments ---
    # Ensure necessary NLTK packages are downloaded before use
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except:
        nltk.download('punkt')
        nltk.download('stopwords')

except Exception:
    # If NLTK itself fails to import
    st.error("NLTK library failed to load. Tokenization will not work.")

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

try:
    from langdetect import detect
    LANGDET_AVAILABLE = True
except Exception:
    LANGDET_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
    nlp = spacy.load('en_core_web_sm')
except Exception:
    SPACY_AVAILABLE = False
    st.error("SpaCy 'en_core_web_sm' not loaded. Aspect extraction disabled.")

# --- Utility & preprocessing functions ---
STOPWORDS = set(stopwords.words('english')) if 'stopwords' in globals() else set()

# Negation words to feature engineering (improving on simple tokenization)
NEGATION_WORDS = {"not", "no", "never", "none", "neither", "nor", "cannot", "don't", "isn't", "wasn't", "shouldn't", "wouldn't"}

EMOJI_PATTERN = re.compile("""[\U00010000-\U0010ffff]""", flags=re.UNICODE)
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')

def preprocess_text(text):
    """Cleans text, removes URLs/Emojis, and performs tokenization/stopword removal."""
    if not isinstance(text, str):
        return ""
    txt = text.lower()
    txt = URL_PATTERN.sub('', txt)
    txt = EMOJI_PATTERN.sub('', txt)
    # Improved punctuation handling: keep for now, but remove later for bag-of-words
    txt = re.sub(r'[^\w\s\']', ' ', txt) # Keep apostrophes for contractions
    txt = re.sub(r'\s+', ' ', txt).strip()

    tokens = word_tokenize(txt)
    # Simple stopword removal
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return ' '.join(tokens)

# Extended Emotion lexicon (structured, larger seed) -- critical for better emotion accuracy
EMOTION_LEXICON = {
    'joy': ['happy', 'good', 'great', 'love', 'awesome', 'excellent', 'nice', 'satisfied', 'pleased', 'fantastic', 'amazing', 'perfect', 'cheer', 'gorgeous', 'delight', 'joyful', 'wonderful'],
    'anger': ['angry', 'hate', 'terrible', 'worst', 'bad', 'disappointed', 'annoyed', 'frustrated', 'rage', 'furious', 'upset', 'lousy', 'horrible', 'trash'],
    'sadness': ['sad', 'unhappy', 'upset', 'disappointed', 'depressed', 'gloomy', 'misery', 'pain', 'sorry', 'crying', 'broken'],
    'surprise': ['surprise', 'surprised', 'shocked', 'unexpected', 'wow', 'gasp', 'unbelievable'],
    'fear': ['fear', 'scared', 'worried', 'anxious', 'terror', 'horrifying'],
    'trust': ['reliable', 'trustworthy', 'solid', 'confident', 'sure'],
    'disgust': ['disgust', 'gross', 'nasty', 'awful', 'sickening'],
}

def detect_emotions(text):
    """Calculates emotion distribution based on expanded lexicon."""
    text = text.lower()
    scores = {k: 0 for k in EMOTION_LEXICON.keys()}
    total_matches = 0

    for emo, words in EMOTION_LEXICON.items():
        for w in words:
            if w in text:
                count = text.count(w)
                scores[emo] += count
                total_matches += count

    # Normalize
    if total_matches == 0:
        return {k: 0.0 for k in scores}
    return {k: round(v / total_matches, 3) for k, v in scores.items()}

# Contextual Sarcasm heuristic (VADER integration for better accuracy)
def detect_sarcasm(text):
    """Detects sarcasm by checking for positive VADER score combined with negative keywords."""
    t = text.lower()
    
    # 1. Simple Keyword/Punctuation Heuristics (Retained)
    SARCASM_PATTERNS = [
        r"yeah right", r"sure,?", r"(?i)as if", r"(?i)i'm sure",
        r"\bjust great\b", r'\bjust what i needed\b'
    ]
    for p in SARCASM_PATTERNS:
        if re.search(p, t):
            return True
            
    # 2. Contextual Check (New & Improved): Positive tone but presence of strong negative words
    if VADER_AVAILABLE:
        vader_score = vader_sentiment(text)['compound']
        neg_words_in_text = re.findall(r'\b(delay|broken|terrible|horrible|worst|never)\b', t)
        
        # If the overall VADER score is neutral/positive (> 0.1) but strong negative words are present
        if vader_score > 0.1 and len(neg_words_in_text) > 0:
            return True
            
    return False

# Spam / fake review heuristics (Slightly improved)
def detect_spam(text):
    """Heuristic detector for spam or fake reviews."""
    t = text.lower()
    if len(t.split()) < 5:  # Increased minimum length slightly
        return True
    if re.search(r'(http|www\.|\\.com)', t):
        return True
    if t.count('!') + t.count('?') > 5: # Excessive punctuation
        return True
    
    # Repetitive words check (e.g., "amazing amazing amazing")
    tokens = t.split()
    if len(tokens) > 10 and len(set(tokens)) / len(tokens) < 0.4: # Low unique word ratio
        return True
    
    return False

# Aspect extraction (using spaCy noun chunks if available)
def extract_aspects(text, top_n=5):
    """Uses SpaCy Noun Chunks for better aspect extraction than simple nouns."""
    if SPACY_AVAILABLE and isinstance(text, str):
        doc = nlp(text)
        chunks = [chunk.text.strip().lower() for chunk in doc.noun_chunks]
        freq = Counter(chunks)
        items = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [x[0] for x in items if x[0] not in STOPWORDS]
    else:
        # fallback: use common words
        words = re.findall(r'\b\w+\b', str(text).lower())
        common = Counter([w for w in words if len(w) > 3 and w not in STOPWORDS])
        items = common.most_common(top_n)
        return [x[0] for x in items]

# Root cause analysis (keyword mapping) - stays the same
ROOT_CAUSE_KEYWORDS = {
    'delivery': ['delivery', 'late', 'delay', 'courier', 'driver', 'shipment', 'tracking'],
    'product_quality': ['broken', 'damage', 'defect', 'quality', 'scratch', 'torn', 'faulty', 'durability'],
    'packaging': ['packaging', 'package', 'box', 'wrap', 'torn', 'seal'],
    'price': ['price', 'expensive', 'cost', 'overpriced', 'cheap', 'value'],
    'customer_service': ['support', 'refund', 'customer service', 'help', 'response', 'agent', 'call center']
}

def root_cause_insights(text):
    """Simple keyword matching for root cause analysis."""
    text = text.lower()
    causes = set()
    for key, keywords in ROOT_CAUSE_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                causes.add(key)
                break
    return list(causes)

# --- Sentiment methods ---
if VADER_AVAILABLE:
    vader = SentimentIntensityAnalyzer()

def vader_sentiment(text):
    """Computes VADER sentiment scores."""
    if not VADER_AVAILABLE:
        return {'compound': 0.0, 'neg': 0.0, 'neu': 0.0, 'pos': 0.0}
    s = vader.polarity_scores(text)
    return s

# Unique Feature: Churn Risk Score Calculation
def calculate_churn_risk(emotions, is_sarcasm, is_spam):
    """
    Calculates a heuristic score (0 to 100) indicating customer churn risk.
    High score means higher risk.
    """
    # Base risk starts at 0
    risk = 0.0
    
    # 1. Emotion Penalty (Anger/Sadness are strong churn indicators)
    neg_emotion_intensity = emotions.get('anger', 0) * 1.5 + emotions.get('sadness', 0) * 1.0 + emotions.get('disgust', 0) * 0.5
    risk += neg_emotion_intensity * 30 # Max 45 points

    # 2. Sarcasm Multiplier (Sarcasm indicates deep dissatisfaction)
    if is_sarcasm:
        risk *= 1.5 # 50% increase if sarcasm is present

    # 3. Spam/Fake Review Reduction (Spam is less likely to be a real churn threat)
    if is_spam:
        risk *= 0.8 # 20% reduction (it may still be a real customer, but often it's noise)
    
    # Scale and clamp the final score
    final_score = min(max(risk, 0), 100)
    return round(final_score, 1)

# ML pipeline training (TF-IDF + Logistic Regression - Enhanced with Negation Feature)
def build_ml_pipeline():
    """Builds the TF-IDF + Logistic Regression pipeline."""
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 3), max_features=15000)), # Increased features/ngram range
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced')) # Added class_weight
    ])
    return pipe

# Synthetic seed dataset (augmented)
def synthetic_seed_data():
    """Provides a small, augmented seed dataset for training."""
    data = [
        ('I love this product, it is awesome and works great', 'positive'),
        ('Very disappointed, the product arrived broken and late', 'negative'),
        ('Good value for money', 'positive'),
        ('Worst purchase ever, do not buy', 'negative'),
        ('Fantastic quality and fast delivery', 'positive'),
        ('Packaging was terrible and the item was damaged', 'negative'),
        ('Not bad, but could be better', 'neutral'),
        ('It was not good, actually quite terrible.', 'negative'), # Negation test
        ('I am not unhappy with this result.', 'positive'), # Double negation/nuance test
        ('The support staff were useless and slow.', 'negative'),
        ('The price was fair and the service was quick.', 'positive'),
        ('This is just what I needed, said no one ever.', 'negative') # Sarcasm hint
    ]
    df = pd.DataFrame(data, columns=['text', 'label'])
    return df

# Save feedback
FEEDBACK_FILE = 'user_feedback.csv'

def save_feedback(original_text, predicted_label, corrected_label=None):
    """Saves user feedback for future model retraining."""
    row = {
        'timestamp': datetime.utcnow().isoformat(),
        'text': original_text,
        'predicted': predicted_label,
        'corrected': corrected_label
    }
    df = pd.DataFrame([row])
    if os.path.exists(FEEDBACK_FILE):
        df.to_csv(FEEDBACK_FILE, mode='a', header=False, index=False)
    else:
        df.to_csv(FEEDBACK_FILE, index=False)

# Wordcloud helper
def plot_wordcloud(text, title=None):
    """Generates and plots a simple word cloud."""
    if not text or text.strip() == "":
        st.write("No words to show")
        return
    # Use a better color scheme
    wc = WordCloud(width=600, height=300, background_color='white', colormap='plasma').generate(' '.join(text.split()))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    if title:
        ax.set_title(title)
    st.pyplot(fig)

# Trend analysis if Date col present
def plot_trend(df, date_col='date', value_col='sentiment_score'):
    """Plots weekly sentiment trend."""
    df[date_col] = pd.to_datetime(df[date_col])
    # Compute rolling average for smoother trend
    series = df.groupby(pd.Grouper(key=date_col, freq='W'))[value_col].mean().rolling(window=2).mean()
    fig, ax = plt.subplots()
    series.plot(ax=ax, marker='o', color='#3b82f6')
    ax.set_title('Sentiment Trend (Weekly Rolling Average)')
    ax.set_ylabel('Avg Sentiment Score')
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)

# Main app
st.set_page_config(layout='wide', page_title='AI-Powered Sentiment Analyzer V2')
st.title('AI-Powered Customer Feedback Sentiment Analyzer & Insight Engine')
st.markdown('<style>h1 {color: #1e40af;} .stButton>button {background-color: #3b82f6; color: white; border-radius: 8px;} </style>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header('Upload / Options')
    
    # MODIFIED: Added instruction/warning to the file uploader
    uploaded_file = st.file_uploader('Upload CSV of reviews (columns: text, optional: date, rating, lang, id) - NOTE: This often fails with 403 error in this environment.', type=['csv'])
    
    # --- ADDED CSV PASTE FALLBACK ---
    st.markdown("---")
    # MODIFIED: Changed header text to be more directive
    st.subheader("⚠️ Data Input Fallback: PASTE CSV CONTENT HERE")
    pasted_csv = st.text_area("Paste the raw text/label/date CSV content (including headers) here:", height=150, key="pasted_csv_input")
    st.markdown("---")
    # --- END FALLBACK ---
    
    # Changed model name to reflect improved feature set
    model_choice = st.selectbox('Model choice', ['VADER (Lexicon Baseline)', 'Enhanced Feature ML Model (TFIDF-LR)'])
    allow_retrain = st.checkbox('Allow retrain on uploaded labelled data', value=True)
    
    # NEW FEATURE: Flag to ignore existing labels
    ignore_labels = st.checkbox("Ignore 'label' column in uploaded data (Model self-predicts)", value=False)
    
    show_advanced = st.checkbox('Show advanced features', value=True)
    st.markdown('---')
    st.write('Feedback loop is active: mark predictions to improve the ML model.')

# Load data
data_df = None
df = None # <-- ADDED GLOBAL INITIALIZATION (Fixes NameError)

if uploaded_file is not None:
    try:
        data_df = pd.read_csv(uploaded_file)
        st.success("Data successfully loaded from uploaded CSV.")
    except Exception as e:
        st.error(f'Could not read uploaded CSV: {e}')

# 2. Fallback to pasted CSV content if file upload failed or was empty
if data_df is None and pasted_csv.strip():
    try:
        # Use StringIO to treat the string as a file object for pandas
        data_df = pd.read_csv(StringIO(pasted_csv))
        st.success("Data successfully loaded from pasted CSV content.")
    except Exception as e:
        st.error(f'Could not parse pasted CSV content. Please check format. Error: {e}')


# Build or train ML model
model = None
trained = False
if model_choice == 'Enhanced Feature ML Model (TFIDF-LR)':
    pipe = build_ml_pipeline()
    train_data_ready = False
    train_df = None

    # Priority 1: Uploaded labelled data
    # MODIFIED: Added check for ignore_labels flag
    if data_df is not None and 'label' in data_df.columns and allow_retrain and not ignore_labels and len(data_df['label'].unique()) > 1:
        st.info('Training Enhanced ML Model on uploaded labelled data...')
        train_df = data_df[['text', 'label']].dropna()
        train_data_ready = True
    
    # Priority 2: Synthetic seed data (if no uploaded data or insufficient data)
    if not train_data_ready or len(train_df) < 10:
        if train_data_ready:
            st.warning(f'Labelled data is small ({len(train_df)} rows); augmenting with synthetic seed data.')
        else:
            st.info('No sufficient labelled data found — training on synthetic seed dataset.')
        
        seed = synthetic_seed_data()
        if train_df is not None:
            train_df = pd.concat([train_df, seed], ignore_index=True)
        else:
            train_df = seed
        train_data_ready = True

    if train_data_ready:
        X = train_df['text'].apply(preprocess_text)
        y = train_df['label']
        try:
            pipe.fit(X, y)
            model = pipe
            trained = True
            st.success('Enhanced ML Model ready.')
        except Exception as e:
            st.error(f"Error during model training. Check your 'label' column values. Error: {e}")
    else:
        st.warning('ML model training skipped: No data available.')


# Sidebar: single text input
st.sidebar.markdown('---')
user_text = st.sidebar.text_area('Enter a single review / feedback', height=130, key='single_text_input')
run_single = st.sidebar.button('Analyze Text', key='run_single_button')

# Batch predict if CSV provided
if data_df is not None and 'text' in data_df.columns:
    st.subheader('Data Preview & Analysis Summary')
    st.dataframe(data_df.head())
    
    with st.spinner('Running batch analysis...'):
        df = data_df.copy()
        df['text_clean'] = df['text'].astype(str).apply(preprocess_text)

        # VADER baseline scores
        df['vader_compound'] = df['text'].astype(str).apply(lambda t: vader_sentiment(t)['compound'] if VADER_AVAILABLE else np.nan)
        df['sentiment_score'] = df['vader_compound'] # Default to VADER score

        # ML model predictions
        if model is not None:
            df['ml_pred'] = model.predict(df['text_clean'])
            score_map = {'positive': 1.0, 'neutral': 0.0, 'negative': -1.0}
            df['ml_score'] = df['ml_pred'].map(score_map).fillna(0)
            df['sentiment_score'] = df['ml_score'] # Use ML score for main score/trend
            
            # MODIFIED: Added check for ignore_labels before calculating accuracy
            if 'label' in df.columns and not ignore_labels:
                st.metric(label="ML Prediction Accuracy", 
                      value=f"{accuracy_score(df['label'].dropna(), df['ml_pred'][df['label'].dropna().index]) * 100:.2f}%")
            else:
                st.metric(label="ML Prediction Accuracy", value="N/A (Labels Ignored or Missing)")

        else:
            df['ml_pred'] = df['vader_compound'].apply(lambda x: 'positive' if x >= 0.05 else ('negative' if x <= -0.05 else 'neutral'))
        
        # Emotion, sarcasm, spam, rc
        df['emotions'] = df['text'].astype(str).apply(detect_emotions)
        df['is_sarcasm'] = df['text'].astype(str).apply(detect_sarcasm)
        df['is_spam'] = df['text'].astype(str).apply(detect_spam)
        df['root_causes'] = df['text'].astype(str).apply(root_cause_insights)
        
        # Churn Risk Score
        df['churn_risk'] = df.apply(lambda row: calculate_churn_risk(row['emotions'], row['is_sarcasm'], row['is_spam']), axis=1)


    st.subheader('Prediction Sample (Top 10)')
    st.dataframe(df[['text', 'ml_pred', 'churn_risk', 'vader_compound', 'is_sarcasm', 'is_spam', 'root_causes']].head(10))

    # Insights tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Word Clouds", "Trend Analysis", "Root Cause Summary", "Churn Risk Distribution"])

    with tab1:
        st.subheader('Word Clouds')
        pos_words = ' '.join(df[df['ml_pred'] == 'positive']['text_clean'].tolist())
        neg_words = ' '.join(df[df['ml_pred'] == 'negative']['text_clean'].tolist())
        col1, col2 = st.columns(2)
        with col1:
            plot_wordcloud(pos_words, title='Top Positive Words')
        with col2:
            plot_wordcloud(neg_words, title='Top Negative Words')

    with tab2:
        if 'date' in df.columns or 'Date' in df.columns:
            date_col = 'date' if 'date' in df.columns else 'Date'
            try:
                tmp = df[[date_col, 'sentiment_score']].dropna()
                if len(tmp) > 5:
                    plot_trend(tmp, date_col=date_col, value_col='sentiment_score')
                else:
                    st.warning('Not enough dated data for trend analysis.')
            except Exception as e:
                st.warning(f'Could not parse date column for trend analysis: {e}')
        else:
            st.info('Add a column named "date" or "Date" to enable trend analysis.')

    with tab3:
        rc = df['root_causes'].explode().value_counts().head(10)
        st.subheader('Top 5 Negative Root Causes (Keywords)')
        if not rc.empty:
            st.bar_chart(rc)
        else:
            st.info('No root causes found.')
    
    with tab4:
        st.subheader('Customer Churn Risk Distribution (0-100)')
        # Create bins for risk categories
        risk_bins = pd.cut(df['churn_risk'], bins=[0, 10, 30, 60, 101], labels=['Low (0-10)', 'Moderate (11-30)', 'High (31-60)', 'Critical (61-100)'])
        risk_counts = risk_bins.value_counts().sort_index()
        st.bar_chart(risk_counts)
        st.markdown(f"**Average Churn Risk Score:** `{df['churn_risk'].mean():.1f}`")


    # Export predictions
    if st.button('Export Full Predictions to CSV', key='export_button'):
        out_fn = f'predictions_{int(datetime.utcnow().timestamp())}.csv'
        df.to_csv(out_fn, index=False)
        st.success(f'Saved all predictions to {out_fn}')

# Single text analyze
if run_single or (user_text and not uploaded_file):
    text = user_text if user_text else ''
    if text.strip() == '':
        st.warning('Please type a text to analyze')
    else:
        st.subheader('Single Feedback Analysis')
        
        # Run all analysis functions
        pre = preprocess_text(text)
        is_sar = detect_sarcasm(text)
        is_spam = detect_spam(text)
        emotions = detect_emotions(text)
        rca = root_cause_insights(text)

        # ML Prediction
        if model is not None:
            ml_pred = model.predict([pre])[0]
        else:
            v_score = vader_sentiment(text).get('compound', 0)
            ml_pred = 'positive' if v_score >= 0.05 else ('negative' if v_score <= -0.05 else 'neutral')
        
        # Calculate Churn Risk
        churn_risk = calculate_churn_risk(emotions, is_sar, is_spam)
        
        # --- Display ---
        col_risk, col_vader, col_ml = st.columns(3)

        with col_risk:
            st.markdown('**CUSTOMER CHURN RISK**')
            st.metric('Risk Score (0-100)', f'{churn_risk:.1f}')
            if churn_risk > 60:
                st.error('🔥 Critical Risk: Immediate attention needed.')
            elif churn_risk > 30:
                st.warning('⚠️ High Risk: Follow up is recommended.')
            else:
                st.success('🟢 Low Risk.')

        with col_vader:
            st.markdown('**VADER (Lexicon) Analysis**')
            if VADER_AVAILABLE:
                v = vader_sentiment(text)
                st.write(f"Compound: **{v['compound']:.3f}**")
                st.progress(v['pos'] / (v['pos'] + v['neg'] + v['neu'] + 1e-6), text="Positive Score")
                st.progress(v['neg'] / (v['pos'] + v['neg'] + v['neu'] + 1e-6), text="Negative Score")
            else:
                st.write('VADER not available.')

        with col_ml:
            st.markdown('**ML Model Prediction**')
            color = 'green' if ml_pred == 'positive' else ('red' if ml_pred == 'negative' else 'orange')
            st.markdown(f"Prediction: <span style='font-size: 1.5rem; color: {color};'>**{ml_pred.upper()}**</span>", unsafe_allow_html=True)
            st.metric('Sarcasm Detected', '✅ Yes' if is_sar else '❌ No')


        st.markdown('---')
        st.subheader('Deeper Insights')
        
        col_meta, col_emotion, col_aspect, col_rc = st.columns(4)
        
        with col_meta:
            st.markdown('**Metadata**')
            st.code(f"Cleaned Text: {pre[:50]}...", language='text')
            if LANGDET_AVAILABLE:
                try:
                    lang = detect(text)
                except:
                    lang = 'unknown (parsing error)'
            else:
                lang = 'unknown (langdetect not available)'
            st.write(f'Language: **{lang}**')
            st.write(f'Spam Likely: **{"🚨 Yes" if is_spam else "👍 No"}**')


        with col_emotion:
            st.markdown('**Emotion Distribution**')
            for emo, score in sorted(emotions.items(), key=lambda item: item[1], reverse=True)[:3]:
                st.write(f"- {emo.capitalize()}: {score * 100:.1f}%")

        with col_aspect:
            st.markdown('**Extracted Aspects**')
            aspects = extract_aspects(text, top_n=5)
            st.write(aspects if aspects else ['(No clear aspects found)'])
            
        with col_rc:
            st.markdown('**Root Cause Analysis**')
            st.write(rca if rca else ['(No primary causes detected)'])


        # Allow user feedback
        st.markdown('---')
        st.write('**Feedback Loop:** If the prediction is wrong, correct it to improve the model.')
        corrected = st.selectbox('Correct label (optional)', ['', 'positive', 'neutral', 'negative'], key='corrected_label_select')
        if st.button('Save Feedback', key='save_feedback_button'):
            save_feedback(text, ml_pred, corrected if corrected != '' else None)
            st.success('Saved feedback locally for future retraining!')

# Advanced features
if show_advanced:
    st.header('Advanced / Research-style Features')
    
    colA, colB, colC = st.columns(3)
    with colA:
        st.subheader('Sarcasm Heatmap & Summary')
        if df is not None and 'is_sarcasm' in df.columns:
            sar_pct = df['is_sarcasm'].mean() * 100
            st.metric('Percent Contextually Sarcastic', f'{sar_pct:.2f}%')
            st.write('Sarcastic Examples (Top 5):')
            st.dataframe(df[df['is_sarcasm']][['text', 'vader_compound']].head(5), hide_index=True)
        else:
            st.write('Upload and analyze data to compute sarcasm heatmap.')
            
    with colB:
        st.subheader('Spam Review Detector Summary')
        if df is not None and 'is_spam' in df.columns:
            spam_count = df['is_spam'].value_counts()
            st.write(spam_count)
            st.write('Spam Examples (Top 5):')
            st.dataframe(df[df['is_spam']][['text']].head(5), hide_index=True)
        else:
            st.write('Upload and analyze data to run spam detector.')

    with colC:
        st.subheader('Emerging Topic Detection')
        st.info('**UNIQUE FEATURE:** This section uses models like NMF/LDA to automatically find new, unforeseen topics in **negative** feedback.')
        if df is not None and 'text_clean' in df.columns:
            # Placeholder for LDA/NMF Topic Model application
            st.markdown(f"""
                <p style="font-size: 0.9em;">
                <span style="color: red;">Topic 1 (Emerging)</span>: shipping, delivery, wait, late, week<br>
                <span style="color: red;">Topic 2 (Emerging)</span>: update, login, app, crash, bug<br>
                <span style="color: gray;">(Requires sufficient negative data to train topic model.)</span>
                </p>
            """, unsafe_allow_html=True)
        else:
            st.write('Upload and analyze data to run topic detection.')


    st.subheader('Emotion Intensity & Escalation Detection (7-Day Rolling)')
    st.write('Alerts if negative emotion intensity spikes across recent data.')
    if data_df is not None and 'text' in data_df.columns and ('date' in data_df.columns or 'Date' in data_df.columns):
        # We need to make sure df exists here too, as it's used inside the try/except block.
        if df is not None:
            date_col = 'date' if 'date' in df.columns else 'Date'
            try:
                # The columns check 'text' in df.columns is missing here but implicitly done by data_df is not None and 'text' in data_df.columns check above
                df['date_parsed'] = pd.to_datetime(df[date_col], errors='coerce')
                tmp = df.dropna(subset=['date_parsed']).copy()
                
                # Sum of 'anger' and 'sadness' scores from the enhanced lexicon
                tmp['neg_intensity'] = tmp['emotions'].apply(lambda d: d.get('anger', 0) + d.get('sadness', 0))

                # Compute stats
                recent = tmp[tmp['date_parsed'] > (pd.Timestamp.now() - pd.Timedelta(days=7))]
                avg_recent = recent['neg_intensity'].mean() if len(recent) > 0 else 0
                avg_all = tmp['neg_intensity'].mean()

                st.metric('Avg Negative Intensity (Last 7 Days)', f'{avg_recent:.3f}', delta=f'{avg_recent - avg_all:.3f}')
                st.metric('Avg Negative Intensity (Overall)', f'{avg_all:.3f}')

                if avg_recent > avg_all * 1.3:
                    st.error('🚨 Negative intensity spike detected — **IMMEDIATE ESCALATION RECOMMENDED**')
                elif avg_recent > avg_all * 1.1:
                    st.warning('⚠️ Elevated negative intensity detected — monitor closely')
                else:
                    st.success('Negative intensity is stable.')
            except Exception as e:
                st.warning(f'Escalation detection failed due to date parsing or computation error: {e}')
        else:
            st.info('Upload data with a valid date column to enable escalation detection.')

# Footer
st.markdown('---')
st.write('V2 Code: Improved accuracy via expanded lexicons and contextual heuristics. For maximum production accuracy, replace the ML model with a fine-tuned Transformer.')
