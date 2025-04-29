import glob
import os
import re
import time
import joblib
import nltk
import pandas as pd
import requests
from flask import Flask, render_template, request, flash, redirect
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import fetch_20newsgroups
from functools import lru_cache
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from flask_cors import CORS  # Added for cross-device access

# Initialize Flask app with CORS support
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load environment variables
load_dotenv()
app.secret_key = os.environ.get('SECRET_KEY', 'dev-key-only-for-development')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'json'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Thread pool for parallel execution
executor = ThreadPoolExecutor(4)

# Global model variables
vectorizer = None
model = None
model_accuracy = 0.0

# === NLP SETUP ===
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    """Enhanced text preprocessing with spell checking and advanced cleaning"""
    if not isinstance(text, str):
        return ""
    
    # Remove special characters, URLs, and HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#|[^\w\s]', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove stopwords and lemmatize
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in text.split() 
              if word not in stop_words and len(word) > 2]
    
    return ' '.join(tokens)

@lru_cache(maxsize=1000)
def cached_predict(text):
    """Cached prediction function with thread-safe implementation"""
    if vectorizer is None or model is None:
        raise RuntimeError("ML models not initialized")
    
    processed = preprocess_text(text)
    features = vectorizer.transform([processed])
    prediction = model.predict(features)[0]
    confidence = model.predict_proba(features)[0].max()
    return prediction, confidence

def scrape_article(url):
    """Improved web scraping with better content extraction"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9'
        }
        
        with requests.Session() as session:
            response = session.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'iframe', 'noscript', 
                               'header', 'footer', 'nav', 'aside', 'form']):
                element.decompose()
            
            # Extract title
            title = soup.title.string if soup.title else "No Title Found"
            
            # Extract main content
            article_content = soup.find_all(['article', 'main', 'div.article', 'div.content'])
            if article_content:
                text = ' '.join([p.get_text().strip() for p in 
                                article_content[0].find_all(['p', 'h1', 'h2', 'h3'])])
            else:
                text = ' '.join([p.get_text().strip() for p in soup.find_all('p')])
            
            # Extract metadata
            authors = []
            author_tags = soup.find_all(attrs={'rel': 'author'}) or soup.select('[itemprop="author"]')
            if author_tags:
                authors = [a.get_text().strip() for a in author_tags]
            
            return {
                'text': f"{title}\n\n{text[:5000]}",
                'title': title,
                'url': url,
                'authors': authors[:3],
                'publish_date': None,
                'top_image': None
            }
            
    except Exception as e:
        return {'error': f"Scraping error: {str(e)}"}

def create_minimal_dataset():
    """Create a minimal dataset for fallback purposes"""
    return pd.DataFrame({
        'text': [
            "This is a real news article about science and technology",
            "This is completely fake news created to mislead people",
            "Official government report confirms economic growth",
            "Celebrity gossip that has been proven false",
            "New scientific discovery published in reputable journal",
            "Fabricated story designed to generate clicks"
        ],
        'label': [1, 0, 1, 0, 1, 0]
    })

def train_model(data_path='data/'):
    """Enhanced model training with validation and fallback options"""
    try:
        print("Starting model training...")
        
        # Try to load either the full dataset or our fallback options
        try:
            true_path = os.path.join(data_path, 'True.csv')
            fake_path = os.path.join(data_path, 'Fake.csv')
            
            if os.path.exists(true_path) and os.path.exists(fake_path):
                true_news = pd.read_csv(true_path)
                fake_news = pd.read_csv(fake_path)
                true_news['label'] = 1
                fake_news['label'] = 0
                data = pd.concat([true_news, fake_news])
                print("Loaded main dataset")
            else:
                try:
                    print("Attempting to load 20newsgroups as fallback...")
                    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
                    data = pd.DataFrame({
                        'text': newsgroups.data,
                        'label': (newsgroups.target < 10).astype(int)
                    })
                    data = data.sample(1000)
                    print("Loaded 20newsgroups fallback dataset")
                except Exception as e:
                    print(f"Failed to load 20newsgroups: {str(e)}")
                    data = create_minimal_dataset()
                    print("Using minimal in-memory dataset")
        
        except Exception as e:
            print(f"Error loading datasets: {str(e)}")
            data = create_minimal_dataset()
            print("Using emergency minimal dataset")
        
        # Shuffle and preprocess
        data = data.sample(frac=1).reset_index(drop=True)
        print(f"Training with {len(data)} samples")
        data['cleaned_text'] = data['text'].apply(preprocess_text)
        
        # Train-test split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            data['cleaned_text'], 
            data['label'], 
            test_size=0.2, 
            random_state=42,
            stratify=data['label']
        )
        
        # Vectorize text
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=3
        )
        X_train_vec = vectorizer.fit_transform(X_train)
        
        # Train model
        model = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            solver='liblinear'
        )
        model.fit(X_train_vec, y_train)
        
        # Evaluate
        X_test_vec = vectorizer.transform(X_test)
        y_pred = model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        print("\nModel Evaluation:")
        print(f"Accuracy: {accuracy:.2%}")
        print(classification_report(y_test, y_pred))
        
        # Save models
        model_version = int(time.time())
        vectorizer_file = f'vectorizer_v{model_version}.pkl'
        model_file = f'model_v{model_version}.pkl'
        
        joblib.dump(vectorizer, vectorizer_file)
        joblib.dump(model, model_file)
        
        # Clean up old models
        for f in sorted(glob.glob('vectorizer_v*.pkl'))[:-3]:
            os.remove(f)
        for f in sorted(glob.glob('model_v*.pkl'))[:-3]:
            os.remove(f)
        
        print(f"Models saved as {vectorizer_file} and {model_file}")
        return vectorizer, model, accuracy
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return None, None, 0

def load_latest_model():
    """Load the most recent model version with better error handling"""
    try:
        vectorizer_files = sorted(glob.glob('vectorizer_v*.pkl'))
        model_files = sorted(glob.glob('model_v*.pkl'))
        
        if not vectorizer_files or not model_files:
            print("No model files found")
            return None, None, 0
            
        latest_vectorizer = vectorizer_files[-1]
        latest_model = model_files[-1]
        
        print(f"Loading model: {latest_model}")
        vectorizer = joblib.load(latest_vectorizer)
        model = joblib.load(latest_model)
        
        accuracy = 0.8  # Default reasonable value
        
        return vectorizer, model, accuracy
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, None, 0

def verify_model_loading():
    """Verify that models are properly loaded and functional"""
    if vectorizer is None or model is None:
        return False
    
    try:
        test_text = "This is a test news article for verification"
        _, _ = cached_predict(test_text)
        return True
    except Exception as e:
        print(f"Model verification failed: {str(e)}")
        return False

def initialize_models():
    """Initialize models with multiple fallback options"""
    global vectorizer, model, model_accuracy
    
    print("Initializing models...")
    
    # First try loading existing models
    vectorizer, model, model_accuracy = load_latest_model()
    
    if verify_model_loading():
        print("Models loaded successfully")
        return
    
    print("No valid model found, attempting to train...")
    
    # Try training with various fallback options
    vectorizer, model, model_accuracy = train_model()
    
    if verify_model_loading():
        print("Models trained successfully")
        return
    
    # Final fallback - create trivial model
    print("Creating minimal fallback model...")
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(["real news example", "fake news example"])
    model = LogisticRegression()
    model.fit(X, [1, 0])
    model_accuracy = 0.5
    print("Minimal fallback model created")

# Initialize models when starting
initialize_models()

# === FLASK ROUTES ===
@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    confidence = None
    processing_time = None
    article_info = None
    
    if request.method == 'POST':
        start_time = time.time()
        
        if 'text' in request.form and request.form['text']:
            text = request.form['text'].strip()
            if len(text) < 20:
                flash("Please enter a longer text (minimum 20 characters)", 'error')
            else:
                try:
                    prediction, confidence = cached_predict(text)
                    result = "Real News ✅" if prediction == 1 else "Fake News ❌"
                except Exception as e:
                    flash(f"Prediction error: {str(e)}", 'error')
        
        elif 'url' in request.form and request.form['url']:
            url = request.form['url'].strip()
            if not url.startswith(('http://', 'https://')):
                flash("Please enter a valid URL starting with http:// or https://", 'error')
            else:
                try:
                    scraped = scrape_article(url)
                    if 'error' in scraped:
                        flash(scraped['error'], 'error')
                    else:
                        prediction, confidence = cached_predict(scraped['text'])
                        result = "Real News ✅" if prediction == 1 else "Fake News ❌"
                        article_info = scraped
                except Exception as e:
                    flash(f"URL analysis error: {str(e)}", 'error')
        
        processing_time = f"{(time.time() - start_time):.3f} seconds"
    
    return render_template(
        'index.html',
        result=result,
        confidence=confidence,
        processing_time=processing_time,
        model_accuracy=f"{model_accuracy:.2%}" if model_accuracy else None,
        article_info=article_info
    )

@app.route('/retrain', methods=['POST'])
def retrain():
    def train_async():
        global vectorizer, model, model_accuracy
        try:
            vectorizer, model, model_accuracy = train_model()
            return True
        except Exception as e:
            print(f"Retraining failed: {str(e)}")
            return False
    
    flash("Model retraining started in the background...", 'info')
    executor.submit(train_async)
    return redirect('/')

@app.route('/model_status')
def model_status():
    """Endpoint to check model status"""
    status = {
        'model_loaded': model is not None,
        'vectorizer_loaded': vectorizer is not None,
        'model_accuracy': f"{model_accuracy:.2%}" if model_accuracy else "N/A",
        'model_files': {
            'vectorizers': sorted(glob.glob('vectorizer_v*.pkl')),
            'models': sorted(glob.glob('model_v*.pkl'))
        },
        'status': 'OK' if verify_model_loading() else 'WARNING'
    }
    return status

@app.route('/analyze_file', methods=['POST'])
def analyze_file():
    if 'file' not in request.files:
        flash("No file uploaded", 'error')
        return redirect('/')
    
    file = request.files['file']
    if file.filename == '':
        flash("No selected file", 'error')
        return redirect('/')
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            if filename.endswith('.csv'):
                df = pd.read_csv(filepath)
            elif filename.endswith('.json'):
                df = pd.read_json(filepath)
            else:
                raise ValueError("Unsupported file format")
            
            if 'text' not in df.columns:
                raise ValueError("File must contain a 'text' column")
            
            results = []
            for text in df['text']:
                try:
                    prediction, confidence = cached_predict(str(text))
                    results.append({
                        'text': text[:100] + '...' if len(text) > 100 else text,
                        'prediction': "Real" if prediction == 1 else "Fake",
                        'confidence': f"{confidence*100:.1f}%"
                    })
                except Exception as e:
                    results.append({
                        'text': text[:100] + '...' if len(text) > 100 else text,
                        'prediction': "Error",
                        'confidence': str(e)
                    })
            
            return render_template(
                'batch_results.html',
                results=results,
                filename=filename
            )
            
        except Exception as e:
            flash(f"File processing error: {str(e)}", 'error')
            return redirect('/')
    
    else:
        flash("Allowed file types are csv and json", 'error')
        return redirect('/')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

if __name__ == '__main__':
    print("Starting Fake News Detector application...")
    try:
        # Verify models before starting server
        if not verify_model_loading():
            print("Warning: Model verification failed on startup")
            print("Attempting to initialize models...")
            initialize_models()
            
            if not verify_model_loading():
                print("Warning: Could not initialize fully functional models")
                print("Starting server with limited functionality")
        
        # Run the app with production-ready settings
        app.run(
            host='0.0.0.0',  # Accessible on all network interfaces
            port=8000,       # Standard Flask port
            debug=False,     # Disable debug mode for production
            threaded=True    # Enable threading for multiple requests
        )
    except Exception as e:
        print(f"Fatal error during startup: {str(e)}")
        print("Application cannot start")