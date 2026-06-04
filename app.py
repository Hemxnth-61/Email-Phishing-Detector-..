import streamlit as st
import pickle
import re
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Phishing Detector",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Phishing Email Detector")
st.markdown("Detect phishing emails using ML (trained by my_detector.py)")

# Load the saved model
@st.cache_resource
def load_model():
    """Load model trained by my_detector.py"""
    try:
        with open('phishing_model.pkl', 'rb') as f:
            data = pickle.load(f)
        return data['model'], data['vectorizer'], data['accuracy']
    except FileNotFoundError:
        st.error("❌ Model file not found! Please run 'python my_detector.py' first.")
        st.stop()

# Load
model, vectorizer, accuracy = load_model()

# Sidebar info
st.sidebar.markdown("### 📊 Model Info")
st.sidebar.metric("Model Accuracy", f"{accuracy*100:.2f}%")
st.sidebar.markdown("**Algorithm:** Multinomial Naive Bayes")
st.sidebar.markdown("**Features:** TF-IDF (2000)")
st.sidebar.markdown("**Trained by:** my_detector.py")

# Main tabs
tab1, tab2, tab3 = st.tabs(["🔍 Analyze", "📚 How It Works", "🧪 Examples"])

# TAB 1: Analyze
with tab1:
    st.markdown("### Paste your email below:")
    
    email_text = st.text_area(
        "Email Content:",
        height=200,
        placeholder="Paste the email you want to analyze..."
    )
    
    if st.button("🔍 Analyze Email", use_container_width=True):
        if not email_text.strip():
            st.warning("⚠️ Please paste an email!")
        else:
            # Analyze
            email_lower = email_text.lower()
            email_tfidf = vectorizer.transform([email_lower])
            prediction = model.predict(email_tfidf)[0]
            confidence = model.predict_proba(email_tfidf)[0]
            
            # Extract links
            url_pattern = r'https?://[^\s]+'
            links = re.findall(url_pattern, email_text)
            
            # Display result
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.error("⚠️ PHISHING DETECTED")
                    st.metric("Risk Level", "HIGH", delta="Dangerous")
                else:
                    st.success("✅ SAFE EMAIL")
                    st.metric("Risk Level", "LOW", delta="Safe")
            
            with col2:
                st.metric(
                    "Safe Probability",
                    f"{confidence[0]*100:.1f}%"
                )
                st.metric(
                    "Phishing Probability",
                    f"{confidence[1]*100:.1f}%"
                )
            
            # Links found
            st.markdown("### 🔗 Links Found")
            if links:
                st.warning(f"Found {len(links)} link(s):")
                for i, link in enumerate(links, 1):
                    st.code(link)
            else:
                st.info("No links found in email")
            
            # Detailed analysis
            st.markdown("### 📋 Detailed Analysis")
            col1, col2, col3 = st.columns(3)
            
            col1.metric("Email Length", f"{len(email_text)} characters")
            col2.metric("Word Count", len(email_text.split()))
            col3.metric("Links Count", len(links))


# TAB 2: How It Works
with tab2:
    st.markdown("### 🤖 How the Model Works")
    
    st.markdown("""
    #### 1. **Training (my_detector.py)**
    - Loads 18,650 emails
    - Cleans and splits data (80% train, 20% test)
    - Converts text to numbers using TF-IDF
    - Trains Naive Bayes classifier
    - Saves model to `phishing_model.pkl`
    
    #### 2. **Deployment (app.py)**
    - Loads the saved model
    - Takes user input (email text)
    - Vectorizes with same TF-IDF
    - Makes prediction
    - Shows results
    
    #### 3. **TF-IDF Vectorization**
    - Converts email text into 2000 numerical features
    - Words that appear in phishing emails get HIGH scores
    - Common words get LOW scores
    - Result: Each email becomes a vector of numbers
    
    #### 4. **Naive Bayes Classifier**
    - Learns patterns from training data
    - Calculates P(Phishing | Email)
    - Fast, simple, effective for text
    """)
    
    st.markdown("### 📊 Model Performance")
    
    metrics_data = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
        "Score": ["93.0%", "93.1%", "88.8%", "90.9%"]
    }
    st.table(pd.DataFrame(metrics_data))


# TAB 3: Examples
with tab3:
    st.markdown("### 🧪 Try These Examples")
    
    examples = {
        "✅ Safe - Meeting": """Hi team,

Let's schedule our weekly standup for tomorrow at 2 PM.
I've attached the agenda for your review.

Best regards,
John""",
        
        "⚠️ Phishing - Urgency": """URGENT ACTION REQUIRED!

Your account has been compromised. 
Click here immediately to verify your identity:
http://verify-account-urgent-now.tk/login

DO NOT IGNORE THIS!""",
        
        "✅ Safe - Project Update": """Project Status Update

Hi all,

Q3 deliverables are on track. The team has made great progress.
Next meeting: Friday 3 PM in Conference Room B.

Regards,
Manager""",
        
        "⚠️ Phishing - Prize": """CONGRATULATIONS! YOU'VE WON!

You've been selected to claim a $500,000 prize!
Click below to claim now:
http://prize-claim-xyz.ru/winner

Limited time offer!"""
    }
    
    for title, content in examples.items():
        if st.button(f"Try: {title}", use_container_width=True, key=title):
            st.session_state.example_text = content
    
    if 'example_text' in st.session_state:
        st.text_area("", value=st.session_state.example_text, height=150, disabled=True, key="display")
        
        # Analyze it
        email_lower = st.session_state.example_text.lower()
        email_tfidf = vectorizer.transform([email_lower])
        prediction = model.predict(email_tfidf)[0]
        confidence = model.predict_proba(email_tfidf)[0]
        
        col1, col2 = st.columns(2)
        with col1:
            if prediction == 1:
                st.error(f"⚠️ PHISHING - {confidence[1]*100:.1f}% confidence")
            else:
                st.success(f"✅ SAFE - {confidence[0]*100:.1f}% confidence")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ using Streamlit | ML Model: TF-IDF + Naive Bayes</p>
    <p><small>Model trained: my_detector.py | App deployed: app.py</small></p>
</div>
""", unsafe_allow_html=True)