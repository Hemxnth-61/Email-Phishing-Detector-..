# Step 1: Load the dataset
from datasets import load_dataset
import pandas as pd

print("Loading dataset...")
dataset = load_dataset('zefang-liu/phishing-email-dataset')

# Convert to pandas dataframe
df = dataset['train'].to_pandas()

print("\n✓ Dataset loaded!")
print(f"Shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

print(f"\nColumn names:")
print(df.columns.tolist())

print(f"\nEmail Type distribution:")
print(df['Email Type'].value_counts())
# Step 2: Clean data
print("\n" + "="*50)
print("STEP 2: Clean Data")
print("="*50)

# Remove missing values
df = df.dropna(subset=['Email Text'])
print(f"After removing NaN: {len(df)} emails")

# Convert to lowercase
df['Email Text'] = df['Email Text'].str.lower()
print("✓ Converted text to lowercase")

# Create binary labels (0 = Safe, 1 = Phishing)
df['label'] = (df['Email Type'] == 'Phishing Email').astype(int)

print("\nCleaned data sample:")
print(df[['Email Text', 'label']].head(3))

print(f"\nLabel distribution:")
print(df['label'].value_counts())
# Step 3: Split data
print("\n" + "="*50)
print("STEP 3: Split Data (Train/Test)")
print("="*50)

from sklearn.model_selection import train_test_split

X = df['Email Text'].to_numpy()  # Convert to numpy array
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,           # 20% for testing
    random_state=42,         # For reproducibility
    stratify=y               # Keep same ratio in train/test
)

print(f"Training set: {len(X_train)} emails")
print(f"Testing set: {len(X_test)} emails")
print(f"Ratio: {len(X_train)/len(X_test):.1f}:1")

print(f"\nTraining set - Safe: {(y_train == 0).sum()}, Phishing: {(y_train == 1).sum()}")
print(f"Testing set - Safe: {(y_test == 0).sum()}, Phishing: {(y_test == 1).sum()}")
# Step 4: TF-IDF Vectorization
print("\n" + "="*50)
print("STEP 4: TF-IDF Vectorization")
print("="*50)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_features=2000,      # Keep top 2000 words
    min_df=2,               # Word must appear in at least 2 emails
    max_df=0.95,            # Word shouldn't appear in 95%+ of emails
    ngram_range=(1, 2),     # Single words + two-word phrases
    stop_words='english'    # Remove common words like "the", "a", "is"
)

# Transform training data
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"Training shape: {X_train_tfidf.shape}")
print(f"Testing shape: {X_test_tfidf.shape}")

# See some features
features = vectorizer.get_feature_names_out()
print(f"\nTotal features created: {len(features)}")
print(f"Sample features (first 20):")
print(features[:20])
# Step 5: Train the model
print("\n" + "="*50)
print("STEP 5: Train Naive Bayes Model")
print("="*50)

from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB(alpha=0.1)
model.fit(X_train_tfidf, y_train)

print("✓ Model trained!")
print(f"Model type: {type(model)}")
# Step 6: Test and evaluate
print("\n" + "="*50)
print("STEP 6: Evaluate Model")
print("="*50)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report

# Make predictions
y_pred = model.predict(X_test_tfidf)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"  Accuracy:  {accuracy*100:.2f}%")
print(f"  Precision: {precision*100:.2f}%")
print(f"  Recall:    {recall*100:.2f}%")
print(f"  F1-Score:  {f1*100:.2f}%")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(f"  True Negatives:  {cm[0,0]}")
print(f"  False Positives: {cm[0,1]}")
print(f"  False Negatives: {cm[1,0]}")
print(f"  True Positives:  {cm[1,1]}")

# Detailed report
print(f"\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Safe', 'Phishing']))
# Step 7: Try on custom emails
print("\n" + "="*50)
print("STEP 7: Test on Custom Emails")
print("="*50)

test_emails = [
    "Hi, let's schedule a meeting tomorrow at 2pm",
    "URGENT: Click here to verify your account immediately!",
    "Your password has been changed. If this wasn't you, contact us.",
    "Meeting notes from today attached below",
    "CONFIRM your bank details now or account will be suspended",
]

for email in test_emails:
    # Vectorize
    email_tfidf = vectorizer.transform([email])
    
    # Predict
    prediction = model.predict(email_tfidf)[0]
    confidence = model.predict_proba(email_tfidf)[0]
    
    label = "🔴 PHISHING" if prediction == 1 else "🟢 SAFE"
    phishing_prob = confidence[1] * 100
    
    print(f"\n{label}")
    print(f"Email: \"{email[:60]}...\"")
    print(f"Confidence: {phishing_prob:.1f}%")
    # Step 8: Save the model
print("\n" + "="*50)
print("STEP 8: Save Model")
print("="*50)

import pickle

with open('phishing_model.pkl', 'wb') as f:
    pickle.dump({
        'model': model,
        'vectorizer': vectorizer,
        'accuracy': accuracy
    }, f)

print("✓ Model saved to 'phishing_model.pkl'")
print(f"✓ You can now use this in the web app!")
