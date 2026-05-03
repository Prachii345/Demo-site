<div align="center">

<img src="https://img.shields.io/badge/MechSwap-AI%20Powered-1D6B72?style=for-the-badge&logo=robot&logoColor=white" />
<img src="https://img.shields.io/badge/Built%20With-Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/Deployed%20On-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
<img src="https://img.shields.io/badge/Status-In%20Progress-orange?style=for-the-badge" />
<img src="https://img.shields.io/badge/Recommendation%20System-✅%20Complete-brightgreen?style=for-the-badge" />

<br/><br/>

# 🏭 MechSwap — AI Intelligence Layer

### *Bringing Machine Learning to India's Industrial Machinery Marketplace*

**[🚀 Live Demo](https://demo-site-mechswap.streamlit.app)** • **[🌐 MechSwap Platform](https://mechswap.in)** • **[📬 Contact](mailto:pprachi.be23@thapar.edu)**

<br/>

> *"The industrial machinery market in India is worth billions — yet buyers still search blindly and fake listings cost sellers trust and time. This project changes that."*

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Current Progress](#-current-progress)
- [Recommendation System](#-recommendation-system-completed)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [Demo](#-demo)
- [Installation](#-installation)
- [Upcoming Features](#-upcoming-features)
- [Author](#-author)

---

## 🧩 Overview

MechSwap is a **global marketplace for buying and selling used industrial machinery**. This project adds a complete **AI/ML intelligence layer** on top of the existing platform — making it smarter, safer, and more personalized for every user.

Three core problems being solved:

| Problem | Solution | Status |
|---|---|---|
| Buyers can't find relevant machinery | ✅ Machinery Recommendation Engine | **Complete** |
| Fake listings erode platform trust | 🔄 Automated Fraud Detection System | Coming Soon |
| Sellers don't know fair market price | 🔄 ML-based Price Prediction | Coming Soon |

---

## 📊 Current Progress

```
Phase 1 — Recommendation Engine      ████████████████████  100% ✅
Phase 2 — Fraud Detection            ░░░░░░░░░░░░░░░░░░░░    0% 🔄
Phase 3 — Price Prediction           ░░░░░░░░░░░░░░░░░░░░    0% 🔄
```

---

## ✅ Recommendation System (Completed)

> *"Find the right machine — not just any machine."*

A **hybrid recommendation engine** combining two approaches to give the most accurate and personalized machinery suggestions to every buyer.

### How It Works

**Approach 1 — Content-Based Filtering**
- Converts machinery descriptions and metadata into numerical vectors using **TF-IDF vectorization**
- Computes **cosine similarity** between vectors to find machines with similar attributes
- Works even for new users with no history — just input a machine they viewed and get similar ones
- Best for: *"Show me machines similar to this one"*

**Approach 2 — Collaborative Filtering (SVD)**
- Builds a **user-item interaction matrix** from buyer behavior — views, inquiries, purchases
- Uses **Singular Value Decomposition** to discover hidden preference patterns
- Finds machines that similar buyers liked — without looking at machine content at all
- Best for: *"Show me what buyers like me purchased"*

**Hybrid Combination**
- Both scores are weighted and combined for the final recommendation
- Content-based handles cold start (new users) — Collaborative handles personalization (returning users)
- Best of both worlds — accurate from day one, smarter over time

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.10+ |
| **ML / Data** | Scikit-learn, Surprise (SVD), Pandas, NumPy |
| **NLP** | TF-IDF Vectorizer, Cosine Similarity |
| **Web App** | Streamlit |
| **Visualization** | Matplotlib, Seaborn |
| **Database** | SQLite3 |
| **Deployment** | Streamlit Community Cloud |

---

## 📁 Project Structure

```
mechswap-ai/
│
├── app.py                          # Main Streamlit application
├── requirements.txt                # Dependencies
│
├── recommendation/
│   ├── content_based.py            # TF-IDF + Cosine Similarity
│   ├── collaborative.py            # SVD Matrix Factorization
│   └── hybrid.py                   # Combined recommendation logic
│
├── data/
│   ├── listings.csv                # Machinery listing dataset
│   └── interactions.csv            # User interaction data
│
└── models/
    └── svd_model.pkl               # Saved collaborative filter model
```

---

## ⚙️ How It Works

### Content-Based Filtering

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Convert machinery descriptions to TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(listings['description'])

# Find similar machines using cosine similarity
similarity_matrix = cosine_similarity(tfidf_matrix)

def get_similar_machines(machine_id, top_n=5):
    scores = list(enumerate(similarity_matrix[machine_id]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return scores[1:top_n+1]
```

### Collaborative Filtering (SVD)

```python
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate

# Train SVD on user-item interaction data
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(
    interactions[['user_id', 'machine_id', 'rating']], reader
)

model = SVD(n_factors=50, n_epochs=20, random_state=42)
cross_validate(model, data, measures=['RMSE'], cv=5)

# Predict rating for unseen user-machine pair
prediction = model.predict(user_id, machine_id)
```

### Hybrid Combination

```python
def hybrid_recommend(user_id, machine_id, top_n=5):
    # Get scores from both models
    content_scores = get_similar_machines(machine_id, top_n)
    collab_scores  = get_collab_recommendations(user_id, top_n)

    # Weighted combination — more weight to collaborative
    final_scores = 0.4 * content_scores + 0.6 * collab_scores
    return sorted(final_scores, reverse=True)[:top_n]
```

---

## 🎬 Demo

### 🔗 [Live on Streamlit → demo-site-mechswap.streamlit.app](https://demo-site-mechswap.streamlit.app)

**What you can do on the demo right now:**

- 🔍 Enter a machine type → get **top 5 personalized recommendations**
- 👤 Enter a user ID → see **collaborative filtering** suggestions
- 🔀 Toggle between content-based, collaborative, and hybrid modes

---

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/your-username/mechswap-ai.git
cd mechswap-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

**requirements.txt**
```
streamlit>=1.28.0
scikit-learn>=1.3.0
scikit-surprise>=1.1.3
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

---

## 🔮 Upcoming Features

### 🚨 Phase 2 — Fake Listing Detection *(Coming Soon)*
- Rule-based filters for obvious fraud — price anomaly, missing images, short descriptions
- Isolation Forest anomaly detection for subtle fake patterns
- Fraud probability score (0–1) assigned to every new listing automatically

### 💰 Phase 3 — Price Prediction *(Coming Soon)*
- XGBoost regression model trained on listing features
- Suggests fair price range to sellers before listing
- Helps buyers verify market value instantly

### 🔍 Phase 4 — NLP Semantic Search *(Planned)*
- Sentence-BERT embeddings for natural language search
- Buyers describe what they need in plain language
- System finds relevant machinery regardless of exact keyword match

---

## 👩‍💻 Author

<div align="center">

**Prachi**
*AI/ML Engineer Intern @ MechSwap*
*B.E. Electronics & Communication — Thapar Institute of Engineering & Technology*
*CGPA: 8.31 | Batch of 2027*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin)](https://linkedin.com)
[![LeetCode](https://img.shields.io/badge/LeetCode-500%2B%20Problems-FFA116?style=for-the-badge&logo=leetcode)](https://leetcode.com)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-1D6B72?style=for-the-badge&logo=googlechrome&logoColor=white)](https://your-portfolio.com)

*Selected for McKinsey Forward Program 2025 · AlgoUniversity Fellowship (Top 800/20,000)*

</div>

---

<div align="center">

**⭐ Star this repo to follow the progress — fraud detection and price prediction coming next!**

*Built with 💙 for MechSwap — making India's industrial marketplace smarter*

</div>
