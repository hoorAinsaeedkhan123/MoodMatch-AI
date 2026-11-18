# 🎧 MoodMatch AI — Entertainment Vibe Predictor

MoodMatch AI is an interactive machine-learning quiz. Users answer six light-hearted preference questions and receive:
- A predicted entertainment-vibe category
- A playful estimated vibe-energy score
- A chart showing the model's probabilities across five categories
- An option to explore the sample training dataset and evaluation metrics

## 🤖 Machine-learning concepts
- **Multiclass Logistic Regression:** predicts one of five vibe categories.
- **Linear Regression:** estimates a continuous 0–100 vibe-energy score.
- **StandardScaler:** scales input features before model training.
- **Train/test split:** evaluates each model on held-out examples.
- **Accuracy and MAE:** reports simple classification and regression metrics.

## 🧩 Vibe categories
- 🌿 Chill Explorer
- 🚀 Adventure Seeker
- 🕵️ Mystery Lover
- 🎨 Creative Dreamer
- 🎉 Social Spark

## ▶️ Run locally (Windows)
1. Download and extract this repository.
2. Open the `MoodMatch_AI` folder in VS Code.
3. Open **Terminal → New Terminal**.
4. Install dependencies:

   ```bash
   py -m pip install -r requirements.txt
   ```

5. Run the app:

   ```bash
   py -m streamlit run app.py
   ```

6. Open the local URL printed in the terminal if your browser does not open automatically.

## 📁 Project structure
```text
MoodMatch_AI/
├── .streamlit/
│   └── config.toml
├── data/
│   └── vibe_training_data.csv
├── tests/
│   └── test_models.py
├── app.py
├── ml_models.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 📌 Dataset note
The included dataset is **synthetic demo data** generated for this educational project. It is designed to demonstrate a complete ML workflow without collecting personal information. The labels follow playful, predefined patterns with added noise. The model is not a validated psychological assessment and should not be used to infer someone's real personality or mental health.

## 💼 Portfolio description
Built MoodMatch AI, an interactive entertainment-vibe predictor using Python, Streamlit, multiclass Logistic Regression, and Linear Regression. Implemented feature scaling, train/test evaluation, probability visualizations, and an explorable dataset.

## 📚 What I learned
- Preparing tabular data for machine learning
- Training classification and regression models
- Comparing predictions and evaluation metrics
- Connecting scikit-learn models to an interactive Streamlit interface
