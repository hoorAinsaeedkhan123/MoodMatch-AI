import streamlit as st
import pandas as pd
import plotly.express as px
from ml_models import FEATURES, load_data, train_models, predict_vibe

st.set_page_config(page_title="MoodMatch AI", page_icon="🎧", layout="wide")

st.markdown("""
<style>
.stApp {background: linear-gradient(135deg, #101322 0%, #17152b 55%, #10242c 100%); color: #f4f4f8;}
.block-container {padding-top: 2rem; max-width: 1150px;}
.hero {padding: 1.6rem 1.8rem; border: 1px solid #393653; border-radius: 22px;
       background: rgba(255,255,255,0.055); margin-bottom: 1.2rem;}
.hero h1 {font-size: 2.7rem; margin-bottom: .2rem;}
.small-muted {color: #b9b7ca;}
.result {padding: 1.2rem; border-radius: 18px; background: rgba(255,255,255,0.07);
         border: 1px solid #494365;}
div[data-testid="stMetric"] {background: rgba(255,255,255,0.06); padding: 14px; border-radius: 14px;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_models():
    return train_models()

classifier, regressor, metrics = get_models()

st.markdown("""
<div class="hero">
  <h1>🎧 MoodMatch AI</h1>
  <p class="small-muted">A playful machine-learning quiz that discovers your entertainment vibe.</p>
  <p>Answer six quick questions and let two ML models predict your vibe category and a fun energy score.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("🧠 The ML behind it")
    st.write("**Logistic Regression** predicts one of five entertainment-vibe categories.")
    st.write("**Linear Regression** estimates a playful energy score from 0 to 100.")
    st.write("**StandardScaler** puts the input features on a comparable scale.")
    st.divider()
    st.caption("This is an entertainment demo, not a scientific personality test.")

left, right = st.columns([1, 1.15], gap="large")
with left:
    st.subheader("✨ Build your vibe")
    music = st.slider("How energetic is your usual music taste?", 1, 5, 3,
                      help="1 = calm and soft · 5 = high-energy")
    social = st.slider("How social do you feel on a free day?", 1, 5, 3,
                       help="1 = solo time · 5 = group plans")
    movie = st.slider("What kind of movie intensity do you enjoy?", 1, 5, 3,
                      help="1 = light and cozy · 5 = intense and thrilling")
    creativity = st.slider("How much do you enjoy making or imagining things?", 1, 5, 3)
    pace = st.slider("What weekend pace sounds best?", 1, 5, 3,
                     help="1 = slow and relaxed · 5 = packed with activities")
    mystery = st.slider("How much do you enjoy puzzles, twists, and mysteries?", 1, 5, 3)
    submitted = st.button("✨ Reveal my vibe", type="primary", use_container_width=True)

with right:
    st.subheader("🔮 Your AI result")
    if submitted:
        answers = {
            "music_energy": music,
            "social_style": social,
            "movie_intensity": movie,
            "creativity": creativity,
            "weekend_pace": pace,
            "mystery_interest": mystery,
        }
        vibe, probs, energy = predict_vibe(answers, classifier, regressor)
        descriptions = {
            "Chill Explorer": ("🌿", "You enjoy a calmer pace, relaxed discoveries, and room to recharge."),
            "Adventure Seeker": ("🚀", "You like energetic plans, exciting stories, and trying something new."),
            "Mystery Lover": ("🕵️", "You enjoy clues, plot twists, puzzles, and stories that keep you guessing."),
            "Creative Dreamer": ("🎨", "You enjoy imagination, creative ideas, and a little wonder in everyday life."),
            "Social Spark": ("🎉", "You get energy from shared experiences, lively plans, and fun conversations."),
        }
        emoji, description = descriptions[vibe]
        st.markdown(f'<div class="result"><h2>{emoji} {vibe}</h2><p>{description}</p></div>',
                    unsafe_allow_html=True)
        st.metric("Your estimated vibe-energy score", f"{energy:.0f}/100")
        st.caption("The score is a playful model estimate, not a measurement of your real mood.")
        prob_df = pd.DataFrame({
            "Vibe": list(probs.keys()),
            "Model confidence": [float(v) for v in probs.values()]
        }).sort_values("Model confidence", ascending=True)
        fig = px.bar(prob_df, x="Model confidence", y="Vibe", orientation="h",
                     range_x=[0, 1], title="How the classifier compared the five vibes")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", height=330,
                          xaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
        st.info("Try changing one or two answers and reveal your vibe again to see how the prediction changes.")
    else:
        st.markdown("""
        <div class="result">
          <h2>Ready when you are ✨</h2>
          <p>Choose your answers on the left, then click <b>Reveal my vibe</b>.</p>
          <p class="small-muted">Your result includes a predicted category, an estimated score, and a probability chart.</p>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        st.write("**Five possible vibes**")
        st.write("🌿 Chill Explorer · 🚀 Adventure Seeker · 🕵️ Mystery Lover")
        st.write("🎨 Creative Dreamer · 🎉 Social Spark")

st.divider()
with st.expander("📊 Explore the sample dataset"):
    data = load_data()
    st.write("The app is trained on a small synthetic dataset included with this project.")
    st.dataframe(data.head(12), use_container_width=True)
    counts = data["vibe"].value_counts().rename_axis("Vibe").reset_index(name="Examples")
    chart = px.pie(counts, names="Vibe", values="Examples", title="Training examples by vibe")
    chart.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(chart, use_container_width=True)

with st.expander("🧪 Model evaluation (held-out test data)"):
    st.write(f"**Logistic Regression accuracy:** {metrics['accuracy']:.1%}")
    st.write(f"**Linear Regression mean absolute error:** {metrics['mae']:.1f} points")
    st.caption("These metrics evaluate models on generated demo data. They do not establish real-world personality accuracy.")

st.caption("Built with Python · Streamlit · pandas · scikit-learn · Plotly")
