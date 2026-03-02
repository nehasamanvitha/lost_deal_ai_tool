import pandas as pd
import streamlit as st
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- UI ----------------
st.set_page_config(page_title="Adaptive Lost Deals AI", layout="wide")
st.markdown("<h1 style='text-align:center; color:#4B0082;'>🚀 Adaptive Lost Deals AI Tool</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Works with ANY dataset: auto-detects text, predicts lost reasons, and shows charts.</p>", unsafe_allow_html=True)
st.write("---")

# ---------------- Sidebar ----------------
st.sidebar.header("Settings & Filters")
uploaded_file = st.sidebar.file_uploader("Upload CSV/TSV", type=["csv"])
show_charts = st.sidebar.checkbox("Show Charts", True)

# ---------------- Load CSV ----------------
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, sep="\t", engine='python')
    except:
        df = pd.read_csv(uploaded_file, engine='python')
    df.columns = [c.strip() for c in df.columns]

    st.subheader("📄 Uploaded CSV Preview")
    st.dataframe(df.head())

    # ---------------- Detect Text Column ----------------
    text_cols = df.select_dtypes(include='object').columns.tolist()
    note_col = None
    for col in text_cols:
        if any(k in col.lower() for k in ["note", "comment", "description", "stage"]):
            note_col = col
            break
    if note_col is None and text_cols:
        note_col = text_cols[0]  # fallback to first text column
    if note_col is None:
        df['Notes'] = "No text available"
        note_col = 'Notes'
    else:
        df.rename(columns={note_col: 'Notes'}, inplace=True)

    # ---------------- Load AI ----------------
    st.subheader("🤖 Loading AI Model...")
    classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

    # ---------------- Predict Lost Reason ----------------
    def predict_reason(note):
        note = str(note).lower()
        if any(k in note for k in ["price", "budget"]):
            return "Price/Budget"
        elif any(k in note for k in ["competitor", "switched"]):
            return "Competitor"
        elif any(k in note for k in ["feature", "missing"]):
            return "Feature"
        elif any(k in note for k in ["approval", "internal"]):
            return "Internal Approval"
        else:
            result = classifier(note)[0]['label']
            return "Negative Feedback" if "NEGATIVE" in result else "Other"

    df['Predicted Lost Reason'] = df['Notes'].apply(predict_reason)
    st.subheader("✅ Predicted Lost Reasons")
    st.dataframe(df)

    # ---------------- Charts ----------------
    if show_charts:
        st.subheader("📊 Analysis")
        if 'Predicted Lost Reason' in df.columns:
            fig1, ax1 = plt.subplots()
            df['Predicted Lost Reason'].value_counts().plot.pie(
                autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
            plt.ylabel('')
            st.pyplot(fig1)

        for col in ['Competitor', 'Sector', 'Revenue', 'Employees']:
            if col in df.columns and not df[col].dropna().empty:
                st.subheader(f"Analysis by {col}")
                fig, ax = plt.subplots()
                try:
                    sns.countplot(x=col, data=df, palette="pastel", order=df[col].value_counts().index)
                    plt.xticks(rotation=45)
                except:
                    df[col].plot(kind='bar', color='purple')
                st.pyplot(fig)

    # ---------------- Download ----------------
    df.to_csv("lost_deals_adaptive.csv", index=False, sep="\t")
    st.download_button("💾 Download CSV with Predictions", open("lost_deals_adaptive.csv", "rb"), "lost_deals_adaptive.csv", "text/csv")
    st.success("🎉 Your Adaptive Lost Deals AI is ready!")