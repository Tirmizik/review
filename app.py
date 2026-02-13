import streamlit as st
import pandas as pd
from rapidfuzz import fuzz

# ---- Load Data ----
reviews_df = pd.read_csv(r"C:\Users\hakima\Desktop\Project\review\data\reviews_behavior.csv")
reviews_df.columns = reviews_df.columns.str.strip()

# ---- Check required columns ----
required_cols = ['app_name', 'content', 'behavior_score']
if not all(col in reviews_df.columns for col in required_cols):
    st.error(f"Your CSV must have columns: {required_cols}")
else:
    st.title("App UX/UI Analysis - Smart Merged Feedback")
    st.write("Strengths and weaknesses merged, subsets removed, differing words combined, duplicates cleared.")

    # ---- Select App ----
    app_list = reviews_df['app_name'].unique()
    selected_app = st.selectbox("Select an App", app_list)

    # ---- Filter reviews for selected app ----
    app_reviews = reviews_df[reviews_df['app_name'] == selected_app]

    # ---- Classify reviews by behavior score ----
    strengths = app_reviews[app_reviews['behavior_score'] ==5]['content'].dropna().tolist()
    weaknesses = app_reviews[app_reviews['behavior_score'] ==1]['content'].dropna().tolist()

    # ---- Smart merge function with:
    # 1) Subset removal
    # 2) Near-duplicate merge
    # 3) Differing word merge
    # 4) Duplicate word removal
    def merge_feedback(feedback_list, threshold=80):
        merged = []
        used = set()
        
        for i, base in enumerate(feedback_list):
            if i in used:
                continue
            group = [base]
            used.add(i)
            for j, other in enumerate(feedback_list):
                if j != i and j not in used:
                    if fuzz.token_sort_ratio(base.lower(), other.lower()) >= threshold:
                        group.append(other)
                        used.add(j)

            # ---- Remove subsets ----
            group_sorted = sorted(group, key=lambda x: len(x))
            final_group = []
            for phrase in group_sorted:
                if not any(phrase in other for other in final_group):
                    final_group.append(phrase)

            # ---- Merge differing words ----
            if len(final_group) == 1:
                merged_phrase = final_group[0]
            else:
                base_words = final_group[0].split()
                for phrase in final_group[1:]:
                    phrase_words = phrase.split()
                    merged_words = []
                    for bw, pw in zip(base_words, phrase_words):
                        if bw.lower() != pw.lower():
                            merged_words.append(f"{bw}/{pw}")
                        else:
                            merged_words.append(bw)
                    # Add remaining words if lengths differ
                    if len(base_words) < len(phrase_words):
                        merged_words += phrase_words[len(base_words):]
                    elif len(phrase_words) < len(base_words):
                        merged_words += base_words[len(phrase_words):]
                    base_words = merged_words
                merged_phrase = ' '.join(base_words)

            # ---- Remove duplicate words while preserving order ----
            seen = set()
            final_phrase = []
            for word in merged_phrase.split():
                if word.lower() not in seen:
                    final_phrase.append(word)
                    seen.add(word.lower())
            merged.append(' '.join(final_phrase))

        return merged

    # ---- Merge strengths & weaknesses ----
    top_strengths = merge_feedback(strengths)
    top_weaknesses = merge_feedback(weaknesses)

    # ---- Display ----
    st.subheader("✅ Strengths")
    if top_strengths:
        for s in top_strengths:
            st.write(f"- {s}")
    else:
        st.write("No strengths found.")

    st.subheader("❌ Weaknesses")
    if top_weaknesses:
        for w in top_weaknesses:
            st.write(f"- {w}")
    else:
        st.write("No weaknesses found.")
