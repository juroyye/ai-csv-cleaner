import streamlit as st
import pandas as pd
import io

from csv_cleaner import (
    profile_dataframe,
    generate_cleaning_plan,
    apply_cleaning_plan,
    generate_cleaning_report
)

st.set_page_config(page_title="AI CSV Cleaner", layout="wide")

st.title("AI CSV Cleaner")
st.write("Upload a CSV file and the AI Cleaner will profile, clean, and document all changes.")

uploaded_file = st.file_uploader("Drag & Drop your CSV file here", type=["csv"])

if uploaded_file:
    st.subheader("📄 Original Dataset Preview")
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

  
    with st.spinner("Analyzing and cleaning your data…"):
        profile = profile_dataframe(df)
        plan = generate_cleaning_plan(profile)
        cleaned_df, changes = apply_cleaning_plan(df, plan)
        report_text = generate_cleaning_report(profile, plan, changes)

    st.success("Cleaning complete!")

    
    st.subheader("✨ Cleaned Dataset Preview")
    st.dataframe(cleaned_df.head())

  
    cleaned_csv = cleaned_df.to_csv(index=False).encode("utf-8")

 
    report_bytes = report_text.encode("utf-8")

    st.download_button(
        label="⬇️ Download Cleaned CSV",
        data=cleaned_csv,
        file_name="cleaned_dataset.csv",
        mime="text/csv"
    )

    st.download_button(
        label="⬇️ Download Cleaning Report",
        data=report_bytes,
        file_name="cleaning_report.txt",
        mime="text/plain"
    )

   
    with st.expander("📘 View Cleaning Report"):
        st.text(report_text)
