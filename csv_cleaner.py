import os
import sys
import json

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

def classify_dtype(series, total_rows):

    dtype = str(series.dtype)

    if dtype in ["int64", "float64", "UInt32", "UInt64"]:
        return "number"

    try:
       converted = pd.to_numeric(series.dropna(), errors="coerce")
       non_numeric = converted.isna().sum()
       total_non_na = series.dropna().shape[0]

        
       if total_non_na > 0 and (non_numeric / total_non_na) < 0.1:
            return "number"
    except:
        pass

    if "datetime64" in dtype:
        return "date"

   
    sample = series.dropna().astype(str).head(20)
    parsed = sample.apply(lambda x: pd.to_datetime(x, errors="coerce"))
    if len(sample) > 0 and (parsed.notna().sum() / len(sample)) >= 0.8:
        return "date"

 
    bool_like = {"true", "false", "yes", "no", "0", "1", "true", "false"}
    unique_vals = set(str(x).lower() for x in series.dropna().unique())
    if unique_vals.issubset(bool_like):
        return "boolean"


    if dtype == "bool":
        return "boolean"

   
    unique_count = series.nunique(dropna=True)
    if total_rows > 0 and (unique_count / total_rows) < 0.05:
        return "categorical"


    return "text"

def profile_dataframe(df):
    total_rows = df.shape[0]
    total_cols = df.shape[1]

    
    structure = {
        "column_names": list(df.columns),
        "column_order": list(df.columns),
        "num_rows": total_rows,
        "num_columns": total_cols,
        "dtypes": {}
    }

     
    for col in df.columns:
        structure["dtypes"][col] = classify_dtype(df[col], total_rows)
