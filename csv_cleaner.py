import os
import sys
import json
import re

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

            # STATISTICAL METADATA
    stats = {
        "missing_pct": {},
        "unique_value_counts": {}
    }

    for col in df.columns:
        col_data = df[col]

        # missing percentage
        missing = col_data.isna().sum()
        stats["missing_pct"][col] = round((missing / total_rows) * 100, 2) if total_rows > 0 else 0

        # unique counts (excluding NaN)
        stats["unique_value_counts"][col] = col_data.nunique(dropna=True)

            # SAMPLE ROWS (first 20 rows as dictionaries)
    sample_rows = []

    sample_df = df.head(20)
    for _, row in sample_df.iterrows():
        sample_rows.append(row.to_dict())

    
    return {
        "structure": structure,
        "stats": stats,
        "samples": sample_rows
    }

def generate_cleaning_plan(profile):
    prompt = f"""
You are a data cleaning planner.  
Return ONLY valid JSON.  
NO commentary.  
NO text outside the JSON.  

JSON schema:
{{
  "actionable_steps": [
    {{
      "column": "column_name",
      "action": "standardize_case | strip_whitespace | unify_date_format | normalize_categorical"
    }}
  ],
  "issues_for_review": []
}}

Here is the dataset profile:
{json.dumps(profile, indent=2)}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": prompt}
        ]
    )

    text = response.choices[0].message.content

    # Extract JSON using regex (GPT-safe extraction)
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if not json_match:
        raise ValueError("No JSON found in model output.")

    json_str = json_match.group(0)

    try:
        plan = json.loads(json_str)
    except:
        raise ValueError("Model returned invalid JSON:\n" + json_str)

    return plan


def build_cleaning_prompt(profile):
    return f"""
You are an expert data-cleaning planner.

Your job is to read the dataset profile and create two things:

1. actionable_steps
   - Only include these actions:
     - standardize_case
     - strip_whitespace
     - unify_date_format (must use YYYY-MM-DD)
     - normalize_categorical

2. issues_for_review
   - Anything uncertain, risky, ambiguous, or requiring human judgment
   - Do NOT apply transformations you're unsure about
   - Examples: invalid ages, unclear category meanings, ambiguous dates, outliers, mixed numeric/text columns

RULES:
- You MUST return valid JSON.
- Do NOT include explanations, only structured JSON.
- actionable_steps should ONLY include allowed actions.
- issues_for_review captures everything else.
- Never delete rows or guess values in v1.
- Never input missing values.
- Never change numeric values.
- Never hallucinate columns.

Here is the dataset profile you must analyze:
{json.dumps(profile, indent=2)}
"""


def apply_cleaning_plan(df, plan):
    df = df.copy()
    changes_applied = []

    
    for step in plan.get("actionable_steps", []):
        col = step.get("column")
        action = step.get("action")

        if col not in df.columns:
            continue  

        if action == "standardize_case":
            df[col] = df[col].astype(str).str.lower()
            changes_applied.append(f"Standardized case in column '{col}'.")

        elif action == "strip_whitespace":
            df[col] = df[col].astype(str).str.strip()
            changes_applied.append(f"Stripped whitespace in column '{col}'.")

        elif action == "unify_date_format":
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%Y-%m-%d")
            changes_applied.append(f"Unified date format in column '{col}' to YYYY-MM-DD.")

        elif action == "normalize_categorical":
            df[col] = df[col].astype(str).str.lower().str.strip()
            changes_applied.append(f"Normalized categorical values in column '{col}'.")

        return df, changes_applied


def generate_cleaning_report(profile, plan, changes_applied):
    issues = plan.get("issues_for_review", [])

    report_prompt = f"""
You are an expert data quality analyst.  
Generate a clear, concise cleaning report based on the following information.

---------------------
PROFILE SUMMARY
---------------------
{json.dumps(profile, indent=2)}

---------------------
ACTIONS APPLIED
---------------------
{json.dumps(changes_applied, indent=2)}

---------------------
ISSUES FLAGGED FOR REVIEW
---------------------
{json.dumps(issues, indent=2)}

Write a professional cleaning report in plain text.  
Include these sections:

1. Overview of the dataset
2. Summary of all changes applied
3. Issues flagged for human review
4. Recommendations for next steps  
5. Any warnings about data quality risks

Do NOT output JSON.  
The final output must be natural language ONLY.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You write professional data quality reports."},
            {"role": "user", "content": report_prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        print("Usage: python csv_cleaner.py <csv_file_path>")
        sys.exit(1)

    file_path = sys.argv[1]

    try:
        df = pd.read_csv(file_path)
        print(f"Loaded file: {file_path}")
    except Exception as e:
        print(f"Failed to load CSV: {e}")
        sys.exit(1)

 
    profile = profile_dataframe(df)
    print("Profile generated.")


   
    plan = generate_cleaning_plan(profile)
    print("Cleaning plan generated.")


    cleaned_df, changes_applied = apply_cleaning_plan(df, plan)
    print("Cleaning applied.")

   
    report_text = generate_cleaning_report(profile, plan, changes_applied)
    print("Cleaning report generated.")


   
    cleaned_filename = f"cleaned_{os.path.basename(file_path)}"
    cleaned_df.to_csv(cleaned_filename, index=False)
    print(f"Cleaned CSV saved to: {cleaned_filename}")

   
    report_filename = "cleaning_report.txt"
    with open(report_filename, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"Cleaning report saved to: {report_filename}")
    print("Process complete.")

