# Libraries
import pandas as pd
import streamlit as st


# Load Files
@st.cache_data
def load_file(uploaded_file) -> pd.DataFrame | None:
    name = uploaded_file.name.lower()

    try:
        if name.endswith(".csv"):
            for sep in [",", ";", "\t", "|"]:
                try:
                    df = pd.read_csv(uploaded_file, sep=sep)
                    if df.shape[1] >= 2:
                        break
                    uploaded_file.seek(0)
                except Exception:
                    uploaded_file.seek(0)

        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)

        elif name.endswith(".parquet"):
            df = pd.read_parquet(uploaded_file)

        elif name.endswith(".json"):
            df = pd.read_json(uploaded_file)

        elif name.endswith(".txt"):
            for sep in ["\t", ",", ";", "|", " "]:
                try:
                    df = pd.read_csv(uploaded_file, sep=sep)
                    if df.shape[1] >= 2:
                        break
                    uploaded_file.seek(0)
                except Exception:
                    uploaded_file.seek(0)
        else:
            st.error("Unsupported format. Please use CSV, Excel, Parquet, JSON, or TXT.")
            return None

        return df

    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None


@st.cache_data
def parse_dataframe(df: pd.DataFrame) -> pd.DataFrame | None:
    if df.shape[1] < 2:
        st.error("The file must have at least 2 columns: dates and returns.")
        return None

    date_col = None
    for col in df.columns:
        try:
            parsed = pd.to_datetime(df[col])
            date_col = col
            df[col] = parsed
            break
        except Exception:
            continue

    if date_col is None:
        st.error("No valid date column found.")
        return None

    return_col = None
    for col in df.columns:
        if col == date_col:
            continue
        try:
            df[col] = pd.to_numeric(df[col], errors="raise")
            return_col = col
            break
        except Exception:
            continue

    if return_col is None:
        st.error("No numeric returns column found.")
        return None

    result = df[[date_col, return_col]].rename(
        columns={date_col: "date", return_col: "return"}
    )
    result = result.dropna().sort_values("date").reset_index(drop=True)
    result = result.set_index("date")
    result.index = pd.to_datetime(result.index)

    return result
