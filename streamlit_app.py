import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
import sklearn
from sklearn.preprocessing import MultiLabelBinarizer
# Load libraries
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import graphviz
from sklearn.tree import export_graphviz
from scipy.stats import zscore
import random
#from explainerdashboard import ClassifierExplainer
import plotly.graph_objects as go
import plotly.express as px
import json
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from xgboost import plot_tree
import mlflow
import mlflow.sklearn
import io






st.set_page_config(
    page_title="Insiders Scoopr",
    page_icon="📈",
    layout="wide"
)

# All other Streamlit code goes below this!
st.markdown("# Insiders Scoop")
st.write()
st.write()
st.write("### 🎯 Objective – Insiders Scoop")

st.write("""
**Insiders Scoop** is designed to help users explore and better understand SEC Form 4 insider trading data. The app aims to **predict the likely range** (low, medium, or high) of transaction values based on reported attributes like ownership type, transaction codes, and share activity.

Beyond prediction, the app provides **educational insights** to make insider trades more transparent—highlighting patterns, clarifying confusing terminology, and addressing common issues found in raw SEC filings.

Ultimately, **Insiders Scoop empowers users** to interpret insider trading behavior with confidence, whether they’re investors, students, or simply curious about market dynamics.
""")

st.write()
st.write()
st.write()


page = st.sidebar.selectbox("Select Page",["Introduction 📘","Visualization 📊","Prediction",])
st.image("trade.jpeg")
st.write()
st.write()
st.write()


df = pd.read_csv('insider_sample_df.csv')

df1 = pd.read_csv('insider_sample_df.csv')

df1 = df1[df1["TRANS_PRICEPERSHARE"] <= 1000]


# Create TRADE_VALUE column
df1["TRADE_VALUE"] = df1["TRANS_PRICEPERSHARE"] * df1["TRANS_SHARES"]


df = df1


if page == "Introduction 📘":

    st.subheader("01 Introduction 📘")
    st.video("https://youtu.be/dn7uIqQDHag?si=wVmSIF91aWpZDoD8")
    st.write()
    st.write()
    st.write()


    st.markdown("#### 📊 About the Data")

    st.write("""
    This app is powered by the SEC’s Financial Statement Data Sets, which contain structured data extracted from corporate financial reports filed with the U.S. Securities and Exchange Commission using XBRL (eXtensible Business Reporting Language). The data specifically focuses on insider trading activity disclosed in Form 4 filings, offering a flattened format to make it easier to analyze trends across companies and time periods.

    It's important to note that this data is submitted directly by company insiders, executives, and directors. Because these filings are self-reported, the accuracy of the data relies on the honesty and compliance of the individuals submitting the reports. The SEC does not verify each entry in real time, and errors may also be introduced during the extraction or compilation process.

    While the data is useful for understanding patterns in insider trading, it should not be treated as a definitive source for investment decisions without reviewing the full official filings. The goal of this dataset is to support transparency and enable public analysis — not to replace due diligence since this is also a sample of a much larger dataset.
    """)



    st.markdown("### 🧠 Learn About a Column")

    # Column options (excluding obvious ones)
    column_info = {
        "ACCESSION_NUMBER": {
            "desc": "Unique identifier for the SEC filing.",
            "values": {}
        },
        "FILING_DATE": {
            "desc": "Date the form was filed with the SEC.",
            "values": {}
        },
        "PERIOD_OF_REPORT": {
            "desc": "The reporting period covered by the filing.",
            "values": {}
        },
        "DOCUMENT_TYPE": {
            "desc": "Type of SEC filing (Form 3, 4, or 5, and their amendments).",
            "values": {
                "Form 3": "Initial Statement of Beneficial Ownership of Securities",
                "Form 3/A": "Amendment of a previous Form 3",
                "Form 4": "Statement of Changes of Beneficial Ownership of Securities",
                "Form 4/A": "Amendment of a previous Form 4",
                "Form 5": "Annual Statement of Beneficial Ownership of Securities",
                "Form 5/A": "Amendment of a previous Form 5"
            }
        },
        "ISSUERTRADINGSYMBOL": {
            "desc": "Ticker symbol of the issuing company.",
            "values": {}
        },
        "ISSUERNAME": {
            "desc": "Full name of the issuing company.",
            "values": {}
        },
        "RPTOWNER_RELATIONSHIP": {
            "desc": "The filer’s relationship to the company (e.g., Officer, Director).",
            "values": {}
        },
        "RPTOWNER_TITLE": {
            "desc": "The title or position held by the reporting person.",
            "values": {}
        },
        "RPTOWNER_STATE": {
            "desc": "State of the reporting person's address.",
            "values": {}
        },
        "RPTOWNER_CITY": {
            "desc": "City of the reporting person's address.",
            "values": {}
        },
        "RPTOWNER_ZIPCODE": {
            "desc": "ZIP code of the reporting person.",
            "values": {}
        },
        "TRANS_PRICEPERSHARE": {
            "desc": "Price per share during the transaction.",
            "values": {}
        },
        "SHRS_OWND_FOLWNG_TRANS": {
            "desc": "Shares owned after the transaction.",
            "values": {}
        },
        "DIRECT_INDIRECT_OWNERSHIP": {
            "desc": "Ownership type (direct or indirect).",
            "values": {
                "D": "Direct ownership",
                "I": "Indirect ownership"
            }
        },
        "TRANS_FORM_TYPE": {
            "desc": "Subtype of the transaction form.",
            "values": {}
        },
        "SECURITY_TITLE": {
            "desc": "Type of security traded (e.g., Common Stock).",
            "values": {}
        },
        "TRANS_SHARES": {
            "desc": "Number of shares involved in the transaction.",
            "values": {}
        },
        "TRANS_ACQUIRED_DISP_CD": {
            "desc": "Indicates if shares were acquired or disposed.",
            "values": {
                "A": "Acquired",
                "D": "Disposed"
            }
        },
        "LATITUDE": {
            "desc": "Latitude of the filer’s location.",
            "values": {}
        },
        "LONGITUDE": {
            "desc": "Longitude of the filer’s location.",
            "values": {}
        },
        "QUARTER": {
            "desc": "Quarter and year of the filing (e.g., 2024Q1).",
            "values": {}
        }
    }


    # Let user select a column
    selected_col = st.selectbox("Choose a column to learn about:", options=list(column_info.keys()))

    # Let user choose explanation type
    explanation_type = st.radio("Select explanation type:", ["Quick Overview", "Detailed Explanation"])

    # Show result
    if selected_col:
        st.markdown(f"**{selected_col}**")
        st.write(column_info[selected_col]["desc"])
        
        if explanation_type == "Detailed Explanation" and column_info[selected_col]["values"]:
            st.markdown("**Possible Values:**")
            for key, val in column_info[selected_col]["values"].items():
                st.markdown(f"• `{key}` = {val}")



    st.markdown("##### 🔍 Data Preview")
    rows = st.slider("Select a number of rows to display",5,20,5)
    st.dataframe(df.head(rows))



    st.markdown("##### ⚠️ Missing Values")
    missing = df.isnull().sum()
    st.write(missing)

    if missing.sum() == 0:
        st.success("✅ No missing values found")
    else:
        st.warning("⚠️ you have missing values")



    st.markdown("##### 📈 Summary Statistics")
    if st.button("Show Describe Table"):
        st.dataframe(df.describe())















elif page == "Visualization 📊":

    ## Step 03 - Data Viz
    st.subheader("02 Data Viz  📊")

    # Get numeric columns only (concrete data is numeric)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    # Dropdowns for user selection
    col_x = st.selectbox("Select X-axis variable", numeric_cols, index=0)
    col_y = st.selectbox("Select Y-axis variable", numeric_cols, index=1)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8= st.tabs(["Scatter Plot 🔹", "Line Chart 📈", "Correlation Heatmap 🔥","Bar Chart 📊","Pie Chart 🥧","Pairwise Relationships 🧩","Insider Trading Map 🗺️","📅 Trade Value Over Time"
])

    with tab1:
        st.subheader(f"Scatter Plot: {col_x} vs {col_y}")
        fig_scatter, ax_scatter = plt.subplots()
        sns.scatterplot(x=df[col_x], y=df[col_y], ax=ax_scatter)
        ax_scatter.set_xlabel(col_x)
        ax_scatter.set_ylabel(col_y)
        st.pyplot(fig_scatter)

    with tab2:
        st.subheader(f"Line Chart: {col_y} over {col_x}")
        fig_line, ax_line = plt.subplots()
        df_sorted = df.sort_values(by=col_x)
        ax_line.plot(df_sorted[col_x], df_sorted[col_y], marker='o')
        ax_line.set_xlabel(col_x)
        ax_line.set_ylabel(col_y)
        st.pyplot(fig_line)

    with tab3:
        st.subheader("Correlation Heatmap")
        fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
        corr_matrix = df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax_corr)
        ax_corr.set_title("Correlation Matrix")
        st.pyplot(fig_corr)

    with tab4:
        st.subheader("Bar Chart 📊")

        # Force X-axis to be categorical for bar chart
        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
        bar_x = st.selectbox("Choose a categorical X-axis", cat_cols, key="bar_col_x")

        # Let user choose Y-axis: Count, Percentage, or Numeric column to aggregate
        bar_y_options = ["Count", "Percentage"] + numeric_cols
        bar_y = st.selectbox("Choose Y-axis (metric)", bar_y_options, key="bar_col_y")

        # Use top 10 categories to avoid clutter
        top_x_vals = df[bar_x].value_counts().dropna().head(10).index
        df_filtered = df[df[bar_x].isin(top_x_vals)]

        if bar_y == "Count":
            y_vals = df_filtered[bar_x].value_counts().sort_index()
            y_label = "Count"
        elif bar_y == "Percentage":
            y_vals = (df_filtered[bar_x].value_counts(normalize=True) * 100).sort_index()
            y_label = "Percentage (%)"
        else:
            y_vals = df_filtered.groupby(bar_x)[bar_y].mean().sort_index()
            y_label = f"Average {bar_y}"

        fig_bar, ax_bar = plt.subplots()
        sns.barplot(x=y_vals.index, y=y_vals.values, ax=ax_bar)

        ax_bar.set_xlabel(bar_x)
        ax_bar.set_ylabel(y_label)
        ax_bar.set_title(f"{y_label} by {bar_x}")
        ax_bar.tick_params(axis='x', rotation=45)
        st.pyplot(fig_bar)


    with tab5:
        st.subheader("Pie Chart 🥧")

        # Choose a categorical column
        selected_pie_col = st.selectbox("Choose a column for pie chart", cat_cols, key="pie_col")

        # Top 10 categories only
        pie_counts = df[selected_pie_col].value_counts().dropna().head(10)

        fig_pie, ax_pie = plt.subplots()

        # Plot wedges without long labels
        wedges, texts, autotexts = ax_pie.pie(
            pie_counts.values,
            autopct='%1.1f%%',
            startangle=90,
            textprops=dict(color="white")
        )

        ax_pie.axis("equal")  # Equal aspect ratio
        ax_pie.set_title(f"Top 10 {selected_pie_col}")

        # Add legend with category names on the side
        ax_pie.legend(
            wedges, 
            pie_counts.index, 
            title="Category", 
            loc="center left", 
            bbox_to_anchor=(1, 0.5)
        )

        st.pyplot(fig_pie)


    with tab6:
        st.subheader("Pairwise Relationships 🧩")

        st.markdown("This plot shows relationships between all numerical columns.")

        # Sample data if too large
        sample_df = df[numeric_cols].dropna()
        if len(sample_df) > 500:
            sample_df = sample_df.sample(500, random_state=42)

        # Create the pairplot
        st.spinner("Generating pairwise plots...")
        fig_pair = sns.pairplot(sample_df)
        st.pyplot(fig_pair)

    with tab7:
        st.subheader("Insider Trading Map 🗺️")

        if "LATITUDE" in df.columns and "LONGITUDE" in df.columns:
            # Drop rows with missing coordinates
            map_df = df[["LATITUDE", "LONGITUDE"]].dropna()

            if map_df.empty:
                st.warning("No valid coordinates available to map.")
            else:
                st.map(map_df.rename(columns={"LATITUDE": "lat", "LONGITUDE": "lon"}))
        else:
            st.error("Latitude and Longitude columns not found in the dataset.")

    with tab8:
        st.subheader("📅 Quarterly Insider Trade Value (2024–2025)")

        # Ensure FILING_DATE is datetime
        df1["FILING_DATE"] = pd.to_datetime(df1["FILING_DATE"], errors='coerce')

        # Filter for years 2024 and 2025
        df1_filtered = df1[df1["FILING_DATE"].dt.year.isin([2024, 2025])].copy()

        # Extract Year and Quarter
        df1_filtered["YEAR"] = df1_filtered["FILING_DATE"].dt.year
        df1_filtered["QUARTER"] = df1_filtered["FILING_DATE"].dt.quarter

        # Create label like "2024 Q1"
        df1_filtered["YEAR_QUARTER"] = df1_filtered["YEAR"].astype(str) + " Q" + df1_filtered["QUARTER"].astype(str)

        # Group and sum
        quarterly_data = df1_filtered.groupby("YEAR_QUARTER")["TRADE_VALUE"].sum().reset_index()

        # Ensure correct quarter order
        quarter_order = [f"{y} Q{q}" for y in [2024, 2025] for q in range(1, 5)]
        quarterly_data = quarterly_data.set_index("YEAR_QUARTER").reindex(quarter_order).reset_index()

        # Plot
        fig, ax = plt.subplots()
        ax.plot(quarterly_data["YEAR_QUARTER"], quarterly_data["TRADE_VALUE"], marker='o')
        ax.set_title("Insider Trade Value by Quarter (2024–2025)")
        ax.set_xlabel("Quarter")
        ax.set_ylabel("Total Trade Value ($)")
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)















elif page == "Prediction":
    st.subheader("04 Prediction")

    model_choice = st.selectbox(
        "🧠 Choose a model",
        ["Linear Regression", "Logistic Regression", "K-Nearest Neighbors (KNN)", "Tree", "MLflow Tracking"]
    )

    # 🔁 1️⃣ Start fresh
    encoded_df = df.copy()

    # 🔠 2️⃣ Encode selected categorical columns
    columns_to_encode = [
        "DOCUMENT_TYPE",
        "DIRECT_INDIRECT_OWNERSHIP",
        "TRANS_ACQUIRED_DISP_CD"
    ]
    encoded_df = pd.get_dummies(encoded_df, columns=columns_to_encode, drop_first=False)

    # 🗑️ 3️⃣ Drop irrelevant or unencodable columns
    encoded_df = encoded_df.drop(columns=["RPTOWNER_TITLE", "RPTOWNER_RELATIONSHIP"], errors="ignore")

    # 🔢 4️⃣ Try to convert all object-type columns to numeric
    for col in encoded_df.columns:
        if encoded_df[col].dtype == 'object':
            encoded_df[col] = pd.to_numeric(encoded_df[col], errors='coerce')

    # ❌ 5️⃣ Drop columns with any NaNs (from failed conversions)
    encoded_df = encoded_df.dropna(axis=1, how='any')

    
    list_var = encoded_df.columns.tolist()

    col1, col2 = st.columns(2)

    # 🧠 7️⃣ Model config
    if model_choice == "Linear Regression":
        st.markdown("### Model Configuration")


        with col1:
            target_selection = st.selectbox("🎯 Select Target Variable (Y)", options=list_var)

        with col2:
            x_options = [col for col in list_var if col != target_selection]
            features_selection = st.multiselect("📊 Select Features (X)", options=x_options, default=x_options)


        test_size = st.slider("Select test set size (%)", 10, 50, 20, step=5) / 100.0

        X = encoded_df[features_selection]
        y = encoded_df[target_selection]

        st.write("### Features (X) preview")
        st.dataframe(X.head())
        st.write("### Target (y) preview")
        st.dataframe(y.head())


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        st.write(f"✅ **Train size:** {X_train.shape[0]} rows")
        st.write(f"✅ **Test size:** {X_test.shape[0]} rows")

        
        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        selected_metrics = st.multiselect(
            "Select metrics to display",
            ["Mean Squared Error (MSE)", "Root Mean Squared Error (RMSE)", "Mean Absolute Error (MAE)", "R² Score"],
            default=["Mean Absolute Error (MAE)"]
        )


        if "Mean Squared Error (MSE)" in selected_metrics:
            st.write(f"- **MSE**: {metrics.mean_squared_error(y_test, predictions):,.2f}")
        if "Root Mean Squared Error (RMSE)" in selected_metrics:
            st.write(f"- **RMSE**: {metrics.root_mean_squared_error(y_test, predictions):,.2f}")
        if "Mean Absolute Error (MAE)" in selected_metrics:
            st.write(f"- **MAE**: {metrics.mean_absolute_error(y_test, predictions):,.2f}")
        if "R² Score" in selected_metrics:
            st.write(f"- **R² Score**: {metrics.r2_score(y_test, predictions):,.3f}")

        fig, ax = plt.subplots()
        ax.scatter(y_test, predictions, alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "--r")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted")
        st.pyplot(fig)













    elif model_choice == "Logistic Regression":
        st.markdown("### Model Configuration (Logistic Regression)")

        log_df = df.copy()

        # --- Allow user to toggle between automatic vs manual binning ---
        use_custom_bins = st.checkbox("🛠️ Use Custom Bins for Categorization (PPS, Shares, Trade Value)", value=False)

        if use_custom_bins:
            st.markdown("#### 🔢 Custom Bin Thresholds")

            # User-defined thresholds
            pps_cut = st.number_input("PPS Threshold (e.g. Low vs High)", value=50.0)
            shares_cut = st.number_input("Shares Threshold (e.g. Low vs High)", value=5000)
            tv_cut = st.number_input("Trade Value Threshold (Price * Shares)", value=250_000)

            # Apply user-defined binning
            log_df["PPS_Category"] = pd.cut(log_df["TRANS_PRICEPERSHARE"], bins=[-float("inf"), pps_cut, float("inf")], labels=["Low", "High"])
            log_df["Shares_Category"] = pd.cut(log_df["TRANS_SHARES"], bins=[-float("inf"), shares_cut, float("inf")], labels=["Low", "High"])
            log_df["TradeValue_Category"] = pd.cut(
                log_df["TRANS_PRICEPERSHARE"] * log_df["TRANS_SHARES"],
                bins=[-float("inf"), tv_cut, float("inf")],
                labels=["Low", "High"]
            )

            # Set fake bin edges for reporting
            pps_bins = [-float("inf"), pps_cut, float("inf")]
            shares_bins = [-float("inf"), shares_cut, float("inf")]
            tv_bins = [-float("inf"), tv_cut, float("inf")]

        else:
            # Use automatic quantile-based binning (qcut)
            log_df["PPS_Category"], pps_bins = pd.qcut(log_df["TRANS_PRICEPERSHARE"], q=2, labels=["Low", "High"], retbins=True)
            log_df["Shares_Category"], shares_bins = pd.qcut(log_df["TRANS_SHARES"], q=2, labels=["Low", "High"], retbins=True)
            log_df["TradeValue_Category"], tv_bins = pd.qcut(
                log_df["TRANS_PRICEPERSHARE"] * log_df["TRANS_SHARES"],
                q=2, labels=["Low", "High"], retbins=True
            )

        # Convert categories to string
        log_df["PPS_Category"] = log_df["PPS_Category"].astype(str)
        log_df["Shares_Category"] = log_df["Shares_Category"].astype(str)
        log_df["TradeValue_Category"] = log_df["TradeValue_Category"].astype(str)

        # Let user pick binary target variable
        binary_cols = [col for col in log_df.columns if log_df[col].nunique() == 2 and log_df[col].dtype in ['object', 'bool']]
        selected_binary_col = st.selectbox("🎯 Select Binary Target Variable", binary_cols)

        # Show bin ranges based on selection
        if selected_binary_col == 'PPS_Category':
            st.markdown("#### 🧾 Category Breakdown (Quantile Bins)")
            s = f"""
            <p><strong>Price Per Share Bins (Low–High):</strong></p>
            <ul>
            <li><span style='color: white;'>${pps_bins[0]:,.2f} – ${pps_bins[1]:,.2f} (Low)</span></li>
            <li><span style='color: white;'>${pps_bins[1]:,.2f} – ${pps_bins[2]:,.2f} (High)</span></li>
            </ul>
            """
            st.markdown(s, unsafe_allow_html=True)

            st.write()
            st.write()

        elif selected_binary_col == 'Shares_Category':

            st.markdown("#### 🧾 Category Breakdown (Quantile Bins)")
            s = f"""
            <p><strong>Shares Bins (Low–High):</strong></p>
            <ul>
            <li><span style='color: white;'>{shares_bins[0]:,.2f} – {shares_bins[1]:,.2f} (Low)</span></li>
            <li><span style='color: white;'>{shares_bins[1]:,.2f} – {shares_bins[2]:,.2f} (High)</span></li>
            </ul>
            """
            st.markdown(s, unsafe_allow_html=True)

            st.write()
            st.write()


        elif selected_binary_col == 'TradeValue_Category':
            st.markdown("#### 🧾 Category Breakdown (Quantile Bins)")
            s = f"""
            <p><strong>Trade Value Bins (Low–High):</strong></p>
            <ul>
            <li><span style='color: white;'>${tv_bins[0]:,.2f} – ${tv_bins[1]:,.2f} (Low)</span></li>
            <li><span style='color: white;'>${tv_bins[1]:,.2f} – ${tv_bins[2]:,.2f} (High)</span></li>
            </ul>
            """
            st.markdown(s, unsafe_allow_html=True)

            st.write()
            st.write()
        







        # Create binary target column (1 if value == first category, else 0)
        log_df["TARGET_BINARY"] = (log_df[selected_binary_col] == log_df[selected_binary_col].unique()[0]).astype(int)

        # Display what 0 and 1 stand for
        label_0 = log_df[selected_binary_col].unique()[0]
        label_1 = log_df[selected_binary_col].unique()[1]

        st.markdown(f"""
        #### 🧬 Target Encoding Explanation  
        The target variable has been encoded as:

        - **0 = {label_0}**  
        - **1 = {label_1}**
        """)


        y = log_df["TARGET_BINARY"]

        # One-hot encode categorical features
        columns_to_encode = ["DOCUMENT_TYPE", "DIRECT_INDIRECT_OWNERSHIP"]
        log_df = pd.get_dummies(log_df, columns=columns_to_encode, drop_first=True)

        # Drop unneeded columns (including the original target)
        # Start with known columns to drop
        drop_cols = ["RPTOWNER_TITLE", "RPTOWNER_RELATIONSHIP"]

        # Add raw target column
        drop_cols.append(selected_binary_col)

        # Add one-hot encoded version of the selected column (if it exists)
        for col in log_df.columns:
            if col.startswith(selected_binary_col + "_"):
                drop_cols.append(col)

        # Drop them all
        log_df = log_df.drop(columns=drop_cols, errors="ignore")




        # Convert object columns to numeric
        for col in log_df.columns:
            if log_df[col].dtype == "object":
                log_df[col] = pd.to_numeric(log_df[col], errors="coerce")

        # Drop NaNs
        log_df = log_df.dropna(axis=1)

        # Features
        X = log_df.drop(columns=["TARGET_BINARY"])

        # Drop raw numeric columns if they were used to generate the category target
        if selected_binary_col == "TradeValue_Category":
            log_df = log_df.drop(columns=["TRANS_PRICEPERSHARE", "TRANS_SHARES", "TRADE_VALUE"], errors="ignore")
        elif selected_binary_col == "PPS_Category":
            log_df = log_df.drop(columns=["TRANS_PRICEPERSHARE"], errors="ignore")
        elif selected_binary_col == "Shares_Category":
            log_df = log_df.drop(columns=["TRANS_SHARES"], errors="ignore")


        # Normalize
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)



        st.write("### 🧾 Feature Columns Used in Model")
        st.write(X.columns.tolist())



        #Distribution chart
        class_counts = y.value_counts()
        fig, ax = plt.subplots()
        ax.bar(class_counts.index.astype(str), class_counts.values, color=['#1f77b4', '#ff7f0e'])
        ax.set_title("Class Distribution")
        ax.set_xlabel("Class (0 or 1)")
        ax.set_ylabel("Count")
        st.pyplot(fig)        

        # Train-test split
        
        test_size = st.slider("Test Set Size (%)", 10, 50, 20, step=5) / 100.0
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)

        # Train logistic regression
        
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        
        st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.3f}")
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        # Confusion matrix
        fig, ax = plt.subplots()
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)













    elif model_choice == "K-Nearest Neighbors (KNN)":
        st.markdown("### Model Configuration (KNN Classifier)")

        knn_df = df.copy()
        # --- Allow user to toggle between automatic vs manual binning ---
        use_custom_bins = st.checkbox("🛠️ Use Custom Bins for Categorization (PPS, Shares, Trade Value)", value=False)

        if use_custom_bins:
            st.markdown("#### 🔢 Custom Bin Thresholds")
            pps_cut = st.number_input("PPS Threshold (e.g. Low vs High)", value=50.0)
            shares_cut = st.number_input("Shares Threshold (e.g. Low vs High)", value=5000)
            tv_cut = st.number_input("Trade Value Threshold (Price * Shares)", value=250_000)

            knn_df["PPS_Category"] = pd.cut(knn_df["TRANS_PRICEPERSHARE"], bins=[-float("inf"), pps_cut, float("inf")], labels=["Low", "High"])
            knn_df["Shares_Category"] = pd.cut(knn_df["TRANS_SHARES"], bins=[-float("inf"), shares_cut, float("inf")], labels=["Low", "High"])
            knn_df["TradeValue_Category"] = pd.cut(
                knn_df["TRANS_PRICEPERSHARE"] * knn_df["TRANS_SHARES"],
                bins=[-float("inf"), tv_cut, float("inf")],
                labels=["Low", "High"]
            )

            pps_bins = [-float("inf"), pps_cut, float("inf")]
            shares_bins = [-float("inf"), shares_cut, float("inf")]
            tv_bins = [-float("inf"), tv_cut, float("inf")]
        else:
            knn_df["PPS_Category"], pps_bins = pd.qcut(knn_df["TRANS_PRICEPERSHARE"], q=2, labels=["Low", "High"], retbins=True)
            knn_df["Shares_Category"], shares_bins = pd.qcut(knn_df["TRANS_SHARES"], q=2, labels=["Low", "High"], retbins=True)
            knn_df["TradeValue_Category"], tv_bins = pd.qcut(
                knn_df["TRANS_PRICEPERSHARE"] * knn_df["TRANS_SHARES"],
                q=2, labels=["Low", "High"], retbins=True
            )

        # Convert categories to string
        knn_df["PPS_Category"] = knn_df["PPS_Category"].astype(str)
        knn_df["Shares_Category"] = knn_df["Shares_Category"].astype(str)
        knn_df["TradeValue_Category"] = knn_df["TradeValue_Category"].astype(str)

        # Let user pick target column
        binary_cols = [col for col in knn_df.columns if knn_df[col].nunique() == 2 and knn_df[col].dtype in ['object', 'bool']]
        selected_binary_col = st.selectbox("🎯 Select Binary Target Variable", binary_cols)

        # Show bin ranges
        if selected_binary_col == 'PPS_Category':
            st.markdown("#### 🧾 Category Breakdown (PPS)")
            st.markdown(f"""
            <ul>
            <li><span style='color: white;'>${pps_bins[0]:,.2f} – ${pps_bins[1]:,.2f} (Low)</span></li>
            <li><span style='color: white;'>${pps_bins[1]:,.2f} – ${pps_bins[2]:,.2f} (High)</span></li>
            </ul>
            """, unsafe_allow_html=True)

        elif selected_binary_col == 'Shares_Category':
            st.markdown("#### 🧾 Category Breakdown (Shares)")
            st.markdown(f"""
            <ul>
            <li><span style='color: white;'>{shares_bins[0]:,.0f} – {shares_bins[1]:,.0f} (Low)</span></li>
            <li><span style='color: white;'>{shares_bins[1]:,.0f} – {shares_bins[2]:,.0f} (High)</span></li>
            </ul>
            """, unsafe_allow_html=True)

        elif selected_binary_col == 'TradeValue_Category':
            st.markdown("#### 🧾 Category Breakdown (Trade Value)")
            st.markdown(f"""
            <ul>
            <li><span style='color: white;'>${tv_bins[0]:,.0f} – ${tv_bins[1]:,.0f} (Low)</span></li>
            <li><span style='color: white;'>${tv_bins[1]:,.0f} – ${tv_bins[2]:,.0f} (High)</span></li>
            </ul>
            """, unsafe_allow_html=True)








        # Create binary target column (1 if value == first category, else 0)
        knn_df["TARGET_BINARY"] = (knn_df[selected_binary_col] == knn_df[selected_binary_col].unique()[0]).astype(int)
        

        # Drop raw numeric columns if they were used to generate the category target
        if selected_binary_col == "TradeValue_Category":
            knn_df = knn_df.drop(columns=["TRANS_PRICEPERSHARE", "TRANS_SHARES", "TRADE_VALUE"], errors="ignore")
        elif selected_binary_col == "PPS_Category":
            knn_df = knn_df.drop(columns=["TRANS_PRICEPERSHARE"], errors="ignore")
        elif selected_binary_col == "Shares_Category":
            knn_df = knn_df.drop(columns=["TRANS_SHARES"], errors="ignore")

        # Display what 0 and 1 stand for
        label_0 = knn_df[selected_binary_col].unique()[0]
        label_1 = knn_df[selected_binary_col].unique()[1]

        st.markdown(f"""
        #### 🧬 Target Encoding Explanation  
        The target variable has been encoded as:

        - **0 = {label_0}**  
        - **1 = {label_1}**
        """)




        y = knn_df["TARGET_BINARY"]

        # Encode selected categorical features
        columns_to_encode = ["DOCUMENT_TYPE", "DIRECT_INDIRECT_OWNERSHIP"]
        knn_df = pd.get_dummies(knn_df, columns=columns_to_encode, drop_first=True)

        # Drop unneeded columns, including the original target
        drop_cols = ["RPTOWNER_TITLE", "RPTOWNER_RELATIONSHIP", selected_binary_col]
        knn_df = knn_df.drop(columns=drop_cols, errors="ignore")

        # Convert object columns to numeric if possible
        for col in knn_df.columns:
            if knn_df[col].dtype == "object":
                knn_df[col] = pd.to_numeric(knn_df[col], errors="coerce")

        knn_df = knn_df.dropna(axis=1)


        # Drop raw target column
        drop_cols.append(selected_binary_col)

        # Drop dummy-encoded version if it exists
        for col in knn_df.columns:
            if col.startswith(selected_binary_col + "_"):
                drop_cols.append(col)

        knn_df = knn_df.drop(columns=drop_cols, errors="ignore")


        # Define features
        X = knn_df.drop(columns=["TARGET_BINARY"])

        # Normalize
       
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)


        st.write("### 🧾 Feature Columns Used in Model")
        st.write(X.columns.tolist())


        #Distribution chart
        class_counts = y.value_counts()
        fig, ax = plt.subplots()
        ax.bar(class_counts.index.astype(str), class_counts.values, color=['#1f77b4', '#ff7f0e'])
        ax.set_title("Class Distribution")
        ax.set_xlabel("Class (0 or 1)")
        ax.set_ylabel("Count")
        st.pyplot(fig)        






        # Train-test split
        
        test_size = st.slider("Test Set Size (%)", 10, 50, 20, step=5) / 100.0
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)


        use_grid = st.checkbox("🔍 Use GridSearchCV to find best K?", value=False)

        if use_grid:
            st.write("Searching for best K using GridSearchCV...")
            k_range = list(range(1, 21))
            param_grid = {"n_neighbors": k_range}
            grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring="accuracy")
            grid.fit(X_train, y_train)
            best_k = grid.best_params_["n_neighbors"]
            st.write(f"✅ Best K found: {best_k}")
            knn_model = grid.best_estimator_

            # Plot
            results_df = pd.DataFrame(grid.cv_results_)
            plt.figure(figsize=(8, 4))
            plt.plot(results_df["param_n_neighbors"], results_df["mean_test_score"], marker='o')
            plt.title("K vs. Cross-Validated Accuracy")
            plt.xlabel("Number of Neighbors (K)")
            plt.ylabel("Mean CV Accuracy")
            plt.grid(True)
            st.pyplot(plt)
        else:
            k = st.slider("Select K (number of neighbors)", 1, 20, 5)
            knn_model = KNeighborsClassifier(n_neighbors=k)
            knn_model.fit(X_train, y_train)

        y_pred = knn_model.predict(X_test)

        # Evaluate
        st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.3f}")
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        # Confusion matrix
     
        fig, ax = plt.subplots()
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)


















    elif model_choice == "Tree":

        tree = df.copy()

        # --- Allow user to toggle between automatic vs manual binning ---
        st.markdown("### ⚙️ Choose How to Categorize Your Data")

        use_custom_bins = st.checkbox("🛠️ Use Custom Bins for Categorization (PPS, Shares, Trade Value)", value=False)
        category_labels = ["Very Low", "Low", "Medium", "High", "Very High"]

        if use_custom_bins:
            st.markdown("#### 🔢 Custom Bin Thresholds")

            pps_cuts = st.text_input("PPS Custom Thresholds (comma-separated): Patern (Very Low, Low, Medium, High, Very High)", "100,300,600,700")
            shares_cuts = st.text_input("Shares Custom Thresholds (comma-separated): Patern (Very Low, Low, Medium, High, Very High)", "1000,5000,10000,100000")
            tv_cuts = st.text_input("Trade Value Custom Thresholds (comma-separated): Patern (Very Low, Low, Medium, High, Very High)", "50000,100000,250000,500000")

            pps_bins = [-float("inf")] + [float(x.strip()) for x in pps_cuts.split(",")] + [float("inf")]
            shares_bins = [-float("inf")] + [float(x.strip()) for x in shares_cuts.split(",")] + [float("inf")]
            tv_bins = [-float("inf")] + [float(x.strip()) for x in tv_cuts.split(",")] + [float("inf")]

            tree["PPS_Category"] = pd.cut(tree["TRANS_PRICEPERSHARE"], bins=pps_bins, labels=category_labels).astype(str)
            tree["Shares_Category"] = pd.cut(tree["TRANS_SHARES"], bins=shares_bins, labels=category_labels).astype(str)
            tree["TradeValue_Category"] = pd.cut(tree["TRANS_PRICEPERSHARE"] * tree["TRANS_SHARES"], bins=tv_bins, labels=category_labels).astype(str)

        else:
            tree["PPS_Category"], pps_bins = pd.qcut(tree["TRANS_PRICEPERSHARE"], q=5, labels=category_labels, retbins=True)
            tree["Shares_Category"], shares_bins = pd.qcut(tree["TRANS_SHARES"], q=5, labels=category_labels, retbins=True)
            tree["TradeValue_Category"], tv_bins = pd.qcut(tree["TRANS_PRICEPERSHARE"] * tree["TRANS_SHARES"], q=5, labels=category_labels, retbins=True)

            tree["PPS_Category"] = tree["PPS_Category"].astype(str)
            tree["Shares_Category"] = tree["Shares_Category"].astype(str)
            tree["TradeValue_Category"] = tree["TradeValue_Category"].astype(str)

        potential_targets = []
        for col in tree.columns:
            nunique = tree[col].nunique()
            if tree[col].dtype in ['object', 'bool'] and 2 <= nunique <= 10:
                potential_targets.append(col)

        


        selected_target = st.selectbox("🎯 Select a Target Variable (Classification)", potential_targets)

        category_labels = ["Very Low", "Low", "Medium", "High", "Very High"]

        # Show bin ranges only for derived categories
        if selected_target in ['PPS_Category', 'Shares_Category', 'TradeValue_Category']:
            st.markdown("#### 🧾 Category Bin Ranges + Encoding")

            if selected_target == 'PPS_Category':
                selected_bins = pps_bins
            elif selected_target == 'Shares_Category':
                selected_bins = shares_bins
            else:
                selected_bins = tv_bins

            label_map = {}
            for i, label in enumerate(category_labels):
                bin_range = f"{selected_bins[i]:,.2f} – {selected_bins[i+1]:,.2f}"
                label_map[label] = i
                st.write(f"**{i}** → {label} = ${bin_range}")

            # Encode
            tree["TARGET"] = tree[selected_target].map(label_map)
            y = tree["TARGET"]

            # Distribution chart
            class_counts = y.value_counts().sort_index()
            fig, ax = plt.subplots()
            ax.bar([category_labels[i] for i in class_counts.index], class_counts.values)
            ax.set_title("Target Category Distribution")
            ax.set_xlabel("Category")
            ax.set_ylabel("Count")
            st.pyplot(fig)

        else:
            # Fallback if target is not a derived categorical column (binary encoding)
            tree["TARGET"] = (tree[selected_target] == tree[selected_target].unique()[0]).astype(int)
            y = tree["TARGET"]

            # Show what 0 and 1 mean
            st.markdown("#### 🔢 Target Encoding (Binary)")
            st.write(f"**1** → {tree[selected_target].unique()[0]}")
            st.write(f"**0** → not {tree[selected_target].unique()[0]}")

            # Distribution chart
            class_counts = y.value_counts().sort_index()
            fig, ax = plt.subplots()
            ax.bar(class_counts.index.astype(str), class_counts.values, color=["#4CAF50", "#FFC107"])
            ax.set_title("Target Class Distribution")
            ax.set_xlabel("Class (0 or 1)")
            ax.set_ylabel("Count")
            st.pyplot(fig)




        # ✅ Now drop the raw inputs used for Trade Value
        tree = tree.drop(columns=["TRANS_PRICEPERSHARE", "TRANS_SHARES", "TRADE_VALUE"], errors="ignore")

        st.markdown("### Tree / Ensemble Configuration")

        model_type = st.radio("Select Model Type", ["Decision Tree", "Bagging", "Boosting","XGBoost","Random Forest"])


        # Drop target and encode categorical
        X = tree.drop(columns=[selected_target, "TARGET"], errors="ignore")




        # Remove object-type leftovers
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.to_numeric(X[col], errors="coerce")
        X = X.dropna(axis=1)
        


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


        # Create Decision Tree classifer object
        if model_type == "Decision Tree":
            clf = DecisionTreeClassifier()

        #creating bagging clf
        elif model_type == "Bagging":
            st.markdown("#### 🔢 Bagging Parameters")

            n_estimators = st.slider("Number of Estimators (Trees)", min_value=10, max_value=300, value=100, step=10)
            random_state = st.number_input("Random State (Seed for Reproducibility)", value=42, step=1)

            clf = BaggingClassifier(
                estimator=DecisionTreeClassifier(),
                n_estimators=n_estimators,
                random_state=random_state
            )

        elif model_type == "Boosting":
            st.markdown("#### 🚀 Boosting Parameters (Gradient Boosting)")

            n_estimators = st.slider("Number of Estimators (Boosting Rounds)", 10, 300, 100, step=10)
            learning_rate = st.number_input("Learning Rate", value=0.1, min_value=0.01, max_value=1.0, step=0.01)
            max_depth = st.slider("Max Tree Depth", 1, 10, 3)
            random_state = st.number_input("Random State", value=42)

            clf = GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=random_state
            )

        if model_type == "Random Forest":
            st.markdown("#### 🌲 Random Forest Parameters")

            n_estimators = st.slider("Number of Trees", 10, 300, 100, step=10)
            max_depth = st.slider("Max Tree Depth", 1, 20, 5)
            random_state = st.number_input("Random State", value=42)

            clf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state
            )

        if model_type == "XGBoost":
            st.markdown("#### ⚡ XGBoost Parameters")

            n_estimators = st.slider("Number of Boosting Rounds", 10, 300, 100, step=10)
            learning_rate = st.number_input("Learning Rate", value=0.1, min_value=0.01, max_value=1.0, step=0.01)
            max_depth = st.slider("Max Tree Depth", 1, 10, 3)
            random_state = st.number_input("Random State", value=42)

            clf = XGBClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=random_state,
                use_label_encoder=False,
                eval_metric="logloss"
            )


            


        # Train Decision Tree Classifer
        clf = clf.fit(X_train,y_train)

        #Predict the response for test dataset
        y_pred = clf.predict(X_test)

        # Show performance
    
        st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.3f}")
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        
        # Plot confusion matrix

        label_list = sorted(y.unique())  # Should be [0, 1, 2, 3, 4]
        class_names = [category_labels[i] for i in label_list]

        fig, ax = plt.subplots()
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",  xticklabels=class_names,yticklabels=class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        
        
        if model_type == "Decision Tree":

            # Visualize decision tree and feature importances only on button click
            st.markdown("### 🌳 Optional: Visualize Decision Tree and Feature Importances")
            st.warning("⚠️ This process may take some time. Your browser may ask if you want to wait — choose **Wait** to continue.")

            if st.button("📊 Click to Visualize Tree + Top 10 Features"):
                with st.spinner("Generating visualizations... Please wait."):
                    # Visualize the decision tree
                    feature_names = X.columns
                    class_names = [str(cls) for cls in sorted(y.unique())]  # E.g., ['0', '1', '2', ...] or ['Very Low', ..., 'Very High']
                    st.write("### 🧾 Feature Columns Used in Model")
                    st.write(X.columns.tolist())

                    dot_data = export_graphviz(
                        clf,
                        out_file=None,
                        feature_names=feature_names,
                        class_names=class_names,
                        filled=True,
                        rounded=True,
                        special_characters=True
                    )

                    graph = graphviz.Source(dot_data)
                    st.markdown("### 🧠 Visualized Decision Tree")
                    st.graphviz_chart(dot_data)

                    # Get feature importances
                    importances = clf.feature_importances_
                    feature_names = X_test.columns

                    fi_df = pd.DataFrame({
                        "feature": feature_names,
                        "importance": importances
                    }).sort_values(by="importance", ascending=False)

                    # Plot
                    fig = px.bar(fi_df.head(10), x="importance", y="feature", orientation="h", title="Top 10 Feature Importances")
                    st.plotly_chart(fig)

        elif model_type == "Boosting":


            # Visualize decision tree and feature importances only on button click
            st.markdown("### 🌳 Optional: Visualize Decision Tree and Feature Importances")
            st.warning("⚠️ This process may take some time. Your browser may ask if you want to wait — choose **Wait** to continue.")

            if st.button("📊 Click to Visualize Tree + Top 10 Features"):
                with st.spinner("Generating visualizations... Please wait."):
                    # Visualize the decision tree
                    feature_names = X.columns
                    class_names = [str(cls) for cls in sorted(y.unique())]  # E.g., ['0', '1', '2', ...] or ['Very Low', ..., 'Very High']
                    st.write("### 🧾 Feature Columns Used in Model")
                    st.write(X.columns.tolist())

                    dot_data = export_graphviz(
                        clf.estimators_[0, 0],
                        out_file=None,
                        feature_names=feature_names,
                        class_names=class_names,
                        filled=True,
                        rounded=True,
                        special_characters=True
                    )

                    graph = graphviz.Source(dot_data)
                    st.markdown("### 🧠 Visualized Decision Tree")
                    st.graphviz_chart(dot_data)

                    # Get feature importances
                    importances = clf.feature_importances_
                    feature_names = X_test.columns

                    fi_df = pd.DataFrame({
                        "feature": feature_names,
                        "importance": importances
                    }).sort_values(by="importance", ascending=False)

                    # Plot
                    fig = px.bar(fi_df.head(10), x="importance", y="feature", orientation="h", title="Top 10 Feature Importances")
                    st.plotly_chart(fig)


        elif model_type == "Bagging":
            # Visualize decision tree and feature importances only on button click
            st.markdown("### 🌳 Optional: Visualize Decision Tree and Feature Importances")
            st.warning("⚠️ This process may take some time. Your browser may ask if you want to wait — choose **Wait** to continue.")

            if st.button("📊 Click to Visualize One Tree + Top 10 Features (Bagging)"):
                with st.spinner("Visualizing one of the trees..."):
                    # Visualize one tree
                    estimator = clf.estimators_[0]
                    dot_data = export_graphviz(
                        estimator,
                        out_file=None,
                        feature_names=X.columns,
                        class_names=[str(cls) for cls in sorted(y.unique())],
                        filled=True,
                        rounded=True,
                        special_characters=True
                    )
                    st.graphviz_chart(dot_data)

                    # Average feature importances
                    importances = np.mean([tree.feature_importances_ for tree in clf.estimators_], axis=0)

                    fi_df = pd.DataFrame({
                        "feature": X.columns,
                        "importance": importances
                    }).sort_values(by="importance", ascending=False)

                    fig = px.bar(fi_df.head(10), x="importance", y="feature", orientation="h", title="Top 10 Feature Importances (Bagging)")
                    st.plotly_chart(fig)

        

        if model_type in ["Random Forest", "XGBoost"]:
            st.markdown("### 🌳 Optional: Visualize One Tree and Feature Importances")
            st.warning("⚠️ This may take time. Your browser may ask if you want to wait — choose **Wait**.")

            if st.button("📊 Click to Visualize One Tree + Top 10 Features"):
                with st.spinner("Generating visualizations..."):
                    if model_type == "Random Forest":
                        # Visualize one tree from the Random Forest
                        st.markdown("#### 🌲 Visualizing One Tree from Random Forest")
                        single_tree = clf.estimators_[0]

                        dot_data = export_graphviz(
                            single_tree,
                            out_file=None,
                            feature_names=X.columns,
                            class_names=[str(cls) for cls in sorted(y.unique())],
                            filled=True,
                            rounded=True,
                            special_characters=True
                        )
                        st.graphviz_chart(dot_data)


                        importances = clf.feature_importances_
                        feature_names = X.columns

                        fi_df = pd.DataFrame({
                            "feature": feature_names,
                            "importance": importances
                        }).sort_values(by="importance", ascending=False)

                        fig = px.bar(fi_df.head(10), x="importance", y="feature", orientation="h", title="Top 10 Feature Importances")
                        st.plotly_chart(fig)


                    elif model_type == "XGBoost":
                        # Visualize one XGBoost tree
                        st.markdown("#### 🔝 Top 10 Feature Importances")
                        importances = clf.feature_importances_
                        feature_names = X.columns

                        fi_df = pd.DataFrame({
                            "feature": feature_names,
                            "importance": importances
                        }).sort_values(by="importance", ascending=False)

                        fig = px.bar(fi_df.head(10), x="importance", y="feature", orientation="h")
                        st.plotly_chart(fig)
















    elif model_choice == "MLflow Tracking":
        tree = df.copy()

        # --- Allow user to toggle between automatic vs manual binning ---
        st.markdown("### ⚙️ Choose How to Categorize Your Data")

        use_custom_bins = st.checkbox("🛠️ Use Custom Bins for Categorization (PPS, Shares, Trade Value)", value=False)
        category_labels = ["Very Low", "Low", "Medium", "High", "Very High"]

        if use_custom_bins:
            st.markdown("#### 🔢 Custom Bin Thresholds")

            
            pps_cuts = st.text_input("PPS Custom Thresholds (comma-separated): Patern (Very Low, Low, Medium, High, Very High)", "100,300,600,700")
            shares_cuts = st.text_input("Shares Custom Thresholds (comma-separated): Patern (Very Low, Low, Medium, High, Very High)", "1000,5000,10000,100000")
            tv_cuts = st.text_input("Trade Value Custom Thresholds (comma-separated): Patern (Very Low, Low, Medium, High, Very High)", "50000,100000,250000,500000")

            pps_bins = [-float("inf")] + [float(x.strip()) for x in pps_cuts.split(",")] + [float("inf")]
            shares_bins = [-float("inf")] + [float(x.strip()) for x in shares_cuts.split(",")] + [float("inf")]
            tv_bins = [-float("inf")] + [float(x.strip()) for x in tv_cuts.split(",")] + [float("inf")]

            tree["PPS_Category"] = pd.cut(tree["TRANS_PRICEPERSHARE"], bins=pps_bins, labels=category_labels).astype(str)
            tree["Shares_Category"] = pd.cut(tree["TRANS_SHARES"], bins=shares_bins, labels=category_labels).astype(str)
            tree["TradeValue_Category"] = pd.cut(tree["TRANS_PRICEPERSHARE"] * tree["TRANS_SHARES"], bins=tv_bins, labels=category_labels).astype(str)

        else:
            tree["PPS_Category"], pps_bins = pd.qcut(tree["TRANS_PRICEPERSHARE"], q=5, labels=category_labels, retbins=True)
            tree["Shares_Category"], shares_bins = pd.qcut(tree["TRANS_SHARES"], q=5, labels=category_labels, retbins=True)
            tree["TradeValue_Category"], tv_bins = pd.qcut(tree["TRANS_PRICEPERSHARE"] * tree["TRANS_SHARES"], q=5, labels=category_labels, retbins=True)

            tree["PPS_Category"] = tree["PPS_Category"].astype(str)
            tree["Shares_Category"] = tree["Shares_Category"].astype(str)
            tree["TradeValue_Category"] = tree["TradeValue_Category"].astype(str)

        potential_targets = []
        for col in tree.columns:
            nunique = tree[col].nunique()
            if tree[col].dtype in ['object', 'bool'] and 2 <= nunique <= 10:
                potential_targets.append(col)

        


        selected_target = st.selectbox("🎯 Select a Target Variable (Classification)", potential_targets)

        category_labels = ["Very Low", "Low", "Medium", "High", "Very High"]

        # Show bin ranges only for derived categories
        if selected_target in ['PPS_Category', 'Shares_Category', 'TradeValue_Category']:
            st.markdown("#### 🧾 Category Bin Ranges + Encoding")

            if selected_target == 'PPS_Category':
                selected_bins = pps_bins
            elif selected_target == 'Shares_Category':
                selected_bins = shares_bins
            else:
                selected_bins = tv_bins

            label_map = {}
            for i, label in enumerate(category_labels):
                bin_range = f"{selected_bins[i]:,.2f} – {selected_bins[i+1]:,.2f}"
                label_map[label] = i
                st.write(f"**{i}** → {label} = ${bin_range}")

            # Encode
            tree["TARGET"] = tree[selected_target].map(label_map)
            y = tree["TARGET"]

            # Distribution chart
            class_counts = y.value_counts().sort_index()
            fig, ax = plt.subplots()
            ax.bar([category_labels[i] for i in class_counts.index], class_counts.values)
            ax.set_title("Target Category Distribution")
            ax.set_xlabel("Category")
            ax.set_ylabel("Count")
            st.pyplot(fig)

        else:
            # Fallback if target is not a derived categorical column (binary encoding)
            tree["TARGET"] = (tree[selected_target] == tree[selected_target].unique()[0]).astype(int)
            y = tree["TARGET"]

            # Show what 0 and 1 mean
            st.markdown("#### 🔢 Target Encoding (Binary)")
            st.write(f"**1** → {tree[selected_target].unique()[0]}")
            st.write(f"**0** → not {tree[selected_target].unique()[0]}")

            # Distribution chart
            class_counts = y.value_counts().sort_index()
            fig, ax = plt.subplots()
            ax.bar(class_counts.index.astype(str), class_counts.values, color=["#4CAF50", "#FFC107"])
            ax.set_title("Target Class Distribution")
            ax.set_xlabel("Class (0 or 1)")
            ax.set_ylabel("Count")
            st.pyplot(fig)




        # ✅ Now drop the raw inputs used for Trade Value
        tree = tree.drop(columns=["TRANS_PRICEPERSHARE", "TRANS_SHARES", "TRADE_VALUE"], errors="ignore")

        st.markdown("### Tree / Ensemble Configuration")

        # Drop target and encode categorical
        X = tree.drop(columns=[selected_target, "TARGET"], errors="ignore")




        # Remove object-type leftovers
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.to_numeric(X[col], errors="coerce")
        X = X.dropna(axis=1)
        


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


        n_estimators = st.slider("Number of Estimators", 10, 300, 100, step=10)
        max_depth = st.slider("Max Depth", 1, 15, 5)
        learning_rate = st.slider("Learning Rate (for boosting models)", 0.01, 1.0, 0.1, step=0.01)
        random_state = st.number_input("Random State", value=42)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

        # Define models
        models = {
            "Decision Tree": DecisionTreeClassifier(max_depth=max_depth, random_state=random_state),
            "Random Forest": RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=random_state),
            "XGBoost": XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, use_label_encoder=False, eval_metric="logloss", random_state=random_state),
            "Bagging": BaggingClassifier(n_estimators=n_estimators, random_state=random_state)
        }

        
        import tempfile

        st.markdown("## 📊 Results")
        best_model = None
        best_score = 0

        for name, model in models.items():
            with mlflow.start_run(run_name=name):
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)

                mlflow.log_params(model.get_params())
                mlflow.log_metric("accuracy", acc)

                # ✅ TEMP SAFE LOGGING FOR STREAMLIT CLOUD
                with tempfile.TemporaryDirectory() as tmp_dir:
                    mlflow.sklearn.log_model(model, artifact_path=name)

                st.write(f"{name} Accuracy: **{acc:.4f}**")

                if acc > best_score:
                    best_score = acc
                    best_model = name

        









        st.markdown("### 🧾 Summary Report")

        st.markdown(f"""
        - 📌 **Selected Target**: `{selected_target}`
        - 🔍 **Encoding**: {'Custom Bins' if use_custom_bins else 'Quantile Binning'}
        - 🧪 **Train/Test Split**: 80/20
        - ⚙️ **Models Compared**:
        - Decision Tree
        - Random Forest
        - Gradient Boosting
        - XGBoost
        - Bagging
        - 🥇 **Best Model**: `{best_model}`  
        - 🎯 **Best Accuracy**: **{best_score:.4f}**
        """)


        if st.button("📥 Download Report Summary"):
            report = io.StringIO()
            report.write("Model Performance Report\n")
            report.write("========================\n")
            report.write(f"Selected Target: {selected_target}\n")
            report.write(f"Train/Test Split: 80/20\n")
            report.write("Models Compared:\n")
            for name in models:
                report.write(f"- {name}\n")
            report.write(f"\nBest Model: {best_model}\n")
            report.write(f"Best Accuracy: {best_score:.4f}\n")
            
            st.download_button("Download as .txt", report.getvalue(), file_name="ml_report.txt")



        st.success(f"🥇 Best Model: **{best_model}** with Accuracy: **{best_score:.4f}**")





        if st.checkbox("🔍 Show MLflow tracking info"):
            st.markdown("### 📂 View Tracked MLflow Runs")

            # Search MLflow runs from the current experiment
            runs_df = mlflow.search_runs(order_by=["metrics.accuracy DESC"])


            if runs_df.empty:
                st.info("No runs tracked yet. Train a model to populate MLflow logs.")
            else:
                # Show relevant columns
                display_cols = ["run_id", "params_max_depth", "params_n_estimators", "metrics_accuracy", "tags.mlflow.runName"]
                filtered_cols = [col for col in display_cols if col in runs_df.columns]

                st.dataframe(runs_df[filtered_cols].rename(columns={
                    "run_id": "Run ID",
                    "params_max_depth": "Max Depth",
                    "params_n_estimators": "N Estimators",
                    "metrics_accuracy": "Accuracy",
                    "tags.mlflow.runName": "Model Name"
                }))
            csv = runs_df[filtered_cols].to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download MLflow Logs (CSV)", csv, "mlflow_runs.csv", "text/csv")


