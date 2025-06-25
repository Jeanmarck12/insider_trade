import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
import sklearn

st.set_page_config(
    page_title="Insider Trade Value Predictor",
    page_icon="📈",
    layout="wide"
)

# All other Streamlit code goes below this!
st.markdown("#### 🎯 Objective")
st.write("This app allows users to explore and understand SEC Form 4 insider trading data, with the goal of predicting the dollar value of transactions based on reported attributes such as ownership type, transaction codes, and share activity.")

st.write()
st.write()
st.write()


page = st.sidebar.selectbox("Select Page",["Introduction 📘","Visualization 📊","Prediction","Automated Report 📑"])
st.image("trade.jpeg")
st.write()
st.write()
st.write()


df = pd.read_csv('dataset_new_csv.csv')
df1 = pd.read_csv('dataset_new_csv.csv')

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

    While the data is useful for understanding patterns in insider trading, it should not be treated as a definitive source for investment decisions without reviewing the full official filings. The goal of this dataset is to support transparency and enable public analysis — not to replace due diligence.
    """)



    st.markdown("### 🧠 Learn About a Column")

    # Column options (excluding obvious ones)
    column_info = {
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
        "TRANS_CODE": {
            "desc": "Code representing the type of insider transaction.",
            "values": {
                "A": "Grant, award or other acquisition (Rule 16b-3(d))",
                "C": "Conversion of derivative security",
                "D": "Disposition to issuer (Rule 16b-3(e))",
                "E": "Expiration of short derivative position",
                "F": "Payment of exercise price or tax liability (Rule 16b-3)",
                "G": "Bona fide gift",
                "H": "Expiration/cancellation of long derivative position with value received",
                "I": "Discretionary transaction (Rule 16b-3(f))",
                "J": "Other acquisition or disposition",
                "L": "Small acquisition (Rule 16a-6)",
                "M": "Exercise/conversion of derivative security (Rule 16b-3)",
                "O": "Exercise of out-of-the-money derivative security",
                "P": "Open market or private purchase of non-derivative or derivative security",
                "S": "Open market or private sale of non-derivative or derivative security",
                "U": "Disposition in a change of control transaction",
                "W": "Acquisition/disposition by will or inheritance",
                "X": "Exercise of in-the-money or at-the-money derivative security",
                "Z": "Deposit to or withdrawal from voting trust"
            }
        },

        "DIRECT_INDIRECT_OWNERSHIP": {
            "desc": "Ownership type (direct or indirect).",
            "values": {
                "D": "Direct ownership",
                "I": "Indirect ownership"
            }
        },
        "TRANS_ACQUIRED_DISP_CD": {
            "desc": "Indicates if shares were acquired or disposed.",
            "values": {
                "A": "Acquired",
                "D": "Disposed"
            }
        },
        "TRANS_TIMELINESS": {
            "desc": "Whether the filing was on time or late.",
            "values": {
                "L": "Late filing",
                "T": "Timely filing"
            }
        },
        "NATURE_OF_OWNERSHIP": {
            "desc": "How the shares are owned (e.g., through a trust).",
            "values": {}
        },
        "SHRS_OWND_FOLWNG_TRANS": {
            "desc": "Shares owned after the transaction.",
            "values": {}
        },
        "SECURITY_TITLE": {
            "desc": "Type of security traded (e.g., Common Stock).",
            "values": {}
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
    "SHRS_OWND_FOLWNG_TRANS": {
        "desc": "Shares owned after the transaction occurred.",
        "values": {}
    },
    "DIRECT_INDIRECT_OWNERSHIP_x": {
        "desc": "Type of ownership (Direct or Indirect).",
        "values": {
            "D": "Direct",
            "I": "Indirect"
        }
    },
    "TRANS_FORM_TYPE_x": {
        "desc": "Subtype of the transaction form.",
        "values": {}
    },
    "SECURITY_TITLE": {
        "desc": "Name of the security involved in the trade.",
        "values": {}
    },
    "TRANS_SHARES": {
        "desc": "Number of shares involved in the transaction.",
        "values": {}
    },
    "TRANS_PRICEPERSHARE": {
        "desc": "Price per share during the transaction.",
        "values": {}
    },
    "UNDLYNG_SEC_TITLE": {
        "desc": "Title of the underlying security (for options).",
        "values": {}
    },
    "UNDLYNG_SEC_SHARES": {
        "desc": "Number of shares for the underlying security.",
        "values": {}
    },
    "TRANS_FORM_TYPE_y": {
        "desc": "Another form type field (may duplicate x).",
        "values": {}
    },
    "LATITUDE": {
        "desc": "Latitude of the filer’s location.",
        "values": {}
    },
    "LONGITUDE": {
        "desc": "Longitude of the filer’s location.",
        "values": {}
    },
    "TRADE_VALUE": {
        "desc": "Total value of the transaction.",
        "values": {}
    },
    "FILING_YEAR": {
        "desc": "Year the filing was submitted.",
        "values": {}
    },
    "FILING_MONTH": {
        "desc": "Month the filing was submitted.",
        "values": {}
    },
    "FILING_DAYOFWEEK": {
        "desc": "Day of the week the filing was submitted.",
        "values": {}
    },
    "REPORT_YEAR": {
        "desc": "Year the transaction occurred.",
        "values": {}
    },
    "REPORT_MONTH": {
        "desc": "Month the transaction occurred.",
        "values": {}
    },
    "REPORT_DAYOFWEEK": {
        "desc": "Day of the week the transaction occurred.",
        "values": {}
    },
    "DAYS_DELAY": {
        "desc": "Delay (in days) between transaction date and report filing.",
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


    df['TRANS_FORM_TYPE_x'] = df['TRANS_FORM_TYPE_x'].astype(str)
    df['TRANS_FORM_TYPE_y'] = df['TRANS_FORM_TYPE_y'].astype(str)


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

        # Filter only 2024 and 2025
        df1_filtered = df1[df1["REPORT_YEAR"].isin([2024, 2025])].copy()

        # Make sure REPORT_MONTH is integer
        df1_filtered["REPORT_MONTH"] = pd.to_numeric(df1_filtered["REPORT_MONTH"], errors='coerce')

        # Create a 'Quarter' column
        df1_filtered["QUARTER"] = pd.cut(
            df1_filtered["REPORT_MONTH"],
            bins=[0, 3, 6, 9, 12],
            labels=["Q1", "Q2", "Q3", "Q4"],
            right=True
        )

        # Combine year and quarter for a nice label
        df1_filtered["YEAR_QUARTER"] = df1_filtered["REPORT_YEAR"].astype(str) + " " + df1_filtered["QUARTER"].astype(str)

        # Group by YEAR_QUARTER and sum trade value
        quarterly_data = df1_filtered.groupby("YEAR_QUARTER")["TRADE_VALUE"].sum().reset_index()

        # Sort the labels properly (not alphabetically)
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
        ["Linear Regression", "Logistic Regression", "K-Nearest Neighbors (KNN)", "Decision Tree", "Deep Learning", "MLflow Tracking"]
    )




    if model_choice == "Linear Regression":
        df2 = df.copy()

        st.markdown("### Model Configuration")
        list_var = list(df2.columns)

        col1, col2 = st.columns(2)
        with col1:
            target_selection = st.selectbox("🎯 Select Target Variable (Y)", options=list_var, index=len(list_var) - 1)
        with col2:
            x_options = [col for col in list_var if col != target_selection]
            features_selection = st.multiselect("📊 Select Features (X)", options=x_options, default=x_options)

        test_size = st.slider("Select test set size (%)", 10, 50, 20, step=5) / 100.0

        X = df2[features_selection]
        y = df2[target_selection]

        st.write("### Features (X) preview")
        st.dataframe(X.head())
        st.write("### Target (y) preview")
        st.dataframe(y.head())

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        st.write(f"✅ **Train size:** {X_train.shape[0]} rows")
        st.write(f"✅ **Test size:** {X_test.shape[0]} rows")

        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        selected_metrics = st.multiselect(
            "Select metrics to display",
            ["Mean Squared Error (MSE)", "Root Mean Squared Error (RMSE)", "Mean Absolute Error (MAE)", "R² Score"],
            default=["Mean Absolute Error (MAE)"]
        )

        from sklearn import metrics
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




elif page == "Automated Report 📑":
    st.subheader("03 Automated Report")

    detail_level = st.radio(
        "Select level of detail:",
        ["Basic Overview", "Standard EDA", "Advanced EDA"],
        index=0
    )

    if st.button("Generate Report"):
        with st.spinner("Generating report..."):

            # Always show summary stats and missing values
            st.markdown("### Summary Statistics 📊")
            st.dataframe(df.describe())

            st.markdown("### Missing Values")
            missing = df.isnull().sum()
            st.write(missing)
            if missing.sum() == 0:
                st.success("✅ No missing values found")
            else:
                st.warning("⚠️ Missing values detected")

            # If Standard or Advanced, add correlation + pairplot
            if detail_level in ["Standard EDA", "Advanced EDA"]:
                st.markdown("### Correlation Matrix 🔥")
                fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
                sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax_corr)
                ax_corr.set_title("Correlation Matrix")
                st.pyplot(fig_corr)

                st.markdown("### Pairplot (sampled if large) 🔍")
                if df.shape[0] > 500:
                    df_sample = df.sample(500, random_state=42)
                    st.info("Pairplot shown for a 500-row sample (for speed).")
                else:
                    df_sample = df
                fig_pair = sns.pairplot(df_sample)
                st.pyplot(fig_pair.figure)

            # If Advanced, add additional analysis
            if detail_level == "Advanced EDA":
                st.markdown("### Constant Columns 🚩")
                constant_cols = [col for col in df.columns if df[col].nunique() == 1]
                if constant_cols:
                    st.warning(f"⚠️ Constant columns: {constant_cols}")
                else:
                    st.success("✅ No constant columns")

                st.markdown("### High Cardinality Columns 🚩")
                cardinality = {col: df[col].nunique() for col in df.columns}
                high_card_cols = [col for col, uniq in cardinality.items() if uniq > 50]
                if high_card_cols:
                    st.warning(f"⚠️ High cardinality columns: {high_card_cols}")
                else:
                    st.success("✅ No high cardinality columns")

                st.markdown("### Highly Correlated Pairs (> 0.85) 🚩")
                corr_matrix = df.corr().abs()
                upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                high_corr = [
                    (col1, col2, upper.loc[col1, col2])
                    for col1 in upper.columns
                    for col2 in upper.index
                    if pd.notnull(upper.loc[col1, col2]) and upper.loc[col1, col2] > 0.85
                ]
                if high_corr:
                    for col1, col2, corr_val in high_corr:
                        st.warning(f"⚠️ {col1} and {col2} correlation: {corr_val:.2f}")
                else:
                    st.success("✅ No highly correlated pairs")

                st.markdown("### Outlier Detection (z-score > 3) 🚩")
                from scipy.stats import zscore
                z_scores = np.abs(zscore(df.select_dtypes(include=np.number)))
                outliers = (z_scores > 3).sum(axis=0)
                outlier_cols = {col: int(count) for col, count in zip(df.select_dtypes(include=np.number).columns, outliers) if count > 0}
                if outlier_cols:
                    st.warning(f"⚠️ Outliers detected:\n{outlier_cols}")
                else:
                    st.success("✅ No significant outliers detected")

            # Download summary
            csv = df.describe().to_csv(index=True).encode()
            st.download_button(
                label="📥 Download Summary Statistics CSV",
                data=csv,
                file_name="summary_statistics.csv",
                mime='text/csv'
            )


