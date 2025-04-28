import streamlit as st
import pandas as pd
import numpy as np
import io
from sklearn.linear_model import LinearRegression  # For simple forecasting
import plotly.express as px

# --- Helper Functions ---
def parse_csv_data(data: str) -> pd.DataFrame:
    """Parses CSV data into a Pandas DataFrame, handling commas inside quotes."""
    try:
        df = pd.read_csv(io.StringIO(data))
        return df
    except Exception as e:
        raise ValueError(f"Could not parse CSV: {e}")


def get_years(df: pd.DataFrame) -> list[str]:
    """Gets a sorted list of year columns from the DataFrame."""
    return sorted([col for col in df.columns if col.isdigit()], key=int)


def get_total_sales_by_year(df: pd.DataFrame, years: list[str]) -> dict[str, int]:
    """Calculates total sales for each year in the given list."""
    return {year: df[year].sum() for year in years}


def generate_forecast(data: dict[str, int], periods: int = 3) -> list[dict[str, any]]:
    """Generates a simple linear forecast."""
    years = list(data.keys())
    sales = list(data.values())

    X = np.array(range(len(years))).reshape((-1, 1))
    y = np.array(sales)

    model = LinearRegression()
    model.fit(X, y)

    forecast = []
    for i in range(periods):
        future_idx = len(years) + i
        future_year = int(years[-1]) + i + 1
        predicted = model.predict([[future_idx]])[0]
        forecast.append({'year': str(future_year), 'value': round(predicted), 'isEstimate': True})

    historical = [{'year': y, 'value': v, 'isEstimate': False} for y, v in data.items()]
    return historical + forecast

# --- Streamlit App ---
st.title("Sales Data Analysis")

uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    file_content = uploaded_file.read().decode('utf-8')
    try:
        df = parse_csv_data(file_content)
    except ValueError as err:
        st.error(err)
        st.stop()

    # Validate required columns
    required_cols = {'Cat', 'Maker'}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        st.error(f"Missing required columns: {', '.join(missing)}")
        st.stop()

    # Identify year columns
    years = get_years(df)
    if not years:
        st.error("No numeric year columns found in the data.")
        st.stop()

    # --- Sidebar Filters ---
    categories = df['Cat'].dropna().unique().tolist()
    manufacturers = df['Maker'].dropna().unique().tolist()

    selected_categories = st.sidebar.multiselect("Select Categories", categories, default=categories)
    selected_manufacturers = st.sidebar.multiselect("Select Manufacturers", manufacturers, default=manufacturers)
    year_range = st.sidebar.slider(
        "Select Year Range",
        int(years[0]),
        int(years[-1]),
        (int(years[0]), int(years[-1]))
    )

    # Filter rows by category and maker
    filtered_df = df[
        (df['Cat'].isin(selected_categories)) &
        (df['Maker'].isin(selected_manufacturers))
    ]

    # Determine which years to include
    filtered_years = [y for y in years if year_range[0] <= int(y) <= year_range[1]]

    try:
        # Calculate totals and forecast
        total_sales_by_year = get_total_sales_by_year(filtered_df, filtered_years)
        forecast_data = generate_forecast(total_sales_by_year, periods=3)

        # --- Key Metrics ---
        total_sales_all = sum(total_sales_by_year.values())
        last_year = filtered_years[-1]
        last_year_sales = total_sales_by_year[last_year]
        prev_year = filtered_years[-2] if len(filtered_years) >= 2 else last_year
        prev_year_sales = total_sales_by_year.get(prev_year, 0)
        yoy_growth = ((last_year_sales - prev_year_sales) / prev_year_sales * 100) if prev_year_sales else None

        # Top category by sales (using filtered data)
        cat_sales = filtered_df.groupby('Cat')[filtered_years].sum().sum(axis=1)
        top_cat = cat_sales.idxmax()
        top_cat_sales = int(cat_sales.max())

        # Forecast future values
        future_forecast = [f for f in forecast_data if f['isEstimate']]
        expected_growth = ((future_forecast[-1]['value'] - last_year_sales) / last_year_sales * 100) if last_year_sales else None

        # Display summary metrics
        st.subheader("Summary Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Sales", f"{total_sales_all:,}", "All time EV sales from 2015 to 2024")
        if yoy_growth is not None:
            m2.metric("YoY Growth", f"{yoy_growth:+.1f}%%", f"{prev_year} → {last_year}")
        else:
            m2.metric("YoY Growth", "N/A")
        m3.metric(f"{last_year} Sales", f"{last_year_sales:,}")
        m4.metric("Top Category", f"{top_cat} ({top_cat_sales:,} vehicles)")

        # --- Charts ---
        st.subheader("Total Sales by Year")
        sales_data = pd.DataFrame.from_dict(total_sales_by_year, orient='index', columns=['Sales'])
        st.bar_chart(sales_data)

        st.subheader("Sales Forecast")
        forecast_df = pd.DataFrame(forecast_data)
        fig = px.line(
            forecast_df,
            x='year',
            y='value',
            color='isEstimate',
            labels={'year': 'Year', 'value': 'Sales'},
            title='Sales Forecast'
        )
        st.plotly_chart(fig)

        # Forecast summary table
        st.subheader("Forecast Summary")
        fs_df = pd.DataFrame({
            'Year': [int(f['year']) for f in future_forecast],
            'Sales': [f['value'] for f in future_forecast]
        }).set_index('Year')
        st.table(fs_df)
        if expected_growth is not None:
            st.write(f"**Expected Growth:** {expected_growth:+.1f}%")

        # Top 10 manufacturers (using filtered data)
        st.subheader("Top 10 Manufacturers by Sales")
        maker_sales = filtered_df.groupby('Maker')[filtered_years].sum().sum(axis=1)
        top10 = maker_sales.nlargest(10).reset_index()
        top10.columns = ['Maker', 'Sales']
        fig_top10 = px.bar(top10, x='Maker', y='Sales', title='Top 10 Manufacturers')
        st.plotly_chart(fig_top10)

        # --- Display Filtered Data ---
        st.subheader("Filtered Data")
        display_cols = ['Cat', 'Maker'] + filtered_years
        st.dataframe(filtered_df[display_cols])

    except Exception as exc:
        st.error(f"An unexpected error occurred during analysis: {exc}")
        st.stop()
else:
    st.info("Please upload a CSV file to start.")
