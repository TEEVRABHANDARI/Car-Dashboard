import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from PIL import Image

# Cache data loading and preprocessing
@st.cache_data
def load_and_preprocess_data():
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("Loading data...")
        df = pd.read_csv("data/cardekho_dataset.csv")
        progress_bar.progress(20)
        
        status_text.text("Cleaning data...")
        df = df.dropna(subset=['selling_price', 'brand', 'model', 'vehicle_age'])
        numeric_cols = ['selling_price', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        df = df.dropna()
        progress_bar.progress(40)
        
        status_text.text("Engineering features...")
        df['engine_power_ratio'] = df['max_power'] / df['engine'].replace(0, 1)
        df['mileage_per_year'] = df['km_driven'] / df['vehicle_age'].clip(1)
        progress_bar.progress(60)
        
        status_text.text("Encoding categories...")
        encode_cols = ['brand', 'model', 'seller_type', 'fuel_type', 'transmission_type']
        le_dict = {}
        for col in encode_cols:
            le = LabelEncoder()
            df[col+'_encoded'] = le.fit_transform(df[col])
            le_dict[col] = le
        progress_bar.progress(80)
        
        return df, le_dict
        
    except Exception as e:
        st.error(f"Error in data processing: {str(e)}")
        st.stop()
    finally:
        progress_bar.empty()
        status_text.empty()

@st.cache_resource
def train_model(df):
    try:
        X = df[['brand_encoded', 'model_encoded', 'vehicle_age', 'km_driven', 
               'fuel_type_encoded', 'transmission_type_encoded', 'mileage', 
               'engine', 'max_power', 'seats', 'engine_power_ratio', 
               'mileage_per_year']]
        y = df['selling_price']
        
        model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
        model.fit(X, y)
        return model
    except Exception as e:
        st.error(f"Model training failed: {str(e)}")
        st.stop()

# Load data and model
df, le_dict = load_and_preprocess_data()
model = train_model(df)

# Streamlit UI
logo = Image.open("images/logo.png")
col_logo, col_title = st.columns([1, 4])
with col_logo:
    st.image(logo, width=120)
with col_title:
    st.markdown("### Smart Car Valuation Advisor")
    st.markdown("Your Trusted Automotive Pricing Expert")

st.sidebar.header("🔍 Search Filters")
brand = st.sidebar.selectbox("Brand", sorted(df['brand'].unique()))
model_list = sorted(df[df['brand'] == brand]['model'].unique())
model_selected = st.sidebar.selectbox("Model", model_list)

@st.cache_data
def filter_data(df, brand, model_selected):
    return df[(df['brand'] == brand) & (df['model'] == model_selected)]

filtered_cars = filter_data(df, brand, model_selected)

if not filtered_cars.empty:
    with st.spinner("Generating predictions..."):
        X_pred = filtered_cars[[
            'brand_encoded', 'model_encoded', 'vehicle_age', 'km_driven',
            'fuel_type_encoded', 'transmission_type_encoded', 'mileage',
            'engine', 'max_power', 'seats', 'engine_power_ratio',
            'mileage_per_year'
        ]]
        filtered_cars['predicted_price'] = model.predict(X_pred)
        filtered_cars['price_diff'] = filtered_cars['selling_price'] - filtered_cars['predicted_price']
        filtered_cars['price_status'] = filtered_cars['price_diff'].apply(
            lambda x: "✅ Fair" if abs(x) < 50000 else "❌ Overpriced" if x > 0 else "💸 Underpriced"
        )
    
    filtered_cars['car_display_name'] = filtered_cars['car_name'] + " : " + filtered_cars['engine'].astype(str) + " CC"

    # Results Overview
    st.header(f"Results Overview for {brand.title()} {model_selected.title()}")
    
    # Key Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Listings Found", len(filtered_cars))
    with col2:
        st.metric("Average Price", f"₹{filtered_cars['selling_price'].mean():,.0f}")
    with col3:
        fair_pct = (filtered_cars['price_status'] == "✅ Fair").mean() * 100
        st.metric("Fair Pricing", f"{fair_pct:.1f}%")

    # Market Distribution
    st.subheader("Market Distribution")
    dist_col1, dist_col2 = st.columns(2)
    with dist_col1:
        fuel_counts = filtered_cars['fuel_type'].value_counts().reset_index()
        fig = px.bar(fuel_counts, x='fuel_type', y='count', 
                     title="Fuel Type Distribution", color='fuel_type')
        st.plotly_chart(fig, use_container_width=True)
    
    with dist_col2:
        transmission_counts = filtered_cars['transmission_type'].value_counts().reset_index()
        fig = px.bar(transmission_counts, x='transmission_type', y='count',
                     title="Transmission Type Distribution", color='transmission_type')
        st.plotly_chart(fig, use_container_width=True)

    # Engine Analysis
    st.subheader("Engine Specifications")
    eng_col1, eng_col2, eng_col3 = st.columns(3)
    with eng_col1:
        st.metric("Engine Variants", filtered_cars['engine'].nunique())
    with eng_col2:
        st.metric("Most Common", f"{filtered_cars['engine'].mode()[0]} CC")
    with eng_col3:
        st.metric("Average Size", f"{filtered_cars['engine'].mean():,.0f} CC")

    # Price Comparison
    st.subheader("Price Analysis")
    fig = px.scatter(filtered_cars, x='predicted_price', y='selling_price',
                    color='price_status', hover_name='car_display_name',
                    labels={'predicted_price': 'Predicted Price (₹)', 
                            'selling_price': 'Actual Price (₹)'},
                    hover_data=['vehicle_age', 'km_driven'])
    st.plotly_chart(fig, use_container_width=True)

    # Detailed Listings
    st.subheader("Detailed Listings")
    st.dataframe(filtered_cars[['car_display_name', 'vehicle_age', 'km_driven', 
                              'selling_price', 'predicted_price', 'price_status']]
                .rename(columns={'car_display_name': 'Car Model'}),
                height=300)

    # Deep Market Analysis
    st.header("Deep Market Analysis")
    
    # Depreciation Analysis
    st.subheader("Depreciation Trends")
    dep_col1, dep_col2 = st.columns(2)
    with dep_col1:
        fig = px.line(filtered_cars.groupby('vehicle_age')['selling_price'].mean().reset_index(),
                     x='vehicle_age', y='selling_price', 
                     title="Price Depreciation by Age",
                     labels={'vehicle_age': 'Age (years)', 'selling_price': 'Price (₹)'})
        st.plotly_chart(fig, use_container_width=True)
    with dep_col2:
        fig = px.histogram(filtered_cars, x='vehicle_age', nbins=15,
                          title="Age Distribution",
                          labels={'vehicle_age': 'Years Old'})
        st.plotly_chart(fig, use_container_width=True)

    # Mileage Analysis
    st.subheader("Mileage Insights")
    mil_col1, mil_col2 = st.columns(2)
    with mil_col1:
        fig = px.scatter(filtered_cars, x='km_driven', y='selling_price',
                        trendline="lowess", 
                        title="Mileage vs Price Relationship",
                        labels={'km_driven': 'Kilometers Driven', 'selling_price': 'Price (₹)'})
        st.plotly_chart(fig, use_container_width=True)
    with mil_col2:
        fig = px.box(filtered_cars, y='km_driven', 
                    title="Mileage Distribution",
                    labels={'km_driven': 'Kilometers'})
        st.plotly_chart(fig, use_container_width=True)

    # Feature Correlations
    st.subheader("Feature Relationships")
    base_cols = ['selling_price', 'vehicle_age', 'km_driven', 'engine', 'max_power']
    safe_cols = [col for col in base_cols if col in filtered_cars.columns]
    
    if 'predicted_price' in filtered_cars.columns:
        safe_cols.append('predicted_price')
    
    if len(safe_cols) > 1:
        fig = px.imshow(filtered_cars[safe_cols].corr(),
                        labels=dict(color="Correlation"),
                        color_continuous_scale='RdBu',
                        title="Feature Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Not enough data to show correlations")

else:
    st.warning("No matching cars found. Try different filters.")
st.caption("Note: Analysis based on historical market data and predictive modeling")