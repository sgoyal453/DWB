import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import requests
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from geopy.distance import great_circle
import math
from folium.plugins import MarkerCluster


# Streamlit page configuration
st.set_page_config(page_title="Conflict Zone Interactive Map", layout="wide")
st.title("Conflict Zone Interactive Map")

# Sidebar form for submitting warnings
st.sidebar.header("Submit a Warning")
with st.sidebar.form("warning_form"):
    user_name = st.text_input("Your Name")
    location = st.text_input("Location of Civil Unrest")
    warning_message = st.text_area("Warning Message")
    submit_warning = st.form_submit_button("Submit Warning")
   
if "alerts" not in st.session_state:
    st.session_state["alerts"] = []

if submit_warning:
    if user_name and location and warning_message:
        new_alert = {
            "name": user_name,
            "location": location,
            "message": warning_message,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        st.session_state["alerts"].append(new_alert)
        st.sidebar.success("Warning submitted successfully!")
    else:
        st.sidebar.error("Please fill in all fields before submitting.")

@st.cache_data(ttl=24*3600)
def fetch_acled_data():
    email = 'anayrshukla@berkeley.edu'
    access_key = '2XIE9PEBQ!n4lt!OVEJ1'
    url = 'https://api.acleddata.com/acled/read'
    params = {'email': email, 'key': access_key, 'limit': 500, 'format': 'json'}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = pd.DataFrame(response.json().get("data", []))
        timestamp = datetime.now().strftime("%Y-%m-%d")
        data.to_csv(f"acled_data_{timestamp}.csv", index=False)
        return data
    else:
        st.error(f"API Error: {response.status_code} - {response.text}")
        return None

@st.cache_data
def load_and_process_data():
    data = fetch_acled_data()
    if data is None:
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d")
            data = pd.read_csv(f"acled_data_{timestamp}.csv")
        except FileNotFoundError:
            st.error("No recent data file found")
            return None
   
    data['fatalities'] = pd.to_numeric(data['fatalities'], errors='coerce')
    data['latitude'] = pd.to_numeric(data['latitude'], errors='coerce')
    data['longitude'] = pd.to_numeric(data['longitude'], errors='coerce')
    data = data.dropna(subset=['fatalities', 'latitude', 'longitude'])
   
    if 'admin1' in data.columns:
        data.rename(columns={'admin1': 'city'}, inplace=True)
    return data

def train_random_forest(data):
    # Prepare features and target
    features = ['latitude', 'longitude', 'event_type', 'sub_event_type', 'actor1', 'location']
    target = 'fatalities'
    data = data.dropna(subset=features + [target])

    X = data[features]
    y = data[target]

    # Preprocessing for numerical and categorical data
    categorical_features = ['event_type', 'sub_event_type', 'actor1', 'location']
    numerical_features = ['latitude', 'longitude']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(random_state=42))
    ])

    # Train-test split and model training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    X_test['predicted_fatalities'] = predictions
    X_test['risk_level'] = X_test['predicted_fatalities'].apply(
        lambda x: "black" if x > 50 else "red" if x > 5 else "orange" if x > 0 else "green"
    )
    return X_test

def calculate_safe_directions(data, current_location):
    directions = {"N": True, "NE": True, "E": True, "SE": True, "S": True, "SW": True, "W": True, "NW": True}
    for _, other_location in data.iterrows():
        if current_location.equals(other_location):
            continue
        distance = great_circle(
            (current_location['latitude'], current_location['longitude']),
            (other_location['latitude'], other_location['longitude'])
        ).miles
        if distance < 50:  # Example threshold for proximity
            lat_diff = other_location['latitude'] - current_location['latitude']
            lon_diff = other_location['longitude'] - current_location['longitude']
            if lat_diff > 0 and abs(lat_diff) > abs(lon_diff): directions["N"] = False
            elif lat_diff < 0 and abs(lat_diff) > abs(lon_diff): directions["S"] = False
            if lon_diff > 0 and abs(lon_diff) > abs(lat_diff): directions["E"] = False
            elif lon_diff < 0 and abs(lon_diff) > abs(lat_diff): directions["W"] = False
            if lat_diff > 0 and lon_diff > 0: directions["NE"] = False
            elif lat_diff > 0 and lon_diff < 0: directions["NW"] = False
            elif lat_diff < 0 and lon_diff > 0: directions["SE"] = False
            elif lat_diff < 0 and lon_diff < 0: directions["SW"] = False

    return [dir for dir, safe in directions.items() if safe]

def create_map(data):
    m = folium.Map(location=[data['latitude'].mean(), data['longitude'].mean()], zoom_start=6)
    marker_cluster = MarkerCluster().add_to(m)

    for _, row in data.iterrows():
        safe_directions = calculate_safe_directions(data, row)
        popup_content = f"""
        <b>Location:</b> {row['location']}<br>
        <b>Country:</b> {row['country']}<br>
        <b>City:</b> {row.get('city', 'N/A')}<br>
        <b>Predicted Fatalities:</b> {row['predicted_fatalities']:.2f}<br>
        <b>Risk Level:</b> {row['risk_level']}<br>
        <b>Safe Directions:</b> {', '.join(safe_directions) if safe_directions else 'No Safe Directions'}<br>
        """
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=6,
            color=row['risk_level'],
            fill=True,
            fill_opacity=0.8,
            popup=folium.Popup(popup_content, max_width=300)
        ).add_to(marker_cluster)
    return m

# Streamlit tabs
tab1, tab2 = st.tabs(["Map View", "Alerts"])
with tab1:
    data = load_and_process_data()
    if data is not None:
        predictions = train_random_forest(data)
        conflict_map = create_map(predictions)
        st_folium(conflict_map, width=1200, height=700)

with tab2:
    st.subheader("User-Submitted Alerts")
    for alert in st.session_state["alerts"]:
        st.write(f"""
        **Name:** {alert['name']}  
        **Location:** {alert['location']}  
        **Message:** {alert['message']}  
        **Timestamp:** {alert['timestamp']}  
        """)


