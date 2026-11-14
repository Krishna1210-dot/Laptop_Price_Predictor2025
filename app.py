import streamlit as st
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
@st.cache_resource
def load_model():

    preprocessor = joblib.load(
        "laptop_preprocessor.joblib"
    )

    model = XGBRegressor()

    model.load_model("xgb_model.json")

    return preprocessor, model

@st.cache_data
def load_dataset():
    return joblib.load("df.joblib")


preprocessor, model = load_model()
df = load_dataset()


st.title("💻 Laptop Price Predictor")
st.write("Welcome to my Laptop Price Prediction System!")
st.info("Enter laptop specifications to estimate its price.")

brand = st.selectbox("Enter the laptop brand:",sorted(df['brand'].unique()))

year=st.slider("Enter the laptop's release year:",sorted(df['release_year'].unique())[0],sorted(df['release_year'].unique())[-1],sorted(df['release_year'].unique())[2])

os=st.selectbox("Enter the laptop's OS:",sorted(df['os'].unique()))

# ff=st.selectbox("Enter the laptop's Form Factor:",sorted(df['form_factor'].unique()))
ff=st.selectbox("Enter the laptop's Form Factor:",['Gaming','Notebook'])

cpubrand=st.selectbox("Enter the laptop's CPU brand:",sorted(df['cpu_brand'].unique()))

cpufamily=st.selectbox("Enter the laptop's Processor:",sorted(df['cpu_family'].unique()))


gpubrand=st.selectbox("Enter the laptop's GPU brand:",sorted(df['gpu_brand'].unique()))
gpufamily=st.selectbox("Enter the laptop's Graphics:",sorted(df['gpu_family'].unique()))


vram=st.select_slider("Enter the laptop GPU's Vram:",options=sorted(df['vram_gb'].unique()))

# ram=st.slider("Enter the laptop's RAM:",sorted(df['ram_gb'].unique())[0],sorted(df['ram_gb'].unique())[-7],sorted(df['ram_gb'].unique())[2])
ram=st.select_slider("Enter the laptop's RAM:",options=[2,4,8,12,16,24,32,64])


storage_type=st.selectbox("Enter the laptop's Storage Type:",sorted(df['storage_type'].unique()))

storage_gb=st.select_slider("Enter the laptop's Storage Size:",options=sorted(df['storage_gb'].unique()))


storage_drive_count=st.number_input("Enter the laptop's Storage Driver Count:",min_value=1,max_value=5)

display=st.selectbox("Enter the laptop's Display Type:",sorted(df['display_type'].unique()))

hz=st.select_slider("Enter the laptop's Refresh Rate:",options=sorted(df['refresh_hz'].unique()))


weight=st.slider("Enter the laptop's weight:",sorted(df['weight_kg'].unique())[0],sorted(df['weight_kg'].unique())[-1],sorted(df['weight_kg'].unique())[2])

res=st.selectbox("Enter the laptop's Resolution",[
            "1280x720", "1366x768", "1600x900", "1920x1080", 
            "1920x1200", "2160x1440", "2256x1504", "2560x1440", 
            "2560x1600", "2880x1800", "3000x2000", "3200x1800", 
            "3456x2234", "3840x2160", "3840x2400"
        ])

screensize=st.select_slider("Enter the laptop's Screen Size",options=[11.6, 12.5, 13.3, 14.0, 15.0, 15.6, 16.0, 17.3, 18.0])


X=res.split('x')[0]
Y=res.split('x')[1]
ppi=((int(X)**2+int(Y)**2)**0.5)/float(screensize)

# query=np.array([brand,year,os,ff,cpubrand,gpubrand,vram,ram,storage_type,storage_gb,storage_drive_count,display,hz,weight,cpufamily,gpufamily,ppi])

# query=query.reshape(1,17)

button = st.button("Predict Price")
if button:
    if ff=='Notebook':
     ff='Mainstream'

    query_dic = {
        'brand': [brand],
        'release_year': [year],
        'os': [os],
        'form_factor': [ff],
        'cpu_brand': [cpubrand],
        'gpu_brand': [gpubrand],
        'vram_gb': [vram],
        'ram_gb': [ram],
        'storage_type': [storage_type],
        'storage_gb': [storage_gb],
        'storage_drive_count': [storage_drive_count],
        'display_type': [display],
        'refresh_hz': [hz],
        'weight_kg': [weight],
        'cpu_family': [cpufamily],
        'gpu_family': [gpufamily], 
        'ppi': [ppi]
    }


    query=pd.DataFrame(query_dic)

    processed_data = preprocessor.transform(query)
    prediction=model.predict(processed_data)

# --- MARKET CALIBRATION LAYER ---
# The training dataset lacks explicit currency and market metadata.
# As a result, the model outputs an uncalibrated baseline value.
# These manual scaling factors calibrate the raw predictions to match 
# localized Indian retail pricing based on chassis tier and specifications.
    if ff=='Mainstream' or vram==0:
        final_prediction=(prediction[0])*46.8
    elif (gpufamily== 'RTX 30 50' and vram==4):
        final_prediction=(prediction[0])*49.5
    elif (gpufamily== 'RTX 30 50' and vram==6):
            final_prediction=(prediction[0])*50
    else:
        final_prediction=(prediction[0])*52
    
    st.balloons()
    st.success(f"Estimated Laptop Price: ₹{final_prediction:,.2f}")