import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="NVIDIA/AMD FPS Estimator", page_icon="🎮", layout="centered")

# --- 2. GAME NAME TRANSLATOR ---
GAME_TRANSLATION = {
    "awayout": "A Way Out",
    "airmechstrike": "AirMech Strike",
    "apexlegends": "Apex Legends",
    "battlefield4": "Battlefield 4",
    "battletech": "Battle Tech",
    "callofdutyww2": "Call of Duty: WW 2",
    "counterstrikeglobaloffensive": "Counter Strike: Global Offensive",
    "destiny2": "Destiny 2",
    "dota2": "Dota 2",
    "farcry5": "Far Cry 5",
    "fortnite": "Fortnite",
    "frostpunk": "Frostpunk",
    "grandtheftauto5": "Grand Theft Auto 5",
    "leagueoflegends": "League of Legends",
    "overwatch": "Overwatch",
    "pathofexile": "Path of Exile",
    "playerunknownsbattlegrounds": "Player Unknowns Battlegrounds",
    "radicalheights": "Radical Heights",
    "rainbowsixsiege": "Rainbow Six: Siege",
    "seaofthieves": "Sea of Thieves",
    "starcraft2": "Star Craft 2",
    "totalwar3kingdoms": "Total War: 3 Kingdoms",
    "warframe": "Warframe",
    "worldoftanks": "World of Tanks",
}


def clean_game_name(text):
    """Converts dirty dataset names into pretty UI names"""
    # 1. Removing b'' artifacts
    text = str(text)
    if text.startswith("b'") or text.startswith('b"'):
        text = text[2:-1]

    # 2. Lowercase for matching
    lower_text = text.lower().strip()

    # 3. Checking Dictionary 
    if lower_text in GAME_TRANSLATION:
        return GAME_TRANSLATION[lower_text]

    # 4. Fallback (If not in dictionary, just Title Case it)
    # Tries to insert space before numbers (e.g., worldoftanks -> World of Tanks)
    import re
    text_with_spaces = re.sub(r'(?<=[a-zA-Z])(?=\d)', ' ', text)
    return text_with_spaces.title()


def clean_hardware_name(text):
    text = str(text)
    if text.startswith("b'") or text.startswith('b"'):
        text = text[2:-1]
    return text.title()


# --- 3. LOADING ASSETS ---
@st.cache_data
def load_data():
    # Loading raw files
    gpu_df = pd.read_csv('assets/gpu_specs_lookup.csv')
    cpu_df = pd.read_csv('assets/cpu_specs_lookup.csv')

    with open('assets/fps_model.pkl', 'rb') as f: model = joblib.load(f)
    with open('assets/game_encoder.pkl', 'rb') as f: game_encoder = pickle.load(f)
    with open('assets/model_columns.pkl', 'rb') as f: model_columns = pickle.load(f)

    # Cleaning Hardware Names immediately
    gpu_df['GpuName'] = gpu_df['GpuName'].apply(clean_hardware_name)
    cpu_df['CpuName'] = cpu_df['CpuName'].apply(clean_hardware_name)

    return gpu_df, cpu_df, model, game_encoder, model_columns


try:
    gpu_df, cpu_df, model, game_encoder, model_columns = load_data()
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# --- 4. UI HEADER ---
st.title("🎮 FPS Bottleneck Predictor")
st.markdown("### Powered by ML | CPU & GPU Performance Analysis")
st.markdown("---")

# --- 5. SIDEBAR INPUTS ---
st.sidebar.header("⚙️ Configuration")

# A. GPU Selection
gpu_list = sorted(gpu_df['GpuName'].unique())
# Defaulting to RTX 3060 if available, else first item
default_gpu = next((x for x in gpu_list if "3060" in x), gpu_list[0])
selected_gpu = st.sidebar.selectbox("Graphics Card", gpu_list, index=gpu_list.index(default_gpu))

# B. CPU Selection
cpu_list = sorted(cpu_df['CpuName'].unique())
default_cpu = next((x for x in cpu_list if "i5" in x or "Ryzen 5" in x), cpu_list[0])
selected_cpu = st.sidebar.selectbox("Processor", cpu_list, index=cpu_list.index(default_cpu))

# C. Game Selection
# Creating a map: { "Pretty Name": "Original_Dirty_Name" }
pretty_to_dirty_map = {}
for dirty_name in game_encoder.classes_:
    pretty_name = clean_game_name(dirty_name)
    pretty_to_dirty_map[pretty_name] = dirty_name

# Displaying the PRETTY names in the dropdown
selected_game_pretty = st.sidebar.selectbox("Game Title", sorted(pretty_to_dirty_map.keys()))

# Get the DIRTY name to send to the model
selected_game_original = pretty_to_dirty_map[selected_game_pretty]

# D. Resolution
res_options = {'1080p (Full HD)': 1.0, '1440p (2K)': 0.75, '4k (Ultra HD)': 0.55}
selected_res_name = st.sidebar.selectbox("Target Resolution", list(res_options.keys()))
res_multiplier = res_options[selected_res_name]

# --- 6. PREDICTION ENGINE ---
if st.button("🚀 Analyze Performance", type="primary"):

    # 1. Retrieve Specs
    gpu_specs = gpu_df[gpu_df['GpuName'] == selected_gpu].iloc[0]
    cpu_specs = cpu_df[cpu_df['CpuName'] == selected_cpu].iloc[0]

    # 2. Build Input
    input_data = {}
    for col in gpu_specs.index:
        if col != 'GpuName': input_data[col] = gpu_specs[col]
    for col in cpu_specs.index:
        if col != 'CpuName': input_data[col] = cpu_specs[col]

    # Add Game (Encoded) & Res (Neutral)
    input_data['GameName_Enc'] = game_encoder.transform([selected_game_original])[0]
    input_data['GameResolution_Enc'] = 2

    # 3. Align Columns
    input_df = pd.DataFrame(columns=model_columns)
    input_df.loc[0] = 0
    for col in model_columns:
        if col in input_data:
            input_df.loc[0, col] = input_data[col]

    # 4. Predict
    base_fps = model.predict(input_df)[0]
    final_fps = int(base_fps * res_multiplier)

    # --- 7. DISPLAY RESULTS ---
    st.markdown(f"### 🎯 Estimated Performance: **{final_fps} FPS**")

    # Dynamic Color Bar
    bar_color = "red" if final_fps < 30 else "orange" if final_fps < 60 else "green"
    st.markdown(f"<style>.stProgress > div > div > div > div {{ background-color: {bar_color}; }}</style>",
                unsafe_allow_html=True)
    st.progress(min(final_fps / 300, 1.0))

    # Details
    c1, c2, c3 = st.columns(3)
    c1.metric("Resolution", selected_res_name.split(" ")[0])
    c2.metric("Settings", "High/Ultra")

    if final_fps > 144:
        c3.metric("Status", "Competitive 🏆")
    elif final_fps > 60:
        c3.metric("Status", "Smooth 🌊")
    else:
        c3.metric("Status", "Playable ⚠️")

    # Bottleneck Insight
    st.divider()
    gpu_power = gpu_specs.get('GpuFP32Performance', 0)

    if final_fps < 60 and gpu_power > 15:
        st.warning(
            f"⚠️ **CPU Limitation:** The {selected_gpu} is powerful, but the {selected_cpu} is limiting performance.")
    elif final_fps > 200:
        st.success(f"⚡ **Maximum Performance:** Your rig is crushing {selected_game_pretty}!")
    else:

        st.info("✅ **Balanced Build:** CPU and GPU are well matched.")

