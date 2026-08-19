import os
import time
import functools
import urllib.request
import zipfile
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F
import pandas as pd

# --- INITIALIZE SESSION STATE FOR ECO-TRACKER ---
if 'scan_history' not in st.session_state:
    st.session_state.scan_history = []

# --- MODEL WEIGHTS AUTO-DOWNLOAD CONFIGURATION ---
EFF_DROPBOX_URL = "https://www.dropbox.com/scl/fi/f1lxqlmh2tbtcjmonu1h0/efficientnet_b3_best.zip?rlkey=13upilam4lijh7teqcpmvml7k&st=avoe1uui&dl=1"
RESNET_DROPBOX_URL = "https://www.dropbox.com/scl/fi/9c9otu14o4kn9eqj0m792/resnet50_best-1.zip?rlkey=d5ilztslx74kfws3ppllyvzwg&st=f1wkrz5m&dl=1"

def download_weights_if_missing():
    eff_exists = any('efficient' in f.lower() and f.endswith('.pth') for f in os.listdir('.'))
    if not eff_exists:
        with st.spinner("Downloading EfficientNet-B3 weights..."):
            temp_file = "eff_weights.tmp"
            urllib.request.urlretrieve(EFF_DROPBOX_URL, temp_file)
            if zipfile.is_zipfile(temp_file):
                with zipfile.ZipFile(temp_file, 'r') as zip_ref:
                    zip_ref.extractall(".")
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            else:
                os.rename(temp_file, "efficientnet_b3_best.pth")

    res_exists = any('resnet' in f.lower() and f.endswith('.pth') for f in os.listdir('.'))
    if not res_exists:
        with st.spinner("Downloading ResNet50 weights..."):
            temp_file = "res_weights.tmp"
            urllib.request.urlretrieve(RESNET_DROPBOX_URL, temp_file)
            if zipfile.is_zipfile(temp_file):
                with zipfile.ZipFile(temp_file, 'r') as zip_ref:
                    zip_ref.extractall(".")
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            else:
                os.rename(temp_file, "resnet50_best.pth")

# --- CUSTOM DECORATORS ---

def measure_execution_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        return result, elapsed_time
    return wrapper

def handle_inference_errors(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"Execution Error during inference: {str(e)}")
            return None
    return wrapper

# --- ADVANCED SMART RECYCLING KNOWLEDGE BASE ---

CLASS_NAMES = ['biodegradable', 'cardboard', 'glass', 'metal', 'paper', 'plastic']

SMART_RECYCLING_AGENT_DB = {
    'biodegradable': {
        'bin': '💚 Organic / Green Compost Bin',
        'co2_num': 0.8,
        'co2_saved_per_kg': '0.8 kg CO₂e',
        'energy_equivalency': 'Powers an LED bulb for 12 hours',
        'do_list': [
            'Place directly in backyard compost or green municipal bins.',
            'Keep free from plastic wraps, bags, or twist ties.',
            'Chop large organic matter to accelerate decomposition.'
        ],
        'dont_list': [
            'Do not mix with treated wood, painted materials, or plastic packaging.',
            'Avoid adding meat or dairy to home compost bins to prevent pests.'
        ],
        'upcycling_ideas': [
            '🌱 Backyard Composting: Convert scraps into nutrient-rich humus for garden beds.',
            '🍵 Fertilizer Tea: Steep vegetable ends in water to make a nitrogen-rich liquid plant feed.',
            '🍋 Citrus All-Purpose Cleaner: Infuse leftover lemon/orange peels in white vinegar for 2 weeks.',
            '🪴 Seed Starter Pots: Use hollowed citrus halves or eggshells to germinate garden seeds.'
        ]
    },
    'cardboard': {
        'bin': '📦 Blue Paper & Cardboard Bin',
        'co2_num': 3.1,
        'co2_saved_per_kg': '3.1 kg CO₂e',
        'energy_equivalency': 'Saves 4,000 kWh of electricity per ton recycled',
        'do_list': [
            'Flatten all boxes completely to maximize bin volume.',
            'Remove plastic bubble wrap, packing peanuts, and heavy packing tape.',
            'Keep cardboard dry at all times.'
        ],
        'dont_list': [
            'Do not recycle cardboard contaminated with food grease.',
            'Do not include wax-coated produce boxes.'
        ],
        'upcycling_ideas': [
            '📦 Drawer & Closet Organizers: Cut and assemble box sections into custom storage dividers.',
            '🌾 Sheet Mulch Layer: Place non-printed cardboard under soil to suppress weeds naturally.',
            '🐱 Pet Scratching Post: Glue tightly coiled cardboard strips into custom cat scratch pads.',
            '🖼️ Picture Frame Backing: Use heavy-duty corrugated board for sturdy art backing frames.'
        ]
    },
    'glass': {
        'bin': '🍾 Glass Collection Bin',
        'co2_num': 0.3,
        'co2_saved_per_kg': '0.3 kg CO₂e',
        'energy_equivalency': 'Powers a computer for 25 minutes per bottle',
        'do_list': [
            'Rinse thoroughly to remove food or liquid residue.',
            'Separate by color (Clear, Amber, Green) if required locally.',
            'Keep metal lids attached or sorted in metal bins.'
        ],
        'dont_list': [
            'Do not mix container glass with window panes, mirrors, or Pyrex cookware.',
            'Do not throw broken glass in open bins without protective wrapping.'
        ],
        'upcycling_ideas': [
            '🫙 Dry Food Pantry Storage: Wash and sterilize jars to store rice, beans, spices, or snacks.',
            '🕯️ Candle Holders & Lanterns: Insert tealights or fairy lights into clear jars for decor.',
            '🌱 Propagation Vases: Fill glass bottles with water to root plant cuttings.',
            '🎨 Mosaic Crafts: Use safely smoothed glass pieces for decorative mosaic coasters.'
        ]
    },
    'metal': {
        'bin': '🥫 Yellow Metal & Can Recycling Bin',
        'co2_num': 9.0,
        'co2_saved_per_kg': '9.0 kg CO₂e (Aluminum)',
        'energy_equivalency': 'Saves 95% of energy required to process raw aluminum ore',
        'do_list': [
            'Rinse out food tins, soup cans, and soda cans thoroughly.',
            'Push loose metal lids inside the empty can and squeeze shut.',
            'Aerosol cans must be 100% completely empty before recycling.'
        ],
        'dont_list': [
            'Do not recycle full or pressurized spray cans.',
            'Do not mix propane tanks or electronic scrap in household metal bins.'
        ],
        'upcycling_ideas': [
            '✏️ Desk Stationery Holders: Wrap clean tin cans with fabric or twine to organize pens.',
            '🪴 Hanging Herb Planters: Punch drainage holes in the bottom of cans for kitchen herb gardens.',
            '🕯️ Lantern Candle Holders: Punch decorative hole patterns into tin cans to diffuse candlelight.',
            '🐦 Bird Feeder: Convert empty tin cans with a wooden perch into outdoor bird feeders.'
        ]
    },
    'paper': {
        'bin': '📄 Paper & Office Waste Bin',
        'co2_num': 1.8,
        'co2_saved_per_kg': '1.8 kg CO₂e',
        'energy_equivalency': 'Saves 17 trees per ton of paper recycled',
        'do_list': [
            'Keep paper dry, clean, and flat.',
            'Recycle newspapers, magazines, mail, envelopes, and paper bags.',
            'Staples and small paperclips are generally acceptable.'
        ],
        'dont_list': [
            'Do not recycle paper towels, tissues, thermal receipt paper, or wax paper.',
            'Do not recycle foil-lined wrapping paper or glitter cards.'
        ],
        'upcycling_ideas': [
            '📝 Scratchpads & Notebooks: Bind single-sided printed pages into handy scrap memo pads.',
            '📦 Eco Packing Material: Shred old newspapers to cushion fragile items for shipping.',
            '📜 Handmade Recycled Paper: Blend paper scraps with water to press fresh custom stationery.',
            '🎨 Origami & Crafts: Use colorful magazine pages for paper folding or papier-mâché.'
        ]
    },
    'plastic': {
        'bin': '♻️ Rigid Plastic Container Bin',
        'co2_num': 1.5,
        'co2_saved_per_kg': '1.5 kg CO₂e',
        'energy_equivalency': 'Saves 5,774 kWh of energy per ton recycled',
        'do_list': [
            'Check resin identification code (#1 PET and #2 HDPE are widely accepted).',
            'Rinse all containers clean of soaps, sauces, and liquids.',
            'Reattach bottle caps tightly after rinsing.'
        ],
        'dont_list': [
            'Do not place plastic bags or film wrap in standard curb bins.',
            'Do not recycle black plastic containers in facilities using optical sorting.'
        ],
        'upcycling_ideas': [
            '🪴 Self-Watering Planters: Cut plastic bottles in half and invert the top to create wick planters.',
            '🪛 Scoop & Funnels: Trim sturdy plastic jugs into garden soil scoops or oil funnels.',
            '🔌 Cable Organizers: Cut plastic bottle sections into clips to manage power cords.',
            '🐦 Vertical Garden Tower: Stack linked plastic bottles along a wall for vertical plant growth.'
        ]
    }
}

# --- MODEL LOADING WITH CACHING & AUTO-DOWNLOAD ---

@st.cache_resource
def load_models():
    download_weights_if_missing()

    eff_path = next((f for f in os.listdir('.') if 'efficient' in f.lower() and f.endswith('.pth')), 'efficientnet_b3_best.pth')
    eff_model = models.efficientnet_b3(weights=None)
    eff_model.classifier[1] = nn.Linear(eff_model.classifier[1].in_features, len(CLASS_NAMES))
    eff_ckpt = torch.load(eff_path, map_location='cpu')
    eff_model.load_state_dict(eff_ckpt['model_state_dict'] if isinstance(eff_ckpt, dict) and 'model_state_dict' in eff_ckpt else eff_ckpt)
    eff_model.eval()

    res_path = next((f for f in os.listdir('.') if 'resnet' in f.lower() and f.endswith('.pth')), None)
    res_model = None
    if res_path and os.path.exists(res_path):
        res_model = models.resnet50(weights=None)
        res_model.fc = nn.Linear(res_model.fc.in_features, len(CLASS_NAMES))
        res_ckpt = torch.load(res_path, map_location='cpu')
        res_model.load_state_dict(res_ckpt['model_state_dict'] if isinstance(res_ckpt, dict) and 'model_state_dict' in res_ckpt else res_ckpt)
        res_model.eval()

    return eff_model, res_model

# --- INFERENCE PIPELINE ---

@handle_inference_errors
@measure_execution_time
def predict_ensemble(eff_model, res_model, input_tensor):
    with torch.no_grad():
        eff_probs = F.softmax(eff_model(input_tensor), dim=1)[0]
        
        if res_model is not None:
            res_probs = F.softmax(res_model(input_tensor), dim=1)[0]
            ensemble_probs = (eff_probs + res_probs) / 2.0
        else:
            res_probs = eff_probs
            ensemble_probs = eff_probs
            
        confidence, predicted_idx = torch.max(ensemble_probs, 0)
        
    return predicted_idx.item(), confidence.item(), eff_probs, res_probs, ensemble_probs

# --- STREAMLIT USER INTERFACE ---

st.set_page_config(page_title="AI Smart Recycling Agent", layout="centered", page_icon="♻️")

st.title("🤖 AI Smart Recycling Recommendation Agent")
st.write("Upload an image to get dual-model classification alongside material-specific recycling guidelines and environmental impact estimates.")

# Sidebar Configuration & Eco Dashboard
st.sidebar.header("⚙️ Agent Settings")
location_preset = st.sidebar.selectbox("Select Recycling Facility Type:", ["Standard Municipal Curbside", "Specialized Sorting Plant", "Home Backyard System"])

st.sidebar.markdown("---")
st.sidebar.header("🏆 Session Eco-Tracker")
total_scans = len(st.session_state.scan_history)
total_co2 = sum([item["CO2 Saved (kg)"] for item in st.session_state.scan_history])

if total_scans == 0:
    badge = "🌱 Newbie Recycler"
elif total_scans < 3:
    badge = "🌿 Eco Apprentice"
elif total_scans < 8:
    badge = "⭐ Recycling Champion"
else:
    badge = "👑 Zero-Waste Legend"

st.sidebar.metric("Total Items Scanned", total_scans)
st.sidebar.metric("Total CO₂ Saved", f"{total_co2:.2f} kg")
st.sidebar.info(f"**Current Rank:** {badge}")

if total_scans > 0:
    df_history = pd.DataFrame(st.session_state.scan_history)
    csv_data = df_history.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label="📥 Export Waste Audit (CSV)",
        data=csv_data,
        file_name="waste_audit_log.csv",
        mime="text/csv"
    )

try:
    eff_model, res_model = load_models()
except Exception as e:
    st.error(f"Error loading model weights: {e}")
    st.stop()

transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

uploaded_file = st.file_uploader("Choose a waste material photo...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Waste Image', use_container_width=True)
    
    if st.button('Analyze & Generate Smart Recommendation', type='primary'):
        with st.spinner('Agent analyzing visual characteristics & matching disposal rules...'):
            input_tensor = transform(image).unsqueeze(0)
            prediction_output = predict_ensemble(eff_model, res_model, input_tensor)
            
            if prediction_output is not None:
                (result_tuple, elapsed_time) = prediction_output
                predicted_idx, confidence, eff_probs, res_probs, ensemble_probs = result_tuple
                
                predicted_class = CLASS_NAMES[predicted_idx]
                agent_info = SMART_RECYCLING_AGENT_DB[predicted_class]

                st.session_state.scan_history.append({
                    "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "Material": predicted_class.upper(),
                    "Confidence": f"{confidence * 100:.2f}%",
                    "CO2 Saved (kg)": agent_info['co2_num'],
                    "Target Bin": agent_info['bin']
                })

                st.success(f"**Detected Material:** {predicted_class.upper()} ({confidence * 100:.2f}% Confidence)")
                st.caption(f"⚡ Analysis time: **{elapsed_time:.3f} seconds** | Facility Context: **{location_preset}**")
                
                tab1, tab2, tab3, tab4 = st.tabs(["📋 Disposal Steps", "🌱 Eco Impact", "💡 Upcycling", "🔍 Model Confidence"])

                with tab1:
                    st.subheader("Actionable Disposal Steps")
                    st.info(f"**Target Destination:** {agent_info['bin']}")
                    
                    col_do, col_dont = st.columns(2)
                    with col_do:
                        st.markdown("**✅ DO:**")
                        for item in agent_info['do_list']:
                            st.write(f"- {item}")
                            
                    with col_dont:
                        st.markdown("**❌ DON'T:**")
                        for item in agent_info['dont_list']:
                            st.write(f"- {item}")

                with tab2:
                    st.subheader("Estimated Environmental Saved Impact")
                    col_metric1, col_metric2 = st.columns(2)
                    col_metric1.metric("CO₂ Offset", agent_info['co2_saved_per_kg'])
                    col_metric2.metric("Energy Conservation", "High Efficiency")
                    st.write(f"**Energy Equivalent:** {agent_info['energy_equivalency']}")

                with tab3:
                    st.subheader("💡 Creative Upcycling & Reuse Alternatives")
                    for idea in agent_info['upcycling_ideas']:
                        st.write(f"- {idea}")

                with tab4:
                    st.subheader("Model Metrics & Predictions")
                    col_eff, col_res = st.columns(2)
                    col_eff.metric("EfficientNet-B3", CLASS_NAMES[torch.argmax(eff_probs).item()].upper(), f"{torch.max(eff_probs).item()*100:.1f}%")
                    if res_model is not None:
                        col_res.metric("ResNet50", CLASS_NAMES[torch.argmax(res_probs).item()].upper(), f"{torch.max(res_probs).item()*100:.1f}%")
                    else:
                        col_res.metric("ResNet50", "Offline / Single Model Mode", "N/A")

                    st.markdown("---")
                    st.write("**Ensemble Category Probability Distribution:**")
                    prob_df = pd.DataFrame({
                        "Material": [c.capitalize() for c in CLASS_NAMES],
                        "Probability (%)": (ensemble_probs * 100).cpu().numpy()
                    }).sort_values(by="Probability (%)", ascending=True)
                    
                    st.bar_chart(prob_df, x="Material", y="Probability (%)", horizontal=True)