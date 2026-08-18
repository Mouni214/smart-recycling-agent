import os
import time
import functools
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F

# --- CUSTOM DECORATORS ---

def measure_execution_time(func):
    """Decorator to measure and return the execution time of a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        return result, elapsed_time
    return wrapper

def handle_inference_errors(func):
    """Decorator to catch unexpected errors during inference safely."""
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
        'co2_saved_per_kg': '0.8 kg CO₂e',
        'energy_equivalency': 'Powers a LED bulb for 12 hours',
        'do_list': [
            'Place directly in backyard compost or green municipal bins.',
            'Keep free from plastic wraps, bags, or twist ties.',
            'Chop large organic matter to accelerate decomposition.'
        ],
        'dont_list': [
            'Do not mix with treated wood, painted materials, or plastic packaging.',
            'Avoid adding meat or dairy to home compost bins to prevent pests.'
        ],
        'upcycling_idea': '🌱 Convert organic food scraps into nutrient-rich garden compost or liquid fertilizer tea.'
    },
    'cardboard': {
        'bin': '📦 Blue Paper & Cardboard Bin',
        'co2_saved_per_kg': '3.1 kg CO₂e',
        'energy_equivalency': 'Saves 4,000 kWh of electricity per ton recycled',
        'do_list': [
            'Flatten all boxes completely to maximize bin volume.',
            'Remove plastic bubble wrap, packing peanuts, and heavy packing tape.',
            'Keep cardboard dry at all times.'
        ],
        'dont_list': [
            'Do not recycle cardboard contaminated with food grease (e.g., greasy pizza box bottoms).',
            'Do not include wax-coated produce boxes.'
        ],
        'upcycling_idea': '📦 Reusable storage boxes, weed-barrier sheet mulch for gardening, or cat scratching pads.'
    },
    'glass': {
        'bin': '🍾 Glass Collection Bin',
        'co2_saved_per_kg': '0.3 kg CO₂e',
        'energy_equivalency': 'Saves enough energy to power a computer for 25 minutes per bottle',
        'do_list': [
            'Rinse thoroughly to remove food or liquid residue.',
            'Separate by color (Clear, Amber, Green) if required by local recycling centers.',
            'Keep metal lids attached or sorted in metal bins.'
        ],
        'dont_list': [
            'Do not mix container glass with window panes, mirrors, light bulbs, or Pyrex cookware.',
            'Do not throw broken glass in open bins without protective wrapping.'
        ],
        'upcycling_idea': '🫙 Reuse jars as food storage containers, DIY flower vases, or candle holders.'
    },
    'metal': {
        'bin': '🥫 Yellow Metal & Can Recycling Bin',
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
        'upcycling_idea': '🎨 Use clean tin cans as desk organizers, cutlery holders, or potted plant containers.'
    },
    'paper': {
        'bin': '📄 Paper & Office Waste Bin',
        'co2_saved_per_kg': '1.8 kg CO₂e',
        'energy_equivalency': 'Saves 17 trees per ton of paper recycled',
        'do_list': [
            'Keep paper dry, clean, and flat.',
            'Recycle newspapers, magazines, mail, envelopes, and paper bags.',
            'Staples and small paperclips are generally acceptable in modern facilities.'
        ],
        'dont_list': [
            'Do not recycle paper towels, tissues, thermal receipt paper, or wax paper.',
            'Do not recycle foil-lined wrapping paper or glitter cards.'
        ],
        'upcycling_idea': '📝 Use clean single-sided prints for scratch notes or paper mache art projects.'
    },
    'plastic': {
        'bin': '♻️ Rigid Plastic Container Bin',
        'co2_saved_per_kg': '1.5 kg CO₂e',
        'energy_equivalency': 'Saves 5,774 kWh of energy per ton of plastic recycled',
        'do_list': [
            'Check resin identification code (#1 PET and #2 HDPE are widely accepted).',
            'Rinse all containers clean of soaps, sauces, and liquids.',
            'Reattach bottle caps tightly after rinsing.'
        ],
        'dont_list': [
            'Do not place plastic bags, bubble wrap, or film wrap in standard curb bins (use supermarket drop-offs).',
            'Do not recycle black plastic containers in facilities using optical sorting.'
        ],
        'upcycling_idea': '🪴 Convert plastic bottles into self-watering planters or small seed-starting pots.'
    }
}

# --- MODEL LOADING WITH CACHING ---

@st.cache_resource
def load_models():
    eff_model = models.efficientnet_b3(weights=None)
    eff_model.classifier[1] = nn.Linear(eff_model.classifier[1].in_features, len(CLASS_NAMES))
    eff_ckpt = torch.load('efficientnet_b3_best.pth', map_location='cpu')
    eff_model.load_state_dict(eff_ckpt['model_state_dict'] if isinstance(eff_ckpt, dict) and 'model_state_dict' in eff_ckpt else eff_ckpt)
    eff_model.eval()

    res_path = 'resnet50_best (1).pth' if os.path.exists('resnet50_best (1).pth') else 'resnet50_best.pth'
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
        res_probs = F.softmax(res_model(input_tensor), dim=1)[0]
        ensemble_probs = (eff_probs + res_probs) / 2.0
        confidence, predicted_idx = torch.max(ensemble_probs, 0)
        
    return predicted_idx.item(), confidence.item(), eff_probs, res_probs

# --- STREAMLIT USER INTERFACE ---

st.set_page_config(page_title="AI Smart Recycling Agent", layout="centered", page_icon="♻️")

st.title("🤖 AI Smart Recycling Recommendation Agent")
st.write("Upload an image to get dual-model classification alongside material-specific recycling guidelines and environmental impact estimates.")

# Sidebar Configuration for Smart Customization
st.sidebar.header("⚙️ Agent Settings")
location_preset = st.sidebar.selectbox("Select Recycling Facility Type:", ["Standard Municipal Curbside", "Specialized Sorting Plant", "Home Backyard System"])

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
                predicted_idx, confidence, eff_probs, res_probs = result_tuple
                
                predicted_class = CLASS_NAMES[predicted_idx]
                agent_info = SMART_RECYCLING_AGENT_DB[predicted_class]

                st.success(f"**Detected Material:** {predicted_class.upper()} ({confidence * 100:.2f}% Confidence)")
                st.caption(f"⚡ Analysis time: **{elapsed_time:.3f} seconds** | Facility Context: **{location_preset}**")
                
                # --- TABBED RECOMMENDATION SECTION ---
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
                    st.subheader("Upcycling & Reuse Alternative")
                    st.write(agent_info['upcycling_idea'])

                with tab4:
                    st.subheader("Ensemble Model Breakdown")
                    col_eff, col_res = st.columns(2)
                    col_eff.metric("EfficientNet-B3", CLASS_NAMES[torch.argmax(eff_probs).item()].upper(), f"{torch.max(eff_probs).item()*100:.1f}%")
                    col_res.metric("ResNet50", CLASS_NAMES[torch.argmax(res_probs).item()].upper(), f"{torch.max(res_probs).item()*100:.1f}%")