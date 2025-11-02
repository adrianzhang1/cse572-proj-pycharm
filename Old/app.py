import streamlit as st
import pandas as pd
import re

# --- Helper: Parse frozenset strings into Python sets ---
def parse_frozenset(s):
    return set(re.findall(r"'(.*?)'", s))

# --- Helper: RGB to hex conversion ---
def rgb_to_hex(rgb_str):
    match = re.match(r'rgb\((\d+),(\d+),(\d+)\)', color_str)
    if not match:
        return "#000000"
    r, g, b = map(int, match.groups())
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)

# --- Helper: HTML for color swatch ---
def color_with_swatch(color_str):
    hex_color = rgb_to_hex(color_str)
    return f'''
        <div style="
            width: 700px;
            height: 100px;
            background: {hex_color};
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
            color: #000;
            margin: 0px;
            box-sizing: border-box;
        ">
            {color_str}
        </div>
    '''

# --- Filtered CSV loading ---
@st.cache_data
def load_filtered_data(min_conf, min_lift):
    usecols = ['antecedents', 'consequents', 'support', 'confidence', 'lift', 'jaccard', 'conviction']
    df = pd.read_csv("kimono_association_rules 0.01.csv", usecols=usecols)

    df = df[df['confidence'] >= min_conf]
    df = df[df['lift'] >= min_lift]

    df['antecedents'] = df['antecedents'].apply(parse_frozenset)
    df['consequents'] = df['consequents'].apply(parse_frozenset)

    return df

# --- Streamlit UI ---
st.title("Association Rule Visualizer")

# Sidebar filters
min_conf = st.sidebar.slider("Minimum Confidence", 0.0, 1.0, 0.8, 0.01)
min_lift = st.sidebar.slider("Minimum Lift", 0.0, 5.0, 1.1, 0.1)

# Load filtered rules
df = load_filtered_data(min_conf, min_lift)

if df.empty:
    st.warning("No rules match the current filters.")
    st.stop()

# Rule selection
st.markdown(f"**Total Rules:** {len(df)}")
index = st.slider("Select a rule", 0, len(df) - 1, 0)
rule = df.iloc[index]

# Swatch rendering
ante = [color_with_swatch(c) for c in rule['antecedents']]
cons = [color_with_swatch(c) for c in rule['consequents']]

st.markdown("### Antecedents", unsafe_allow_html=True)
st.markdown(''.join(f"{c}<br>" for c in ante), unsafe_allow_html=True)

st.markdown("### Consequents", unsafe_allow_html=True)
st.markdown(''.join(f"{c}<br>" for c in cons), unsafe_allow_html=True)

# Show metrics
st.markdown("### Metrics")
for metric in ['support', 'confidence', 'lift', 'jaccard', 'conviction']:
    st.write(f"**{metric.capitalize()}**: {rule[metric]:.4f}")
