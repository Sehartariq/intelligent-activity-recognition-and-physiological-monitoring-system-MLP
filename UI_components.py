# ui_components.py
import streamlit as st

# ------------------------
# Animated logo / glowing circle with uploaded image
# ui_components.py
import streamlit as st

import streamlit as st
import time
from PIL import Image

def animated_logo(title="Intelligent Health Monitoring", logo_path=None):
    """Display a logo with a glowing effect at the top of the app."""
    st.markdown(f"<h1 style='text-align:center;'>{title}</h1>", unsafe_allow_html=True)
    if logo_path:
        img = Image.open(logo_path)
        st.image(img, width=120, output_format="PNG")
        # Glow effect via CSS animation
        st.markdown("""
            <style>
            .glow {
                width: 120px;
                height: 120px;
                border-radius: 50%;
                margin: 0 auto;
                box-shadow: 0 0 30px 15px rgba(255,100,100,0.6);
                animation: pulse 2s infinite;
            }
            @keyframes pulse {
                0% { box-shadow: 0 0 10px 5px rgba(255,100,100,0.2); }
                50% { box-shadow: 0 0 40px 20px rgba(255,100,100,0.7); }
                100% { box-shadow: 0 0 10px 5px rgba(255,100,100,0.2); }
            }
            </style>
            <div class="glow"></div>
        """, unsafe_allow_html=True)

def section_header(title):
    st.markdown(f"<h3 style='text-align:center;'>{title}</h3>", unsafe_allow_html=True)

class loading_spinner:
    """Context manager for a custom spinner"""
    def __init__(self, message="Processing..."):
        self.message = message
    def __enter__(self):
        self.spinner = st.spinner(self.message)
        self.spinner.__enter__()
    def __exit__(self, exc_type, exc_value, traceback):
        self.spinner.__exit__(exc_type, exc_value, traceback)
