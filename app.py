# ═══════════════════════════════════════════════════════════════════
# app.py — DermaAI Pakistan v2.0
# Multi-Provider AI Dermatology Assistant
# Providers: Google Gemini · Groq Cloud · Alibaba Qwen
# Deploy: Hugging Face Spaces | Streamlit Cloud | Local (Windows/Linux)
# ═══════════════════════════════════════════════════════════════════

import os
import re
import json
import base64
import warnings
from io import BytesIO
from typing import Optional, Dict, List, Any

warnings.filterwarnings("ignore")

# ─── Detect environment (HF Spaces sets SPACE_ID automatically) ─────
IS_HF_SPACES: bool = bool(os.environ.get("SPACE_ID", ""))

# ─── Page config MUST be first Streamlit call ───────────────────────
import streamlit as st

st.set_page_config(
    page_title="DermaAI Pakistan",
    page_icon="⚕",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "DermaAI Pakistan — Advanced AI Dermatology Assistant v2.0"},
)

import numpy as np
from PIL import Image, ImageFilter

# ─── Conditional heavy dependencies ─────────────────────────────────
try:
    import torch
    import torchvision.models as tvm
    import torchvision.transforms as T
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    GRADCAM_OK = True
except ImportError:
    GRADCAM_OK = False

try:
    import google.generativeai as genai
    GEMINI_OK = True
except ImportError:
    GEMINI_OK = False

try:
    from openai import OpenAI as _OpenAIClient
    OPENAI_OK = True
except ImportError:
    OPENAI_OK = False

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    RAG_OK = True
except ImportError:
    RAG_OK = False

# ─── Local modules ────────────────────────────────────────────────────
try:
    from knowledge_base import SKIN_KNOWLEDGE, get_all_documents
    from medicines_pakistan import get_medicines, get_care_tips
except ImportError as _e:
    st.error(f"Missing knowledge_base.py or medicines_pakistan.py — ensure both are in the same directory. ({_e})")
    st.stop()


# ═══════════════════════════════════════════════════════════════════
# PROVIDER & MODEL REGISTRY
# ═══════════════════════════════════════════════════════════════════

PROVIDERS: Dict[str, Dict] = {
    "gemini": {
        "name": "Google Gemini",
        "short": "Gemini",
        "icon": "G",
        "color": "#4285f4",
        "models": [
            # ── Gemini 2.x (current stable — these work) ──────────────────
            {"id": "gemini-2.0-flash",             "name": "Gemini 2.0 Flash ⚡ (Recommended)", "vision": True},
            {"id": "gemini-2.0-flash-exp",         "name": "Gemini 2.0 Flash Exp",              "vision": True},
            {"id": "gemini-2.5-flash-preview-05-20","name": "Gemini 2.5 Flash Preview",          "vision": True},
            {"id": "gemini-2.5-pro-preview-05-06", "name": "Gemini 2.5 Pro Preview",             "vision": True},
            # ── Gemini 1.5 (reliable fallbacks) ──────────────────────────
            {"id": "gemini-1.5-flash",             "name": "Gemini 1.5 Flash",                  "vision": True},
            {"id": "gemini-1.5-flash-8b",          "name": "Gemini 1.5 Flash-8B (Fastest)",     "vision": True},
            {"id": "gemini-1.5-pro",               "name": "Gemini 1.5 Pro (Best Quality)",     "vision": True},
        ],
        "default": "gemini-2.0-flash",
        "env_key": "GEMINI_API_KEY",
        "key_hint": "AIza...  ·  aistudio.google.com (free 15 req/min)",
        "available": GEMINI_OK,
        "req_lib": "google-generativeai",
    },
    "groq": {
        "name": "Groq Cloud (Free)",
        "short": "Groq",
        "icon": "G",
        "color": "#f97316",
        "models": [
            # Vision-capable (use for image analysis)
            {"id": "meta-llama/llama-4-scout-17b-16e-instruct", "name": "Llama 4 Scout 17B 🖼 (Best vision)", "vision": True},
            {"id": "llama-3.2-90b-vision-preview",              "name": "Llama 3.2 90B 🖼 (High quality)",   "vision": True},
            {"id": "llama-3.2-11b-vision-preview",              "name": "Llama 3.2 11B 🖼 (Fast vision)",    "vision": True},
            # Text-only (use for chat only)
            {"id": "llama-3.3-70b-versatile",                   "name": "Llama 3.3 70B (text only)",         "vision": False},
            {"id": "llama-3.1-8b-instant",                      "name": "Llama 3.1 8B Instant (text only)",  "vision": False},
            {"id": "mixtral-8x7b-32768",                        "name": "Mixtral 8x7B (text only)",          "vision": False},
            {"id": "gemma2-9b-it",                              "name": "Gemma 2 9B (text only)",            "vision": False},
        ],
        "default": "meta-llama/llama-4-scout-17b-16e-instruct",
        "env_key": "GROQ_API_KEY",
        "key_hint": "gsk_...  ·  console.groq.com (100% free)",
        "base_url": "https://api.groq.com/openai/v1",
        "available": OPENAI_OK,
        "req_lib": "openai",
    },
    "qwen": {
        "name": "Alibaba Qwen",
        "short": "Qwen",
        "icon": "Q",
        "color": "#ff6800",
        "models": [
            # Vision-capable (use for image analysis)
            {"id": "qwen-vl-max",               "name": "Qwen-VL Max 🖼 (Best vision)",      "vision": True},
            {"id": "qwen-vl-plus",              "name": "Qwen-VL Plus 🖼 (Fast vision)",     "vision": True},
            {"id": "qwen2.5-vl-72b-instruct",   "name": "Qwen2.5-VL 72B 🖼 (Latest)",       "vision": True},
            {"id": "qwen2.5-vl-7b-instruct",    "name": "Qwen2.5-VL 7B 🖼 (Lightweight)",   "vision": True},
            # Text-only (use for chat only)
            {"id": "qwen-plus",                 "name": "Qwen Plus (text only)",             "vision": False},
            {"id": "qwen-turbo",                "name": "Qwen Turbo (fast, text only)",      "vision": False},
        ],
        "default": "qwen-vl-max",
        "env_key": "QWEN_API_KEY",
        "key_hint": "sk-...  ·  bailian.console.aliyun.com (free credits)",
        "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "available": OPENAI_OK,
        "req_lib": "openai",
    },
}

SEVERITY_COLORS = {
    "Low": "#22c55e", "Medium": "#f59e0b",
    "High": "#ef4444", "Critical": "#dc2626",
    "کم": "#22c55e", "درمیانہ": "#f59e0b",
    "زیادہ": "#ef4444", "نازک": "#dc2626",
}

# ═══════════════════════════════════════════════════════════════════
# BILINGUAL UI STRINGS
# ═══════════════════════════════════════════════════════════════════

UI: Dict[str, Dict] = {
    "en": {
        "title": "DermaAI Pakistan",
        "tagline": "Multi-Provider AI Dermatology  ·  Bilingual  ·  Pakistan Medicine Database",
        "lang_label": "Language",
        "provider_label": "AI Provider",
        "model_label": "Model",
        "api_key_label": "API Key",
        "key_env_note": "Key loaded from environment",
        "tab_analysis": "Skin Analysis",
        "tab_chat": "AI Chat",
        "tab_about": "About",
        "upload_header": "Upload Skin Image",
        "upload_hint": "JPEG / PNG · Max 5 MB · Clear close-up photo",
        "analyze_btn": "Analyze Condition",
        "analyzing": "Running multi-agent AI analysis...",
        "no_image": "Please upload a skin image first.",
        "no_key": "Please enter an API key in the sidebar.",
        "result_header": "Analysis Results",
        "condition": "Condition",
        "severity": "Severity",
        "confidence": "Confidence",
        "affected_area": "Area",
        "heatmap_header": "Attention Heatmap (XAI)",
        "heatmap_caption": "Highlighted regions show where the AI focused during analysis.",
        "medicines_tab": "Medicines",
        "care_tab": "Care Tips",
        "doctor_tab": "See Doctor",
        "chat_header": "Chat with DermaAI",
        "chat_placeholder": "Ask about your skin condition, treatment, or care...",
        "chat_hint": "e.g.  'Is this medicine safe during pregnancy?'  or  'How long will this take to heal?'",
        "disclaimer": "DermaAI is for informational purposes only. Always consult a certified dermatologist for diagnosis and treatment. In an emergency, visit the nearest hospital.",
        "rx": "Rx Required",
        "otc": "OTC",
        "price": "Price (PKR)",
        "usage": "Usage",
        "caution": "Caution",
        "brands": "Brands (PK)",
        "no_results": "Upload an image and click Analyze to see results here.",
        "demo_mode": "Demo Mode — Enter an API key for live AI analysis",
        "urgent_msg": "URGENT: This condition requires immediate specialist attention.",
        "chat_context": "Context loaded for",
        "chat_no_context": "Tip: Analyse an image first for context-aware responses.",
        "clear_chat": "Clear Chat",
    },
    "ur": {
        "title": "ڈرما اے آئی پاکستان",
        "tagline": "کثیر فراہم کنندہ مصنوعی ذہانت  ·  دو زبانی  ·  پاکستانی ادویات",
        "lang_label": "زبان",
        "provider_label": "AI فراہم کنندہ",
        "model_label": "ماڈل",
        "api_key_label": "API کلید",
        "key_env_note": "کلید ماحول سے لوڈ ہوئی",
        "tab_analysis": "جلد کا تجزیہ",
        "tab_chat": "AI چیٹ",
        "tab_about": "تعارف",
        "upload_header": "جلد کی تصویر اپلوڈ کریں",
        "upload_hint": "JPEG / PNG · زیادہ سے زیادہ 5 MB",
        "analyze_btn": "تجزیہ کریں",
        "analyzing": "کثیر ایجنٹ تجزیہ ہو رہا ہے...",
        "no_image": "براہ کرم پہلے جلد کی تصویر اپلوڈ کریں۔",
        "no_key": "براہ کرم سائیڈبار میں API کلید درج کریں۔",
        "result_header": "نتائج",
        "condition": "بیماری",
        "severity": "شدت",
        "confidence": "یقین",
        "affected_area": "حصہ",
        "heatmap_header": "توجہ کا نقشہ (XAI)",
        "heatmap_caption": "روشن حصہ وہ جگہ ہے جہاں مصنوعی ذہانت نے توجہ دی۔",
        "medicines_tab": "ادویات",
        "care_tab": "دیکھ بھال",
        "doctor_tab": "ڈاکٹر سے ملیں",
        "chat_header": "ڈرما اے آئی سے بات",
        "chat_placeholder": "اپنی جلد کی بیماری یا علاج کے بارے میں پوچھیں...",
        "chat_hint": "مثلاً: 'کیا یہ دوا حمل میں محفوظ ہے؟'",
        "disclaimer": "ڈرما اے آئی صرف معلوماتی مقاصد کے لیے ہے۔ تشخیص کے لیے ماہر جلد سے ملیں۔",
        "rx": "نسخہ ضروری",
        "otc": "بغیر نسخے",
        "price": "قیمت (روپے)",
        "usage": "استعمال",
        "caution": "احتیاط",
        "brands": "برانڈز",
        "no_results": "تصویر اپلوڈ کریں اور تجزیہ دبائیں۔",
        "demo_mode": "ڈیمو موڈ — مکمل تجزیے کے لیے API کلید درج کریں",
        "urgent_msg": "فوری: اس حالت کے لیے فوری ماہر کا معائنہ ضروری ہے۔",
        "chat_context": "سیاق و سباق لوڈ ہوا",
        "chat_no_context": "پہلے تصویر کا تجزیہ کریں تاکہ بہتر جواب ملے۔",
        "clear_chat": "چیٹ صاف کریں",
    },
}


# ═══════════════════════════════════════════════════════════════════
# CSS — MODERN MEDICAL DARK THEME
# ═══════════════════════════════════════════════════════════════════

def inject_css():
    st.markdown("""
<style>
/* ── Reset & Root ─────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

[data-testid="stAppViewContainer"] {
    background: linear-gradient(160deg, #05091a 0%, #080d1f 60%, #050810 100%) !important;
    min-height: 100vh;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060a1c 0%, #050810 100%) !important;
    border-right: 1px solid rgba(59,130,246,0.12) !important;
}
section.main { background: transparent !important; }
.block-container { padding-top: 1.5rem !important; }

/* ── Typography ───────────────────────────────────────── */
html, body, [class*="css"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
h1, h2, h3 { color: #e2e8f0 !important; letter-spacing: -0.3px; }
p, li, div, span { color: #94a3b8; }
.urdu { direction: rtl; text-align: right; line-height: 2.1; }

/* ── Header Card ──────────────────────────────────────── */
.derm-header {
    position: relative;
    background: linear-gradient(135deg,
        rgba(12,22,55,0.97) 0%,
        rgba(15,30,75,0.95) 50%,
        rgba(10,18,50,0.97) 100%
    );
    border: 1px solid rgba(59,130,246,0.18);
    border-radius: 20px;
    padding: 2rem 2.5rem 1.8rem;
    margin-bottom: 1.5rem;
    overflow: hidden;
}
.derm-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent 0%, #3b82f6 30%, #8b5cf6 70%, transparent 100%);
}
.derm-header::after {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(59,130,246,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.derm-header h1 {
    font-size: 2.1rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.35rem 0 !important;
    line-height: 1.2 !important;
}
.derm-header .tagline {
    color: #475569 !important;
    font-size: 0.88rem;
    margin: 0 0 1rem 0;
    font-weight: 400;
}
.badge-row { display: flex; gap: 0.4rem; flex-wrap: wrap; }
.hdr-badge {
    background: rgba(59,130,246,0.08);
    border: 1px solid rgba(59,130,246,0.2);
    color: #60a5fa !important;
    padding: 0.18rem 0.7rem;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 500;
}
.hdr-badge.green {
    background: rgba(34,197,94,0.08);
    border-color: rgba(34,197,94,0.2);
    color: #4ade80 !important;
}
.hdr-badge.purple {
    background: rgba(139,92,246,0.08);
    border-color: rgba(139,92,246,0.2);
    color: #a78bfa !important;
}

/* ── Glass Card ───────────────────────────────────────── */
.glass-card {
    background: rgba(10,18,42,0.82);
    border: 1px solid rgba(25,45,90,0.7);
    border-radius: 16px;
    padding: 1.4rem;
    margin: 0.6rem 0;
    transition: border-color 0.25s ease;
}
.glass-card:hover { border-color: rgba(59,130,246,0.25); }
.card-title {
    font-size: 0.78rem;
    font-weight: 700;
    color: #60a5fa !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.9rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(25,45,90,0.9);
}

/* ── Condition Display ────────────────────────────────── */
.condition-wrapper {
    background: linear-gradient(135deg,
        rgba(29,78,216,0.1) 0%,
        rgba(124,58,237,0.08) 100%
    );
    border: 1px solid rgba(59,130,246,0.25);
    border-radius: 14px;
    padding: 1rem 1.3rem;
    margin-bottom: 1rem;
    display: inline-block;
}
.condition-name {
    font-size: 1.35rem;
    font-weight: 700;
    color: #f1f5f9 !important;
    margin: 0;
}
.severity-pill {
    display: inline-flex;
    align-items: center;
    padding: 0.2rem 0.75rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    margin-left: 0.6rem;
    vertical-align: middle;
    letter-spacing: 0.03em;
}
.conf-label {
    font-size: 0.8rem;
    color: #475569 !important;
    margin: 0 0 0.2rem 0;
}
.conf-value {
    font-size: 1.25rem;
    font-weight: 700;
    margin: 0 0 0.3rem 0;
}
.conf-track {
    background: rgba(10,20,45,0.9);
    border-radius: 6px;
    height: 7px;
    overflow: hidden;
    margin-bottom: 0.8rem;
}
.conf-fill {
    height: 100%;
    border-radius: 6px;
    background: linear-gradient(90deg, #2563eb, #60a5fa);
}
.description-box {
    background: rgba(8,14,34,0.7);
    border: 1px solid rgba(25,45,90,0.7);
    border-radius: 12px;
    padding: 1rem;
    color: #cbd5e1 !important;
    line-height: 1.75;
    font-size: 0.93rem;
}
.urgent-box {
    background: rgba(220,38,38,0.08);
    border: 1px solid rgba(220,38,38,0.3);
    border-left: 3px solid #dc2626;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    color: #fca5a5 !important;
    font-size: 0.9rem;
    font-weight: 600;
    margin-top: 0.7rem;
}

/* ── Medicine Card ────────────────────────────────────── */
.med-card {
    background: rgba(8,14,34,0.9);
    border: 1px solid rgba(25,45,90,0.65);
    border-left: 3px solid #3b82f6;
    border-radius: 12px;
    padding: 1rem 1.15rem;
    margin: 0.5rem 0;
    transition: all 0.2s ease;
}
.med-card:hover {
    border-color: rgba(59,130,246,0.35);
    border-left-color: #60a5fa;
    background: rgba(12,22,50,0.9);
}
.med-name {
    font-size: 0.98rem;
    font-weight: 700;
    color: #60a5fa !important;
    margin: 0 0 0.35rem;
}
.med-line {
    font-size: 0.83rem;
    color: #475569 !important;
    margin: 0.18rem 0;
}
.med-usage { color: #94a3b8 !important; font-size: 0.88rem; margin: 0.35rem 0; }
.med-caution { color: #fde68a !important; font-size: 0.83rem; margin: 0.3rem 0; }
.pill {
    display: inline-block;
    padding: 0.15rem 0.6rem;
    border-radius: 6px;
    font-size: 0.72rem;
    font-weight: 600;
    margin: 0.25rem 0.2rem 0 0;
}
.pill-rx    { background: rgba(239,68,68,0.1);  color: #fca5a5 !important; border: 1px solid rgba(239,68,68,0.2); }
.pill-otc   { background: rgba(34,197,94,0.1);  color: #86efac !important; border: 1px solid rgba(34,197,94,0.2); }
.pill-price { background: rgba(59,130,246,0.1); color: #93c5fd !important; border: 1px solid rgba(59,130,246,0.2); }

/* ── Care Tips ────────────────────────────────────────── */
.care-item {
    display: flex;
    align-items: flex-start;
    gap: 0.65rem;
    padding: 0.38rem 0;
    font-size: 0.9rem;
    color: #cbd5e1 !important;
    line-height: 1.6;
}
.care-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #22c55e;
    flex-shrink: 0;
    margin-top: 0.5rem;
}

/* ── Doctor Warning ───────────────────────────────────── */
.doctor-box {
    background: rgba(245,158,11,0.05);
    border: 1px solid rgba(245,158,11,0.2);
    border-left: 3px solid #f59e0b;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    color: #fde68a !important;
    font-size: 0.9rem;
    line-height: 1.75;
}
.doctor-box.critical {
    background: rgba(239,68,68,0.05);
    border-color: rgba(239,68,68,0.2);
    border-left-color: #ef4444;
    color: #fca5a5 !important;
}

/* ── Upload Area ──────────────────────────────────────── */
.upload-placeholder {
    text-align: center;
    padding: 3.5rem 1.5rem;
    color: #334155;
}
.upload-icon {
    font-size: 2.8rem;
    margin-bottom: 0.75rem;
    opacity: 0.5;
}
.upload-placeholder h3 {
    color: #475569 !important;
    font-size: 0.95rem;
    margin: 0 0 0.3rem;
    font-weight: 500;
}
.upload-placeholder p {
    color: #334155 !important;
    font-size: 0.82rem;
    margin: 0;
}

/* ── Demo / Error Banner ──────────────────────────────── */
.demo-banner {
    background: rgba(245,158,11,0.06);
    border: 1px solid rgba(245,158,11,0.2);
    border-radius: 10px;
    padding: 0.6rem 1rem;
    color: #fde68a !important;
    font-size: 0.83rem;
}
.error-banner {
    background: rgba(239,68,68,0.06);
    border: 1px solid rgba(239,68,68,0.2);
    border-radius: 10px;
    padding: 0.6rem 1rem;
    color: #fca5a5 !important;
    font-size: 0.83rem;
    margin-top: 0.5rem;
    word-break: break-word;
}

/* ── Disclaimer ───────────────────────────────────────── */
.disclaimer-box {
    background: rgba(239,68,68,0.04);
    border: 1px solid rgba(239,68,68,0.15);
    border-radius: 10px;
    padding: 0.7rem 1rem;
    color: #fca5a5 !important;
    font-size: 0.82rem;
    margin-top: 1.2rem;
    line-height: 1.6;
}

/* ── Status Indicators ────────────────────────────────── */
.status-row { display: flex; flex-direction: column; gap: 0.3rem; margin-top: 0.5rem; }
.status-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.78rem;
    color: #475569 !important;
}
.dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
.dot-green  { background: #22c55e; box-shadow: 0 0 4px #22c55e66; }
.dot-yellow { background: #f59e0b; }
.dot-red    { background: #ef4444; }

/* ── Provider Badge ───────────────────────────────────── */
.provider-indicator {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(59,130,246,0.08);
    border: 1px solid rgba(59,130,246,0.18);
    border-radius: 8px;
    padding: 0.25rem 0.65rem;
    font-size: 0.75rem;
    color: #60a5fa !important;
    margin-bottom: 0.5rem;
}

/* ── Streamlit Overrides ──────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.6rem !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    box-shadow: 0 4px 14px rgba(37,99,235,0.22) !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.01em !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%) !important;
    box-shadow: 0 6px 20px rgba(59,130,246,0.32) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:disabled {
    background: rgba(25,40,80,0.5) !important;
    color: #334155 !important;
    box-shadow: none !important;
    transform: none !important;
}

div[data-testid="stFileUploader"] {
    background: rgba(8,14,34,0.7) !important;
    border: 2px dashed rgba(59,130,246,0.25) !important;
    border-radius: 14px !important;
    transition: border-color 0.2s !important;
}
div[data-testid="stFileUploader"]:hover {
    border-color: rgba(59,130,246,0.5) !important;
}
div[data-testid="stFileUploader"] label { color: #475569 !important; font-size: 0.85rem !important; }

.stTabs [data-baseweb="tab-list"] {
    background: rgba(8,14,34,0.85) !important;
    border-radius: 12px !important;
    padding: 0.22rem !important;
    border: 1px solid rgba(25,45,90,0.7) !important;
    gap: 0.15rem !important;
}
.stTabs [data-baseweb="tab"] {
    color: #475569 !important;
    border-radius: 9px !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    padding: 0.45rem 1.1rem !important;
    transition: all 0.15s !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%) !important;
    color: #fff !important;
    box-shadow: 0 2px 8px rgba(37,99,235,0.3) !important;
}

textarea, input[type="text"], input[type="password"] {
    background: rgba(8,14,34,0.92) !important;
    color: #e2e8f0 !important;
    border: 1px solid rgba(25,45,90,0.8) !important;
    border-radius: 10px !important;
    font-size: 0.9rem !important;
}
textarea:focus, input:focus {
    border-color: rgba(59,130,246,0.45) !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.08) !important;
}

div[data-testid="stSelectbox"] > div > div {
    background: rgba(8,14,34,0.92) !important;
    border: 1px solid rgba(25,45,90,0.8) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}
div[data-testid="stExpander"] {
    background: rgba(8,14,34,0.8) !important;
    border: 1px solid rgba(25,45,90,0.6) !important;
    border-radius: 12px !important;
}

[data-testid="stMetricValue"] { color: #60a5fa !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: #475569 !important; }

/* Chat messages styling */
[data-testid="stChatMessage"] {
    background: rgba(8,14,34,0.6) !important;
    border: 1px solid rgba(25,45,90,0.5) !important;
    border-radius: 14px !important;
    padding: 0.8rem 1rem !important;
    margin: 0.3rem 0 !important;
}
[data-testid="stChatInput"] textarea {
    background: rgba(8,14,34,0.95) !important;
    border: 1px solid rgba(25,45,90,0.8) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: rgba(8,14,34,0.5); }
::-webkit-scrollbar-thumb { background: rgba(59,130,246,0.25); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(59,130,246,0.45); }

/* Sidebar labels */
[data-testid="stSidebar"] label { color: #475569 !important; font-size: 0.82rem !important; }
[data-testid="stSidebar"] small { color: #374151 !important; }
[data-testid="stSidebar"] .stMarkdown p { color: #475569 !important; font-size: 0.82rem; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def _image_to_bytes(image_pil: Image.Image, quality: int = 88) -> bytes:
    buf = BytesIO()
    image_pil.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _image_to_b64(image_pil: Image.Image) -> str:
    return base64.b64encode(_image_to_bytes(image_pil)).decode("utf-8")


def _parse_json_response(text: str) -> dict:
    """Robustly extract JSON from an AI response string.
    Returns a diagnostic dict (never raises) so the caller always gets a real result."""
    original = text
    text = text.strip()
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to find a JSON object anywhere in the text
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    # Return a structured error result instead of raising — caller will surface this clearly
    return {
        "_parse_error": True,
        "_raw_response": original[:400],
        "condition_en": "Response Parse Error",
        "condition_local": "Response Parse Error",
        "condition_key": "unknown",
        "severity": "Low",
        "confidence_percent": 0,
        "affected_area": "",
        "visual_features": "",
        "description": f"AI returned a non-JSON response. Raw: {original[:200]}",
        "urgent": False,
    }


def get_env_key(provider_key: str) -> str:
    """Read API key from environment variable (HF Spaces secrets, .env, etc.)."""
    return os.environ.get(provider_key, "")


def _demo_vision_result(error: str = "") -> dict:
    return {
        "condition_en": "Acne Vulgaris (Demo Mode)",
        "condition_local": "Acne Vulgaris (Demo Mode)",
        "condition_key": "acne",
        "severity": "Low",
        "confidence_percent": 72,
        "affected_area": "Face (cheeks and forehead)",
        "visual_features": "Papules, comedones, mild erythema",
        "description": "Demo mode — enter a valid API key for real AI analysis. This shows sample output for acne vulgaris.",
        "urgent": False,
        "_demo": True,
        "_error": error,
    }


def _make_vision_prompt(language: str) -> str:
    lang_instr = (
        "Respond ONLY in Urdu (اردو) language." if language == "ur"
        else "Respond ONLY in English."
    )
    local_lang = "Urdu" if language == "ur" else "English"
    return f"""You are a board-certified AI dermatology assistant. Carefully analyze this skin image.

{lang_instr}

Return ONLY a valid JSON object — no markdown fences, no preamble, no extra text:
{{
  "condition_en": "Primary skin condition name in English",
  "condition_local": "Condition name in {local_lang}",
  "condition_key": "one of: acne|eczema|psoriasis|fungal_infection|scabies|vitiligo|contact_dermatitis|urticaria|rosacea|melanoma|unknown",
  "severity": "Low|Medium|High|Critical",
  "confidence_percent": 75,
  "affected_area": "Brief description of the affected body area",
  "visual_features": "2-3 key visual features you observed",
  "description": "3-4 sentence clinical description in {local_lang}",
  "urgent": false
}}

Rules:
- If the image is unclear, not skin-related, or you are unsure — set condition_key to "unknown"
- Be conservative: lower confidence scores when uncertain
- Set urgent=true ONLY for melanoma or clear signs of skin cancer
- Output ONLY the JSON object, nothing else."""


# ═══════════════════════════════════════════════════════════════════
# AGENT 1 — VISION AGENT (Multi-Provider)
# ═══════════════════════════════════════════════════════════════════

def _call_gemini_vision(image_pil: Image.Image, prompt: str, model_id: str, api_key: str) -> str:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_id,
        generation_config=genai.types.GenerationConfig(
            temperature=0.2,
            max_output_tokens=1024,
        ),
    )
    img_bytes = _image_to_bytes(image_pil)
    response = model.generate_content(
        [prompt, {"mime_type": "image/jpeg", "data": img_bytes}]
    )
    return response.text


def _call_openai_vision(
    image_pil: Image.Image, prompt: str, model_id: str, api_key: str, base_url: str
) -> str:
    client = _OpenAIClient(api_key=api_key, base_url=base_url)
    img_b64 = _image_to_b64(image_pil)
    resp = client.chat.completions.create(
        model=model_id,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                {"type": "text", "text": prompt},
            ],
        }],
        max_tokens=1024,
        temperature=0.2,
    )
    return resp.choices[0].message.content


def vision_agent(
    image_pil: Image.Image,
    provider: str,
    model_id: str,
    api_key: str,
    language: str,
) -> dict:
    if not api_key:
        return _demo_vision_result()

    cfg = PROVIDERS.get(provider, {})

    # Verify vision support for selected model
    model_info = next((m for m in cfg.get("models", []) if m["id"] == model_id), None)
    if model_info and not model_info.get("vision", False):
        return _demo_vision_result(
            error=f"Model '{model_id}' does not support vision/images. "
                  f"Please select a vision-capable model from the sidebar."
        )

    prompt = _make_vision_prompt(language)
    raw = ""
    try:
        if provider == "gemini":
            if not GEMINI_OK:
                return _demo_vision_result(error="google-generativeai is not installed.")
            raw = _call_gemini_vision(image_pil, prompt, model_id, api_key)
        elif provider in ("groq", "qwen"):
            if not OPENAI_OK:
                return _demo_vision_result(error="openai package is not installed.")
            raw = _call_openai_vision(image_pil, prompt, model_id, api_key, cfg["base_url"])
        else:
            return _demo_vision_result(error=f"Unknown provider: {provider}")

        result = _parse_json_response(raw)

        # If parsing produced an error (non-JSON response), wrap it as a demo result
        if result.get("_parse_error"):
            return _demo_vision_result(
                error=f"AI returned non-JSON response. Check your model/key. Raw: {result.get('_raw_response','')[:200]}"
            )

        # Ensure required keys exist
        result.setdefault("condition_key", "unknown")
        result.setdefault("severity", "Low")
        result.setdefault("confidence_percent", 60)
        result.setdefault("urgent", False)
        return result

    except Exception as exc:
        msg = str(exc)
        # Friendly error messages for common HTTP errors
        if "403" in msg or "PERMISSION_DENIED" in msg.upper() or "permission" in msg.lower():
            friendly = (
                "403 Permission Denied — Your API key may be invalid, "
                "expired, or lack required permissions. "
                "Verify the key at the provider's console and re-enter it."
            )
        elif "429" in msg or "RESOURCE_EXHAUSTED" in msg.upper() or "quota" in msg.lower():
            friendly = "429 Rate Limit — Too many requests. Wait a moment and try again."
        elif "404" in msg or "NOT_FOUND" in msg.upper():
            friendly = (
                f"404 Model Not Found — '{model_id}' may not be available in your region "
                "or for your plan. Try a different model."
            )
        elif "401" in msg or "UNAUTHENTICATED" in msg.upper():
            friendly = "401 Unauthenticated — API key is missing or invalid."
        elif "timeout" in msg.lower() or "timed out" in msg.lower():
            friendly = "Request timed out. The model server may be busy — please try again."
        else:
            friendly = msg
        return _demo_vision_result(error=friendly)


# ═══════════════════════════════════════════════════════════════════
# AGENT 2 — RAG AGENT (FAISS + Sentence Transformers)
# ═══════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading knowledge base (first run only)...")
def build_rag_index():
    if not RAG_OK:
        return None, None, None
    try:
        encoder = SentenceTransformer("all-MiniLM-L6-v2")
        docs = get_all_documents()
        texts = [d["text"] for d in docs]
        embs = encoder.encode(texts, convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(embs)
        idx = faiss.IndexFlatIP(embs.shape[1])
        idx.add(embs)
        return encoder, idx, docs
    except Exception:
        return None, None, None


def rag_agent(query: str, condition_key: str, encoder, index, docs) -> str:
    if encoder is None:
        info = SKIN_KNOWLEDGE.get(condition_key, {})
        if info:
            return (
                f"{info.get('description_en', '')}\n\n"
                f"Causes: {'; '.join(info.get('causes_en', [])[:3])}\n"
                f"Symptoms: {'; '.join(info.get('symptoms_en', [])[:4])}"
            )
        return ""
    try:
        q = f"{condition_key} {query}"
        q_emb = encoder.encode([q], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_emb)
        _, I = index.search(q_emb, k=3)
        return "\n\n".join(docs[i]["text"][:600] for i in I[0] if i < len(docs))
    except Exception:
        return ""


# ═══════════════════════════════════════════════════════════════════
# AGENT 3 — MEDICINE AGENT
# ═══════════════════════════════════════════════════════════════════

def medicine_agent(condition_key: str, language: str):
    return get_medicines(condition_key), get_care_tips(condition_key)


# ═══════════════════════════════════════════════════════════════════
# AGENT 4 — GRAD-CAM / ATTENTION HEATMAP AGENT
# ═══════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_cnn():
    """Load EfficientNet-B0 for Grad-CAM.
    Skipped on HF Spaces (slow CPU download causes 5-min freeze).
    The fast numpy fallback heatmap is used instead."""
    if IS_HF_SPACES:
        # HF free CPU: downloading + running EfficientNet causes ~5 min freeze.
        # The gradient/saturation fallback is used instead — it is instant and meaningful.
        return None
    if not TORCH_OK:
        return None
    try:
        model = tvm.efficientnet_b0(weights="IMAGENET1K_V1")
        model.eval()
        return model
    except Exception:
        return None


def _fallback_heatmap(image_pil: Image.Image) -> Image.Image:
    """Generate a gradient-based attention heatmap without Grad-CAM."""
    img = image_pil.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0

    gray = np.mean(arr, axis=2)
    gx = np.gradient(gray, axis=1)
    gy = np.gradient(gray, axis=0)
    edges = np.sqrt(gx**2 + gy**2)
    edges /= edges.max() + 1e-8

    max_c = arr.max(axis=2)
    min_c = arr.min(axis=2)
    saturation = np.where(max_c > 0, (max_c - min_c) / (max_c + 1e-8), 0.0)

    attention = 0.55 * edges + 0.45 * saturation
    attn_img = Image.fromarray((attention * 255).astype(np.uint8))
    attn_img = attn_img.filter(ImageFilter.GaussianBlur(radius=9))
    attention = np.array(attn_img, dtype=np.float32) / 255.0
    attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)

    # Jet-like colormap: blue→green→yellow→red
    r = np.clip(1.5 * attention - 0.25, 0, 1)
    g = np.clip(1.5 * attention - 0.5,  0, 1) * np.clip(2.0 - 1.5 * attention, 0, 1)
    b = np.clip(0.75 - 1.5 * attention, 0, 1)
    heatmap = np.stack([r, g, b], axis=2)

    overlay = np.clip(0.58 * arr + 0.42 * heatmap, 0, 1)
    return Image.fromarray((overlay * 255).astype(np.uint8))


def gradcam_agent(image_pil: Image.Image, cnn_model) -> Optional[Image.Image]:
    if not TORCH_OK or not GRADCAM_OK or cnn_model is None:
        return _fallback_heatmap(image_pil)
    try:
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        tensor = transform(image_pil).unsqueeze(0)
        with GradCAM(model=cnn_model, target_layers=[cnn_model.features[-1]]) as cam:
            grayscale = cam(
                input_tensor=tensor,
                targets=[ClassifierOutputTarget(0)],
            )[0]
        rgb = np.array(image_pil.convert("RGB").resize((224, 224)), dtype=np.float32) / 255.0
        vis = show_cam_on_image(rgb, grayscale, use_rgb=True)
        return Image.fromarray(vis)
    except Exception:
        return _fallback_heatmap(image_pil)


# ═══════════════════════════════════════════════════════════════════
# AGENT 5 — CHAT AGENT (Multi-Provider)
# ═══════════════════════════════════════════════════════════════════

def _build_system_prompt(
    analysis_context: Optional[dict],
    rag_context: str,
    language: str,
) -> str:
    lang_instr = (
        "Always respond in Urdu (اردو) language."
        if language == "ur"
        else "Always respond in English."
    )
    ctx_line = (
        f"Detected condition: {analysis_context.get('condition_en', 'Unknown')}, "
        f"Severity: {analysis_context.get('severity', 'Unknown')}"
        if analysis_context
        else "No skin image has been analysed yet."
    )
    return f"""You are DermaAI Pakistan — a compassionate, knowledgeable AI dermatology assistant designed for Pakistani patients.
{lang_instr}

Guidelines:
- You are NOT a replacement for a real doctor. Always recommend consulting a dermatologist for serious conditions.
- Provide accurate, evidence-based dermatology information.
- Mention Pakistan-available medicines (brand names) when relevant.
- Be warm, empathetic, and clear. Avoid overly technical jargon.
- For serious conditions (melanoma, spreading infection, anaphylaxis), urge IMMEDIATE specialist consultation.
- Do NOT diagnose definitively — provide information and guidance.
- Keep responses concise and practical.

Current analysis context:
{ctx_line}

Relevant medical knowledge:
{rag_context[:900] if rag_context else "No specific knowledge retrieved."}"""


def _chat_gemini(
    user_msg: str,
    history: list,
    system_prompt: str,
    model_id: str,
    api_key: str,
) -> str:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_id,
        system_instruction=system_prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.65,
            max_output_tokens=1024,
        ),
    )
    gemini_history = [
        {"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]}
        for m in history[-12:]
    ]
    chat = model.start_chat(history=gemini_history)
    return chat.send_message(user_msg).text


def _chat_openai(
    user_msg: str,
    history: list,
    system_prompt: str,
    model_id: str,
    api_key: str,
    base_url: str,
) -> str:
    client = _OpenAIClient(api_key=api_key, base_url=base_url)
    messages = [{"role": "system", "content": system_prompt}]
    for m in history[-12:]:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_msg})
    resp = client.chat.completions.create(
        model=model_id,
        messages=messages,
        max_tokens=1024,
        temperature=0.65,
    )
    return resp.choices[0].message.content


def chat_agent(
    user_msg: str,
    history: list,
    analysis_context: Optional[dict],
    provider: str,
    model_id: str,
    api_key: str,
    language: str,
    encoder,
    index,
    docs,
) -> str:
    if not api_key:
        return (
            "Please enter an API key in the sidebar to use the chat feature."
            if language == "en"
            else "براہ کرم سائیڈبار میں API کلید درج کریں۔"
        )

    cfg = PROVIDERS.get(provider, {})
    condition_key = (analysis_context or {}).get("condition_key", "unknown")
    rag_ctx = rag_agent(user_msg, condition_key, encoder, index, docs)
    system_prompt = _build_system_prompt(analysis_context, rag_ctx, language)

    # For vision models used in chat, prefer the text variant
    chat_model = model_id
    if provider in ("groq", "qwen"):
        model_info = next((m for m in cfg.get("models", []) if m["id"] == model_id), None)
        if model_info and model_info.get("vision", False):
            text_models = [m["id"] for m in cfg.get("models", []) if not m.get("vision")]
            if text_models:
                chat_model = text_models[0]

    try:
        if provider == "gemini":
            if not GEMINI_OK:
                return "google-generativeai is not installed."
            return _chat_gemini(user_msg, history, system_prompt, chat_model, api_key)
        elif provider in ("groq", "qwen"):
            if not OPENAI_OK:
                return "openai package is not installed."
            return _chat_openai(user_msg, history, system_prompt, chat_model, api_key, cfg["base_url"])
        return "Unsupported provider."
    except Exception as exc:
        msg = str(exc)
        if "403" in msg or "permission" in msg.lower():
            return f"API key error (403): {msg}"
        if "429" in msg or "quota" in msg.lower():
            return "Rate limit reached. Please wait a moment and try again."
        return f"Chat error: {msg}"


# ═══════════════════════════════════════════════════════════════════
# UI COMPONENTS
# ═══════════════════════════════════════════════════════════════════

def render_medicine_card(med: dict, lang: str, ui: dict):
    rx_pill = (
        f'<span class="pill pill-rx">⚕ {ui["rx"]}</span>'
        if med.get("rx_required")
        else f'<span class="pill pill-otc">✓ {ui["otc"]}</span>'
    )
    brands = ", ".join(med.get("brands_pk", []))
    usage = med.get(f"usage_{'ur' if lang == 'ur' else 'en'}", "")
    caution = med.get(f"caution_{'ur' if lang == 'ur' else 'en'}", "")
    is_urdu = lang == "ur"
    urdu_cls = ' urdu' if is_urdu else ''

    st.markdown(f"""
<div class="med-card">
  <p class="med-name{urdu_cls}">{med.get('generic', '')}</p>
  <p class="med-line"><b style="color:#334155">{ui['brands']}:</b> {brands}</p>
  <p class="med-line"><b style="color:#334155">Type:</b> {med.get('type', '')}</p>
  <p class="med-usage{urdu_cls}"><b>{ui['usage']}:</b> {usage}</p>
  <p class="med-caution{urdu_cls}">⚠ <b>{ui['caution']}:</b> {caution}</p>
  <span class="pill pill-price">PKR {med.get('price_pkr', 'N/A')}</span>
  {rx_pill}
</div>""", unsafe_allow_html=True)


def render_care_tips(tips: list, lang: str):
    for tip in tips:
        is_rtl = any(ord(c) > 1536 for c in tip[:15])
        span_cls = ' class="urdu"' if is_rtl else ''
        st.markdown(
            f'<div class="care-item"><div class="care-dot"></div>'
            f'<span{span_cls}>{tip}</span></div>',
            unsafe_allow_html=True,
        )


def render_analysis_results(
    result: dict,
    heatmap_img: Optional[Image.Image],
    image_pil: Image.Image,
    lang: str,
    ui: dict,
):
    condition_key = result.get("condition_key", "unknown")
    severity = result.get("severity", "Low")
    sev_color = SEVERITY_COLORS.get(severity, "#94a3b8")
    confidence = int(result.get("confidence_percent", 70))
    kb_info = SKIN_KNOWLEDGE.get(condition_key, {})
    medicines, care_tips = medicine_agent(condition_key, lang)

    st.markdown(
        f'<h2 style="color:#60a5fa;font-size:1.3rem;margin:0 0 1rem">'
        f'{ui["result_header"]}</h2>',
        unsafe_allow_html=True,
    )

    if result.get("_demo"):
        if result.get("_error"):
            # Big, unmissable error — user must see this
            st.error(
                f"⚠ **Vision API Failed** — Real analysis could not be completed.\n\n"
                f"**Error:** {result['_error']}\n\n"
                f"👉 Check your API key in the sidebar and click **Analyze** again."
            )
            st.warning(
                "The result shown below is a **fixed demo/fallback** (Acne Vulgaris, 72%) "
                "— it does NOT reflect your actual skin image. Fix the error above first."
            )
        else:
            st.info(f"⚡ {ui['demo_mode']} — Enter a valid API key for real analysis.")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        cond_name = result.get("condition_local") or result.get("condition_en", "Unknown")
        is_urdu_text = lang == "ur"
        urdu_cls = ' urdu' if is_urdu_text else ''

        st.markdown(
            f'<div class="condition-wrapper">'
            f'<p class="condition-name{urdu_cls}">{cond_name}'
            f'<span class="severity-pill" style="background:{sev_color}22;color:{sev_color};border:1px solid {sev_color}44">'
            f'{severity}</span></p></div>',
            unsafe_allow_html=True,
        )

        conf_color = "#22c55e" if confidence >= 80 else "#f59e0b" if confidence >= 60 else "#ef4444"
        st.markdown(
            f'<p class="conf-label">{ui["confidence"]}</p>'
            f'<p class="conf-value" style="color:{conf_color}">{confidence}%</p>'
            f'<div class="conf-track">'
            f'<div class="conf-fill" style="width:{confidence}%;background:linear-gradient(90deg,{conf_color},{conf_color}aa)"></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        area = result.get("affected_area", "")
        if area:
            st.markdown(
                f'<p style="color:#475569;font-size:0.82rem;margin:0 0 0.5rem">'
                f'<b style="color:#334155">{ui["affected_area"]}:</b> {area}</p>',
                unsafe_allow_html=True,
            )

        desc = result.get("description", "")
        if desc:
            st.markdown(
                f'<div class="description-box{urdu_cls}">{desc}</div>',
                unsafe_allow_html=True,
            )

        if result.get("urgent"):
            st.markdown(
                f'<div class="urgent-box">🚨 {ui["urgent_msg"]}</div>',
                unsafe_allow_html=True,
            )

    with col_right:
        st.image(image_pil, caption="Uploaded Image", use_container_width=True)

    # Heatmap expander
    if heatmap_img:
        with st.expander(f"🌡 {ui['heatmap_header']}", expanded=False):
            st.image(heatmap_img, caption=ui["heatmap_caption"], use_container_width=True)
            if not GRADCAM_OK or not TORCH_OK:
                st.markdown(
                    '<p style="color:#334155;font-size:0.78rem;margin-top:0.5rem">'
                    'Note: Grad-CAM unavailable — showing gradient-based attention map (fallback).</p>',
                    unsafe_allow_html=True,
                )

    # Detail tabs
    t_med, t_care, t_doc = st.tabs([
        f"💊 {ui['medicines_tab']}",
        f"💡 {ui['care_tab']}",
        f"👨‍⚕️ {ui['doctor_tab']}",
    ])

    with t_med:
        if medicines:
            for med in medicines:
                render_medicine_card(med, lang, ui)
        else:
            st.info("No specific medicine data available. Please consult a dermatologist.")
        st.markdown(f'<div class="disclaimer-box">⚠ {ui["disclaimer"]}</div>', unsafe_allow_html=True)

    with t_care:
        if care_tips:
            st.markdown('<div class="glass-card"><div class="card-title">Care & Prevention</div>', unsafe_allow_html=True)
            render_care_tips(care_tips, lang)
            st.markdown("</div>", unsafe_allow_html=True)

    with t_doc:
        doc_en = kb_info.get("when_to_see_doctor_en", "Consult a dermatologist for proper evaluation.")
        doc_ur = kb_info.get("when_to_see_doctor_ur", "")
        is_critical = severity == "Critical"
        box_cls = "doctor-box critical" if is_critical else "doctor-box"
        icon = "🚨" if is_critical else "👨‍⚕️"
        st.markdown(
            f'<div class="{box_cls}">{icon} {doc_en if lang == "en" else (doc_ur or doc_en)}</div>',
            unsafe_allow_html=True,
        )
        if lang == "en" and doc_ur:
            st.markdown(
                f'<div class="doctor-box urdu" style="margin-top:0.5rem">{doc_ur}</div>',
                unsafe_allow_html=True,
            )


def render_about():
    st.markdown("""
<div class="glass-card">
<div class="card-title">Project Overview</div>
<p style="color:#94a3b8;line-height:1.8">
DermaAI Pakistan is an advanced AI dermatology assistant with a Multi-Agent RAG architecture,
designed for Pakistani patients with full bilingual English / اردو support.
It supports multiple free AI providers so you are never locked to one service.
</p>
</div>

<div class="glass-card" style="margin-top:0.75rem">
<div class="card-title">Multi-Agent Architecture</div>
<ul style="color:#94a3b8;line-height:2;margin:0;padding-left:1.2rem">
<li><b style="color:#60a5fa">Vision Agent</b> — Gemini / Groq Cloud / Qwen multimodal vision: analyses skin images</li>
<li><b style="color:#60a5fa">RAG Agent</b> — FAISS + Sentence Transformers: retrieves relevant medical knowledge</li>
<li><b style="color:#60a5fa">Medicine Agent</b> — Pakistan DRAP formulary: recommends local brand medicines</li>
<li><b style="color:#60a5fa">Heatmap Agent</b> — Grad-CAM (EfficientNet-B0) + gradient fallback: explains AI focus</li>
<li><b style="color:#60a5fa">Chat Agent</b> — Multi-turn bilingual Q&A with RAG context</li>
</ul>
</div>

<div class="glass-card" style="margin-top:0.75rem">
<div class="card-title">Supported AI Providers</div>
<table style="width:100%;border-collapse:collapse;color:#94a3b8;font-size:0.88rem">
<tr style="border-bottom:1px solid rgba(25,45,90,0.6)">
  <th style="text-align:left;padding:0.4rem;color:#60a5fa">Provider</th>
  <th style="text-align:left;padding:0.4rem;color:#60a5fa">Vision Models</th>
  <th style="text-align:left;padding:0.4rem;color:#60a5fa">Free Tier</th>
</tr>
<tr style="border-bottom:1px solid rgba(25,45,90,0.4)">
  <td style="padding:0.4rem">Google Gemini</td>
  <td style="padding:0.4rem">2.0 Flash, 1.5 Flash, 1.5 Pro…</td>
  <td style="padding:0.4rem">15 req/min (AI Studio)</td>
</tr>
<tr style="border-bottom:1px solid rgba(25,45,90,0.4)">
  <td style="padding:0.4rem">Groq Cloud</td>
  <td style="padding:0.4rem">Llama 4 Scout, Llama 3.2 Vision</td>
  <td style="padding:0.4rem">100% Free (console.groq.com)</td>
</tr>
<tr>
  <td style="padding:0.4rem">Alibaba Qwen</td>
  <td style="padding:0.4rem">Qwen-VL Max, Qwen2.5-VL 72B</td>
  <td style="padding:0.4rem">Free quota (bailian.console.aliyun.com)</td>
</tr>
</table>
</div>

<div class="glass-card" style="margin-top:0.75rem">
<div class="card-title">Skin Conditions Covered</div>
<p style="color:#94a3b8;line-height:1.9;margin:0">
Acne Vulgaris · Atopic Dermatitis (Eczema) · Psoriasis · Fungal Infection (Tinea) ·
Scabies · Vitiligo · Contact Dermatitis · Urticaria (Hives) · Rosacea ·
<span style="color:#fca5a5">Melanoma (Critical)</span>
</p>
</div>

<div class="glass-card" style="margin-top:0.75rem">
<div class="card-title">Technical Stack</div>
<ul style="color:#94a3b8;line-height:2;margin:0;padding-left:1.2rem">
<li>Frontend: <b>Streamlit</b></li>
<li>LLMs: <b>Google Gemini API · Groq Cloud API · Alibaba Qwen API</b></li>
<li>Embeddings: <b>all-MiniLM-L6-v2</b> (Sentence Transformers)</li>
<li>Vector DB: <b>FAISS-CPU</b></li>
<li>XAI: <b>EfficientNet-B0 + pytorch-grad-cam</b> (with gradient fallback)</li>
<li>Deploy: <b>Hugging Face Spaces / Streamlit Cloud / Local</b></li>
</ul>
</div>

<div class="glass-card" style="margin-top:0.75rem;border-color:rgba(239,68,68,0.2)">
<div class="card-title" style="color:#fca5a5">Medical Disclaimer</div>
<p style="color:#fca5a5;line-height:1.75;margin:0;font-size:0.88rem">
DermaAI Pakistan is an <b>educational and informational tool only</b>. It is <b>NOT</b> a substitute
for professional medical advice, diagnosis, or treatment. Always consult a licensed,
board-certified dermatologist. In Pakistan: Aga Khan Hospital (Karachi),
SKMCH (Lahore & Peshawar), PIMS (Islamabad), and PMDC-registered clinics.
</p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════

def main():
    inject_css()

    # ── Session state init ───────────────────────────────────────
    for key, default in [
        ("analysis_result", None),
        ("heatmap_img", None),
        ("original_img", None),
        ("chat_history", []),
        ("_last_file_id", ""),   # tracks which file is currently loaded
        ("_img_bytes", b""),     # raw image bytes — avoids PIL lazy-load issues on rerun
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # ── Sidebar ──────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            '<p style="font-size:1.05rem;font-weight:700;color:#60a5fa;margin:0 0 0.2rem">DermaAI Pakistan</p>'
            '<p style="font-size:0.72rem;color:#334155;margin:0 0 1rem">v2.0 · Multi-Provider AI Dermatology</p>',
            unsafe_allow_html=True,
        )
        st.divider()

        # Language
        lang = st.selectbox(
            "Language / زبان",
            options=["en", "ur"],
            format_func=lambda x: "English" if x == "en" else "اردو",
            key="lang_select",
        )
        ui = UI[lang]

        st.divider()

        # Provider selection
        available_providers = [k for k, v in PROVIDERS.items() if v["available"]]
        if not available_providers:
            st.error("No AI provider libraries found. Check requirements.")
            available_providers = list(PROVIDERS.keys())

        provider_names = {k: PROVIDERS[k]["name"] for k in available_providers}
        provider = st.selectbox(
            ui["provider_label"],
            options=list(provider_names.keys()),
            format_func=lambda x: provider_names[x],
            key="provider_select",
        )
        cfg = PROVIDERS[provider]

        # Model selection (only vision models for analysis)
        model_options = cfg["models"]
        model_ids = [m["id"] for m in model_options]
        model_labels = {m["id"]: m["name"] for m in model_options}

        # Default to first vision model
        default_model = next(
            (m["id"] for m in model_options if m.get("vision")),
            model_ids[0],
        )
        if "model_select" not in st.session_state:
            st.session_state.model_select = default_model

        model_id = st.selectbox(
            ui["model_label"],
            options=model_ids,
            format_func=lambda x: model_labels[x],
            key="model_select",
        )

        # Warn if non-vision model selected for analysis
        selected_model_info = next((m for m in model_options if m["id"] == model_id), None)
        if selected_model_info and not selected_model_info.get("vision"):
            st.caption("⚠ This model does not support vision. Select a vision model for image analysis.")

        st.divider()

        # API Key — try env var first
        env_key = get_env_key(cfg["env_key"])
        if env_key:
            st.markdown(
                f'<div style="background:rgba(34,197,94,0.08);border:1px solid rgba(34,197,94,0.2);'
                f'border-radius:8px;padding:0.4rem 0.7rem;font-size:0.78rem;color:#4ade80">'
                f'✓ {ui["key_env_note"]}</div>',
                unsafe_allow_html=True,
            )
            api_key = env_key
        else:
            api_key = st.text_input(
                ui["api_key_label"],
                type="password",
                placeholder=cfg["key_hint"].split("·")[0].strip(),
                key="api_key_input",
            )
            if not api_key:
                st.caption(f"🔗 {cfg['key_hint']}")

        st.divider()

        # Quick guide
        st.markdown(
            '<p style="font-size:0.8rem;font-weight:600;color:#334155;margin-bottom:0.5rem">Quick Guide</p>',
            unsafe_allow_html=True,
        )
        steps = [
            "Get free API key from provider",
            "Select provider & model",
            "Upload clear skin photo",
            "Click Analyze",
            "Chat for follow-up questions",
        ] if lang == "en" else [
            "API کلید حاصل کریں",
            "فراہم کنندہ اور ماڈل منتخب کریں",
            "جلد کی تصویر اپلوڈ کریں",
            "تجزیہ کریں دبائیں",
            "سوالات کے لیے چیٹ کریں",
        ]
        for i, step in enumerate(steps, 1):
            st.markdown(
                f'<p style="font-size:0.78rem;color:#334155;margin:0.15rem 0">'
                f'<span style="color:#3b82f6;font-weight:600">{i}.</span> {step}</p>',
                unsafe_allow_html=True,
            )

        st.divider()

        # System status
        st.markdown(
            '<p style="font-size:0.78rem;font-weight:600;color:#334155;margin-bottom:0.4rem">System Status</p>',
            unsafe_allow_html=True,
        )
        status_items = [
            (GEMINI_OK,    "Gemini SDK"),
            (OPENAI_OK,    "OpenAI SDK (Groq/Qwen)"),
            (RAG_OK,       "RAG (FAISS + SentenceTransformers)"),
            (TORCH_OK,     "PyTorch (Grad-CAM)"),
            (GRADCAM_OK,   "pytorch-grad-cam"),
        ]
        for ok, name in status_items:
            dot = "dot-green" if ok else "dot-yellow"
            label_color = "#334155" if ok else "#1e293b"
            st.markdown(
                f'<div class="status-item"><div class="dot {dot}"></div>'
                f'<span style="color:{label_color}">{name}</span></div>',
                unsafe_allow_html=True,
            )

    # ── Header ────────────────────────────────────────────────────
    st.markdown(f"""
<div class="derm-header">
  <h1>{ui["title"]}</h1>
  <p class="tagline">{ui["tagline"]}</p>
  <div class="badge-row">
    <span class="hdr-badge">{PROVIDERS[provider]['name']}</span>
    <span class="hdr-badge">{model_labels.get(model_id, model_id)}</span>
    <span class="hdr-badge green">RAG</span>
    <span class="hdr-badge purple">XAI Heatmap</span>
    <span class="hdr-badge">EN / اردو</span>
  </div>
</div>""", unsafe_allow_html=True)

    # Load shared resources (cached)
    encoder, index, docs = build_rag_index()
    cnn_model = load_cnn()

    # ── Main tabs ──────────────────────────────────────────────────
    tab_analysis, tab_chat, tab_about = st.tabs([
        ui["tab_analysis"],
        ui["tab_chat"],
        ui["tab_about"],
    ])

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 1: SKIN ANALYSIS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab_analysis:
        upload_col, result_col = st.columns([1, 2], gap="large")

        with upload_col:
            st.markdown(
                f'<h3 style="color:#60a5fa;font-size:1rem;margin-bottom:0.4rem">{ui["upload_header"]}</h3>'
                f'<p style="color:#334155;font-size:0.8rem;margin-bottom:0.7rem">{ui["upload_hint"]}</p>',
                unsafe_allow_html=True,
            )

            uploaded = st.file_uploader(
                label=" ",
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed",
                key="uploader",
            )

            if uploaded:
                # ── Detect new file: clear stale results immediately ──────
                current_file_id = f"{uploaded.name}_{uploaded.size}"
                if st.session_state["_last_file_id"] != current_file_id:
                    st.session_state["_last_file_id"]    = current_file_id
                    st.session_state["analysis_result"]  = None
                    st.session_state["heatmap_img"]      = None
                    st.session_state["original_img"]     = None
                    # Store raw bytes so analyze can re-open a fresh PIL image
                    raw = uploaded.read()
                    st.session_state["_img_bytes"] = raw
                    uploaded.seek(0)   # reset so Image.open() below works

                image_pil = Image.open(uploaded).convert("RGB")
                w, h = image_pil.size
                max_dim = 900
                if max(w, h) > max_dim:
                    ratio = max_dim / max(w, h)
                    image_pil = image_pil.resize(
                        (int(w * ratio), int(h * ratio)), Image.LANCZOS
                    )
                st.session_state.original_img = image_pil
                st.image(image_pil, use_container_width=True, caption="Ready for analysis")

            st.markdown("<br>", unsafe_allow_html=True)
            analyze_clicked = st.button(
                ui["analyze_btn"],
                disabled=(uploaded is None),
                use_container_width=True,
            )

            if analyze_clicked:
                if not uploaded:
                    st.warning(ui["no_image"])
                elif not api_key:
                    st.warning(ui["no_key"])
                else:
                    # ── Always clear stale results before a new analysis ──
                    st.session_state.analysis_result = None
                    st.session_state.heatmap_img     = None

                    # ── Rebuild PIL image from stored raw bytes (avoids lazy-load stale data) ──
                    img_bytes = st.session_state.get("_img_bytes") or b""
                    if img_bytes:
                        fresh_pil = Image.open(BytesIO(img_bytes)).convert("RGB")
                        fw, fh = fresh_pil.size
                        if max(fw, fh) > 900:
                            r = 900 / max(fw, fh)
                            fresh_pil = fresh_pil.resize(
                                (int(fw * r), int(fh * r)), Image.LANCZOS
                            )
                        st.session_state.original_img = fresh_pil
                    image_pil = st.session_state.original_img

                    # ── Single spinner — no intermediate DOM updates = no flicker ──
                    with st.spinner("🤖 Analyzing skin image — please wait..."):
                        result = vision_agent(
                            image_pil, provider, model_id, api_key, lang,
                        )
                        rag_agent(
                            result.get("condition_en", ""),
                            result.get("condition_key", "unknown"),
                            encoder, index, docs,
                        )
                        medicine_agent(result.get("condition_key", "unknown"), lang)
                        heatmap = gradcam_agent(image_pil, cnn_model)

                    # ── Persist results ───────────────────────────────────
                    st.session_state.analysis_result = result
                    st.session_state.heatmap_img     = heatmap

                    # ── If API failed, show a loud error (no rerun — keep messages visible) ──
                    if result.get("_demo") and result.get("_error"):
                        st.error(
                            f"⚠ **Vision API Error** — {result['_error']}\n\n"
                            "Fix the error above, then click **Analyze** again for a real result."
                        )
                    else:
                        # Success — rerun to render results cleanly in the result column
                        st.rerun()

        with result_col:
            if st.session_state.analysis_result and st.session_state.original_img:
                render_analysis_results(
                    st.session_state.analysis_result,
                    st.session_state.heatmap_img,
                    st.session_state.original_img,
                    lang, ui,
                )
            else:
                st.markdown(
                    f'<div class="glass-card upload-placeholder">'
                    f'<div class="upload-icon">🔬</div>'
                    f'<h3>{ui["no_results"]}</h3>'
                    f'<p>Powered by {PROVIDERS[provider]["name"]} · {model_labels.get(model_id, model_id)}</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 2: CHAT
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab_chat:
        st.markdown(
            f'<h3 style="color:#60a5fa;font-size:1rem;margin-bottom:0.5rem">{ui["chat_header"]}</h3>',
            unsafe_allow_html=True,
        )

        if st.session_state.analysis_result:
            cond = st.session_state.analysis_result.get("condition_en", "your skin condition")
            st.markdown(
                f'<p style="color:#334155;font-size:0.8rem;margin-bottom:0.8rem">'
                f'⚡ {ui["chat_context"]}: <b style="color:#60a5fa">{cond}</b></p>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<p style="color:#334155;font-size:0.8rem;margin-bottom:0.8rem">'
                f'💡 {ui["chat_no_context"]}</p>',
                unsafe_allow_html=True,
            )

        # Render chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input (native Streamlit component — sticky at bottom)
        if user_input := st.chat_input(ui["chat_placeholder"]):
            if not api_key:
                st.warning(ui["no_key"])
            else:
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                with st.chat_message("user"):
                    st.markdown(user_input)
                with st.chat_message("assistant"):
                    with st.spinner(""):
                        ai_reply = chat_agent(
                            user_input,
                            st.session_state.chat_history[:-1],
                            st.session_state.analysis_result,
                            provider, model_id, api_key, lang,
                            encoder, index, docs,
                        )
                    st.markdown(ai_reply)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": ai_reply}
                )

        # Clear chat
        if st.session_state.chat_history:
            if st.button(ui["clear_chat"], key="clear_chat_btn"):
                st.session_state.chat_history = []
                st.rerun()

        st.markdown(
            f'<div class="disclaimer-box">⚠ {ui["disclaimer"]}</div>',
            unsafe_allow_html=True,
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 3: ABOUT
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab_about:
        render_about()


if __name__ == "__main__":
    main()
