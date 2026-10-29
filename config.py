"""
Configuration, constants, theme tokens, and metadata for InfraBudget AI.
"""

APP_TITLE = "InfraBudget AI"
APP_SUBTITLE = "Data-Driven Ward-Wise Fund & Resource Allocation Command Center"
MUNICIPALITY_NAME = "Metro City Municipal Corporation (MCMC)"
VERSION = "v2.4 - GovTech Enterprise Edition"

# Color Palette Tokens - Modern GovTech Dark/Light Harmonized Palette
COLORS = {
    "primary": "#2563EB",         # Royal Tech Blue
    "primary_light": "#3B82F6",
    "primary_dark": "#1D4ED8",
    "accent": "#06B6D4",          # Cyan highlight
    "critical": "#EF4444",        # Soft Red
    "critical_bg": "#FEF2F2",
    "critical_border": "#FCA5A5",
    "moderate": "#F59E0B",        # Amber
    "moderate_bg": "#FFFBEB",
    "moderate_border": "#FCD34D",
    "stable": "#10B981",          # Emerald Green
    "stable_bg": "#ECFDF5",
    "stable_border": "#6EE7B7",
    "card_bg": "#FFFFFF",
    "card_border": "#E2E8F0",
    "text_main": "#0F172A",
    "text_muted": "#64748B",
    "bg_light": "#F8FAFC"
}

SECTORS = ["Roads & Pavements", "Water Supply & Sewage", "Sanitation & Waste", "Greenery & Lighting"]

DEFAULT_TOTAL_BUDGET_LAKHS = 3200.0  # ₹32 Crore default pool
MIN_WARD_BUDGET_LAKHS = 40.0
MAX_WARD_BUDGET_LAKHS = 450.0

MOCK_BLOCKCHAIN_GENESIS = "000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f"
