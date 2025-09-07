# streamlit_graph_from_prompt.py
# -----------------------------------------------
# Streamlit app: Prompt → Graph (PNG)
# - Optional Azure OpenAI extraction of equation and range
# - Regex fallback if Azure config/SDK unavailable
# - Safe NumPy eval environment
# -----------------------------------------------

import io
import json
import math
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import streamlit as st

# Optional Azure OpenAI SDK (v1+)
try:
    from openai import AzureOpenAI
except Exception:
    AzureOpenAI = None


# =========================
# Page UI
# =========================
st.set_page_config(page_title="Prompt → Graph (Azure optional)", page_icon="📈", layout="centered")
st.title("📈 Prompt → Graph")
st.caption("Type a math request (e.g., “Plot y = sin(x) from −2π to 2π”). "
           "Optionally use Azure OpenAI to extract the equation and range.")


# =========================
# Azure Config (secrets only)
# =========================
def read_azure_config() -> dict:
    """Read Azure creds strictly from st.secrets. Do not hard-code keys."""
    return {
        "api_key": st.secrets.get("AZURE_API_KEY"),
        "endpoint": st.secrets.get("AZURE_ENDPOINT"),
        "deployment": st.secrets.get("AZURE_DEPLOYMENT"),
        "version": st.secrets.get("AZURE_API_VERSION"),
    }


AZURE_CFG = read_azure_config()


def get_azure_client():
    """Create Azure client only if SDK and secrets are present."""
    if AzureOpenAI is None:
        st.info("AzureOpenAI SDK not installed (pip install openai>=1.13.3). Falling back to regex extractor.")
        return None
    if not all([AZURE_CFG.get("api_key"), AZURE_CFG.get("endpoint"), AZURE_CFG.get("deployment"), AZURE_CFG.get("version")]):
        st.info("Azure config not found in .streamlit/secrets.toml. Using regex extractor.")
        return None
    try:
        return AzureOpenAI(
            api_key=AZURE_CFG["api_key"],
            azure_endpoint=AZURE_CFG["endpoint"],
            api_version=AZURE_CFG["version"],
        )
    except Exception as e:
        st.warning(f"Azure client init failed: {e}")
        return None


# =========================
# Sidebar Controls
# =========================
with st.sidebar:
    st.subheader("Extraction options")
    use_gpt = st.checkbox("Use Azure GPT to extract equation/range", value=True)
    st.caption("If Azure secrets/SDK are missing, the app will use regex extraction.")

prompt = st.text_area(
    "Describe the function to plot",
    value="Plot y = x^2 - 1 from -5 to 5",
    height=120,
)

col1, col2, col3 = st.columns(3)
with col1:
    override_xmin = st.text_input("x_min (optional)")
with col2:
    override_xmax = st.text_input("x_max (optional)")
with col3:
    n_points = st.number_input("points", min_value=200, max_value=5000, value=1000, step=100)


# =========================
# Extraction helpers
# =========================
@dataclass
class Extraction:
    equation: Optional[str]
    x_min: Optional[float]
    x_max: Optional[float]
    source: str  # "gpt" or "regex"


GPT_SYSTEM = (
    "You extract graphable equations from short prompts. "
    "Return STRICT JSON (no prose) with keys: equation (string, like 'y = ...'), "
    "x_min (number or null), x_max (number or null). The equation must be y as a function of x."
)
GPT_USER_TEMPLATE = (
    "Prompt:\n{prompt}\n\n"
    "Return JSON like:\n"
    "{\n  \"equation\": \"y = ...\",\n  \"x_min\": -10,\n  \"x_max\": 10\n}\n"
)

# Strip trailing “from … to … / between … and … / for x in …” phrases
def _strip_trailing_range_phrases(s: str) -> str:
    s = re.sub(r"\bfrom\b.+$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\bbetween\b.+$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\bfor\s*x\s*in\b.+$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\bon\s*\[.*?\]|\bon\s*\(.*?\)", "", s, flags=re.IGNORECASE)
    return s.strip(" ;,")

def sanitize_equation(eq: str) -> str:
    """Ensure 'y = <RHS>' and strip trailing phrases."""
    s = (eq or "").strip()
    if re.match(r"^\s*y\s*=", s, flags=re.IGNORECASE):
        rhs = s.split("=", 1)[1]
    else:
        rhs = s
    rhs = _strip_trailing_range_phrases(rhs)
    return f"y = {rhs}"

def _parse_num_token(tok: Optional[str]) -> Optional[float]:
    """Accept numbers like -2*pi, 3*pi/4, 2π, etc."""
    if tok is None:
        return None
    t = tok.strip().replace("π", "pi")
    try:
        return float(eval(t, {"__builtins__": {}}, {"pi": math.pi, "e": math.e}))
    except Exception:
        return None

_re_num = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"

def extract_with_regex(prompt_text: str) -> Extraction:
    text = prompt_text.strip()

    # Equation
    m = re.search(r"y\s*=\s*([^\n;]+)", text, flags=re.IGNORECASE)
    if m:
        eq = f"y = {m.group(1).strip()}"
    else:
        m = re.search(r"f\s*\(\s*x\s*\)\s*=\s*([^\n;]+)", text, flags=re.IGNORECASE)
        if m:
            eq = f"y = {m.group(1).strip()}"
        else:
            m = re.search(r"(?<![A-Za-z0-9_])(x[^\n;]+)", text)
            eq = f"y = {m.group(1).strip()}" if m else None

    eq = sanitize_equation(eq) if eq else None

    # Range (numeric first)
    x_min = x_max = None
    m = re.search(rf"from\s*({_re_num})\s*to\s*({_re_num})", text, flags=re.IGNORECASE)
    if m:
        x_min, x_max = float(m.group(1)), float(m.group(2))
    else:
        m = re.search(rf"between\s*({_re_num})\s*and\s*({_re_num})", text, flags=re.IGNORECASE)
        if m:
            x_min, x_max = float(m.group(1)), float(m.group(2))
        else:
            # Symbolic tokens like -2π to 2π
            m = re.search(r"from\s*([^\s,;]+)\s*to\s*([^\s,;]+)", text, flags=re.IGNORECASE)
            if m:
                x_min = _parse_num_token(m.group(1))
                x_max = _parse_num_token(m.group(2))
            else:
                m = re.search(r"between\s*([^\s,;]+)\s*and\s*([^\s,;]+)", text, flags=re.IGNORECASE)
                if m:
                    x_min = _parse_num_token(m.group(1))
                    x_max = _parse_num_token(m.group(2))

    return Extraction(eq, x_min, x_max, "regex")

def extract_with_gpt(prompt_text: str) -> Optional[Extraction]:
    client = get_azure_client()
    if client is None:
        return None
    deployment = AZURE_CFG.get("deployment")
    if not deployment:
        return None
    try:
        resp = client.chat.completions.create(
            model=deployment,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": GPT_SYSTEM},
                {"role": "user", "content": GPT_USER_TEMPLATE.format(prompt=prompt_text.strip())},
            ],
        )
        data = json.loads(resp.choices[0].message.content)
        eq = sanitize_equation((data.get("equation") or "").strip())
        x_min = data.get("x_min")
        x_max = data.get("x_max")
        return Extraction(
            eq if eq else None,
            float(x_min) if x_min is not None else None,
            float(x_max) if x_max is not None else None,
            "gpt",
        )
    except Exception as e:
        st.warning(f"Azure extraction failed: {e}")
        return None


# =========================
# Expression → NumPy
# =========================
def to_numpy_expr(equation: str) -> str:
    """Convert 'y = ...' to a numpy-friendly Python expression in x."""
    s = equation.strip()
    if re.match(r"^y\s*=", s, flags=re.I):
        s = s.split("=", 1)[1]
    s = _strip_trailing_range_phrases(s)
    # Normalizations (unicode + math)
    s = (s.replace("π", "pi").replace("·", "*").replace("√", "sqrt")
           .replace("^", "**").replace("ln", "log")
           .replace("²", "**2").replace("³", "**3"))
    # implicit multiplication
    s = re.sub(r"(?<=\d)\s*(?=x)", "*", s)   # 2x -> 2*x
    s = re.sub(r"(?<=\d)\s*\(", "*(", s)     # 2(x+1) -> 2*(x+1)
    # Abs shorthand
    s = s.replace("|x|", "abs(x)")
    return s

def eval_expression(expr: str, x: np.ndarray) -> np.ndarray:
    env = {
        "x": x,
        "pi": np.pi,
        "e": np.e,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "sinh": np.sinh,
        "cosh": np.cosh,
        "tanh": np.tanh,
        "arcsin": np.arcsin,
        "arccos": np.arccos,
        "arctan": np.arctan,
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
        "abs": np.abs,
    }
    y = eval(expr, {"__builtins__": {}}, env)
    y = np.asarray(y)
    y = np.where(np.isfinite(y), y, np.nan)
    return y


# =========================
# Plotting
# =========================
def render_plot(equation: str, x_min: float, x_max: float, points: int) -> io.BytesIO:
    import matplotlib.pyplot as plt
    if x_max <= x_min:
        raise ValueError("x_max must be greater than x_min")

    x = np.linspace(x_min, x_max, int(points))
    expr = to_numpy_expr(equation)
    y = eval_expression(expr, x)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    ax.plot(x, y)
    ax.axhline(0, lw=1)
    ax.axvline(0, lw=1)
    ax.grid(True)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(equation)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


# =========================
# Main button
# =========================
if st.button("Create Graph", use_container_width=True):
    if not prompt.strip():
        st.error("Please enter a prompt.")
        st.stop()

    # 1) Try GPT (if enabled), else regex
    ext = extract_with_gpt(prompt) if use_gpt else None
    if ext is None:
        ext = extract_with_regex(prompt)

    if not ext.equation:
        st.error("Couldn't find an equation to plot. Try 'Plot y = sin(x) from -2π to 2π'.")
        st.stop()

    # 2) x-range precedence: overrides > GPT > regex > default
    def _try_float(s: Optional[str]) -> Optional[float]:
        if s is None or str(s).strip() == "":
            return None
        try:
            return float(str(s).strip())
        except Exception:
            return None

    x_min = _try_float(override_xmin) or ext.x_min or -10.0
    x_max = _try_float(override_xmax) or ext.x_max or 10.0

    # 3) Plot
    try:
        png_buf = render_plot(ext.equation, x_min, x_max, int(n_points))
        st.success(f"Equation: {ext.equation}  •  Range: [{x_min}, {x_max}]  •  Source: {ext.source}")
        st.image(png_buf, caption=f"Graph of {ext.equation}", use_container_width=True)
        st.download_button(
            "⬇️ Download PNG",
            data=png_buf,
            file_name="graph.png",
            mime="image/png",
            use_container_width=True,
        )
    except Exception as e:
        st.exception(e)


# Footer / help
st.caption("This app reads Azure credentials solely from .streamlit/secrets.toml — nothing is hard-coded.")
with st.expander("Example .streamlit/secrets.toml"):
    st.code(
        """AZURE_API_KEY="YOUR_API_KEY"
AZURE_ENDPOINT="https://YOUR-RESOURCE-NAME.cognitiveservices.azure.com"
AZURE_DEPLOYMENT="gpt-5-chat"
AZURE_API_VERSION="2025-01-01-preview" """.strip(),
        language="toml",
    )
