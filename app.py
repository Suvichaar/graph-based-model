# streamlit_graph_from_prompt.py
# -----------------------------------------------
# Streamlit app:
# 1) Prompt → Graph (static PNG) with optional Azure GPT extraction
# 2) Heat Conduction (Fourier vs. Non-Fourier) → Animated GIF
#    + Upload GIF to S3 and return a viewable URL
# -----------------------------------------------

import io
import os
import re
import math
import json
import time
import tempfile
from dataclasses import dataclass
from typing import Optional

import numpy as np
import streamlit as st

# Optional Azure OpenAI SDK (v1+)
try:
    from openai import AzureOpenAI
except Exception:
    AzureOpenAI = None

# Optional boto3 for S3
try:
    import boto3
except Exception:
    boto3 = None


# =========================
# Page UI
# =========================
st.set_page_config(page_title="Prompt → Graph + GIF to S3", page_icon="📈", layout="wide")
st.title("📈 Prompt → Graph  •  🎞️ Heat Conduction GIF → S3")
st.caption("Type a math request (e.g., “Plot y = sin(x) from −2π to 2π”). "
           "Optionally use Azure OpenAI to extract the equation and range. "
           "You can also generate the Fourier vs. Non-Fourier heat conduction GIF and upload it to S3.")


# =========================
# Secrets Helpers
# =========================
def read_azure_config() -> dict:
    return {
        "api_key": st.secrets.get("AZURE_API_KEY"),
        "endpoint": st.secrets.get("AZURE_ENDPOINT"),
        "deployment": st.secrets.get("AZURE_DEPLOYMENT"),
        "version": st.secrets.get("AZURE_API_VERSION"),
    }

def read_aws_config() -> dict:
    return {
        "access_key": st.secrets.get("AWS_ACCESS_KEY"),
        "secret_key": st.secrets.get("AWS_SECRET_KEY"),
        "region": st.secrets.get("AWS_REGION", "us-east-1"),
        "bucket": st.secrets.get("AWS_BUCKET"),
        "prefix": (st.secrets.get("S3_PREFIX") or "media").strip("/"),
        "cdn_media": st.secrets.get("CDN_PREFIX_MEDIA"),   # e.g., https://media.example.com/
        "cdn_html": st.secrets.get("CDN_HTML_BASE"),       # fallback CDN base if desired
    }


AZURE_CFG = read_azure_config()
AWS_CFG = read_aws_config()


# =========================
# Azure client (optional)
# =========================
def get_azure_client():
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
    st.header("Static Graph (PNG)")
    use_gpt = st.checkbox("Use Azure GPT to extract equation/range", value=True)
    st.caption("If Azure secrets/SDK are missing, the app will use regex extraction.")

    st.markdown("---")
    st.header("Heat Conduction GIF → S3")
    frames = st.number_input("Frames", min_value=30, max_value=600, value=120, step=10)
    fps = st.number_input("GIF FPS", min_value=5, max_value=60, value=15, step=1)
    interval_ms = st.number_input("Preview interval (ms)", min_value=10, max_value=200, value=100, step=10)
    # Model parameters (optional)
    D = st.number_input("Diffusion width factor (D)", min_value=0.01, max_value=2.0, value=0.2, step=0.01, format="%.2f")
    speed = st.number_input("Wave speed", min_value=0.0, max_value=1.0, value=0.05, step=0.01, format="%.2f")
    wavelength = st.number_input("Cosine wavelength", min_value=0.5, max_value=20.0, value=3.0, step=0.5, format="%.1f")
    sigma2 = st.number_input("Wave packet variance", min_value=0.2, max_value=10.0, value=2.0, step=0.2, format="%.1f")


# =========================
# Static Graph: Extraction helpers
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

def _strip_trailing_range_phrases(s: str) -> str:
    s = re.sub(r"\bfrom\b.+$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\bbetween\b.+$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\bfor\s*x\s*in\b.+$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\bon\s*\[.*?\]|\bon\s*\(.*?\)", "", s, flags=re.IGNORECASE)
    return s.strip(" ;,")

def sanitize_equation(eq: str) -> str:
    s = (eq or "").strip()
    if re.match(r"^\s*y\s*=", s, flags=re.IGNORECASE):
        rhs = s.split("=", 1)[1]
    else:
        rhs = s
    rhs = _strip_trailing_range_phrases(rhs)
    return f"y = {rhs}"

def _parse_num_token(tok: Optional[str]) -> Optional[float]:
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

    x_min = x_max = None
    m = re.search(rf"from\s*({_re_num})\s*to\s*({_re_num})", text, flags=re.IGNORECASE)
    if m:
        x_min, x_max = float(m.group(1)), float(m.group(2))
    else:
        m = re.search(rf"between\s*({_re_num})\s*and\s*({_re_num})", text, flags=re.IGNORECASE)
        if m:
            x_min, x_max = float(m.group(1)), float(m.group(2))
        else:
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

def get_azure_extraction(prompt_text: str) -> Optional[Extraction]:
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
        return Extraction(eq or None,
                          float(x_min) if x_min is not None else None,
                          float(x_max) if x_max is not None else None,
                          "gpt")
    except Exception as e:
        st.warning(f"Azure extraction failed: {e}")
        return None


# =========================
# Static Graph: Expression → NumPy
# =========================
def to_numpy_expr(equation: str) -> str:
    s = equation.strip()
    if re.match(r"^y\s*=", s, flags=re.I):
        s = s.split("=", 1)[1]
    s = _strip_trailing_range_phrases(s)
    s = (s.replace("π", "pi").replace("·", "*").replace("√", "sqrt")
           .replace("^", "**").replace("ln", "log")
           .replace("²", "**2").replace("³", "**3"))
    s = re.sub(r"(?<=\d)\s*(?=x)", "*", s)   # 2x -> 2*x
    s = re.sub(r"(?<=\d)\s*\(", "*(", s)     # 2(x+1) -> 2*(x+1)
    s = s.replace("|x|", "abs(x)")
    return s

def eval_expression(expr: str, x: np.ndarray) -> np.ndarray:
    env = {
        "x": x, "pi": np.pi, "e": np.e,
        "sin": np.sin, "cos": np.cos, "tan": np.tan,
        "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
        "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan,
        "exp": np.exp, "log": np.log, "sqrt": np.sqrt, "abs": np.abs,
    }
    y = eval(expr, {"__builtins__": {}}, env)
    y = np.asarray(y)
    y = np.where(np.isfinite(y), y, np.nan)
    return y

def render_static_plot_png(equation: str, x_min: float, x_max: float, points: int) -> io.BytesIO:
    import matplotlib.pyplot as plt
    if x_max <= x_min:
        raise ValueError("x_max must be greater than x_min")
    x = np.linspace(x_min, x_max, int(points))
    expr = to_numpy_expr(equation)
    y = eval_expression(expr, x)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    ax.plot(x, y)
    ax.axhline(0, lw=1); ax.axvline(0, lw=1); ax.grid(True)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title(equation)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


# =========================
# Heat Conduction GIF
# =========================
def build_heat_conduction_gif(frames=120, fps=15, interval_ms=100, L=10.0, N=400,
                              D=0.2, speed=0.05, wavelength=3.0, sigma2=2.0) -> bytes:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    x = np.linspace(0, L, N)
    k = 2 * np.pi / wavelength

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Heat Conduction: Fourier vs. Non-Fourier")
    (line1,) = ax1.plot([], [], lw=2, color="red", label="Gaussian pulse")
    (line2,) = ax2.plot([], [], lw=2, color="blue", label="Wave packet")

    for ax, title in zip([ax1, ax2], ["Fourier (Diffusion)", "Non-Fourier (Wave-like)"]):
        ax.set_xlim(0, L); ax.set_ylim(-1.0, 1.5)
        ax.set_xlabel("Position"); ax.set_ylabel("Temperature")
        ax.set_title(title); ax.legend(loc="upper right", frameon=False)

    def init():
        line1.set_data(x, np.full_like(x, np.nan))
        line2.set_data(x, np.full_like(x, np.nan))
        return line1, line2

    def animate(t):
        width = 1.0 + D * t
        y1 = np.exp(-((x - L/2) ** 2) / width) / np.sqrt(width)
        center = L/2 + speed * t
        y2 = np.exp(-((x - center) ** 2) / sigma2) * np.cos(k * (x - speed * t))
        line1.set_data(x, y1); line2.set_data(x, y2)
        return line1, line2

    ani = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=int(frames), interval=int(interval_ms), blit=True
    )

    # Save to a temporary file (Pillow writer), then read bytes
    from matplotlib.animation import PillowWriter
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
            tmp_path = tmp.name
        ani.save(tmp_path, writer=PillowWriter(fps=int(fps)))
        with open(tmp_path, "rb") as f:
            data = f.read()
    finally:
        try:
            plt.close(fig)
        except Exception:
            pass
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    return data


# =========================
# S3 Upload
# =========================
def s3_upload_bytes(data: bytes, key: str, content_type: str = "image/gif") -> Optional[str]:
    if boto3 is None:
        st.error("boto3 not installed. Run: pip install boto3")
        return None
    bucket = AWS_CFG.get("bucket")
    if not bucket:
        st.error("AWS_BUCKET is not set in secrets.")
        return None
    session = boto3.Session(
        aws_access_key_id=AWS_CFG.get("access_key"),
        aws_secret_access_key=AWS_CFG.get("secret_key"),
        region_name=AWS_CFG.get("region"),
    )
    s3 = session.client("s3")
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=data,
        ContentType=content_type,
        ACL="public-read",
        CacheControl="max-age=31536000",
    )
    # Prefer CDN if configured
    cdn = AWS_CFG.get("cdn_media") or AWS_CFG.get("cdn_html")
    if cdn:
        if not cdn.endswith("/"):
            cdn += "/"
        return cdn + key
    # Default public S3 URL
    region = AWS_CFG.get("region", "us-east-1")
    return f"https://{bucket}.s3.{region}.amazonaws.com/{key}"


# =========================
# Layout: Two columns
# =========================
left, right = st.columns([1, 1])

# ---- Left: Static Graph (PNG) ----
with left:
    st.subheader("Prompt → Static Graph (PNG)")
    prompt = st.text_area(
        "Describe the function to plot",
        value="Plot y = x^2 - 1 from -5 to 5",
        height=120,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        override_xmin = st.text_input("x_min (optional)")
    with c2:
        override_xmax = st.text_input("x_max (optional)")
    with c3:
        n_points = st.number_input("points", min_value=200, max_value=5000, value=1000, step=100)

    if st.button("Create Graph (PNG)", use_container_width=True):
        if not prompt.strip():
            st.error("Please enter a prompt.")
            st.stop()

        ext = get_azure_extraction(prompt) if use_gpt else None
        if ext is None:
            ext = extract_with_regex(prompt)

        if not ext.equation:
            st.error("Couldn't find an equation to plot. Try 'Plot y = sin(x) from -2π to 2π'.")
            st.stop()

        def _try_float(s: Optional[str]) -> Optional[float]:
            if s is None or str(s).strip() == "":
                return None
            try:
                return float(str(s).strip())
            except Exception:
                return None

        x_min = _try_float(override_xmin) or ext.x_min or -10.0
        x_max = _try_float(override_xmax) or ext.x_max or 10.0

        try:
            png_buf = render_static_plot_png(ext.equation, x_min, x_max, int(n_points))
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

# ---- Right: Heat Conduction GIF + S3 ----
with right:
    st.subheader("Heat Conduction: GIF → S3")
    st.caption("Generates the same animation as your Colab (Fourier vs. Non-Fourier) and uploads to S3.")

    if st.button("Create GIF & Upload to S3", use_container_width=True):
        with st.spinner("Rendering GIF..."):
            gif_bytes = build_heat_conduction_gif(
                frames=int(frames),
                fps=int(fps),
                interval_ms=int(interval_ms),
                D=float(D),
                speed=float(speed),
                wavelength=float(wavelength),
                sigma2=float(sigma2),
            )

        st.image(gif_bytes, caption="Heat Conduction Comparison (GIF)", use_container_width=True)
        st.download_button(
            "⬇️ Download GIF",
            data=gif_bytes,
            file_name="heat_conduction_comparison.gif",
            mime="image/gif",
            use_container_width=True,
        )

        # Upload to S3
        prefix = AWS_CFG.get("prefix") or "media"
        key = f"{prefix}/heat_conduction_{int(time.time())}.gif"
        with st.spinner(f"Uploading to s3://{AWS_CFG.get('bucket')}/{key}"):
            try:
                url = s3_upload_bytes(gif_bytes, key, content_type="image/gif")
                if url:
                    st.success("Uploaded to S3!")
                    st.write("Public URL:")
                    st.code(url)
            except Exception as e:
                st.exception(e)


# Footer / help
st.caption("This app reads all credentials from `.streamlit/secrets.toml` — nothing is hard-coded.")
with st.expander("Example .streamlit/secrets.toml"):
    st.code(
        """# Azure OpenAI (optional)
AZURE_API_KEY="YOUR_API_KEY"
AZURE_ENDPOINT="https://YOUR-RESOURCE-NAME.cognitiveservices.azure.com"
AZURE_DEPLOYMENT="gpt-5-chat"
AZURE_API_VERSION="2025-01-01-preview"

# AWS / S3
AWS_ACCESS_KEY="YOUR_AWS_ACCESS_KEY_ID"
AWS_SECRET_KEY="YOUR_AWS_SECRET_ACCESS_KEY"
AWS_REGION="ap-south-1"
AWS_BUCKET="suvichaarapp"
S3_PREFIX="media"

# CDN (optional)
CDN_PREFIX_MEDIA="https://media.suvichaar.org/"
# or
CDN_HTML_BASE="https://stories.suvichaar.org/"
""",
        language="toml",
    )
