import io
import os
import re
import math
import json
import time
import tempfile
from dataclasses import dataclass
from typing import Optional, Tuple

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
st.set_page_config(page_title="Prompt → Graph + GIF (to S3)", page_icon="📈", layout="wide")
st.title("📈 Prompt → Graph  •  🎞️ Prompt → GIF (S3 upload optional)")
st.caption("Describe a function like “plot y = x^2 - 1 from -5 to 5” (PNG) or "
           "“animate y = sin(x - 0.5 t) for x in [-2π,2π], t from 0 to 12” (GIF). "
           "Azure GPT extraction is optional; robust regex fallback is built-in.")


# =========================
# Secrets helpers
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
        "cdn_media": st.secrets.get("CDN_PREFIX_MEDIA"),
        "cdn_html": st.secrets.get("CDN_HTML_BASE"),
    }

AZURE_CFG = read_azure_config()
AWS_CFG = read_aws_config()


# =========================
# Optional Azure client
# =========================
def get_azure_client():
    if AzureOpenAI is None:
        st.info("AzureOpenAI SDK not installed (pip install openai>=1.13.3). Using regex extractor.")
        return None
    if not all([AZURE_CFG.get("api_key"), AZURE_CFG.get("endpoint"),
                AZURE_CFG.get("deployment"), AZURE_CFG.get("version")]):
        st.info("Azure secrets not found. Using regex extractor.")
        return None
    try:
        return AzureOpenAI(
            api_key=AZURE_CFG["api_key"],
            azure_endpoint=AZURE_CFG["endpoint"],
            api_version=AZURE_CFG["version"],
        )
    except Exception as e:
        st.info(f"Azure client init failed; falling back to regex. ({e})")
        return None


# =========================
# Sidebar controls
# =========================
with st.sidebar:
    st.header("Options")
    use_gpt = st.checkbox("Use Azure GPT to extract ranges", value=True)
    st.caption("If secrets/SDK are missing or GPT fails, regex extraction is used.")

    st.markdown("---")
    st.subheader("GIF Options (from prompt)")
    frames = st.number_input("Frames", min_value=20, max_value=1000, value=180, step=10)
    fps = st.number_input("FPS", min_value=5, max_value=60, value=20, step=1)
    points = st.number_input("x points", min_value=200, max_value=5000, value=1000, step=100)
    st.caption("If your equation has no ‘t’, we animate a left→right reveal of the curve.")
    st.markdown("---")
    want_upload = st.checkbox("Upload GIF to S3", value=False)
    st.caption("Requires AWS_* secrets; will return a public URL (or CDN if configured).")


# =========================
# Prompt inputs
# =========================
left, right = st.columns([1, 1])

with left:
    st.subheader("Prompt → Static Graph (PNG)")
    prompt_png = st.text_area("Describe the function to plot (PNG)",
                              value="Plot y = x^2 - 1 from -5 to 5",
                              height=100)
    c1, c2 = st.columns(2)
    with c1:
        override_xmin_png = st.text_input("x_min (optional)")
    with c2:
        override_xmax_png = st.text_input("x_max (optional)")

with right:
    st.subheader("Prompt → Animated GIF")
    prompt_gif = st.text_area("Describe the function to animate (GIF)",
                              value="Animate y = sin(x - 0.5 t) for x in [-2π, 2π], t from 0 to 12",
                              height=100)
    g1, g2 = st.columns(2)
    with g1:
        override_xmin_gif = st.text_input("GIF x_min (optional)")
        override_tmin = st.text_input("t_min (optional)")
    with g2:
        override_xmax_gif = st.text_input("GIF x_max (optional)")
        override_tmax = st.text_input("t_max (optional)")


# =========================
# Extraction helpers
# =========================
@dataclass
class Extraction:
    equation: Optional[str]
    x_min: Optional[float]
    x_max: Optional[float]
    t_min: Optional[float]
    t_max: Optional[float]
    source: str  # "gpt" or "regex"

GPT_SYSTEM = (
    "Extract a graphable equation of y in terms of x, optionally depending on t."
    " Respond as STRICT JSON (no prose) with EXACTLY these keys:"
    ' {"equation":"y = ...","x_min":<number|null>,"x_max":<number|null>,"t_min":<number|null>,"t_max":<number|null>}.'
)
GPT_USER_TEMPLATE = (
    "Prompt:\n{prompt}\n\n"
    "Return exactly:\n"
    '{ "equation": "y = ...", "x_min": -10, "x_max": 10, "t_min": 0, "t_max": 6.28 }\n'
)

_re_num = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"

def _strip_range_phrases_anywhere(s: str) -> str:
    """Remove range/interval phrases anywhere in the expression."""
    # 1) Remove 'for <var> in [a,b]' or '(a,b)'
    s = re.sub(r"\bfor\s*[a-zA-Z]\s*in\s*[\(\[][^\)\]]*[\)\]]", "", s, flags=re.IGNORECASE)

    # 2) Remove '<var> from A to B' or 'from A to B' (no var)
    s = re.sub(r"\b[xt]\s*from\s*[^,;]+?\s*to\s*[^,;]+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\bfrom\s*[^,;]+?\s*to\s*[^,;]+", "", s, flags=re.IGNORECASE)

    # 3) Remove 'between A and B'
    s = re.sub(r"\bbetween\s*[^,;]+?\s*and\s*[^,;]+", "", s, flags=re.IGNORECASE)

    # 4) Remove 'on [a,b]' or '(a,b)'
    s = re.sub(r"\bon\s*[\(\[][^\)\]]*[\)\]]", "", s, flags=re.IGNORECASE)

    # 5) Clean leftover commas/semicolons/extra spaces
    s = re.sub(r"\s*[,;]\s*$", "", s).strip(" ,;")
    return s

def sanitize_equation(eq: str) -> str:
    s = (eq or "").strip()
    if re.match(r"^\s*y\s*=", s, flags=re.IGNORECASE):
        rhs = s.split("=", 1)[1]
    else:
        rhs = s
    rhs = _strip_range_phrases_anywhere(rhs)
    return f"y = {rhs}"

def _parse_num_token(tok: Optional[str]) -> Optional[float]:
    if tok is None: return None
    t = tok.strip().replace("π", "pi")
    try:
        return float(eval(t, {"__builtins__": {}}, {"pi": math.pi, "e": math.e}))
    except Exception:
        return None

def _find_range(text: str, var: str) -> Tuple[Optional[float], Optional[float]]:
    pat_num = rf"{var}\s*(?:from|=)\s*({_re_num})\s*(?:to|,)\s*({_re_num})"
    m = re.search(pat_num, text, flags=re.IGNORECASE)
    if m:
        return float(m.group(1)), float(m.group(2))
    # symbolic tokens (e.g., -2π to 2π)
    m = re.search(rf"{var}\s*(?:from|=)\s*([^\s,;]+)\s*(?:to|,)\s*([^\s,;]+)", text, flags=re.IGNORECASE)
    if m:
        return _parse_num_token(m.group(1)), _parse_num_token(m.group(2))
    # in [a,b]
    m = re.search(rf"{var}\s*in\s*[\[\(]\s*([^\s,;]+)\s*,\s*([^\s,;]+)\s*[\]\)]", text, flags=re.IGNORECASE)
    if m:
        return _parse_num_token(m.group(1)), _parse_num_token(m.group(2))
    return None, None

def extract_with_regex(prompt_text: str) -> Extraction:
    text = prompt_text.strip()

    # Equation (allow y=..., f(x)=..., y(x,t)=...)
    m = re.search(r"y\s*=\s*([^\n;]+)", text, flags=re.IGNORECASE)
    if m:
        eq = f"y = {m.group(1).strip()}"
    else:
        m = re.search(r"y\s*\(\s*x\s*(?:,\s*t\s*)?\)\s*=\s*([^\n;]+)", text, flags=re.IGNORECASE)
        if m:
            eq = f"y = {m.group(1).strip()}"
        else:
            m = re.search(r"f\s*\(\s*x\s*\)\s*=\s*([^\n;]+)", text, flags=re.IGNORECASE)
            if m:
                eq = f"y = {m.group(1).strip()}"
            else:
                # bare expression containing x (and maybe t)
                m = re.search(r"(?<![A-Za-z0-9_])(x[^\n;]+)", text)
                eq = f"y = {m.group(1).strip()}" if m else None

    eq = sanitize_equation(eq) if eq else None

    # x-range
    x_min, x_max = _find_range(text, "x")
    # fallback "from ... to ..." (no var) → treat as x-range
    if x_min is None or x_max is None:
        m = re.search(rf"from\s*({_re_num})\s*to\s*({_re_num})", text, flags=re.IGNORECASE)
        if m:
            x_min, x_max = float(m.group(1)), float(m.group(2))
        else:
            m = re.search(r"from\s*([^\s,;]+)\s*to\s*([^\s,;]+)", text, flags=re.IGNORECASE)
            if m:
                x_min, x_max = _parse_num_token(m.group(1)), _parse_num_token(m.group(2))

    # t-range
    t_min, t_max = _find_range(text, "t")

    return Extraction(eq, x_min, x_max, t_min, t_max, "regex")

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
        raw = resp.choices[0].message.content
        data = json.loads(raw or "{}")

        eq = (data.get("equation") or data.get("Equation") or "").strip()
        if not eq:
            return None
        eq = sanitize_equation(eq)

        def _num(v): 
            return float(v) if v is not None else None

        return Extraction(
            eq,
            _num(data.get("x_min") if "x_min" in data else data.get("xMin")),
            _num(data.get("x_max" ) if "x_max" in data else data.get("xMax")),
            _num(data.get("t_min") if "t_min" in data else data.get("tMin")),
            _num(data.get("t_max") if "t_max" in data else data.get("tMax")),
            "gpt",
        )
    except Exception:
        # Quietly fall back to regex
        return None


# =========================
# Expr → NumPy
# =========================
def to_numpy_expr(equation: str) -> str:
    s = equation.strip()
    if re.match(r"^y\s*=", s, flags=re.I):
        s = s.split("=", 1)[1]

    # Remove any range/interval chatter *anywhere*
    s = _strip_range_phrases_anywhere(s)

    # Normalizations (unicode + math)
    s = (s.replace("π", "pi").replace("·", "*").replace("√", "sqrt")
           .replace("^", "**").replace("ln", "log")
           .replace("²", "**2").replace("³", "**3"))

    # Implicit multiplication fixes
    s = re.sub(r"(?<=\d)\s*(?=x)", "*", s)        # 2x -> 2*x
    s = re.sub(r"(?<=\d)\s*(?=t)", "*", s)        # 2t -> 2*t
    s = re.sub(r"(?<=\d)\s*(?=pi\b)", "*", s)     # 2pi -> 2*pi
    s = re.sub(r"\bpi\s*(?=[xt])", "pi*", s)      # pi x -> pi*x, pi t -> pi*t
    s = re.sub(r"(?<=\d)\s*\(", "*(", s)          # 2(x+1) -> 2*(x+1)

    # Abs shorthand
    s = s.replace("|x|", "abs(x)")
    return s


def eval_expression(expr: str, x: np.ndarray, t: float = 0.0) -> np.ndarray:
    env = {
        "x": x, "t": t,
        "pi": np.pi, "e": np.e,
        "sin": np.sin, "cos": np.cos, "tan": np.tan,
        "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
        "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan,
        "exp": np.exp, "log": np.log, "sqrt": np.sqrt, "abs": np.abs,
    }
    y = eval(expr, {"__builtins__": {}}, env)
    y = np.asarray(y)
    y = np.where(np.isfinite(y), y, np.nan)
    return y


# =========================
# Static PNG
# =========================
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
# Prompt → Animated GIF
# =========================
def _estimate_ylim(expr: str, x: np.ndarray, has_t: bool, t_min: float, t_max: float) -> Tuple[float, float]:
    ys = []
    if has_t:
        for tau in np.linspace(t_min, t_max, 7):
            ys.append(eval_expression(expr, x, t=tau))
    else:
        ys.append(eval_expression(expr, x, t=0.0))
    ycat = np.concatenate(ys)
    ycat = ycat[np.isfinite(ycat)]
    if ycat.size == 0:
        return -1.0, 1.0
    lo, hi = np.nanpercentile(ycat, [5, 95])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = np.min(ycat), np.max(ycat)
    margin = 0.1 * (hi - lo + 1e-9)
    return lo - margin, hi + margin

def build_prompt_gif(equation: str, x_min: float, x_max: float, points: int,
                     frames: int, fps: int, t_min: Optional[float], t_max: Optional[float]) -> bytes:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.animation import PillowWriter

    if x_max <= x_min:
        raise ValueError("x_max must be greater than x_min")

    x = np.linspace(x_min, x_max, int(points))
    expr = to_numpy_expr(equation)
    has_t = "t" in expr

    # Default t-range if equation uses t
    if has_t:
        t_min = -2*np.pi if t_min is None else float(t_min)
        t_max =  2*np.pi if t_max is None else float(t_max)
    else:
        t_min = t_max = 0.0

    # Prepare figure
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    (line,) = ax.plot([], [], lw=2)
    ax.axhline(0, lw=1); ax.axvline(0, lw=1); ax.grid(True)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title(equation)

    # Fix y-limits to avoid jitter
    ylo, yhi = _estimate_ylim(expr, x, has_t, t_min, t_max)
    ax.set_xlim(x_min, x_max); ax.set_ylim(ylo, yhi)

    def init():
        line.set_data([], [])
        return (line,)

    def animate(i):
        if has_t:
            tau = t_min + (t_max - t_min) * (i / (frames - 1))
            y = eval_expression(expr, x, t=tau)
            line.set_data(x, y)
        else:
            n = max(2, int(len(x) * (i + 1) / frames))
            xx = x[:n]
            yy = eval_expression(expr, xx, t=0.0)
            line.set_data(xx, yy)
        return (line,)

    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=int(frames),
                                  interval=1000 // max(1, int(fps)), blit=True)

    # Save to a temporary GIF and return bytes
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
            tmp_path = tmp.name
        ani.save(tmp_path, writer=PillowWriter(fps=int(fps)))
        with open(tmp_path, "rb") as f:
            data = f.read()
    finally:
        try: plt.close(fig)
        except Exception: pass
        if tmp_path and os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except Exception: pass

    return data


# =========================
# S3 upload
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
        ACL="public-read",
        ContentType=content_type,
        CacheControl="public, max-age=31536000",
        ContentDisposition="inline",
    )
    cdn = AWS_CFG.get("cdn_media") or AWS_CFG.get("cdn_html")
    if cdn:
        if not cdn.endswith("/"):
            cdn += "/"
        return cdn + key
    region = AWS_CFG.get("region", "us-east-1")
    return f"https://{bucket}.s3.{region}.amazonaws.com/{key}"


# =========================
# Actions
# =========================
with left:
    if st.button("Create Graph (PNG)", use_container_width=True):
        if not prompt_png.strip():
            st.error("Please enter a prompt.")
            st.stop()

        ext = extract_with_gpt(prompt_png) if use_gpt else None
        if ext is None:
            ext = extract_with_regex(prompt_png)

        if not ext.equation:
            st.error("Couldn't find an equation. Try 'Plot y = sin(x) from -2π to 2π'.")
            st.stop()

        def _try_float(s: Optional[str]) -> Optional[float]:
            if s is None or str(s).strip() == "": return None
            try: return float(str(s).strip())
            except Exception: return None

        x_min = _try_float(override_xmin_png) or ext.x_min or -10.0
        x_max = _try_float(override_xmax_png) or ext.x_max or  10.0

        if (x_max - x_min) > 1e6:
            st.error("x-range too large. Pick a smaller interval.")
            st.stop()

        try:
            png_buf = render_static_plot_png(ext.equation, x_min, x_max, int(points))
            st.success(f"Equation: {ext.equation}  •  x ∈ [{x_min}, {x_max}]  •  Source: {ext.source}")
            st.caption(f"Evaluated: `{to_numpy_expr(ext.equation)}`")
            st.image(png_buf, caption=f"Graph of {ext.equation}", use_container_width=True)
            st.download_button("⬇️ Download PNG", data=png_buf, file_name="graph.png",
                               mime="image/png", use_container_width=True)
        except Exception as e:
            st.exception(e)

with right:
    if st.button("Create GIF (from prompt)", use_container_width=True):
        if not prompt_gif.strip():
            st.error("Please enter a prompt.")
            st.stop()

        ext = extract_with_gpt(prompt_gif) if use_gpt else None
        if ext is None:
            ext = extract_with_regex(prompt_gif)

        if not ext.equation:
            st.error("Couldn't find an equation to animate. Try “animate y = sin(x - 0.5 t) …”.")
            st.stop()

        def _try_float(s: Optional[str]) -> Optional[float]:
            if s is None or str(s).strip() == "": return None
            try: return float(str(s).strip())
            except Exception: return None

        x_min = _try_float(override_xmin_gif) or ext.x_min or -2*np.pi
        x_max = _try_float(override_xmax_gif) or ext.x_max or  2*np.pi
        t_min = _try_float(override_tmin)     or ext.t_min
        t_max = _try_float(override_tmax)     or ext.t_max

        if (x_max - x_min) > 1e6:
            st.error("x-range too large. Pick a smaller interval.")
            st.stop()

        try:
            gif_bytes = build_prompt_gif(ext.equation, float(x_min), float(x_max),
                                         int(points), int(frames), int(fps),
                                         t_min, t_max)
            st.success(f"Equation: {ext.equation}  •  x ∈ [{x_min}, {x_max}]"
                       + (f"  •  t ∈ [{t_min}, {t_max}]" if t_min is not None and t_max is not None else "")
                       + f"  •  Source: {ext.source}")
            st.caption(f"Evaluated: `{to_numpy_expr(ext.equation)}`")
            st.image(gif_bytes, caption="Animated GIF", use_container_width=True)
            st.download_button("⬇️ Download GIF", data=gif_bytes, file_name="graph.gif",
                               mime="image/gif", use_container_width=True)

            if want_upload:
                key = f"{AWS_CFG.get('prefix') or 'media'}/graph_{int(time.time())}.gif"
                with st.spinner(f"Uploading to s3://{AWS_CFG.get('bucket')}/{key}"):
                    url = s3_upload_bytes(gif_bytes, key, content_type="image/gif")
                if url:
                    st.success("Uploaded to S3!")
                    st.code(url)
                    st.link_button("Open GIF", url)
        except Exception as e:
            st.exception(e)


# Footer
st.caption("All credentials are read from `.streamlit/secrets.toml`. Azure is optional; regex fallback is built-in.")
with st.expander("Example secrets.toml"):
    st.code(
        """# Azure OpenAI (optional)
AZURE_API_KEY="YOUR_API_KEY"
AZURE_ENDPOINT="https://YOUR-RESOURCE-NAME.cognitiveservices.azure.com"
AZURE_DEPLOYMENT="gpt-5-chat"
AZURE_API_VERSION="2025-01-01-preview"

# AWS / S3 (optional for uploads)
AWS_ACCESS_KEY="YOUR_AWS_ACCESS_KEY_ID"
AWS_SECRET_KEY="YOUR_AWS_SECRET_ACCESS_KEY"
AWS_REGION="ap-south-1"
AWS_BUCKET="suvichaarapp"
S3_PREFIX="media"

# Optional CDN
CDN_PREFIX_MEDIA="https://media.suvichaar.org/"
""",
        language="toml",
    )
