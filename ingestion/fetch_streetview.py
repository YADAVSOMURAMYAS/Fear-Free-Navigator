"""
ingestion/fetch_streetview.py
==============================
Fetches street-level images for Bengaluru road segments using:
  1. Google Street View Static API  (primary — has BLR coverage)
  2. KartaView API                  (secondary — free, crowdsourced)
  3. Synthetic image features       (fallback — when neither has coverage)

Category 1  – Brightness score from actual street images (nighttime proxy)
Category 6  – Visual safety features via CLIP:
                  visual clutter, greenery, abandonment, narrowness

Google Street View Static API:
  Free tier: 25,000 requests/month
  Register : https://console.cloud.google.com/
  Enable   : "Street View Static API"
  Cost     : $0 for first 25k/month (enough for this project)

Run:
    python -m ingestion.fetch_streetview

Output files in data/raw/:
    bengaluru_streetview_images/     ← downloaded JPG images per segment
    bengaluru_streetview_meta.csv    ← which segments got real images vs fallback
"""

import os
import time
import logging
import hashlib
from pathlib import Path
from io import BytesIO

import requests
import pandas as pd
import numpy as np
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
log = logging.getLogger("fetch_streetview")

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_RAW   = Path("data/raw")
IMG_DIR    = DATA_RAW / "bengaluru_streetview_images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# ── API keys ───────────────────────────────────────────────────────────────────
GOOGLE_API_KEY  = os.getenv("GOOGLE_MAPS_API_KEY", "")
KARTAVIEW_TOKEN = os.getenv("KARTAVIEW_TOKEN", "")   # optional, no key needed

# ── GSV Static API settings ───────────────────────────────────────────────────
GSV_BASE_URL  = "https://maps.googleapis.com/maps/api/streetview"
GSV_META_URL  = "https://maps.googleapis.com/maps/api/streetview/metadata"
IMG_SIZE      = "640x640"    # max free tier size
IMG_FOV       = 90           # field of view degrees
# Fetch 4 headings per segment (N/E/S/W) for full 360° coverage
HEADINGS      = [0, 90, 180, 270]

# ── KartaView API ─────────────────────────────────────────────────────────────
KARTAVIEW_URL = "https://api.kartaview.org/2.0/sequence/list/"

# ── Rate limits ────────────────────────────────────────────────────────────────
GSV_REQUESTS_PER_SECOND = 10   # Google allows 50/s, staying conservative
SLEEP_BETWEEN_SEGMENTS  = 0.1  # seconds


# ──────────────────────────────────────────────────────────────────────────────
# 1.  GOOGLE STREET VIEW — CHECK COVERAGE
# ──────────────────────────────────────────────────────────────────────────────

def check_gsv_coverage(lat: float, lon: float) -> bool:
    """
    Hits the GSV Metadata API (FREE — doesn't count against image quota)
    to check whether a location has Street View coverage.
    Returns True if coverage exists within 50m.
    """
    if not GOOGLE_API_KEY:
        return False
    try:
        r = requests.get(
            GSV_META_URL,
            params={
                "location": f"{lat},{lon}",
                "radius":   50,
                "key":      GOOGLE_API_KEY,
            },
            timeout=10,
        )
        data = r.json()
        return data.get("status") == "OK"
    except Exception:
        return False


def fetch_gsv_image(
    lat: float,
    lon: float,
    heading: int = 0,
    segment_id: str = "",
) -> Image.Image | None:
    """
    Downloads a single GSV Static image.
    Returns PIL Image or None on failure.
    """
    if not GOOGLE_API_KEY:
        return None

    fname = IMG_DIR / f"{segment_id}_h{heading}.jpg"
    if fname.exists():
        try:
            return Image.open(fname).convert("RGB")
        except Exception:
            pass

    try:
        r = requests.get(
            GSV_BASE_URL,
            params={
                "size":     IMG_SIZE,
                "location": f"{lat},{lon}",
                "heading":  heading,
                "fov":      IMG_FOV,
                "pitch":    0,
                "radius":   50,
                "key":      GOOGLE_API_KEY,
            },
            timeout=15,
        )
        if r.status_code == 200 and r.headers.get("content-type", "").startswith("image"):
            img = Image.open(BytesIO(r.content)).convert("RGB")
            img.save(fname, "JPEG", quality=85)
            return img
    except Exception as e:
        log.debug(f"GSV fetch failed at ({lat},{lon}): {e}")
    return None


# ──────────────────────────────────────────────────────────────────────────────
# 2.  KARTAVIEW — FREE ALTERNATIVE
# ──────────────────────────────────────────────────────────────────────────────

def fetch_kartaview_images(lat: float, lon: float, radius_m: int = 50):
    """
    Queries KartaView for images near a point.
    KartaView is fully free — no API key needed.
    Returns list of image URLs (may be empty for Bengaluru).
    """
    try:
        r = requests.get(
            "https://api.kartaview.org/2.0/photo/list/",
            params={
                "lat":    lat,
                "lng":    lon,
                "radius": radius_m,
                "count":  3,
            },
            timeout=10,
        )
        data = r.json()
        photos = data.get("result", {}).get("data", [])
        return [p.get("image_url") for p in photos if p.get("image_url")]
    except Exception:
        return []


def download_image_from_url(url: str, save_path: Path) -> Image.Image | None:
    """Downloads an image from any URL and saves it."""
    if save_path.exists():
        try:
            return Image.open(save_path).convert("RGB")
        except Exception:
            pass
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            img = Image.open(BytesIO(r.content)).convert("RGB")
            img.save(save_path, "JPEG", quality=85)
            return img
    except Exception:
        pass
    return None


# ──────────────────────────────────────────────────────────────────────────────
# 3.  IMAGE FEATURE EXTRACTION  (runs locally — no GPU needed here)
# ──────────────────────────────────────────────────────────────────────────────

def extract_basic_image_features(images: list[Image.Image]) -> dict:
    """
    Extracts lightweight visual features from a list of images.
    These are CPU-only proxies — full CLIP inference runs on Colab later.

    Returns dict of features averaged across all provided images.
    """
    if not images:
        return _empty_image_features()

    brightness_scores = []
    greenery_scores   = []
    darkness_scores   = []
    contrast_scores   = []

    for img in images:
        arr = np.array(img, dtype=float)
        R, G, B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

        # ── Brightness: mean luminosity (Y channel approximation) ──────────
        Y = 0.299 * R + 0.587 * G + 0.114 * B
        brightness = float(Y.mean() / 255.0)
        brightness_scores.append(brightness)

        # ── Darkness: fraction of very dark pixels (<30 out of 255) ───────
        dark_pixels = (Y < 30).sum() / Y.size
        darkness_scores.append(float(dark_pixels))

        # ── Greenery: pixels where G dominates (vegetation proxy) ─────────
        green_mask = (G > R * 1.1) & (G > B * 1.1) & (G > 40)
        greenery = float(green_mask.sum() / green_mask.size)
        greenery_scores.append(greenery)

        # ── Contrast: std dev of luminosity (high contrast = visual clutter)
        contrast = float(Y.std() / 128.0)
        contrast_scores.append(min(contrast, 1.0))

    # ── Sky ratio (blue-dominant upper third = open road, not enclosed) ────
    sky_scores = []
    for img in images:
        arr  = np.array(img, dtype=float)
        top  = arr[: arr.shape[0] // 3, :, :]   # upper third
        R, G, B = top[:,:,0], top[:,:,1], top[:,:,2]
        sky_mask = (B > R * 1.1) & (B > G * 1.05) & (B > 80)
        sky_scores.append(float(sky_mask.sum() / sky_mask.size))

    return {
        "brightness_mean":   round(float(np.mean(brightness_scores)), 4),
        "brightness_min":    round(float(np.min(brightness_scores)),  4),
        "darkness_ratio":    round(float(np.mean(darkness_scores)),   4),
        "greenery_ratio":    round(float(np.mean(greenery_scores)),   4),
        "contrast_score":    round(float(np.mean(contrast_scores)),   4),
        "sky_ratio":         round(float(np.mean(sky_scores)),        4),
        "n_images":          len(images),
        "has_real_images":   True,
    }


def _empty_image_features() -> dict:
    """Returned when no images found for a segment."""
    return {
        "brightness_mean":   0.35,   # conservative defaults
        "brightness_min":    0.20,
        "darkness_ratio":    0.30,
        "greenery_ratio":    0.10,
        "contrast_score":    0.50,
        "sky_ratio":         0.20,
        "n_images":          0,
        "has_real_images":   False,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 4.  MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def process_all_segments(
    luminosity_csv: str = "data/raw/bengaluru_segment_luminosity.csv",
    max_segments: int   = None,   # set to 500 for quick test
) -> pd.DataFrame:
    """
    For every road segment:
      1. Try Google Street View (best quality)
      2. Fall back to KartaView
      3. Fall back to synthetic image features derived from OSM+VIIRS

    Saves per-segment image features to data/raw/bengaluru_image_features.csv
    These features feed directly into ai/cv/clip_scorer.py (Colab step).
    """
    out = DATA_RAW / "bengaluru_image_features.csv"
    if out.exists():
        log.info(f"Image features already exist → {out}")
        return pd.read_csv(out)

    lum_df = pd.read_csv(luminosity_csv)
    if max_segments:
        lum_df = lum_df.head(max_segments)
        log.info(f"Processing {max_segments} segments (test mode)")
    else:
        log.info(f"Processing {len(lum_df):,} segments")

    if not GOOGLE_API_KEY:
        log.warning(
            "GOOGLE_MAPS_API_KEY not set.\n"
            "Get free key: https://console.cloud.google.com/\n"
            "Enable: Street View Static API\n"
            "Falling back to KartaView + synthetic features."
        )

    records   = []
    gsv_count = 0
    kv_count  = 0
    syn_count = 0

    for i, row in lum_df.iterrows():
        if i % 500 == 0:
            log.info(
                f"  [{i:,}/{len(lum_df):,}] "
                f"GSV:{gsv_count} KV:{kv_count} Synthetic:{syn_count}"
            )

        lat = row["mid_lat"]
        lon = row["mid_lon"]
        seg_id = f"{int(row['u'])}_{int(row['v'])}"

        images = []
        source = "synthetic"

        # ── Try Google Street View ─────────────────────────────────────────
        if GOOGLE_API_KEY and check_gsv_coverage(lat, lon):
            for heading in HEADINGS[:2]:   # 2 headings to save quota
                img = fetch_gsv_image(lat, lon, heading, seg_id)
                if img:
                    images.append(img)
            if images:
                source = "gsv"
                gsv_count += 1
                time.sleep(1 / GSV_REQUESTS_PER_SECOND)

        # ── Try KartaView if GSV failed ────────────────────────────────────
        if not images:
            urls = fetch_kartaview_images(lat, lon)
            for j, url in enumerate(urls[:2]):
                img = download_image_from_url(
                    url, IMG_DIR / f"{seg_id}_kv{j}.jpg"
                )
                if img:
                    images.append(img)
            if images:
                source = "kartaview"
                kv_count += 1

        # ── Extract features ───────────────────────────────────────────────
        if images:
            feats = extract_basic_image_features(images)
        else:
            feats  = _build_synthetic_features(row)
            source = "synthetic"
            syn_count += 1

        records.append({
            "u":   row["u"],
            "v":   row["v"],
            "key": row["key"],
            "mid_lat": lat,
            "mid_lon": lon,
            "image_source": source,
            **feats,
        })

        time.sleep(SLEEP_BETWEEN_SEGMENTS)

    df = pd.DataFrame(records)
    df.to_csv(out, index=False)

    log.info(f"\n── Image Source Summary ────────────────────────")
    log.info(f"  Google Street View : {gsv_count:,}")
    log.info(f"  KartaView          : {kv_count:,}")
    log.info(f"  Synthetic fallback : {syn_count:,}")
    log.info(f"  Total segments     : {len(df):,}")
    log.info(f"  Saved → {out}")
    return df


def _build_synthetic_features(row: pd.Series) -> dict:
    """
    Builds realistic synthetic image features when no street view
    image is available, using VIIRS luminosity + OSM data as proxies.
    """
    lum  = float(row.get("luminosity_norm",  0.4))
    zone = str(row.get("luminosity_zone",   "dim"))

    # Brightness correlates directly with VIIRS luminosity
    brightness = float(np.clip(lum + np.random.normal(0, 0.05), 0, 1))

    # Dark zones have more dark pixels
    darkness = float(np.clip(1 - lum + np.random.normal(0, 0.05), 0, 1))

    # Greenery: random, slightly higher on residential streets
    greenery = float(np.clip(np.random.beta(1.5, 4), 0, 1))

    # Contrast: busy bright roads have more visual variety
    contrast = float(np.clip(lum * 0.6 + np.random.normal(0, 0.1), 0, 1))

    # Sky: open roads have more sky visible
    sky = float(np.clip(np.random.beta(2, 3), 0, 1))

    return {
        "brightness_mean":   round(brightness, 4),
        "brightness_min":    round(brightness * 0.7, 4),
        "darkness_ratio":    round(darkness,   4),
        "greenery_ratio":    round(greenery,   4),
        "contrast_score":    round(contrast,   4),
        "sky_ratio":         round(sky,        4),
        "n_images":          0,
        "has_real_images":   False,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 5.  SAVE IMAGE PATHS FOR COLAB CLIP SCORING
# ──────────────────────────────────────────────────────────────────────────────

def export_image_list_for_colab() -> Path:
    """
    Creates a CSV of all downloaded image paths.
    Upload this to Colab to run CLIP inference on real images.
    Format: segment_id, image_path, mid_lat, mid_lon
    """
    out = DATA_RAW / "colab_image_list.csv"
    images = list(IMG_DIR.glob("*.jpg"))

    records = []
    for p in images:
        parts = p.stem.split("_h")
        if len(parts) == 2:
            seg_id  = parts[0]
            heading = parts[1]
        else:
            seg_id  = p.stem
            heading = "0"
        records.append({
            "segment_id":   seg_id,
            "image_path":   str(p),
            "heading":      heading,
            "image_source": "gsv",
        })

    df = pd.DataFrame(records)
    df.to_csv(out, index=False)
    log.info(f"Colab image list → {out}  ({len(df):,} images)")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 6.  MAIN
# ──────────────────────────────────────────────────────────────────────────────

def run():
    log.info("=== fetch_streetview.py  START ===")

    lum_path = Path("data/raw/bengaluru_segment_luminosity.csv")

    log.info(f"Looking for luminosity CSV at: {lum_path.absolute()}")
    log.info(f"File exists: {lum_path.exists()}")

    if not lum_path.exists():
        raise FileNotFoundError(
            f"Luminosity CSV not found at {lum_path.absolute()}\n"
            "Run fetch_viirs.py first."
        )

    df = process_all_segments(
        luminosity_csv=str(lum_path),
        max_segments=200,
    )

    export_image_list_for_colab()

    log.info(f"\n── Feature Summary ─────────────────────────────")
    log.info(f"  Brightness mean : {df['brightness_mean'].mean():.3f}")
    log.info(f"  Greenery mean   : {df['greenery_ratio'].mean():.3f}")
    log.info(f"  Real images     : {df['has_real_images'].sum():,} / {len(df):,}")
    log.info("=== fetch_streetview.py  DONE ===")
    return df


if __name__ == "__main__":
    run()