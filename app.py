"""
Atlas AI — Earth Observation Dashboard
======================================
Single-file Streamlit app consolidating the full CDSE notebook pipeline.

UPLOAD TO GITHUB:  app.py  +  requirements.txt   (that's it)
SECRETS:           set CDSE_CLIENT_ID / CDSE_CLIENT_SECRET in the Streamlit
                   Cloud "Secrets" panel — never commit them.
RUN LOCALLY:       streamlit run app.py
"""

import datetime
import math

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import streamlit as st
from geopy.geocoders import Nominatim
from sentinelhub import (SHConfig, DownloadClient, DownloadRequest,
                         SentinelHubCatalog, SentinelHubSession, BBox, CRS)

st.set_page_config(page_title="Atlas AI — EO Dashboard", layout="wide")
PX = 512
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"

# =============================================================================
# AUTH  (secrets → cached config/catalog/client, token cached ~9 min)
# =============================================================================
@st.cache_resource
def get_config():
    cfg = SHConfig()
    try:
        cfg.sh_client_id = st.secrets["CDSE_CLIENT_ID"]
        cfg.sh_client_secret = st.secrets["CDSE_CLIENT_SECRET"]
    except Exception:
        st.error("Missing CDSE credentials. Add CDSE_CLIENT_ID and "
                 "CDSE_CLIENT_SECRET in the app's Secrets panel.")
        st.stop()
    cfg.sh_token_url = ("https://identity.dataspace.copernicus.eu/auth/realms/"
                        "CDSE/protocol/openid-connect/token")
    cfg.sh_base_url = "https://sh.dataspace.copernicus.eu"
    return cfg

@st.cache_resource
def get_catalog():
    return SentinelHubCatalog(config=get_config())

@st.cache_resource
def get_client():
    return DownloadClient(config=get_config())

@st.cache_data(ttl=540)
def get_token():
    return SentinelHubSession(config=get_config()).token["access_token"]

def _auth_check():
    cid = st.secrets.get("CDSE_CLIENT_ID", "")
    csec = st.secrets.get("CDSE_CLIENT_SECRET", "")
    st.caption(f"id ends …{cid[-6:]} · id len {len(cid)} · secret len {len(csec)}")
    try:
        get_token()
        st.success("✅ CDSE auth OK")
    except Exception as e:
        st.error(f"❌ CDSE rejected credentials: {e}")
        st.stop()

_auth_check()

# =============================================================================
# GEOMETRY + GENERIC HELPERS
# =============================================================================
def bbox_local(lat, lon, buf=0.015):
    return [lon - buf, lat - buf, lon + buf, lat + buf]

def pixel_area_m2(bbox, px=PX):
    mnx, mny, mxx, mxy = bbox
    mid = (mny + mxy) / 2
    w = (mxx - mnx) * 111320 * math.cos(math.radians(mid))
    h = (mxy - mny) * 111320
    return (w / px) * (h / px)

@st.cache_data(ttl=86400)
def geocode(name):
    loc = Nominatim(user_agent="atlas_ai_dashboard").geocode(name)
    return (loc.latitude, loc.longitude, loc.address) if loc else None

@st.cache_data(ttl=3600)
def cleanest_s2(lat, lon, start, end):
    bb = BBox(bbox=bbox_local(lat, lon), crs=CRS.WGS84)
    res = get_catalog().search(collection="sentinel-2-l2a", bbox=bb, time=(start, end),
                               fields={"include": ["properties.datetime",
                                                   "properties.eo:cloud_cover"]}, limit=100)
    scenes = list(res)
    if not scenes:
        return None, None
    best = sorted(scenes, key=lambda x: x["properties"].get("eo:cloud_cover", 100))[0]
    return best["properties"]["datetime"].split("T")[0], \
        best["properties"].get("eo:cloud_cover", 0)

@st.cache_data(ttl=3600)
def recent_s1(lat, lon, start, end):
    bb = BBox(bbox=bbox_local(lat, lon), crs=CRS.WGS84)
    res = get_catalog().search(collection="sentinel-1-grd", bbox=bb, time=(start, end),
                               fields={"include": ["properties.datetime"]}, limit=100)
    scenes = list(res)
    if not scenes:
        return None
    return sorted(scenes, key=lambda x: x["properties"]["datetime"],
                  reverse=True)[0]["properties"]["datetime"].split("T")[0]

def _process(payload):
    req = DownloadRequest(url=PROCESS_URL, post_values=payload, request_type="POST",
                          headers={"Content-Type": "application/json",
                                   "Authorization": f"Bearer {get_token()}"})
    return get_client().download([req])[0]

# =============================================================================
# DISPLAY PANELS (PNG evalscripts — visual only)
# =============================================================================
EVALSCRIPTS = {
    "True colour": """//VERSION=3
function setup(){return{input:["B04","B03","B02"],output:{bands:3}};}
function evaluatePixel(s){return[2.5*s.B04,2.5*s.B03,2.5*s.B02];}""",
    "NDVI (vegetation)": """//VERSION=3
function setup(){return{input:["B08","B04"],output:{bands:3}};}
function evaluatePixel(s){let n=(s.B08-s.B04)/(s.B08+s.B04);
if(n<0)return[0.8,0.2,0.2];else if(n<0.2)return[0.9,0.8,0.4];
else if(n<0.5)return[0.4,0.8,0.4];return[0.1,0.5,0.1];}""",
    "NDWI (water)": """//VERSION=3
function setup(){return{input:["B03","B08"],output:{bands:3}};}
function evaluatePixel(s){let w=(s.B03-s.B08)/(s.B03+s.B08);
if(w>0.1)return[0.0,0.3,0.9];let g=0.21*s.B08+0.72*s.B03;return[g*2,g*2,g*2];}""",
}
RADAR_SCRIPT = """//VERSION=3
function setup(){return{input:[{bands:["VV","VH","dataMask"]}],output:{bands:3}};}
function evaluatePixel(s){if(s.dataMask===0)return[0,0,0];
let vv=Math.sqrt(s.VV),vh=Math.sqrt(s.VH);
return[vv*2.0,vh*3.5,(s.VV/(s.VH+0.001))*0.15];}"""

@st.cache_data(ttl=3600)
def fetch_png(lat, lon, date, collection, evalscript):
    payload = {"input": {"bounds": {"bbox": bbox_local(lat, lon),
                                    "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"}},
                         "data": [{"type": collection,
                                   "dataFilter": {"timeRange": {"from": f"{date}T00:00:00Z",
                                                                "to": f"{date}T23:59:59Z"}}}]},
               "output": {"width": PX, "height": PX,
                          "responses": [{"identifier": "default", "format": {"type": "image/png"}}]},
               "evalscript": evalscript}
    return np.asarray(_process(payload), dtype="uint8")

# =============================================================================
# RAW INDICES (FLOAT — the measured core)
# =============================================================================
RAW_BANDS = """//VERSION=3
function setup(){return{input:[{bands:["B02","B03","B04","B08","B11","B12","dataMask"]}],
output:{bands:7,sampleType:"FLOAT32"}};}
function evaluatePixel(s){return[s.B02,s.B03,s.B04,s.B08,s.B11,s.B12,s.dataMask];}"""

@st.cache_data(ttl=3600)
def fetch_indices(lat, lon, date):
    payload = {"input": {"bounds": {"bbox": bbox_local(lat, lon),
                                    "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"}},
                         "data": [{"type": "sentinel-2-l2a",
                                   "dataFilter": {"timeRange": {"from": f"{date}T00:00:00Z",
                                                                "to": f"{date}T23:59:59Z"}}}]},
               "output": {"width": PX, "height": PX,
                          "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]},
               "evalscript": RAW_BANDS}
    arr = _process(payload)
    B02, B03, B04, B08, B11, B12 = [arr[:, :, i].astype("float32") for i in range(6)]
    mask = arr[:, :, 6] > 0
    nd = lambda a, b: (a - b) / (a + b + 1e-10)
    out = {"ndvi": nd(B08, B04), "ndmi": nd(B08, B11), "ndwi": nd(B03, B08),
           "ndbi": nd(B11, B08), "bsi": nd(B11 + B04, B08 + B02),
           "fe": np.where(B02 > 0, B04 / (B02 + 1e-10), np.nan),
           "clay": np.where(B12 > 0, B11 / (B12 + 1e-10), np.nan), "mask": mask}
    for k in ("ndvi", "ndmi", "ndwi", "ndbi", "bsi", "fe", "clay"):
        out[k][~mask] = np.nan
    return out

# =============================================================================
# OPEN APIs (rainfall, solar, power plants, soil/geology)
# =============================================================================
@st.cache_data(ttl=21600)
def fetch_weather(lat, lon, years=12):
    today = datetime.date.today()
    params = {"latitude": lat, "longitude": lon,
              "start_date": f"{today.year - years}-01-01",
              "end_date": (today - datetime.timedelta(days=6)).strftime("%Y-%m-%d"),
              "daily": "precipitation_sum,shortwave_radiation_sum", "timezone": "auto"}
    r = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=90)
    r.raise_for_status()
    d = r.json()["daily"]
    df = pd.DataFrame({"date": pd.to_datetime(d["time"]),
                       "precip": d["precipitation_sum"],
                       "ghi_mj": d["shortwave_radiation_sum"]}).dropna(subset=["precip"])
    df["year"] = df.date.dt.year
    df["month"] = df.date.dt.month
    df["doy"] = df.date.dt.dayofyear
    return df

@st.cache_data(ttl=3600)
def fetch_power_plants(lat, lon, radius_km=50):
    q = f"""[out:json][timeout:60];
    ( nwr["power"="plant"](around:{int(radius_km*1000)},{lat},{lon}); );
    out center tags;"""
    try:
        r = requests.post("https://overpass-api.de/api/interpreter", data={"data": q},
                          headers={"User-Agent": "atlas-ai-dashboard/1.0"}, timeout=90)
        r.raise_for_status()
        els = r.json().get("elements", [])
    except Exception:
        return []
    import re
    def mw(v):
        if not v:
            return None
        m = re.search(r"([\d.]+)\s*(g|m|k)?\s*w", str(v).lower())
        if not m:
            return None
        n, p = float(m.group(1)), m.group(2)
        return n * 1000 if p == "g" else n / 1000 if p == "k" else n
    out = []
    for e in els:
        t = e.get("tags", {})
        plat = e.get("lat") or e.get("center", {}).get("lat")
        plon = e.get("lon") or e.get("center", {}).get("lon")
        if plat is None:
            continue
        src = (t.get("plant:source") or "unknown").split(";")[0].strip().lower()
        out.append({"source": src, "mw": mw(t.get("plant:output:electricity")),
                    "lat": plat, "lon": plon})
    return out

@st.cache_data(ttl=86400)
def fetch_soil(lat, lon):
    props = ["clay", "sand", "silt", "phh2o", "soc"]
    params = [("lon", lon), ("lat", lat), ("depth", "0-5cm"), ("value", "mean")]
    params += [("property", p) for p in props]
    try:
        r = requests.get("https://rest.isric.org/soilgrids/v2.0/properties/query",
                         params=params, timeout=60)
        r.raise_for_status()
        out = {}
        for layer in r.json()["properties"]["layers"]:
            dd = next((x for x in layer["depths"] if x["label"] == "0-5cm"), None)
            if dd and dd["values"].get("mean") is not None:
                out[layer["name"]] = dd["values"]["mean"] / layer["unit_measure"]["d_factor"]
        return out
    except Exception:
        return {}

@st.cache_data(ttl=86400)
def fetch_bedrock(lat, lon):
    try:
        r = requests.get("https://macrostrat.org/api/v2/geologic_units/map",
                         params={"lat": lat, "lng": lon}, timeout=60)
        r.raise_for_status()
        data = r.json().get("success", {}).get("data", [])
        if not data:
            return None
        u = data[0]
        lith = u.get("lith", "")
        if isinstance(lith, list):
            lith = ", ".join(x.get("name", "") if isinstance(x, dict) else str(x) for x in lith)
        return {"name": u.get("name") or "Unnamed", "lithology": lith or "n/a",
                "age": f"{u.get('b_int_name','?')} – {u.get('t_int_name','?')}"}
    except Exception:
        return None

# =============================================================================
# SIDEBAR
# =============================================================================
st.title("🛰️ Atlas AI — Earth Observation Dashboard")
st.caption("Satellite + open-data fusion · demo build")

with st.sidebar:
    st.header("Location & time")
    loc_input = st.text_input("City name or 'lat, lon'", "Nagpur")
    months = st.slider("Look-back window (months)", 1, 36, 6)
    radius = st.slider("Power-plant search radius (km)", 10, 100, 50)
    run = st.button("Run analysis", type="primary")

lat = lon = None
addr = ""
if "," in loc_input and any(c.isdigit() for c in loc_input):
    try:
        a, b = loc_input.split(",")
        lat, lon, addr = float(a), float(b), "manual coordinates"
    except Exception:
        st.sidebar.error("Couldn't parse coordinates.")
elif loc_input:
    g = geocode(loc_input)
    if g:
        lat, lon, addr = g
    else:
        st.sidebar.error("Location not found.")
if lat is not None:
    st.sidebar.success(f"📍 {lat:.3f}, {lon:.3f}\n\n{addr}")

# =============================================================================
# MAIN
# =============================================================================
if not run:
    st.info("Set a location and click **Run analysis**.")
    st.stop()
if lat is None:
    st.warning("Resolve a valid location first.")
    st.stop()

end = datetime.date.today()
start = end - datetime.timedelta(days=months * 30)
S, E = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

tabs = st.tabs(["🛰️ Imagery", "📊 Indices", "🗺️ Land cover", "📈 Change",
                "🌧️ Rainfall", "☀️ Solar", "⚡ Power", "🌍 Soil & geology"])

# ---- TAB: Imagery ----
with tabs[0]:
    s2date, cloud = cleanest_s2(lat, lon, S, E)
    if not s2date:
        st.warning("No Sentinel-2 scene in this window — widen the look-back.")
    else:
        st.caption(f"Sentinel-2: {s2date} · {cloud:.0f}% cloud")
        cols = st.columns(3)
        for col, (name, script) in zip(cols, EVALSCRIPTS.items()):
            with col:
                img = fetch_png(lat, lon, s2date, "sentinel-2-l2a", script)
                fig, ax = plt.subplots(); ax.imshow(img); ax.set_title(name); ax.axis("off")
                st.pyplot(fig)
        s1date = recent_s1(lat, lon, S, E)
        if s1date:
            st.caption(f"Sentinel-1 radar (cloud-proof): {s1date}")
            img = fetch_png(lat, lon, s1date, "sentinel-1-grd", RADAR_SCRIPT)
            fig, ax = plt.subplots(figsize=(5, 5)); ax.imshow(img)
            ax.set_title("Radar (VV/VH)"); ax.axis("off")
            rc1, rc2, rc3 = st.columns([1, 1, 1])
            with rc2:
                st.pyplot(fig, use_container_width=True)

# ---- TAB: Indices ----
with tabs[1]:
    s2date, cloud = cleanest_s2(lat, lon, S, E)
    if not s2date:
        st.warning("No Sentinel-2 scene.")
    else:
        idx = fetch_indices(lat, lon, s2date)
        st.caption(f"Scene {s2date} · {cloud:.0f}% cloud · all values from raw reflectance")
        # single-label classification so the four cards are consistent and sum to ~100%
        ndvi, ndwi, ndbi, bsi = idx["ndvi"], idx["ndwi"], idx["ndbi"], idx["bsi"]
        vmask = np.isfinite(ndvi)
        cls = np.full(ndvi.shape, -1, dtype=int)
        cls[vmask & (ndwi > 0.0)] = 0
        r = vmask & (cls == -1); cls[r & (ndvi > 0.45)] = 1
        r = vmask & (cls == -1); cls[r & (ndvi > 0.25)] = 2
        r = vmask & (cls == -1); cls[r & (ndbi >= bsi)] = 3
        r = vmask & (cls == -1); cls[r] = 4
        tot = max(int(vmask.sum()), 1)
        m = st.columns(4)
        m[0].metric("Vegetation", f"{100*((cls==1)|(cls==2)).sum()/tot:.0f}%")
        m[1].metric("Open water", f"{100*(cls==0).sum()/tot:.0f}%")
        m[2].metric("Bare ground", f"{100*(cls==4).sum()/tot:.0f}%")
        m[3].metric("Built-up", f"{100*(cls==3).sum()/tot:.0f}%")
        st.caption("Single-label classification — these four sum to ~100% (each pixel counted once).")
        panels = [("NDVI — vegetation", "ndvi", "RdYlGn", -0.2, 0.8),
                  ("NDMI — plant moisture", "ndmi", "BrBG", -0.5, 0.5),
                  ("NDWI — water", "ndwi", "Blues", -0.3, 0.6),
                  ("NDBI — built-up", "ndbi", "pink_r", -0.5, 0.5),
                  ("BSI — bare soil/rock", "bsi", "YlOrBr", -0.3, 0.6),
                  ("Fe — iron-oxide (hint)", "fe", "OrRd", None, None),
                  ("Clay — clay-mineral (hint)", "clay", "BuPu", None, None)]
        fig, axes = plt.subplots(2, 4, figsize=(18, 9)); axes = axes.ravel()
        for i, (title, k, cmap, vmin, vmax) in enumerate(panels):
            if vmin is None:
                vmin, vmax = np.nanpercentile(idx[k], [2, 98])
            im = axes[i].imshow(idx[k], cmap=cmap, vmin=vmin, vmax=vmax)
            axes[i].set_title(title, fontsize=10); axes[i].axis("off")
            fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        axes[7].axis("off")
        st.pyplot(fig)
        st.caption("Fe / Clay are surface-alteration screening hints — not mineralogy.")

# ---- TAB: Land cover ----
with tabs[2]:
    s2date, cloud = cleanest_s2(lat, lon, S, E)
    if not s2date:
        st.warning("No Sentinel-2 scene.")
    else:
        idx = fetch_indices(lat, lon, s2date)
        ndvi, ndwi, ndbi, bsi = idx["ndvi"], idx["ndwi"], idx["ndbi"], idx["bsi"]
        valid = np.isfinite(ndvi)
        cls = np.full(ndvi.shape, -1, dtype=int)
        cls[valid & (ndwi > 0.0)] = 0
        rem = valid & (cls == -1); cls[rem & (ndvi > 0.45)] = 1
        rem = valid & (cls == -1); cls[rem & (ndvi > 0.25)] = 2
        rem = valid & (cls == -1); cls[rem & (ndbi >= bsi)] = 3
        rem = valid & (cls == -1); cls[rem] = 4
        CLASSES = [("Open water", "#1f78b4"), ("Dense vegetation", "#1a7d33"),
                   ("Sparse vegetation", "#8cc06a"), ("Built-up", "#7a6f8a"),
                   ("Bare soil/rock", "#d9a066")]
        pa = pixel_area_m2(bbox_local(lat, lon)); total = valid.sum()
        pcts = [100 * (cls == c).sum() / total for c in range(5)]
        has = [(cls == c).sum() * pa / 10000 for c in range(5)]
        cmap = ListedColormap([c for _, c in CLASSES]); cmap.set_bad("white")
        cm = np.ma.masked_where(cls < 0, cls)
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.imshow(cm, cmap=cmap, vmin=-0.5, vmax=4.5)
            ax.set_title(f"Land classification — {s2date}"); ax.axis("off")
            ax.legend(handles=[mpatches.Patch(color=c, label=n) for n, c in CLASSES],
                      loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, frameon=False)
            st.pyplot(fig)
        with c2:
            order = sorted(range(5), key=lambda c: pcts[c])
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.barh([CLASSES[c][0] for c in order], [pcts[c] for c in order],
                    color=[CLASSES[c][1] for c in order], edgecolor="white")
            for i, c in enumerate(order):
                if pcts[c] >= 0.5:
                    ax.text(pcts[c], i, f" {pcts[c]:.0f}% ({has[c]:,.0f} ha)", va="center", fontsize=9)
            ax.set_xlabel("% of area (sums to 100)"); ax.margins(x=0.2)
            st.pyplot(fig)
        st.caption("Rule-based classification. Built-vs-bare is the least certain split.")

# ---- TAB: Change ----
with tabs[3]:
    today = datetime.date.today()
    nw = ((today - datetime.timedelta(days=60)).strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d"))
    ow = ((today - datetime.timedelta(days=425)).strftime("%Y-%m-%d"),
          (today - datetime.timedelta(days=365)).strftime("%Y-%m-%d"))
    od, oc = cleanest_s2(lat, lon, *ow)
    nd_, nc = cleanest_s2(lat, lon, *nw)
    if not od or not nd_:
        st.warning("Need a clear scene in both periods (same season, 1 yr apart). Widen window.")
    else:
        a_old = fetch_indices(lat, lon, od)["ndvi"]
        a_new = fetch_indices(lat, lon, nd_)["ndvi"]
        diff = a_new - a_old
        v = np.isfinite(diff)
        pa = pixel_area_m2(bbox_local(lat, lon))
        loss = (v & (diff <= -0.2)).sum() * pa / 10000
        gain = (v & (diff >= 0.2)).sum() * pa / 10000
        c1, c2, c3 = st.columns(3)
        c1.metric("Vegetation loss", f"{loss:,.0f} ha")
        c2.metric("Vegetation gain", f"{gain:,.0f} ha")
        c3.metric("Mean NDVI", f"{np.nanmean(a_new):.2f}", f"{np.nanmean(a_new)-np.nanmean(a_old):+.2f}")
        fig, ax = plt.subplots(1, 3, figsize=(16, 5.5))
        ax[0].imshow(a_old, cmap="RdYlGn", vmin=-0.2, vmax=0.8); ax[0].set_title(f"Before {od}"); ax[0].axis("off")
        ax[1].imshow(a_new, cmap="RdYlGn", vmin=-0.2, vmax=0.8); ax[1].set_title(f"After {nd_}"); ax[1].axis("off")
        im = ax[2].imshow(diff, cmap="RdBu", vmin=-0.6, vmax=0.6)
        ax[2].set_title("Change (red=loss · blue=gain)"); ax[2].axis("off")
        fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)
        st.pyplot(fig)
        st.caption("Mining: loss = possible clearing/extraction. Agri: loss = stress/harvest.")

# ---- TAB: Rainfall ----
with tabs[4]:
    try:
        df = fetch_weather(lat, lon)
        cy = datetime.date.today().year
        df["cum"] = df.groupby("year")["precip"].cumsum()
        base = df[df.year < cy]; cur = df[df.year == cy]
        ncum = base.groupby("doy")["cum"].mean()
        latest = int(cur.doy.max()); ytd = float(cur["cum"].max())
        norm = float(ncum.loc[:latest].iloc[-1]); pct = 100 * ytd / norm
        c1, c2 = st.columns(2)
        c1.metric("Rainfall year-to-date", f"{ytd:.0f} mm", f"{pct-100:+.0f}% vs normal")
        c2.metric("Annual normal", f"{base.groupby('year')['precip'].sum().mean():.0f} mm")
        bm = base.groupby([base.year, base.month])["precip"].sum().groupby(level=1).mean()
        start12 = (pd.Timestamp(datetime.date.today()) - pd.DateOffset(months=12)).normalize()
        last12 = df[df.date >= start12]
        ma = last12.groupby([last12.date.dt.year, last12.date.dt.month])["precip"].sum()
        labels = [f"{y}-{mo:02d}" for (y, mo) in ma.index]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 5))
        x = np.arange(len(labels))
        a1.bar(x, ma.values, color="#1f9ed4", label="actual")
        a1.plot(x, [bm.get(mo, np.nan) for (_, mo) in ma.index], "o--", color="#c0392b", label="normal")
        a1.set_xticks(x); a1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        a1.set_ylabel("mm"); a1.set_title("Monthly rainfall vs normal"); a1.legend()
        a2.fill_between(ncum.index, base.groupby("doy")["cum"].min(), base.groupby("doy")["cum"].max(),
                        color="#ccc", alpha=.4, label="12-yr range")
        a2.plot(ncum.index, ncum.values, "--", color="#c0392b", label="normal")
        a2.plot(cur.doy, cur["cum"], color="#1f9ed4", lw=2.5, label=f"{cy}")
        a2.set_xlabel("Day of year"); a2.set_ylabel("Cumulative mm")
        a2.set_title(f"{cy} at {pct:.0f}% of normal"); a2.legend()
        st.pyplot(fig)
        st.caption("Source: Open-Meteo (ERA5) · CC BY 4.0 · ~9–25 km modelled.")
    except Exception as e:
        st.error(f"Weather fetch failed: {e}")

# ---- TAB: Solar ----
with tabs[5]:
    try:
        df = fetch_weather(lat, lon)
        df["ghi"] = df["ghi_mj"] / 3.6
        cnt = df.groupby("year")["ghi"].count()
        comp = cnt[cnt >= 360].index
        b = df[df.year.isin(comp)]
        ann = b.groupby("year")["ghi"].sum()
        annual = float(ann.mean()); cv = 100 * ann.std() / annual
        daily = annual / 365; yield_kwh = annual * 0.78
        md = b.groupby("month")["ghi"].mean()
        c1, c2, c3 = st.columns(3)
        c1.metric("Annual GHI", f"{annual:.0f} kWh/m²/yr")
        c2.metric("Daily avg", f"{daily:.1f} kWh/m²/day", f"≈{daily:.1f} peak-sun-hrs")
        c3.metric("Screening yield", f"~{yield_kwh:.0f} kWh/kWp/yr", f"±{cv:.0f}% yearly")
        mn = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 5))
        a1.bar(range(1, 13), [md.get(m, np.nan) for m in range(1, 13)], color="#f39c12")
        a1.axhline(daily, color="#c0392b", ls="--", label=f"avg {daily:.1f}")
        a1.set_xticks(range(1, 13)); a1.set_xticklabels(mn[1:])
        a1.set_ylabel("kWh/m²/day"); a1.set_title("Solar resource by month"); a1.legend()
        a2.bar(ann.index, ann.values, color="#f39c12")
        a2.axhline(annual, color="#c0392b", ls="--", label=f"mean {annual:.0f}")
        a2.set_xlabel("Year"); a2.set_ylabel("kWh/m²/yr")
        a2.set_title(f"Annual resource (±{cv:.0f}%)"); a2.legend()
        st.pyplot(fig)
        st.caption("Source: Open-Meteo · CC BY 4.0. Yield is a screening estimate — "
                   "not tilt-optimised, not bankable.")
    except Exception as e:
        st.error(f"Solar fetch failed: {e}")

# ---- TAB: Power ----
with tabs[6]:
    plants = fetch_power_plants(lat, lon, radius)
    if not plants:
        st.info(f"No utility-scale power plants mapped within {radius} km.")
    else:
        by = {}
        for p in plants:
            s = p["source"]; by.setdefault(s, {"n": 0, "mw": 0.0})
            by[s]["n"] += 1
            if p["mw"]:
                by[s]["mw"] += p["mw"]
        COL = {"coal": "#2b2b2b", "gas": "#8e44ad", "hydro": "#1f9ed4", "solar": "#f39c12",
               "wind": "#5dade2", "nuclear": "#7f8c8d", "biomass": "#27ae60", "unknown": "#bdc3c7"}
        total_mw = sum(v["mw"] for v in by.values())
        c1, c2 = st.columns(2)
        c1.metric("Plants found", len(plants))
        c2.metric("Known capacity", f"{total_mw:,.0f} MW")
        cc1, cc2 = st.columns(2)
        with cc1:
            fig, ax = plt.subplots(figsize=(7, 6))
            for p in plants:
                ax.scatter(p["lon"], p["lat"], s=45 + (p["mw"] or 0) ** 0.5 * 12,
                           c=COL.get(p["source"], "#bdc3c7"), edgecolors="white", alpha=.9)
            ax.scatter(lon, lat, marker="+", s=180, c="black", linewidths=1.8)
            ax.set_title(f"Plants within {radius} km"); ax.grid(alpha=.2)
            ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
            st.pyplot(fig)
        with cc2:
            order = sorted(by, key=lambda k: by[k]["n"], reverse=True)
            fig, ax = plt.subplots(figsize=(7, 6))
            bars = ax.bar(order, [by[s]["n"] for s in order],
                          color=[COL.get(s, "#bdc3c7") for s in order], edgecolor="white")
            ax.set_ylim(0, max(by[s]["n"] for s in order) * 1.3)
            for s, bar in zip(order, bars):
                ax.annotate(f"{by[s]['n']}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            textcoords="offset points", xytext=(0, 4), ha="center", fontweight="bold")
                if by[s]["mw"] > 0:
                    ax.annotate(f"{by[s]['mw']:,.0f} MW",
                                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                                textcoords="offset points", xytext=(0, 22), ha="center",
                                fontsize=9, color="#666")
            ax.set_ylabel("Number of plants"); ax.set_title("Plants by energy source")
            st.pyplot(fig)
        st.caption("Source: OpenStreetMap / ODbL. Capacity excludes untagged plants.")

# ---- TAB: Soil & geology ----
with tabs[7]:
    soil = fetch_soil(lat, lon); rock = fetch_bedrock(lat, lon)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Soil (0–5 cm)")
        if soil and all(k in soil for k in ("sand", "silt", "clay")):
            sand, silt, clay = soil["sand"], soil["silt"], soil["clay"]
            fig, ax = plt.subplots(figsize=(7, 2.4)); left = 0
            for frac, lab, col in [(sand, "Sand", "#d9b382"), (silt, "Silt", "#9ecae1"),
                                   (clay, "Clay", "#a1683a")]:
                ax.barh(0, frac, left=left, color=col, edgecolor="white")
                if frac > 6:
                    ax.text(left + frac / 2, 0, f"{lab}\n{frac:.0f}%", ha="center", va="center",
                            fontweight="bold")
                left += frac
            ax.set_xlim(0, max(left, 100)); ax.set_yticks([]); ax.set_xlabel("% by mass")
            st.pyplot(fig)
            if "phh2o" in soil:
                st.metric("Soil pH", f"{soil['phh2o']:.1f}")
            if "soc" in soil:
                st.metric("Organic carbon", f"{soil['soc']:.1f} g/kg")
        else:
            st.info("No soil data returned.")
        st.caption("Source: SoilGrids v2.0 / ISRIC · ~250 m modelled.")
    with c2:
        st.subheader("Bedrock geology")
        if rock:
            st.write(f"**Unit:** {rock['name']}")
            st.write(f"**Lithology:** {rock['lithology']}")
            st.write(f"**Age:** {rock['age']}")
        else:
            st.info("No mapped bedrock unit returned.")
        st.caption("Source: Macrostrat v2 · mapped surface unit.")
