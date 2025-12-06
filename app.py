import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import folium
from math import isnan

import streamlit as st
from streamlit.components.v1 import html

from scipy.signal import butter, filtfilt, find_peaks, welch

# Asetukset
ACC_PATH = "data/Accelerometer.csv"
LOC_PATH = "data/Location.csv"

t_cut = 20.0  # sekuntia alusta pois


# Apufunktiot
def haversine(lat1, lon1, lat2, lon2):
    R = 6371e3  # m
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


@st.cache_data
def load_and_process():
    acc = pd.read_csv(ACC_PATH, comment="#", sep=",")
    loc = pd.read_csv(LOC_PATH, comment="#", sep=",")

    acc_time_col = "Time (s)"
    loc_time_col = "Time (s)"

    acc["t"] = acc[acc_time_col] - acc[acc_time_col].iloc[0]
    loc["t"] = loc[loc_time_col] - loc[loc_time_col].iloc[0]

    # Trimmaus
    acc = acc[acc["t"] >= t_cut].reset_index(drop=True)
    loc = loc[loc["t"] >= t_cut].reset_index(drop=True)

    # Valitaan komponentti
    ax_col = "Linear Acceleration x (m/s^2)"
    ay_col = "Linear Acceleration y (m/s^2)"
    az_col = "Linear Acceleration z (m/s^2)"

    selected_axis = az_col
    acc["a_sel"] = acc[selected_axis]

    # Näytteenottotaajuus
    dt = acc["t"].diff().median()
    fs = 1.0 / dt

    # Suodatus
    low = 0.5 / (fs / 2.0)
    high = 5.0 / (fs / 2.0)
    order = 4
    b, a = butter(order, [low, high], btype="band")
    acc["a_filt"] = filtfilt(b, a, acc["a_sel"])

    # Askelmäärä
    peak_height = np.std(acc["a_filt"]) * 0.5
    min_distance_seconds = 0.3
    min_distance_samples = int(min_distance_seconds * fs)

    peaks, properties = find_peaks(
        acc["a_filt"],
        height=peak_height,
        distance=min_distance_samples,
    )
    step_count_time = len(peaks)

    # Tehospektri
    x = acc["a_filt"].to_numpy()
    f, Pxx = welch(x, fs=fs, nperseg=min(2048, len(x)))

    freq_min = 0.5
    freq_max = 5.0
    mask = (f >= freq_min) & (f <= freq_max)
    f_region = f[mask]
    P_region = Pxx[mask]

    dominant_idx = np.argmax(P_region)
    dominant_freq = f_region[dominant_idx]
    duration = acc["t"].iloc[-1] - acc["t"].iloc[0]
    step_count_freq = dominant_freq * duration

    # Reitti kartalle / GPS
    lat_col = "Latitude (°)"
    lon_col = "Longitude (°)"
    loc["lat"] = loc[lat_col]
    loc["lon"] = loc[lon_col]

    distances = haversine(
        loc["lat"].values[:-1],
        loc["lon"].values[:-1],
        loc["lat"].values[1:],
        loc["lon"].values[1:],
    )
    total_distance = float(np.sum(distances))

    gps_duration = loc["t"].iloc[-1] - loc["t"].iloc[0]
    mean_speed = total_distance / gps_duration
    mean_speed_kmh = mean_speed * 3.6

    step_length = total_distance / step_count_time if step_count_time > 0 else np.nan

    return {
        "acc": acc,
        "loc": loc,
        "fs": fs,
        "selected_axis": selected_axis,
        "peaks": peaks,
        "f": f,
        "Pxx": Pxx,
        "step_count_time": step_count_time,
        "step_count_freq": step_count_freq,
        "total_distance": total_distance,
        "mean_speed": mean_speed,
        "mean_speed_kmh": mean_speed_kmh,
        "step_length": step_length,
        "dominant_freq": dominant_freq,
        "duration": duration,
    }

# Streamlit
st.title("Kiihtyvyys- ja GPS-analyysi")

data = load_and_process()
acc = data["acc"]
loc = data["loc"]

st.header("Yhteenveto")

summary = pd.DataFrame(
    {
        "Suure": [
            "Askelmäärä (piikkien perusteella)",
            "Askelmäärä (Fourier / dominoiva taajuus)",
            "Kuljettu matka [m]",
            "Keskinopeus [m/s]",
            "Keskinopeus [km/h]",
            "Askelpituus [m/askel]",
            "Dominoiva taajuus [Hz]",
            "Mittauksen kesto [s]",
        ],
        "Arvo": [
            data["step_count_time"],
            data["step_count_freq"],
            data["total_distance"],
            data["mean_speed"],
            data["mean_speed_kmh"],
            data["step_length"],
            data["dominant_freq"],
            data["duration"],
        ],
    }
)

summary["Arvo"] = summary["Arvo"].astype(float).round(2)

st.dataframe(summary, use_container_width=True)


# Suodatettu kiihtyvyys ja piikit
st.header("Suodatettu kiihtyvyys ja askelien piikit")

fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(acc["t"], acc["a_filt"], label="Suodatettu kiihtyvyys")
ax1.plot(acc["t"].iloc[data["peaks"]], acc["a_filt"].iloc[data["peaks"]], "x", label="Piikit")
ax1.set_xlabel("Aika [s]")
ax1.set_ylabel("Kiihtyvyys [m/s²]")
ax1.set_title(f"Suodatettu kiihtyvyyden z-komponentti")
ax1.grid(True)
ax1.legend()
fig1.tight_layout()
st.pyplot(fig1)


# Tehospektri
st.header("Tehospektri")

fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(data["f"], data["Pxx"])
ax2.set_xlim(0, 14)
ax2.set_xlabel("Taajuus [Hz]")
ax2.set_ylabel("Teho")
ax2.set_title("Tehospektri")
ax2.grid(True)
fig2.tight_layout()
st.pyplot(fig2)


#Reitti kartalle
st.header("Reitti kartalla")

center_lat = loc["Latitude (°)"].mean()
center_lon = loc["Longitude (°)"].mean()

m = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles="OpenStreetMap")

route_coords = list(zip(loc["Latitude (°)"], loc["Longitude (°)"]))
folium.PolyLine(route_coords, color="red", weight=4, opacity=0.9).add_to(m)
folium.Marker(route_coords[0], tooltip="Lähtö").add_to(m)
folium.Marker(route_coords[-1], tooltip="Loppu").add_to(m)

html(m._repr_html_(), height=500)
