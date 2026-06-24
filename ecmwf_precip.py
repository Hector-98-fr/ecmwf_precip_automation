import os
import requests
import xarray as xr
import numpy as np
import pandas as pd

from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import tempfile

# =====================================================
# CONFIG
# =====================================================

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COUNTRIES = {
    "Cameroon": {
        "bbox": (8, 17, 1, 14),
        "cities": {
            "Maroua": (10.59, 14.32),
            "Garoua": (9.30, 13.39),
            "Ngaoundere": (7.32, 13.58),
            "Yaounde": (3.87, 11.52),
            "Douala": (4.05, 9.70)
        }
    },
    "Chad": {
        "bbox": (13, 24, 7, 24),
        "cities": {
            "N'Djamena": (12.11, 15.05),
            "Moundou": (8.56, 16.08),
            "Sarh": (9.15, 18.38)
        }
    }
}

today = datetime.utcnow()

base_url = (
    f"https://data.ecmwf.int/forecasts/"
    f"{today:%Y%m%d}/00z/ifs/0p25/oper/"
    f"{today:%Y%m%d}000000"
)

# =====================================================
# DOWNLOAD + IMMEDIATE CLEANUP
# =====================================================

def fetch_and_extract_subset(h, bbox):
    """
    Download GRIB, extract bbox, delete file immediately.
    """

    url = f"{base_url}-{h}h-oper-fc.grib2"

    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tmp:
        tmp_path = tmp.name

    r = requests.get(url, timeout=120)

    if r.status_code != 200:
        print("Missing:", h)
        return None

    with open(tmp_path, "wb") as f:
        f.write(r.content)

    try:
        ds = xr.open_dataset(
            tmp_path,
            engine="cfgrib",
            filter_by_keys={"typeOfLevel": "surface"},
        )

        # Crop immediately → HUGE memory reduction
        min_lon, max_lon, min_lat, max_lat = bbox

        ds = ds.sel(
            longitude=slice(min_lon, max_lon),
            latitude=slice(max_lat, min_lat)  # reversed lat order in ECMWF
        )

        tp = ds["tp"] * 1000  # m → mm

        ds.close()

    finally:
        os.remove(tmp_path)

    return tp

# =====================================================
# PROCESS FORECASTS
# =====================================================

def process_country(name, config):

    bbox = config["bbox"]
    cities = config["cities"]

    daily = []
    dates = []

    prev = None

    print(f"\nProcessing {name}")

    for i, h in enumerate(range(24, 169, 24)):

        print("Hour:", h)

        precip = fetch_and_extract_subset(h, bbox)

        if precip is None:
            continue

        # convert accumulation → incremental
        if prev is None:
            daily_precip = precip
        else:
            daily_precip = precip - prev

        prev = precip
        daily.append(daily_precip)

        dates.append(
            (datetime.utcnow() + timedelta(days=i+1)).strftime("%d-%b")
        )

    total = sum(daily)

    create_map(total, dates, cities, name)
    df = create_table(daily, dates, cities, name)

    print(df)
    print(f"{name} done.")

# =====================================================
# MAP
# =====================================================

def create_map(total_precip, dates, cities, name):

    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.set_extent([
        float(total_precip.longitude.min()),
        float(total_precip.longitude.max()),
        float(total_precip.latitude.min()),
        float(total_precip.latitude.max())
    ])

    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.COASTLINE)

    mesh = ax.pcolormesh(
        total_precip.longitude,
        total_precip.latitude,
        total_precip,
        cmap="YlGnBu",
        transform=ccrs.PlateCarree()
    )

    for city, (lat, lon) in cities.items():
        ax.plot(lon, lat, "ro", transform=ccrs.PlateCarree())
        ax.text(lon + 0.2, lat + 0.2, city, fontsize=8)

    plt.colorbar(mesh, label="7-Day Total Rainfall (mm)")

    plt.title(f"{name} ECMWF 7-Day Rainfall\n{dates[0]} to {dates[-1]}")

    plt.savefig(
        os.path.join(OUTPUT_DIR, f"{name.lower()}_7day_precip.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# =====================================================
# TABLE
# =====================================================

def create_table(daily_precip, dates, cities, name):

    rows = []

    for city, (lat, lon) in cities.items():

        values = []
        total = 0

        for p in daily_precip:

            v = float(p.sel(latitude=lat, longitude=lon, method="nearest"))
            v = round(v, 1)

            values.append(v)
            total += v

        rows.append([city] + values + [round(total, 1)])

    df = pd.DataFrame(
        rows,
        columns=["City"] + dates + ["7DayTotal"]
    )

    df.to_csv(
        os.path.join(OUTPUT_DIR, f"{name.lower()}_precip_table.csv"),
        index=False
    )

    return df

# =====================================================
# MAIN
# =====================================================

def main():

    for name, config in COUNTRIES.items():
        process_country(name, config)

if __name__ == "__main__":
    main()
