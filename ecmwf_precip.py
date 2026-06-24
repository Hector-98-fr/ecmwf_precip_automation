import os
import requests
import xarray as xr
import numpy as np
import pandas as pd

from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# =====================================================
# CONFIG
# =====================================================

OUTPUT_DIR = "outputs"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# Cameroon extent

BBOX = [
    8,
    17,
    1,
    14
]

CITIES = {
    "Maroua": (10.59, 14.32),
    "Garoua": (9.30, 13.39),
    "Ngaoundere": (7.32, 13.58),
    "Yaounde": (3.87, 11.52),
    "Douala": (4.05, 9.70)
}

# =====================================================
# ECMWF URL
# =====================================================

today = datetime.utcnow()

base_url = (
    f"https://data.ecmwf.int/forecasts/"
    f"{today:%Y%m%d}/00z/ifs/0p25/oper/"
    f"{today:%Y%m%d}000000"
)

# =====================================================
# DOWNLOAD
# =====================================================

def download_forecasts():

    files = []

    for h in range(24,169,24):

        url = f"{base_url}-{h}h-oper-fc.grib2"

        local_file = os.path.join(
            OUTPUT_DIR,
            f"day_{h}.grib2"
        )

        print("Downloading", url)

        r = requests.get(
            url,
            timeout=60
        )

        if r.status_code == 200:

            with open(local_file,"wb") as f:
                f.write(r.content)

            files.append(local_file)

        else:

            print(
                "Missing forecast:",
                h
            )

    return files

# =====================================================
# PRECIP EXTRACTION
# =====================================================

def compute_daily_precip(files):

    daily = []

    previous = None

    dates = []

    for i,file in enumerate(files):

        ds = xr.open_dataset(
            file,
            engine="cfgrib",
            filter_by_keys={
                "typeOfLevel":"surface"
            },
            decode_timedelta=False
        )

        precip = ds["tp"] * 1000

        if previous is None:

            daily_precip = precip

        else:

            daily_precip = precip - previous

        previous = precip

        daily.append(
            daily_precip
        )

        dates.append(
            (
                datetime.utcnow()
                + timedelta(days=i+1)
            ).strftime("%d-%b")
        )

        ds.close()

    return daily, dates

# =====================================================
# MAP
# =====================================================

def create_map(
    total_precip,
    dates
):

    fig = plt.figure(
        figsize=(12,8)
    )

    ax = plt.axes(
        projection=ccrs.PlateCarree()
    )

    ax.set_extent(
        BBOX
    )

    ax.add_feature(
        cfeature.BORDERS
    )

    ax.add_feature(
        cfeature.COASTLINE
    )

    mesh = ax.pcolormesh(
        total_precip.longitude,
        total_precip.latitude,
        total_precip,
        cmap="YlGnBu",
        transform=ccrs.PlateCarree()
    )

    for city,(lat,lon) in CITIES.items():

        ax.plot(
            lon,
            lat,
            "ro",
            transform=ccrs.PlateCarree()
        )

        ax.text(
            lon+0.15,
            lat+0.15,
            city,
            fontsize=9,
            transform=ccrs.PlateCarree()
        )

    plt.colorbar(
        mesh,
        label="7-Day Total Rainfall (mm)"
    )

    plt.title(
        f"Cameroon ECMWF 7-Day Rainfall\n"
        f"{dates[0]} to {dates[-1]}"
    )

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "cameroon_7day_precipitation.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# =====================================================
# TABLE
# =====================================================

def create_table(
    daily_precip,
    dates
):

    rows = []

    for city,(lat,lon) in CITIES.items():

        city_values = []

        total = 0

        for precip in daily_precip:

            value = float(
                precip.sel(
                    latitude=lat,
                    longitude=lon,
                    method="nearest"
                )
            )

            value = round(
                value,
                1
            )

            city_values.append(
                value
            )

            total += value

        rows.append(
            [city]
            + city_values
            + [round(total,1)]
        )

    columns = (
        ["City"]
        + dates
        + ["7DayTotal"]
    )

    df = pd.DataFrame(
        rows,
        columns=columns
    )

    df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "cameroon_precipitation_table.csv"
        ),
        index=False
    )

    return df

# =====================================================
# MAIN
# =====================================================

def main():

    print(
        "Downloading ECMWF forecast..."
    )

    files = download_forecasts()

    daily_precip, dates = (
        compute_daily_precip(files)
    )

    total_precip = sum(
        daily_precip
    )

    create_map(
        total_precip,
        dates
    )

    df = create_table(
        daily_precip,
        dates
    )

    print(df)

    print(
        "Finished successfully."
    )

if __name__ == "__main__":
    main()
