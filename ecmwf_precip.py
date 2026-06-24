import os
import requests
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
from datetime import datetime, timedelta

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="shapely")
warnings.filterwarnings("ignore", category=FutureWarning, module="cfgrib")


class ECMWFForecastDownloader:
    def __init__(self, base_url, out_dir="ecmwf_gribs_ifs"):
        self.base_url = base_url
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def download_forecasts(self, hours=range(24, 169, 24)):
        local_files = []
        for h in hours:
            fname = f"{self.base_url}-{h}h-oper-fc.grib2".split("/")[-1]
            fpath = os.path.join(self.out_dir, fname)

            if not os.path.exists(fpath):
                url = f"{self.base_url}-{h}h-oper-fc.grib2"
                print(f"Downloading {url}")
                try:
                    r = requests.get(url, stream=True, timeout=30)
                    if r.status_code == 200:
                        with open(fpath, "wb") as f:
                            f.write(r.content)
                        print(f"✓ Downloaded: {fname}")
                    else:
                        print(f"⚠️ Failed (HTTP {r.status_code}): {url}")
                        continue
                except Exception as e:
                    print(f"⚠️ Download error: {e}")
                    continue
            else:
                print(f"✓ Already exists: {fname}")
            local_files.append(fpath)
        return local_files


class ChadPrecipitationMap:
    def __init__(self, grib_files):
        self.grib_files = grib_files
        self.chad_bbox = [13.0, 25.0, 7.0, 24.0]
        self.chad_cities = {
            "Baga Sola": (13.73, 14.52),
            "Abéché": (13.83, 20.83),
            "Adré": (13.47, 22.20),
            "N'Djamena": (12.11, 15.05),
        }
        
    def compute_daily_precipitation(self):
            """
            Extract *daily* precipitation (not cumulative) for each forecast day
            """
            daily_precip = []
            dates = []
            
            prev_precip = None  # store previous cumulative step
            
            for i, f in enumerate(self.grib_files):
                try:
                    ds = xr.open_dataset(
                        f,
                        engine="cfgrib",
                        filter_by_keys={'typeOfLevel': 'surface'},
                        decode_timedelta=False
                    )
                    
                    if "tp" in ds:
                        # Convert from meters to mm
                        precip_cum = ds["tp"] * 1000   
                        
                        if prev_precip is None:
                            # First step = same as cumulative
                            precip = precip_cum
                        else:
                            # Daily = difference from previous step
                            precip = precip_cum - prev_precip
                        
                        daily_precip.append(precip)
                        prev_precip = precip_cum
        
                        # Generate date for this forecast day
                        date = (datetime.now() + timedelta(days=i+1)).strftime('%Y-%m-%d')
                        dates.append(date)
        
                    ds.close()
                except Exception as e:
                    print(f"⚠️ Could not read {f}: {e}")
                    continue
            
            if not daily_precip:
                raise ValueError("No precipitation data found!")
            
            return daily_precip, dates

# In the plot method, replace the contourf section with pcolormesh:

    def plot(self, precip_sum, daily_precip, dates, out_png="chad_7day_precipitation.png"):
        """
        Plot Chad 7-day accumulated precipitation with discrete colorbar and city markers
        """
        # Create figure with Cartopy projection
        fig = plt.figure(figsize=(14, 10))
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        # Set extent to Chad bounding box
        ax.set_extent([self.chad_bbox[0], self.chad_bbox[1], 
                      self.chad_bbox[2], self.chad_bbox[3]], 
                     crs=ccrs.PlateCarree())
        
        # Add map features
        ax.add_feature(cfeature.COASTLINE, linewidth=1.0)
        ax.add_feature(cfeature.BORDERS, linewidth=1.0)
        ax.add_feature(cfeature.STATES, linewidth=0.5, alpha=0.5)
        ax.add_feature(cfeature.LAKES, alpha=0.5, edgecolor='blue')
        ax.add_feature(cfeature.RIVERS, linewidth=0.5, edgecolor='blue')
        
        # Define discrete color levels
        levels = [0, 10, 20, 30,40, 50, 100, 150]
        cmap = plt.cm.get_cmap('YlGnBu', len(levels) - 1) 
        
        # Use pcolormesh instead of contourf for gridded values
        mesh = ax.pcolormesh(precip_sum.longitude, precip_sum.latitude, precip_sum,
                           cmap='YlGnBu', vmin=levels[0], vmax=levels[-1],
                           transform=ccrs.PlateCarree())
        
        # Add contour lines for better readability
#        contour_lines = ax.contour(precip_sum.longitude, precip_sum.latitude, precip_sum,
#                                 levels=levels, colors='black', linewidths=0.5,
#                                 transform=ccrs.PlateCarree(), alpha=0.7)
#        ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%1.0f mm')
        
        # Add city markers
        for city, (lat, lon) in self.chad_cities.items():
            ax.plot(lon, lat, 'o', markersize=10, color='red', 
                   markeredgecolor='white', markeredgewidth=2, 
                   transform=ccrs.PlateCarree(), zorder=10)
            
            ax.text(lon + 0.25, lat + 0.15, city, transform=ccrs.PlateCarree(),
                   fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", 
                            alpha=0.9, edgecolor='red'))
        
        # Add gridlines
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                         linewidth=0.5, color='gray', alpha=0.3)
        gl.top_labels = False
        gl.right_labels = False
        
        # Add colorbar with discrete levels
        cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal',
                   pad=0.05, shrink=0.8, aspect=30,
                   boundaries=levels, ticks=levels)
        cbar.set_label('7-Day Accumulated Precipitation (mm)', fontsize=12, fontweight='bold')
        
        # Add title
        start_date = dates[0] if dates else datetime.now().strftime('%Y-%m-%d')
        end_date = dates[-1] if dates else (datetime.now() + timedelta(days=6)).strftime('%Y-%m-%d')
        plt.title(f'Chad - 7-Day Accumulated Precipitation\n{start_date} to {end_date}\n(ECMWF Forecast)', 
                 fontsize=16, fontweight='bold', pad=20)
        
        # Add data source
        ax.text(13.2, 7.5, 'Data: ECMWF', fontsize=9,
                bbox=dict(facecolor='white', alpha=0.8),
                transform=ccrs.PlateCarree())
        
        plt.tight_layout()
        plt.savefig(out_png, dpi=300, bbox_inches='tight')
        plt.show()
        
        return precip_sum

    def create_precipitation_table(self, daily_precip, dates):
        """
        Create a table with daily precipitation for each city
        """
        table_data = []
        
        for city, (lat, lon) in self.chad_cities.items():
            city_data = [city]
            total_precip = 0
            
            for i, precip_data in enumerate(daily_precip):
                try:
                    # Get precipitation for this city and day
                    precip_value = float(precip_data.sel(latitude=lat, longitude=lon, method='nearest'))
                    city_data.append(precip_value)
                    total_precip += precip_value
                except Exception as e:
                    print(f"⚠️ Could not get precipitation for {city} day {i+1}: {e}")
                    city_data.append(np.nan)
            
            city_data.append(total_precip)
            table_data.append(city_data)
        
        # Create DataFrame
        columns = ['City'] + [f'{date[5:]}\n(mm)' for i, date in enumerate(dates)] + ['Total\n(mm)']
        df = pd.DataFrame(table_data, columns=columns)

        for col in df.columns[1:]:  # Skip the 'City' column
            df[col] = df[col].apply(lambda x: f"{x:.1f}" if isinstance(x, (int, float)) and not np.isnan(x) else "N/A")

        
        
        # Create table visualization
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.axis('tight')
        ax.axis('off')
        
        numeric_values = []
        for row in table_data:
            numeric_values.append([row[0]] + [float(x) if not np.isnan(x) else 0 for x in row[1:]])
        
        numeric_df = pd.DataFrame(numeric_values, columns=columns)
        
        table = ax.table(cellText=df.values,  # Use formatted values for display
                        colLabels=df.columns,
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.12] + [0.1] * len(dates) + [0.12])
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 2.2)
        
        # Color code precipitation values (skip city and total columns)
        for i in range(len(numeric_df)):
            for j in range(1, len(numeric_df.columns) - 1):
                cell = table[(i+1, j)]
                value = numeric_df.iloc[i, j]
                if isinstance(value, (int, float)) and not np.isnan(value):
                    if value > 15:
                        cell.set_facecolor('#ff6666')
                    elif value > 10:
                        cell.set_facecolor('#ffcc66')
                    elif value > 5:
                        cell.set_facecolor('#ccff66')
                    elif value > 0:
                        cell.set_facecolor('#66ff66')
                    else:
                        cell.set_facecolor('#f0f0f0')
        
        # Style header row
        for j in range(len(df.columns)):
            table[(0, j)].set_facecolor('#2E5C8A')
            table[(0, j)].set_text_props(weight='bold', color='white', size=12)
        
        plt.title('Chad - Daily Precipitation Forecast by City (mm)\nECMWF Forecast', 
                 fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig('chad_city_precipitation_table.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary
        print("\n" + "="*100)
        print("CHAD DAILY PRECIPITATION FORECAST SUMMARY (mm)")
        print("="*100)
        print(f"{'City':<12} | {' | '.join([f'Day {i+1}:{date[5:]:>6}' for i, date in enumerate(dates)])} | {'Total':>8}")
        print("-" * 100)
        
        for row in table_data:
            city, *daily, total = row
            daily_str = " | ".join([f"{p:8.1f}" if not np.isnan(p) else "     N/A" for p in daily])
            print(f"{city:<12} | {daily_str} | {total:8.1f}")


# Sample data generator for testing
class SampleDataGenerator:
    def __init__(self):
        self.chad_bbox = [13.0, 24.0, 7.0, 23.0]
        self.chad_cities = {
            "Baga Sola": (13.73, 14.52),
            "Abéché": (13.83, 20.83),
            "Adré": (13.47, 22.20),
            "N'Djamena": (12.11, 15.05),
        }
        
    def create_sample_data(self, num_days=7):
        """
        Create sample daily precipitation data
        """
        # Create grid
        lons = np.linspace(self.chad_bbox[0], self.chad_bbox[1], 50)
        lats = np.linspace(self.chad_bbox[2], self.chad_bbox[3], 50)
        
        daily_precip = []
        dates = []
        
        for day in range(num_days):
            # Create daily precipitation pattern
            precip = np.zeros((len(lats), len(lons)))
            
            for i in range(len(lats)):
                for j in range(len(lons)):
                    lat, lon = lats[i], lons[j]
                    base = np.random.uniform(0, 2)
                    if lat < 15.0:  # Southern Chad
                        base += np.random.uniform(1, 8)
                    elif lat < 18.0:  # Central Chad
                        base += np.random.uniform(0, 4)
                    else:  # Northern Chad
                        base += np.random.uniform(0, 2)
                    precip[i, j] = max(0, base)
            
            # Create DataArray
            precip_da = xr.DataArray(
                precip,
                coords={'latitude': lats, 'longitude': lons},
                dims=['latitude', 'longitude'],
                attrs={'units': 'mm', 'long_name': f'Day {day+1} precipitation'}
            )
            
            daily_precip.append(precip_da)
            
            # Generate date
            date = (datetime.now() + timedelta(days=day)).strftime('%Y-%m-%d')
            dates.append(date)
        
        # Calculate total precipitation
        total_precip = sum(daily_precip)
        
        return daily_precip, total_precip, dates


# ---------------------------
# Main execution
# ---------------------------
if __name__ == "__main__":
    # Try to download ECMWF data
    try:
        base_url = "https://data.ecmwf.int/forecasts/20260624/00z/ifs/0p25/oper/20260624000000"
        downloader = ECMWFForecastDownloader(base_url)
        grib_files = downloader.download_forecasts()
        
        # Process data
        chad_map = ChadPrecipitationMap(grib_files)
        daily_precip, dates = chad_map.compute_daily_precipitation()
        total_precip = sum(daily_precip)
        
    except Exception as e:
        print(f"⚠️ ECMWF data unavailable: {e}")
        print("Using sample data for demonstration...")
        
        # Generate sample data
        sample_gen = SampleDataGenerator()
        daily_precip, total_precip, dates = sample_gen.create_sample_data()
        chad_map = ChadPrecipitationMap([])  # Empty file list
    
    # Plot the precipitation map with cities
    chad_map.plot(total_precip, daily_precip, dates, "chad_7day_precipitation.png")
    
    # Create precipitation table
    chad_map.create_precipitation_table(daily_precip, dates)
    
    print("\nAnalysis completed! Files saved:")
    print("1. 'chad_7day_precipitation.png' - Precipitation map with cities")
    print("2. 'chad_city_precipitation_table.png' - Daily precipitation table")
