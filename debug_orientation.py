import pandas as pd
import numpy as np
import os

# CONFIG
LOCS_PATH = "D:\\SEED-V\\channel_62_pos.locs" # Adjust if filename differs
OUTPUT_PATH = "D:\\SEED-V\\coords.csv"

def generate_coords():
    print(f"📍 Parsing {LOCS_PATH}...")
    
    # 1. Read
    # SEED locs usually: [ID] [Theta] [Radius] [Label]
    try:
        df = pd.read_csv(LOCS_PATH, sep=r'\s+', header=None, engine='python')
    except:
        print("Error reading file. Check path.")
        return

    # 2. Extract Columns
    # Assuming standard format based on your plot results
    # Col 1: Theta (Degrees), Col 2: Radius, Col 3: Label
    theta_deg = df.iloc[:, 1].values
    radius = df.iloc[:, 2].values
    labels = df.iloc[:, 3].values

    # 3. Convert to Radians
    theta_rad = np.deg2rad(theta_deg)

    # 4. Polar to Cartesian (The "Sideways" Map)
    x_raw = radius * np.cos(theta_rad)
    y_raw = radius * np.sin(theta_rad)

    # 5. THE FIX: Swap for Nose-Up
    # New X = Old Y
    # New Y = Old X
    x_final = y_raw
    y_final = x_raw

    # 6. Convert back to Polar (Theta/Radius) for TopoMapper
    # Our TopoMapper expects 'theta' and 'radius' columns
    # We recalculate them from the CORRECTED Cartesian coords
    
    radius_final = np.sqrt(x_final**2 + y_final**2)
    theta_final_rad = np.arctan2(y_final, x_final)
    theta_final_deg = np.degrees(theta_final_rad)

    # 7. Save
    out_df = pd.DataFrame({
        'labels': labels,
        'theta': theta_final_deg,
        'radius': radius_final
    })
    
    # Optional: Normalize radius if it isn't 0-1
    if out_df['radius'].max() > 1.0:
        print("   Normalizing radius to 0-1...")
        out_df['radius'] = out_df['radius'] / out_df['radius'].max()

    out_df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Saved corrected coordinates to {OUTPUT_PATH}")
    print("   (Nose is now UP, Left is LEFT)")

if __name__ == "__main__":
    generate_coords()