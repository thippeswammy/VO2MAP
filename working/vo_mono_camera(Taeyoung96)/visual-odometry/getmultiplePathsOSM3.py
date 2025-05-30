import os

import folium
import numpy as np
import openrouteservice

# Initialize ORS client
client = openrouteservice.Client(
    key='5b3ce3597851110001cf6248ab9c3fa94a254ed99f97d8193b32b999'
)

# Define origin and destination
origin_name = "Sir M Visvesvaraya Institute of Technology"
destination_name = "Gitam University Bangalore"

# Geocode destination
destination_geocode = client.pelias_search(text=destination_name)
destination_coords = destination_geocode['features'][0]['geometry']['coordinates']  # [lon, lat]

# Hardcoded origin (correctly in [lon, lat] format)
origin = [77.623190, 13.279433]
destination = destination_coords

# Output directory
output_dir = "routes_npy"
os.makedirs(output_dir, exist_ok=True)

# Create map centered between origin and destination
m = folium.Map(
    location=[(origin[1] + destination[1]) / 2, (origin[0] + destination[0]) / 2],
    zoom_start=10,
    tiles='OpenStreetMap'
)

# Route preferences
preferences = ['fastest', 'shortest', 'recommended']
colors = ['blue', 'green', 'red']

# Fetch & save routes
for i, pref in enumerate(preferences):
    try:
        route = client.directions(
            coordinates=[origin, destination],
            profile='driving-car',
            format='geojson',
            preference=pref
        )

        coords = route['features'][0]['geometry']['coordinates']
        latlon_coords = [(lat, lon) for lon, lat in coords]  # Flip for folium

        # Plot
        folium.PolyLine(
            latlon_coords,
            color=colors[i % len(colors)],
            weight=5,
            opacity=0.9,
            tooltip=f"Route: {pref}"
        ).add_to(m)

        # Save to .npy
        latlon_coords_with_id = [(lat, lon, i) for lat, lon in latlon_coords]
        np_array = np.array(latlon_coords_with_id)
        np.save(os.path.join(output_dir, f"route_{pref}.npy"), np_array)
        print(f"Saved route '{pref}' as route_{pref}.npy")

        # Route steps
        steps = route['features'][0]['properties']['segments'][0]['steps']
        for step in steps:
            print(f"{step['instruction']} | Road: {step.get('name', 'N/A')} | Type: {step['type']}")

    except Exception as e:
        print(f"Failed to fetch route for preference '{pref}': {e}")

# Markers
folium.Marker([origin[1], origin[0]], popup=origin_name, icon=folium.Icon(color='black')).add_to(m)
folium.Marker([destination[1], destination[0]], popup=destination_name, icon=folium.Icon(color='black')).add_to(m)

# Save map
m.save("clean_routes_map.html")
print("Map saved as 'clean_routes_map.html'")
