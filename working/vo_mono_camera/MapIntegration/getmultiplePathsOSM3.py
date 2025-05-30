import os

import folium
import numpy as np
import openrouteservice

# Initialize ORS client
client = openrouteservice.Client(
    key='5b3ce3597851110001cf6248ab9c3fa94a254ed99f97d8193b32b999'
)

# Place names
origin_name = "Gitam University Bangalore"
destination_name = "Ballari Karnataka"

# Geocode names to coordinates
origin_geocode = client.pelias_search(text=origin_name)
destination_geocode = client.pelias_search(text=destination_name)
# print('origin_geocode=>', origin_geocode)
origin_coords = origin_geocode['features'][0]['geometry']['coordinates']
destination_coords = destination_geocode['features'][0]['geometry']['coordinates']

origin = origin_coords  # [lon, lat]
destination = destination_coords

# Output directory for .npy files
output_dir = "routes_npy"
os.makedirs(output_dir, exist_ok=True)

# Create base map
m = folium.Map(
    location=[(origin[1] + destination[1]) / 2, (origin[0] + destination[0]) / 2],
    zoom_start=7,
    tiles='OpenStreetMap'
)

# Define route preferences
preferences = ['fastest', 'shortest', 'recommended']
colors = ['blue', 'green', 'red']

# Fetch and plot routes
for i, pref in enumerate(preferences):
    try:
        route = client.directions(
            coordinates=[origin, destination],
            profile='driving-car',
            format='geojson',
            preference=pref
        )

        # Decode coordinates and convert to (lat, lon)
        coords = route['features'][0]['geometry']['coordinates']
        latlon_coords = [(lat, lon) for lon, lat in coords]

        # Plot route
        folium.PolyLine(
            latlon_coords,
            color=colors[i % len(colors)],
            weight=5,
            opacity=0.9,
            tooltip=f"Route: {pref}"
        ).add_to(m)

        # Save as .npy (lat, lon, lane_id)
        latlon_coords_with_id = [(lat, lon, i) for lat, lon in latlon_coords]
        np_array = np.array(latlon_coords_with_id)
        np.save(os.path.join(output_dir, f"route_{pref}.npy"), np_array)
        print(f"Saved route '{pref}' as route_{pref}.npy")

        # Extract and print route steps
        steps = route['features'][0]['properties']['segments'][0]['steps']
        for step in steps:
            print(f"{step['instruction']} | Road: {step.get('name', 'N/A')} | Type: {step['type']}")

    except Exception as e:
        print(f"Failed to fetch route for preference '{pref}': {e}")

# Mark origin and destination
folium.Marker([origin[1], origin[0]], popup=origin_name, icon=folium.Icon(color='black')).add_to(m)
folium.Marker([destination[1], destination[0]], popup=destination_name, icon=folium.Icon(color='black')).add_to(m)

# Save map
m.save("clean_routes_map.html")
print("'clean_routes_map.html'")
