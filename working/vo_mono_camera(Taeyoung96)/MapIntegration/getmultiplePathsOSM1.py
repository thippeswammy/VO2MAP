import folium
import openrouteservice

# Initialize ORS client
client = openrouteservice.Client(
    key='5b3ce3597851110001cf6248ab9c3fa94a254ed99f97d8193b32b999'  # Replace with your valid ORS key
)

# Define coordinates (longitude, latitude)
origin = [77.4964, 12.8356]  # Gitam University, Bangalore
destination = [76.9326, 15.1394]  # Ballari, Karnataka

# Define routing preferences to simulate different route options
preferences = ['fastest', 'shortest', 'recommended']
colors = ['blue', 'green', 'red']

# Create a clean base map
m = folium.Map(
    location=[(origin[1] + destination[1]) / 2, (origin[0] + destination[0]) / 2],
    zoom_start=7,
    tiles='OpenStreetMap'  # Basic tile, only needed to draw the road
)

# Add routes
for i, pref in enumerate(preferences):
    route = client.directions(
        coordinates=[origin, destination],
        profile='driving-car',
        format='geojson',
        preference=pref
    )

    coords = route['features'][0]['geometry']['coordinates']
    coords = [(lat, lon) for lon, lat in coords]  # convert (lon, lat) → (lat, lon)

    folium.PolyLine(
        coords,
        color=colors[i % len(colors)],
        weight=5,
        opacity=0.9
    ).add_to(m)

# Add minimal origin and destination markers
folium.CircleMarker([origin[1], origin[0]], radius=5, color='black', fill=True).add_to(m)
folium.CircleMarker([destination[1], destination[0]], radius=5, color='black', fill=True).add_to(m)

# Save map
m.save("clean_routes_map.html")
print("✅ Clean route map saved as 'clean_routes_map.html'")
