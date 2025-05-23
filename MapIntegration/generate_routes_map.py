import folium
import openrouteservice

# Initialize ORS client
client = openrouteservice.Client(
    key='5b3ce3597851110001cf6248ab9c3fa94a254ed99f97d8193b32b999')

# Define coordinates: (longitude, latitude)
origin = [77.4964, 12.8356]  # Gitam University, Bangalore
destination = [76.9326, 15.1394]  # Ballari, Karnataka

# Define different routing preferences
preferences = ['fastest', 'shortest', 'recommended']
colors = ['blue', 'green', 'red']
routes_geojson = []

# Create base map
m = folium.Map(location=[(origin[1] + destination[1]) / 2, (origin[0] + destination[0]) / 2], zoom_start=7)

for i, pref in enumerate(preferences):
    route = client.directions(
        coordinates=[origin, destination],
        profile='driving-car',
        format='geojson',
        preference=pref
    )

    # Save geojson for plotting
    routes_geojson.append(route)

    coords = route['features'][0]['geometry']['coordinates']
    coords = [(lat, lon) for lon, lat in coords]

    folium.PolyLine(
        coords,
        color=colors[i % len(colors)],
        weight=5,
        opacity=0.8,
        tooltip=f"Preference: {pref}"
    ).add_to(m)

# Markers
folium.Marker([origin[1], origin[0]], popup="Origin").add_to(m)
folium.Marker([destination[1], destination[0]], popup="Destination").add_to(m)

# Save map
m.save("ors_multiple_route_preferences.html")
print("Map saved as 'ors_multiple_route_preferences.html'")
