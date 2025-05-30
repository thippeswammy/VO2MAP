import os
from functools import partial

import folium
import numpy as np
import openrouteservice
import pyproj
from shapely.geometry import LineString, Point
from shapely.ops import transform

# Initialize ORS client
client = openrouteservice.Client(
    key='5b3ce3597851110001cf6248ab9c3fa94a254ed99f97d8193b32b999'
)

# Place names
origin_name = "Gitam University Bangalore"
destination_name = "Ballari Karnataka"

# Geocode names to coordinates
try:
    origin_geocode = client.pelias_search(text=origin_name)
    destination_geocode = client.pelias_search(text=destination_name)
    print('origin_geocode[]=>', origin_geocode['features'])
    print('origin_geocode=>', origin_geocode)
    origin_coords = origin_geocode['features'][0]['geometry']['coordinates']
    destination_coords = destination_geocode['features'][0]['geometry']['coordinates']

    origin = origin_coords  # [lon, lat]
    destination = destination_coords
except Exception as e:
    print(f"❌ Geocoding failed: {e}")
    exit()

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

# Extra info to request
extra_info = ['waytype', 'surface', 'waycategory', 'steepness', 'tollways', 'wayid']


# Function to create buffer polygon (in meters)
def create_buffer_polygon(point, radius=100, resolution=16):
    try:
        project = partial(
            pyproj.transform,
            pyproj.Proj('EPSG:4326'),  # WGS84
            pyproj.Proj('EPSG:3857')  # Mercator for distance calculations
        )
        point_mercator = transform(project, Point(point))
        buffer_mercator = point_mercator.buffer(radius, resolution=resolution)
        project_back = partial(
            pyproj.transform,
            pyproj.Proj('EPSG:3857'),
            pyproj.Proj('EPSG:4326')
        )
        buffer_wgs84 = transform(project_back, buffer_mercator)
        return buffer_wgs84
    except Exception as e:
        print(f"Failed to create buffer polygon: {e}")
        return None


# Fetch POIs (schools) along the route
def get_school_zones(route_coords, buffer_distance=500):
    try:
        route_line = LineString(route_coords)
        route_buffer = create_buffer_polygon(route_line, radius=buffer_distance)
        if route_buffer is None:
            return []
        bbox = route_buffer.bounds  # [minx, miny, maxx, maxy]
        pois = client.places(
            request='pois',
            geojson={'type': 'Polygon', 'coordinates': [list(route_buffer.exterior.coords)]},
            filters={'category_ids': [201]}  # 201 is the OSM category for schools
        )
        return pois.get('features', [])
    except Exception as e:
        print(f"Failed to fetch POIs: {e}")
        return []


# Fetch and plot routes
for i, pref in enumerate(preferences):
    try:
        route = client.directions(
            coordinates=[origin, destination],
            profile='driving-car',
            format='geojson',
            preference=pref,
            extra_info=extra_info,
            instructions=True
        )

        # Process the route
        coords = route['features'][0]['geometry']['coordinates']
        latlon_coords = [(lat, lon) for lon, lat in coords]
        route_name = pref

        # Plot route
        folium.PolyLine(
            latlon_coords,
            color=colors[i % len(colors)],
            weight=5,
            opacity=0.9,
            tooltip=f"Route: {route_name}"
        ).add_to(m)

        # Save route as .npy (lat, lon, route_id)
        latlon_coords_with_id = [(lat, lon, i) for lat, lon in latlon_coords]
        np_array = np.array(latlon_coords_with_id, dtype=np.float64)
        np.save(os.path.join(output_dir, f"route_{route_name}.npy"), np_array)
        print(f"Saved route '{route_name}' as route_{route_name}.npy")

        # Extract and print route steps with extra info
        steps = route['features'][0]['properties']['segments'][0]['steps']
        extras = route['features'][0]['properties'].get('extras', {})
        for step in steps:
            way_points = step['way_points']
            print(f"Instruction: {step['instruction']} | Road: {step.get('name', 'N/A')} | Type: {step['type']}")
            # Extract extra info for the segment
            for extra in extra_info:
                if extra in extras:
                    values = extras[extra]['values']
                    for wp_range, value in values:
                        if wp_range[0] <= way_points[0] <= wp_range[1]:
                            print(f"  {extra}: {value}")
                            if extra == 'wayid':
                                print(f"    (Use wayid {value} to query OSM for speed limits or other tags)")

        # Get school zones
        school_pois = get_school_zones(coords, buffer_distance=500)
        for poi_idx, poi in enumerate(school_pois):
            poi_coords = poi['geometry']['coordinates']
            school_buffer = create_buffer_polygon(Point(poi_coords), radius=100)  # 100m buffer
            if school_buffer is None:
                continue
            school_coords = list(school_buffer.exterior.coords)
            # Save school zone polygon as .npy
            school_array = np.array([(lat, lon, i) for lon, lat in school_coords], dtype=np.float64)
            np.save(os.path.join(output_dir, f"school_zone_{route_name}_{poi_idx}.npy"), school_array)
            print(f"Saved school zone polygon for '{route_name}' as school_zone_{route_name}_{poi_idx}.npy")
            # Plot school zone
            folium.Polygon(
                locations=[(lat, lon) for lon, lat in school_coords],
                color='purple',
                fill=True,
                fill_opacity=0.3,
                tooltip=f"School Zone {poi_idx}"
            ).add_to(m)

        # Extract intersections (approximate four-way intersections)
        for step in steps:
            intersections = step.get('intersections', [])
            for inter_idx, inter in enumerate(intersections):
                bearings = inter.get('bearings', [])
                if len(bearings) >= 4:  # Approximate four-way intersection
                    inter_coords = inter['location']
                    inter_buffer = create_buffer_polygon(Point(inter_coords), radius=50)  # 50m buffer
                    if inter_buffer is None:
                        continue
                    inter_coords_list = list(inter_buffer.exterior.coords)
                    # Save intersection polygon as .npy
                    inter_array = np.array([(lat, lon, i) for lon, lat in inter_coords_list], dtype=np.float64)
                    np.save(os.path.join(output_dir, f"intersection_{route_name}_{inter_idx}.npy"), inter_array)
                    print(
                        f"Saved intersection polygon for '{route_name}' as intersection_{route_name}_{inter_idx}.npy")
                    # Plot intersection
                    folium.Polygon(
                        locations=[(lat, lon) for lon, lat in inter_coords_list],
                        color='orange',
                        fill=True,
                        fill_opacity=0.3,
                        tooltip=f"Intersection {inter_idx} (Bearings: {len(bearings)})"
                    ).add_to(m)

    except Exception as e:
        print(f"Failed to fetch route for preference '{pref}': {e}")
        continue

# Mark origin and destination
folium.Marker([origin[1], origin[0]], popup=origin_name, icon=folium.Icon(color='black')).add_to(m)
folium.Marker([destination[1], destination[0]], popup=destination_name, icon=folium.Icon(color='black')).add_to(m)

# Save map
m.save("enhanced_routes_map.html")
print("Enhanced route map saved as 'enhanced_routes_map.html'")
