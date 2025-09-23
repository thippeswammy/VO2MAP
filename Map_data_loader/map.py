import os

import folium
import numpy as np
import openrouteservice
import requests

# ==============================
# ORS + OSM Map Feature Extractor
# ==============================

# ORS client (replace with your key)
client = openrouteservice.Client(
    key="5b3ce3597851110001cf6248ab9c3fa94a254ed99f97d8193b32b999"
)


def fetch_routes(origin, destination, output_dir="routes_npy"):
    """
    Fetch routes (fastest, shortest, recommended) using ORS,
    save as .npy, and return summary info.
    """
    os.makedirs(output_dir, exist_ok=True)

    preferences = ["fastest", "shortest", "recommended"]
    colors = ["blue", "green", "red"]

    m = folium.Map(
        location=[(origin[1] + destination[1]) / 2, (origin[0] + destination[0]) / 2],
        zoom_start=7,
        tiles="OpenStreetMap"
    )

    routes_info = []

    for i, pref in enumerate(preferences):
        try:
            route = client.directions(
                coordinates=[origin, destination],
                profile="driving-car",
                format="geojson",
                preference=pref
            )
            coords = route["features"][0]["geometry"]["coordinates"]
            latlon_coords = [(lat, lon) for lon, lat in coords]

            # Plot polyline
            folium.PolyLine(
                latlon_coords,
                color=colors[i % len(colors)],
                weight=5,
                opacity=0.9,
                tooltip=f"Route: {pref}"
            ).add_to(m)

            # Save .npy file
            np_array = np.array([(lat, lon, 0) for lat, lon in latlon_coords])
            np.save(os.path.join(output_dir, f"route_{pref}.npy"), np_array)

            routes_info.append({
                "preference": pref,
                "coords": latlon_coords,
                "distance": route["features"][0]["properties"]["summary"]["distance"],
                "duration": route["features"][0]["properties"]["summary"]["duration"]
            })

            print(f"Saved route '{pref}' as route_{pref}.npy")

        except Exception as e:
            print(f"Failed for {pref}: {e}")

    return m, routes_info


import time
from requests.adapters import HTTPAdapter, Retry


def fetch_osm_features(bbox, max_retries=3):
    query = f"""
    [out:json][timeout:60];
    (
      node["amenity"="school"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
      node["traffic_calming"="bump"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
      way["highway"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    );
    out tags center;
    """
    url = "http://overpass-api.de/api/interpreter"

    session = requests.Session()
    retries = Retry(total=max_retries, backoff_factor=2,
                    status_forcelist=[500, 502, 503, 504])
    session.mount("http://", HTTPAdapter(max_retries=retries))

    for attempt in range(max_retries):
        try:
            response = session.get(url, params={"data": query}, timeout=120)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Retry {attempt + 1}/{max_retries} failed: {e}")
            time.sleep(2)
    raise RuntimeError("Overpass query failed after retries")


def coor(origin, destination):
    """
    Main entry point: get routes + OSM features between origin & destination.
    """
    m, routes = fetch_routes(origin, destination)

    # Bounding box from all routes
    lons = [lon for r in routes for _, lon in r["coords"]]
    lats = [lat for r in routes for lat, _ in r["coords"]]
    bbox = [min(lons), min(lats), max(lons), max(lats)]

    # Fetch OSM features
    features = fetch_osm_features(bbox)
    print("DEBUG features:", features)

    # Add markers to map
    elements = features.get("elements", [])
    normalized = []

    for f in elements:
        feature = {
            "id": f.get("id"),
            "type": f.get("type"),
            "lat": f.get("center", {}).get("lat"),
            "lon": f.get("center", {}).get("lon"),
            "tags": f.get("tags", {}),
            # safely extract road name
            "road_name": f.get("tags", {}).get("name", None),
            # highway type (road category)
            "road_type": f.get("tags", {}).get("highway", None),
            # speed limit if available
            "maxspeed": f.get("tags", {}).get("maxspeed", None),
        }
        normalized.append(feature)

    # Save map
    m.save("full_routes_with_features.html")
    print("Map saved as full_routes_with_features.html")

    return routes, normalized


# ==============================
# Example Run
# ==============================
if __name__ == "__main__":
    # Example: Gitam Bengaluru -> Ballari
    origin = (8.4342971002335, 49.015003823272)
    destination = (8.4329721922397, 49.014596344219)

    routes, features = coor(origin, destination)

    print("Routes:", routes[:2])  # print first 2 route summaries
    print("Features:", features[:5])  # print first 5 features
