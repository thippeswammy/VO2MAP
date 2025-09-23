import json
import os

import folium
import numpy as np
import openrouteservice
import requests

# ===========================
# CONFIG
# ===========================
ORS_KEY = "5b3ce3597851110001cf6248ab9c3fa94a254ed99f97d8193b32b999"

# Example coordinates (lon, lat)
# origin = (49.015003823272, 8.4342971002335)  # (lat, lon)
origin = (49.006719195871, 8.4893558806503)
destination = (49.015637395908, 8.4658573451798)
# destination = (49.014596344219, 8.4329721922397)

OUTPUT_DIR = "map_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================== #
# HELPERS                     #
# =========================== #
def query_osm_features(bbox):
    """
    Query OSM Overpass API for roads, amenities, traffic signs, guideposts within bounding box.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    query = f"""
    [out:json][timeout:25];
    (
      way["highway"]({min_lat},{min_lon},{max_lat},{max_lon});
      node["amenity"]({min_lat},{min_lon},{max_lat},{max_lat});
      node["traffic_sign"]({min_lat},{min_lon},{max_lat},{max_lat});
      node["information"]({min_lat},{min_lon},{max_lat},{max_lat});
    );
    out center;
    """
    url = "http://overpass-api.de/api/interpreter"
    resp = requests.get(url, params={"data": query})
    return resp.json()


import time


def query_nearby_amenities(lat, lon, radius=2, sleep=1.0):
    """
    Query OSM for shops, schools, and hospitals within a given radius (meters).
    Includes error handling for empty/invalid responses.
    """
    query = f"""
    [out:json][timeout:25];
    (
      node(around:{radius},{lat},{lon})["amenity"="hospital"];
      node(around:{radius},{lat},{lon})["amenity"="school"];
      node(around:{radius},{lat},{lon})["shop"];
    );
    out center;
    """
    url = "http://overpass-api.de/api/interpreter"
    try:
        resp = requests.get(url, params={"data": query}, timeout=60)
        if resp.status_code != 200:
            print(f"⚠️ Overpass error {resp.status_code} at ({lat},{lon})")
            return {"elements": []}
        try:
            data = resp.json()
        except Exception as e:
            print(f"⚠️ JSON parse error at ({lat},{lon}): {e}")
            return {"elements": []}
        return data
    except Exception as e:
        print(f"⚠️ Request failed at ({lat},{lon}): {e}")
        return {"elements": []}
    finally:
        time.sleep(sleep)  # avoid hitting Overpass too fast


def normalize_feature(el):
    """
    Normalize OSM element into extended schema.
    """
    tags = el.get("tags", {})
    lat, lon = None, None
    if "center" in el:
        lat, lon = el["center"]["lat"], el["center"]["lon"]
    elif "lat" in el and "lon" in el:
        lat, lon = el["lat"], el["lon"]

    category, subtype, road_name, road_type, maxspeed = None, None, None, None, None
    landmark_name, landmark_type, milestone_km, board_text, place_ref = None, None, None, None, None
    zone_type, zone_extent = None, None

    # --- Road ---
    if "highway" in tags:
        category = "road"
        road_type = tags.get("highway")
        road_name = tags.get("name")
        maxspeed = tags.get("maxspeed")

    # --- School / Hospital Zone ---
    if tags.get("amenity") in ["school", "hospital"]:
        category = "zone"
        subtype = tags["amenity"]
        landmark_name = tags.get("name")
        zone_type = tags["amenity"] + "_zone"
        zone_extent = 200

    # --- Landmark ---
    if tags.get("amenity") and category != "zone":
        category = "landmark"
        subtype = tags["amenity"]
        landmark_name = tags.get("name")
        landmark_type = tags["amenity"]

    # --- Milestone / Signs ---
    if tags.get("highway") == "milestone":
        category = "landmark"
        subtype = "milestone"
        milestone_km = float(tags.get("distance", 0))
        board_text = tags.get("ref") or tags.get("name")

    if tags.get("traffic_sign"):
        category = "landmark"
        subtype = "board"
        board_text = tags.get("traffic_sign")

    if tags.get("information") == "guidepost":
        category = "landmark"
        subtype = "board"
        board_text = tags.get("name") or tags.get("ref")

    drift_correction = category in ["zone", "landmark"]
    correction_strength = 0.9 if subtype in ["milestone", "board"] else 0.7 if subtype else 0.5

    return {
        "id": el["id"],
        "type": el["type"],
        "lat": lat,
        "lon": lon,

        "category": category,
        "subtype": subtype,

        "road_name": road_name,
        "road_type": road_type,
        "maxspeed": maxspeed,

        "landmark_name": landmark_name,
        "landmark_type": landmark_type,
        "board_text": board_text,
        "milestone_km": milestone_km,
        "place_ref": place_ref,

        "zone_type": zone_type,
        "zone_extent": zone_extent,

        "drift_correction": drift_correction,
        "correction_strength": correction_strength,

        "tags": tags,
        "source": "OSM"
    }


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ===========================
# MAIN FUNCTION
# ===========================
def coor(origin, destination):
    client = openrouteservice.Client(key=ORS_KEY)

    preferences = ["fastest", "shortest", "recommended"]
    colors = ["blue", "green", "red"]

    routes = []
    m = folium.Map(location=[(origin[0] + destination[0]) / 2, (origin[1] + destination[1]) / 2], zoom_start=14)

    for i, pref in enumerate(preferences):
        try:
            route = client.directions(
                coordinates=[(origin[1], origin[0]), (destination[1], destination[0])],  # ORS expects (lon, lat)
                profile="driving-car",
                format="geojson",
                preference=pref
            )

            coords = [(lat, lon) for lon, lat in route["features"][0]["geometry"]["coordinates"]]
            dist = route["features"][0]["properties"]["segments"][0]["distance"]
            dur = route["features"][0]["properties"]["segments"][0]["duration"]

            # Save .npy
            latlon_coords1 = [(lat, lon, 0) for lat, lon in coords]
            np.save(os.path.join(OUTPUT_DIR, f"route_{pref}.npy"), np.array(latlon_coords1))

            # Compute bounding box
            lats = [lat for lat, lon in coords]
            lons = [lon for lat, lon in coords]
            route_bbox = [min(lons), min(lats), max(lons), max(lats)]

            routes.append({
                "preference": pref,
                "coords": coords,
                "distance_m": dist,
                "duration_s": dur,
                "bbox": route_bbox
            })

            folium.PolyLine(coords, color=colors[i], weight=4, tooltip=pref).add_to(m)
        except Exception as e:
            print(f"Failed route {pref}: {e}")

    if not routes:
        print("No routes found. Exiting.")
        return [], []

    # Overall bounding box for OSM query
    all_lats = [lat for r in routes for lat, _ in r["coords"]]
    all_lons = [lon for r in routes for _, lon in r["coords"]]
    bbox = [min(all_lons), min(all_lats), max(all_lons), max(all_lats)]

    # Query OSM for roads, signs, zones
    osm_data = query_osm_features(bbox)
    features = [normalize_feature(el) for el in osm_data["elements"]]

    # Query nearby amenities along the route
    amenity_features = []
    for r in routes:
        for lat, lon in r["coords"]:
            nearby = query_nearby_amenities(lat, lon, radius=10)
            for el in nearby["elements"]:
                amenity_features.append(normalize_feature(el))

    # Save
    save_json(routes, os.path.join(OUTPUT_DIR, "routes.json"))
    save_json(features + amenity_features, os.path.join(OUTPUT_DIR, "features.json"))

    # Add amenities to map
    for f in amenity_features:
        if f["lat"] and f["lon"]:
            folium.Marker(
                [f["lat"], f["lon"]],
                popup=f"{f['subtype']} : {f['landmark_name'] or f['board_text']}",
                icon=folium.Icon(color="orange", icon="info-sign")
            ).add_to(m)

    m.save(os.path.join(OUTPUT_DIR, "routes_with_features.html"))

    print(f"Map saved. Routes: {len(routes)} Features: {len(features + amenity_features)}")
    return routes, features + amenity_features


# ===========================
# RUN
# ===========================
if __name__ == "__main__":
    routes, features = coor(origin, destination)
