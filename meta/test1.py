import json
import os

import folium
import numpy as np
import openrouteservice
import osmium as osm

# ====================== #
#         CONFIG         #
# ====================== #
ORS_KEY = "5b3ce3597851110001cf6248ab9c3fa94a254ed99f97d8193b32b999"
OSM_FILE = r"karlsruhe-regbez-250915.osm.pbf"
OUTPUT_DIR = "map_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Example coordinates
origin = (49.006719195871, 8.4893558806503)
destination = (49.015637395908, 8.4658573451798)


class FeatureHandler(osm.SimpleHandler):
    def __init__(self, bbox):
        super().__init__()
        self.features = []
        self.bbox = bbox  # [min_lon, min_lat, max_lon, max_lat]

    def in_bbox(self, lat, lon):
        min_lon, min_lat, max_lon, max_lat = self.bbox
        return min_lat <= lat <= max_lat and min_lon <= lon <= max_lon

    def node(self, n):
        if not n.location.valid():
            return
        lat, lon = n.location.lat, n.location.lon
        if not self.in_bbox(lat, lon):
            return

        tags = dict(n.tags)
        interesting_keys = ["amenity", "shop", "traffic_sign", "information", "highway", "name"]
        if not any(k in tags for k in interesting_keys):
            return

        self.features.append(normalize_feature({
            "id": n.id,
            "type": "node",
            "lat": lat,
            "lon": lon,
            "tags": tags
        }))

    def way(self, w):
        if "highway" not in w.tags:
            return

        coords = []
        for n in w.nodes:
            if not n.location.valid():
                continue
            lat, lon = n.location.lat, n.location.lon
            if self.in_bbox(lat, lon):
                coords.append((lat, lon))

        if coords:
            tags = dict(w.tags)
            self.features.append(normalize_feature({
                "id": w.id,
                "type": "way",
                "tags": tags,
                "coords": coords
            }))


def normalize_feature(el):
    tags = el.get("tags", {})
    lat, lon = el.get("lat"), el.get("lon")
    category, subtype, road_name, road_type, maxspeed = None, None, None, None, None
    landmark_name, landmark_type, milestone_km, board_text, place_ref = None, None, None, None, None
    zone_type, zone_extent = None, None

    if "highway" in tags:
        category = "road"
        road_type = tags.get("highway")
        road_name = tags.get("name")
        maxspeed = tags.get("maxspeed")

    if tags.get("amenity") in ["school", "hospital"]:
        category = "zone"
        subtype = tags["amenity"]
        landmark_name = tags.get("name")
        zone_type = tags["amenity"] + "_zone"
        zone_extent = 200

    if tags.get("amenity") and category != "zone":
        category = "landmark"
        subtype = tags["amenity"]
        landmark_name = tags.get("name")
        landmark_type = tags["amenity"]

    if tags.get("highway") == "milestone":
        category = "landmark"
        subtype = "milestone"
        milestone_km = float(tags.get("distance", 0)) if "distance" in tags else None
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


def coor(origin, destination):
    client = openrouteservice.Client(key=ORS_KEY)

    preferences = ["fastest", "shortest", "recommended"]
    colors = ["blue", "green", "red"]
    routes = []

    m = folium.Map(location=[(origin[0] + destination[0]) / 2, (origin[1] + destination[1]) / 2], zoom_start=14)

    for i, pref in enumerate(preferences):
        try:
            route = client.directions(
                coordinates=[(origin[1], origin[0]), (destination[1], destination[0])],
                profile="driving-car",
                format="geojson",
                preference=pref
            )
            coords = [(lat, lon) for lon, lat in route["features"][0]["geometry"]["coordinates"]]
            dist = route["features"][0]["properties"]["segments"][0]["distance"]
            dur = route["features"][0]["properties"]["segments"][0]["duration"]

            np.save(os.path.join(OUTPUT_DIR, f"route_{pref}.npy"), np.array([(lat, lon, 0) for lat, lon in coords]))

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

    all_lats = [lat for r in routes for lat, _ in r["coords"]]
    all_lons = [lon for r in routes for _, lon in r["coords"]]
    bbox = [min(all_lons), min(all_lats), max(all_lons), max(all_lats)]

    handler = FeatureHandler(bbox)
    handler.apply_file(OSM_FILE, locations=True)
    features = handler.features

    with open(os.path.join(OUTPUT_DIR, "routes.json"), "w", encoding="utf-8") as f:
        json.dump(routes, f, indent=2, ensure_ascii=False)

    with open(os.path.join(OUTPUT_DIR, "features.json"), "w", encoding="utf-8") as f:
        json.dump(features, f, indent=2, ensure_ascii=False)

    for f in features:
        if f["lat"] and f["lon"]:
            folium.Marker(
                [f["lat"], f["lon"]],
                popup=f"{f['subtype']} : {f['landmark_name'] or f['board_text']}",
                icon=folium.Icon(color="orange", icon="info-sign")
            ).add_to(m)

    m.save(os.path.join(OUTPUT_DIR, "routes_with_features.html"))

    print(f"Map saved. Routes: {len(routes)} Features: {len(features)}")
    return routes, features


if __name__ == "__main__":
    routes, features = coor(origin, destination)
