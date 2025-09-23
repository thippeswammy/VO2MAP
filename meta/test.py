import json
import os

import folium
import osmium

OSM_FILE = "karlsruhe-regbez-250915.osm.pbf"
OUTPUT_DIR = "map_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

features = []
import osmium as osm


class WayHandler(osm.SimpleHandler):
    def way(self, w):
        coords = []
        for n in w.nodes:
            if not n.location.valid():  # skip invalid nodes
                continue
            coords.append((n.lat, n.lon))

        if coords:  # process only if we got valid coords
            print(f"Way {w.id}: {coords[:3]} ... total {len(coords)} points")


# Parse local OSM file
h = FeatureHandler()
h.apply_file(OSM_FILE, locations=True)  # <-- locations=True is important


class FeatureHandler(osmium.SimpleHandler):
    def node(self, n):
        tags = dict(n.tags)
        if "amenity" in tags or "highway" in tags or "traffic_sign" in tags:
            features.append({
                "id": n.id,
                "lat": n.location.lat,
                "lon": n.location.lon,
                "tags": tags
            })

    def way(self, w):
        tags = dict(w.tags)
        if "highway" in tags:
            coords = [(n.lat, n.lon) for n in w.nodes]
            features.append({
                "id": w.id,
                "coords": coords,
                "tags": tags
            })


# Save features locally
with open(os.path.join(OUTPUT_DIR, "features_local.json"), "w", encoding="utf-8") as f:
    json.dump(features, f, indent=2, ensure_ascii=False)

# Map preview
m = folium.Map(location=[49.0067, 8.4893], zoom_start=13)
for f in features[:500]:  # only show first 500 to keep map light
    if "lat" in f and "lon" in f:
        folium.CircleMarker([f["lat"], f["lon"]], radius=10, color="red").add_to(m)

m.save(os.path.join(OUTPUT_DIR, "local_features_map.html"))
print(f"Extracted {len(features)} features.")
