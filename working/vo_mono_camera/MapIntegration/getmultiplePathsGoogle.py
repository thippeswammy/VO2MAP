import googlemaps
from polyline import decode

gmaps = googlemaps.Client(key="AIzaSyBQwErxtm6_jsoB44Fye7F61GdY38Eq1GE")
origin = "gitam university bangalore"
destination = "ballari karnataka"
directions_result = gmaps.directions(origin, destination, alternatives=True)
for route in directions_result:
    # Each 'route' in directions_result is a dictionary
    # containing information about a single route.

    # Access the legs of the route (segments between waypoints)
    for leg in route['legs']:
        # Access the steps within each leg (individual instructions)
        for step in leg['steps']:
            # Extract the path (polyline) for the step
            polyline = step['polyline']['points']  # Get the encoded polyline string

            # Optionally, decode the polyline into a list of coordinates
            # (requires a polyline decoding library like 'polyline')
            # decoded_polyline = polyline.decode(polyline)
            # print(decoded_polyline)
            print(polyline)

# for route in directions_result:
#         for leg in route['legs']:
#             for step in leg['steps']:
#                 polyline = step['polyline']['points']
#                 decoded_polyline = decode(polyline)
#                 print(decoded_polyline)