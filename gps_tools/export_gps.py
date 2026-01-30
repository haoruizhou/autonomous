import subprocess
import gpmf_parser
import json
import os
import argparse
import sys
import struct

def extract_binary_gpmf(video_path, output_bin):
    """Extracts the GPMF track (stream 3 typically) to a binary file."""
    cmd = [
        'ffmpeg', '-y', 
        '-i', video_path, 
        '-map', '0:3', 
        '-f', 'data', 
        '-codec', 'copy', 
        output_bin
    ]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Extracted binary to {output_bin}")

def process_gpmf_file(bin_path):
    with open(bin_path, 'rb') as f:
        payload = f.read()
    
    gps_checkpoints = []
    current_scale = [1] * 8
    
    for path, key, values in gpmf_parser.parse_gpmf(payload):
        if 'STRM' in path: 
            if key == 'SCAL':
                 if values:
                    if isinstance(values[0], list):
                         # Flatten
                         flat = []
                         for v in values:
                             if isinstance(v, list): flat.extend(v)
                             else: flat.append(v)
                         current_scale = flat
                    else:
                        current_scale = values
            
            elif key == 'GPS9':
                # GPS9: usually 32 bytes per sample (8 ints)
                # Parse raw values if they are bytes
                
                # Expand scales if needed
                while len(current_scale) < 8:
                    current_scale.append(1)
                
                samples = []
                # Check if values is list of bytes (chunks)
                if values and isinstance(values[0], bytes):
                    for chunk in values:
                        if len(chunk) == 32:
                            # 8 ints
                            ints = struct.unpack('>8i', chunk)
                            samples.append(ints)
                elif values and isinstance(values[0], list):
                     samples = values
                
                for row in samples:
                    if len(row) >= 3: # Lat, Lon, Alt
                        try:
                            lat = float(row[0]) / float(current_scale[0])
                            lon = float(row[1]) / float(current_scale[1])
                            alt = float(row[2]) / float(current_scale[2])
                            
                            # Valid range check
                            if abs(lat) > 90 or abs(lon) > 180:
                                continue
                                
                            # Speed? 
                            # If GPS9 has 8 items, speed might be index 3 (2D) and 4 (3D)
                            speed2d = 0
                            if len(row) > 3:
                                speed2d = float(row[3]) / float(current_scale[3])
                            
                            gps_checkpoints.append({
                                'lat': lat,
                                'lon': lon,
                                'alt': alt,
                                'speed': speed2d
                            })
                        except Exception:
                            pass
                            
    return gps_checkpoints

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', help="Path to GoPro MP4")
    parser.add_argument('--output', '-o', help="Output JSON path", default='gps_track.json')
    args = parser.parse_args()
    
    bin_file = "temp_gpmf.bin"
    try:
        extract_binary_gpmf(args.video_path, bin_file)
        gps_points = process_gpmf_file(bin_file)
        
        print(f"Parsed {len(gps_points)} GPS points.")
        
        # Filter 0,0
        valid_points = [p for p in gps_points if abs(p['lat']) > 0.0001 and abs(p['lon']) > 0.0001]
        print(f"Valid points: {len(valid_points)}")
        
        # Reduce density if too many
        if len(valid_points) > 5000:
             step = len(valid_points) // 5000
             valid_points = valid_points[::step]
        
        with open(args.output, 'w') as f:
            json.dump(valid_points, f, indent=2)
            
        print(f"Saved to {args.output}")
        
        # Also generate map.html with embedded data
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>GoPro GPS Track</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
    <style>
        body {{ margin: 0; padding: 0; }}
        #map {{ position: absolute; top: 0; bottom: 0; width: 100%; }}
        #info {{
            position: absolute; top: 10px; right: 10px; z-index: 1000;
            background: white; padding: 10px; border-radius: 5px;
            box-shadow: 0 0 10px rgba(0,0,0,0.2); font-family: sans-serif;
        }}
    </style>
</head>
<body>
    <div id="info">Loading track...</div>
    <div id="map"></div>
    <script>
        var map = L.map('map').setView([0, 0], 2);

        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        }}).addTo(map);

        var gpsData = {json.dumps(valid_points)};

        if (gpsData.length === 0) {{
            document.getElementById('info').innerText = "No GPS points found.";
        }} else {{
            var latlngs = gpsData.map(p => [p.lat, p.lon]);
            var polyline = L.polyline(latlngs, {{color: 'blue', weight: 4}}).addTo(map);
            
            var start = latlngs[0];
            var end = latlngs[latlngs.length - 1];
            
            L.marker(start).addTo(map).bindPopup("Start");
            L.marker(end).addTo(map).bindPopup("End");

            map.fitBounds(polyline.getBounds());
            document.getElementById('info').innerText = `Track loaded: ${{gpsData.length}} points.`;
        }}
    </script>
</body>
</html>"""
        
        with open("map.html", "w") as f:
            f.write(html_content)
        print("Saved to map.html (standalone)")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if os.path.exists(bin_file):
            os.remove(bin_file)

if __name__ == "__main__":
    main()
