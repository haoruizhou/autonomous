import json
import numpy as np
import matplotlib.pyplot as plt

# Load track data
with open('results/track/track_data.json', 'r') as f:
    data = json.load(f)

cl = np.array(data.get('centerline', []))
cones = np.array(data.get('cone_positions', []))

plt.figure(figsize=(12, 12))

# Plot the centerline very faintly so it's just a reference
if len(cl) > 0:
    plt.plot(cl[:, 0], cl[:, 1], color='#e0e0e0', linewidth=2, alpha=0.3, label='Centerline Reference')

# Plot the cones very prominently
if len(cones) > 0:
    plt.scatter(cones[:, 0], cones[:, 1], color='#ff5500', s=300, edgecolors='black', zorder=5, label=f'Cones ({len(cones)})')

plt.title('Autocross Track - Cones Only', fontsize=16, fontweight='bold')
plt.xlabel('Local X (meters)')
plt.ylabel('Local Y (meters)')
plt.legend(loc='upper left', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.4)
plt.axis('equal')  # Ensure geometry isn't stretched
plt.tight_layout()

output_path = 'results/track/track_cones_only.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved to {output_path}")
