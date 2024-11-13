import numpy as np
from skimage import measure
import trimesh

def heightmap_to_3d_mesh(heightmap, output_path="terrain_mesh.obj", max_GB_mem=0.5, km_per_pix=0.5):
    """heightmap to mesh, assumes heigtmap is in meters"""
    # Normalize the heightmap to form a volume where the z-axis represents the height
    heightmap = np.where(heightmap < 0, 0, heightmap) # Filter out ocean and sea
    max_memory_bytes = max_GB_mem * 1024 ** 3 

    x_size, y_size = heightmap.shape

    # Calculate z_levels to fit within 4GB memory
    z_levels = int(max_memory_bytes / heightmap.size)
    max_height = np.max(heightmap)
    
    # Check if z_levels exceed the max height
    if z_levels > max_height:
        z_levels = int(max_height)  # If there are more z_levels than max height, then just set it equal

    # Create a 3D volume: stack multiple layers of the height map along the z-axis
    volume = np.zeros((x_size, y_size, z_levels), dtype=np.uint8)

    # Fill the 3D volume based on the height map
    for i in range(x_size):
        for j in range(y_size):
            height = heightmap[i, j]
            num_layers = int(np.clip(height / max_height * z_levels, 0, z_levels-1))  # Number of layers for this height
            volume[i, j, :num_layers] = 1  # Fill the volume up to the corresponding height

    # Apply marching cubes
    verts, faces, normals, _ = measure.marching_cubes(volume)

    # Scale the vertices to reflect real-world distances
    verts[:, 0] = verts[:, 0] * km_per_pix
    verts[:, 1] = verts[:, 1] * km_per_pix
    verts[:, 2] = verts[:, 2] * (max_height / z_levels) / 1000  # Convert height from meters to kilometers
                
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)

    mesh.export(output_path)
    return output_path

# Example usage:
if __name__ == "__main__":
    # Example heightmap (replace with your actual 2D heightmap)
    from pathlib import Path
    hm_path = Path('assets/defaults/height_map.npy')
    heightmap = np.load(hm_path)
    heightmap_to_3d_mesh(heightmap, output_path="terrain_mesh.obj")
