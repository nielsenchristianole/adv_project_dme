import os
import json
import tyro
from tqdm import tqdm
from pathlib import Path
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class MapProjector:
    """
    Class to project latitude and longitude coordinates to a sphere and then orthographically to a plane.
    """
    def __init__(
        self,
        output_dir: os.PathLike, 
        visualize: bool = False, 
        radius: float = 1, # Earth's radius is 6378137 meters if we want the sphere to be the correct size
        planes_file: os.PathLike | str  = None,
        inp_npy_file: os.PathLike = None
        ):
        """
        Initialize the MapProjector class.
        inp_npy_file and planes_file are optional arguments, the MapProjector can also be run with planes and lat_lon_arr as np.array arguments.
        
        Args:
            output_dir (os.PathLike): Path to the output directory to save the results.
            visualize (bool): Whether to visualize the sphere to plane projection.
            radius (float): Radius of the sphere to project the points onto.
            planes_file (os.PathLike | str, optional): Can be either:
                1. Path to a .npy file with rows of (center_xyz, width, height) for each plane. The chosen w and h should be chosen as same scale as the radius.
                2. 'test' to use a test plane at (0,0) that projects the entire hemisphere.
                3. None will generate random planes from 0.2 to 1 scale of the radius size.
            inp_npy_file (os.PathLike, optional): Path to the .npy file containing (lat, lon) in the first two columns.
        """
        self.output_dir = Path(output_dir)
        self.visualize = visualize
        self.radius = radius 
        self.planes_file = planes_file
        self.inp_npy = inp_npy_file

    def project_lat_lon_to_sphere(self, lat_lon_arr: np.ndarray) -> np.ndarray:
        """
        Project latitude and longitude coordinates to a sphere with a given radius.

        Args:
            lat_lon_arr (np.ndarray): Array of latitude and longitude coordinates (n x 2 array).

        Returns:
            np.ndarray: Array of 3D points on the sphere (n x 3 array).
        """
        # Convert latitude and longitude to radians
        lat = np.radians(lat_lon_arr[:, 0])
        lon = np.radians(lat_lon_arr[:, 1])

        x = self.radius * np.cos(lat) * np.cos(lon)
        y = self.radius * np.cos(lat) * np.sin(lon)
        z = self.radius * np.sin(lat)

        return np.column_stack((x, y, z))

    def normalize(self, v) -> np.ndarray:
        """
        Normalize a vector.

        Returns:
            np.ndarray: Normalized vector
        """
        return v / np.linalg.norm(v)

    def construct_basis(self, normal: np.ndarray) -> np.ndarray:
        """
        Construct an orthonormal basis given a normal vector (the z-axis of the plane).
        The normal vector defines the direction of the plane's normal.

        Args:
            normal (np.ndarray): The normal vector of the plane.

        Returns:
            np.ndarray: Orthonormal basis for the plane
        """

        normal = self.normalize(normal)

        # Find an arbitrary vector that is not parallel to the normal
        if np.abs(normal[0]) > np.abs(normal[1]):
            tangent_x = np.array([-normal[2], 0, normal[0]])
        else:
            tangent_x = np.array([0, normal[2], -normal[1]])

        # Normalize the tangent vector
        tangent_x = self.normalize(tangent_x)

        # Compute the second tangent vector (y-axis of the plane) using the cross product
        tangent_y = np.cross(normal, tangent_x)

        return np.array([tangent_x, tangent_y, normal])

    def orthographic_projection_matrix(self, center_xyz: np.ndarray) -> np.ndarray:
        """
        Compute the orthographic projection matrix that projects points onto
        the tangent plane at `center_xyz`.

        Args:
            center_xyz (np.ndarray): The center of the plane in Cartesian coordinates (x,y,z)

        Returns:
            np.ndarray: A matrix that projects points onto the plane.
        """

        basis = self.construct_basis(center_xyz)

        # The projection matrix transforms the coordinates into the plane's local frame
        # This matrix maps the points into the local frame, then performs the projection.
        projection_matrix = basis.T

        return projection_matrix

    def project_and_filter(self, points: np.ndarray, center_xyz: np.ndarray, width: float, height: float) -> np.ndarray:
        """
        Project points onto a tangent plane and filter those that lie within the defined width and height,
        and that are on the same hemisphere as the center.

        Args:
            points (np.ndarray): Array of 3D points in Cartesian coordinates on a sphere (n x 3 array).
            center_xyz (np.ndarray): The center of the plane in Cartesian coordinates.
            width (float): Width of the tangent plane.
            height (float): Height of the tangent plane.

        Returns:
            np.ndarray: The 3D points orthographically projected onto the tangent plane with the given center, height and width.
        """

        # Compute the projection matrix
        projection_matrix = self.orthographic_projection_matrix(center_xyz)

        # Apply the projection matrix to the points
        projected_points = np.dot(points, projection_matrix[:, :2])  # Only take the x and y components

        # Filter points based on the width and height of the plane
        half_width = width / 2.0
        half_height = height / 2.0
        in_bounds = (
            (projected_points[:, 0] >= -half_width) & (projected_points[:, 0] <= half_width) &
            (projected_points[:, 1] >= -half_height) & (projected_points[:, 1] <= half_height)
        )

        # Also filter points on the opposite hemisphere (dot product < 0 means on the other side)
        dot_products = np.dot(points, self.normalize(center_xyz))
        on_same_hemisphere = dot_products > 0

        # Combine the two filters
        valid_points = in_bounds & on_same_hemisphere

        return projected_points[valid_points]

    def process_planes(self, planes: np.ndarray, points: np.ndarray) -> list[dict]:
        """
        Project 3D points to multiple planes and save the results to .npy files and a metadata JSON file.
        The center of each plane has to be xyz coordinates in the spherical world coordinates.
        
        Args:
            planes (np.ndarray): Array of shape (n, 5) where each row is (center_xyz, width, height).
            points (np.ndarray): Array of 3D points in Cartesian coordinates (n x 3 array).

        Returns:
            list[dict]: List of metadata dictionaries for each plane with center_xyz, width, height, and npy_filename.
        """
        
        # Ensure output directory exists
        npy_output_folder = self.output_dir / 'planes'
        npy_output_folder.mkdir(parents=True, exist_ok=True)

        # Metadata for the JSON file
        metadata = []

        # Loop through each plane
        for i, plane in enumerate(tqdm(planes, desc="Processing planes")):
            center_latlon, width, height = plane[:2], plane[2], plane[3]
        
            center_xyz = np.squeeze(self.project_lat_lon_to_sphere(np.array([center_latlon])))
            
            # Project and filter the points for this plane
            projected_points = self.project_and_filter(points, center_xyz, width, height)

            # Save the projected points to a .npy file
            npy_filename = f'plane_{i}_projected_points.npy'
            np.save(npy_output_folder / npy_filename, projected_points)

            # Add metadata to the JSON list
            metadata.append({
                'center_xyz': center_xyz.tolist(),
                'width': width,
                'height': height,
                'npy_filename': npy_filename
            })

        # Save the metadata to a JSON file
        json_filename = self.output_dir / 'planes_metadata.json'
        with open(json_filename, 'w') as json_file:
            json.dump(metadata, json_file, indent=4)

        print(f'Processed {len(planes)} planes. Results saved to {self.output_dir}.')
        return metadata

    def visualize_sphere_to_plane_projection(self, sphere_coords: np.ndarray, processed_plane: dict):
        """
        Visualize the projection of points on a sphere to a plane.

        Args:
            sphere_coords (np.ndarray): Coordinates of points on the sphere.
            processed_plane (dict): Metadata of the processed plane with center_xyz, width, height, and npy_filename.
        """
        import matplotlib.pyplot as plt

        def plot_plane(ax, processed_plane, projected_points):
            center_xyz = np.array(processed_plane['center_xyz'])
            normal = center_xyz / np.linalg.norm(center_xyz)
            width = processed_plane['width']
            height = processed_plane['height']

            # Construct basis vectors
            basis = self.construct_basis(normal)
            tangent_x, tangent_y = basis[0], basis[1]

            # Calculate the four corners of the plane
            half_width = width / 2
            half_height = height / 2
            corners = [
                center_xyz + half_width * tangent_x + half_height * tangent_y,
                center_xyz - half_width * tangent_x + half_height * tangent_y,
                center_xyz - half_width * tangent_x - half_height * tangent_y,
                center_xyz + half_width * tangent_x - half_height * tangent_y
            ]

            # Plot the plane
            plane = Poly3DCollection([corners], color='cyan', alpha=0.5)
            ax.add_collection3d(plane)

            # Plot the normal vector
            ax.quiver(*center_xyz, *normal, color='g', label='Plane Normal')

            # Plot the projected points on the plane
            transformed_points = center_xyz + projected_points[:, 0][:, np.newaxis] * tangent_x + projected_points[:, 1][:, np.newaxis] * tangent_y
            ax.scatter(transformed_points[:, 0], transformed_points[:, 1], transformed_points[:, 2], color='r', label='Projected Points')

        # Load the projected points
        projected_points = np.load(self.output_dir / 'planes' / processed_plane['npy_filename'])

        # Plot the sphere points
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(*sphere_coords.T, color='b', label='Sphere Points')

        # Plot the plane normal
        plot_plane(ax, processed_plane, projected_points)

        # Set axis limits
        ax.set_xlim(-self.radius, self.radius)
        ax.set_ylim(-self.radius, self.radius)
        ax.set_zlim(-self.radius, self.radius)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Projection of Points on Sphere to Plane')
        ax.legend()

        plt.show()

    def generate_plane_samples(self, num_planes: int) -> np.ndarray:
        """
        Generate random planes defined by center coordinates and width and height.
        
        Args:
            num_planes (int): Number of random planes to generate.

        Returns:
            np.ndarray: Array of shape (num_planes, 5) where each row is (center_latlon, width, height).
        """

        # Generate latitudes and longitudes between -90 and 90 and -180 and 180 respectively
        latitudes = np.random.uniform(-90, 90, num_planes)
        longitudes = np.random.uniform(-180, 180, num_planes)

        # Generate random width and height between 0 and 1
        width = self.radius * np.random.uniform(0.2, 1, num_planes)
        height = self.radius * np.random.uniform(0.2, 1, num_planes)
        
        plane_samples = np.column_stack((latitudes, longitudes, width, height))
        
        return plane_samples

    def run(self, lat_lon_arr: np.ndarray = None, planes: np.ndarray = None):
        """
        Run the map projection pipeline.
        
        Args:
            lat_lon_arr (np.ndarray, optional): Array of latitude and longitude coordinates (n x 2 array).
            planes (np.ndarray, optional): Array of planes with center_latlon, width, and height (n x 4 array).
        """
        if lat_lon_arr is None:
            lat_lon_arr = np.load(self.inp_npy)
        sphere_coords = self.project_lat_lon_to_sphere(lat_lon_arr)

        if self.planes_file is None and planes is None:
            print('Generating random planes')
            planes = self.generate_plane_samples(5)
        elif str(self.planes_file) == 'test':
            print('Using a test plane at (0,0)')
            planes = np.array([[0, 0, self.radius * 2, self.radius * 2]])
        elif planes is None:
            planes = np.load(self.planes_file)
            
        metadata = self.process_planes(planes, sphere_coords)

        if self.visualize:
            for i, plane_meta in enumerate(metadata):
                self.visualize_sphere_to_plane_projection(sphere_coords, plane_meta)


if __name__ == '__main__':
    tyro.cli(MapProjector).run()