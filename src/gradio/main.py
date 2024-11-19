import os
import random
import itertools
from typing import Literal, Optional, List, Union

import cv2
import numpy as np
import gradio as gr
import shapely
from src.gradio.mesh_map import heightmap_to_3d_mesh
from src.gradio.demo_types import TOWN_TYPE, TOWN_TYPES, Town, RoadGraph, RoadNode, RoadEdge
from src.gradio.display_map import plot_map
from src.gradio.gradio_utils import TownNameSampler
from src.gradio.gradio_configs import CLOSE_ICON
from src.gradio.path_finder import AStar, GreedyBestFirst, BridgeEuclideanHeuristic

# from src.gradio.town_generator import TownGenerator
from src.gradio.height_shape_generator import HeightShapeGenerator


RESOLUTION=2000

# Run app with example data (True) or not (False)
USE_EXAMPLE_DATA = False
EXAMPLE_DATA_PATH = 'assets/defaults/example_data'

# Query parameters
SHAPE_SIZE_MIN = 0.0
SHAPE_SIZE_MAX = 0.5
DDIM_STEPS_MIN = 10
DDIM_STEPS_MAX = 200
    
with gr.Blocks() as demo:

    # used to generate names for towns
    town_name_sampler = gr.State(TownNameSampler)
    # state to keep track of the generation method
    town_generation_method = gr.State('Random')

    if not USE_EXAMPLE_DATA:
        height_shape_generator = HeightShapeGenerator()
        # Town generator object
        # town_generator = TownGenerator(config_path = "out/city_gen/config.yaml",
        #                                model_config_path="out/city_gen/model_config.yaml",
        #                                checkpoint_path="out/city_gen/checkpoint.ckpt")
        
    # current states
    shape_state = gr.State(None)
    height_map_state = gr.State(None)
    towns_state = gr.State(list)
    towns_to_connect_state = gr.State(list)
    roads_state = gr.State(RoadGraph.empty)
    current_map_image = gr.State(None) # TODO: Cache the map wo polygon plotted a better way
    
    poly_points_state = gr.State(list)
    poly_closed_state = gr.State(False)
    
    example_data_nr = gr.State(None)

    canvas_edit_mode = gr.State('shape')

    town_updater = gr.State(2) # TODO: Very big hack, but I want to sleep
    
    with gr.Row():

        # config column
        with gr.Column(scale = 2):
            title = gr.HTML("<h1>Fantasy map generator</h1>")
            with gr.Row():
                # ----------------------------- Shape generation ----------------------------- #
                with gr.Column():
                    run_generate_shape_button = gr.Button('Generate map outline')
                    with gr.Accordion('Advanced outline config', open=False):
                        shape_size_slider = gr.Slider(value=0.5, label='Island Size', minimum=0.0, maximum=1.0, step=0.01, interactive=True)
                        shape_quality_slider = gr.Slider(value=1.0, label='Generation Speed(0) vs Quality(1)', minimum=0.0, maximum=1.0, step=0.01, interactive=True)
                
                # --------------------------- Height map generation -------------------------- #
                
                with gr.Column():
                    run_generate_height_map_button = gr.Button('Generate Height Map')
                    with gr.Accordion('Advanced height map config', open=True):
                        height_quality_slider = gr.Slider(value=1.0, label='Generation Speed(0) vs Quality(1)', minimum=0.0, maximum=1.0, step=0.01, interactive=True)

                # ------------------------------ Town generation ----------------------------- #
                with gr.Column():
                    with gr.Row():
                        run_generate_town_button = gr.Button('Generate Town(s)', scale = 3)
                        reset_town_button = gr.Button('Remove all Towns', scale=2, variant='secondary')
                    with gr.Accordion('Town config', open=True):
                        town_config_random_num_town_slider = gr.Slider(value=5, label='Number of Towns to Add', minimum=1, maximum=30, step=1, interactive=True)
                        
                        with gr.Row():
                            town_config_custom_town_type_radio = gr.Radio([("Random", "random")] + [(c.capitalize(), c) for c in TOWN_TYPES], value='random', label='Town Type', interactive=True)
                            town_config_custom_town_feature_radio = gr.Radio([('Random','random'), ('Coastal','coastal'), ('Inland','inland')], value = 'random', label='Town Features', interactive = True)

                # ------------------------------ Road generation ----------------------------- #
                with gr.Column():
                    with gr.Row():
                        run_generate_road_button = gr.Button('Connect towns with roads', scale = 3)
                        reset_roads_button = gr.Button('Remove all roads', scale = 2)
                    with gr.Accordion('Road config', open=False):
                        # which towns to connect
                        with gr.Accordion('Towns to connect'):
                            @gr.render(inputs=[towns_state], triggers=[towns_state.change])
                            def _(towns):
                                if len(towns) == 0:
                                    gr.Markdown('No towns to connect')
                                else:
                                    for i, town in enumerate(towns):

                                        with gr.Row():
                                            checkbox = gr.Checkbox(value=False, label='', interactive=True, show_label=False, container=False, scale=1, min_width=1000)                                            
                                            text_box = gr.Textbox(value=town['town_name'], lines=1, max_lines=1, interactive=True, show_label=False, container=False, scale=5)
                                            def add_town_to_connect(connect: bool, towns: list, town_name: str):
                                                if connect:
                                                    if not town_name in towns:
                                                        towns.append(town_name)
                                                else:
                                                    if town_name in towns:
                                                        towns.remove(town_name)
                                                return towns
                                            checkbox.input(add_town_to_connect, inputs=[checkbox, towns_to_connect_state, text_box], outputs=[towns_to_connect_state])
                        
                        with gr.Accordion('Advanced road heuristic config', open=False):
                            road_config_road_cost = gr.Number(1, label='Road Cost')
                            road_config_slope_factor = gr.Number(10, label='Max Slope Cost Factor')
                            road_config_bridge_factor = gr.Number(100, label='Bridge Cost Factor')
                                        
        # -------------------------------- Display map ------------------------------- #
        with gr.Column(scale = 3):
            with gr.Tab('Map'):
                output_image = gr.Image(
                    value=plot_map(return_step='empty', resolution=RESOLUTION),
                    label='Map',
                    interactive=False,
                    image_mode='RGB',
                    show_download_button=False)                
                with gr.Row():
                    gr.DownloadButton('Download PNG')
                    gr.DownloadButton('Download JSON')
                
            with gr.Tab('3d Mesh'):
                output_mesh_hm = gr.Model3D(
                        value=None,
                        interactive=False
                    )
                generate_mesh_button = gr.Button('Generate 3D Mesh')
                gr.DownloadButton('Download 3D Mesh')
        
            # ----------------------------- Intervention Tabs ---------------------------- #
            gr.HTML("<H2>Custom editor</H2><p>Click on the map to edit the landscape</p>")
            edit_shape_tab = gr.Tab('Outline editor')
            with edit_shape_tab:
                gr.HTML("<p> Click to draw an area that should be changed\n WARNING! This will remove any existing towns and roads</p>")
                with gr.Row():                  
                    close_btn = gr.Button('Close Polygon') # Changes to Open Polygon when closed
                    clear_btn = gr.Button('Clear Points')
                with gr.Row():
                    poly_add_land_btn = gr.Button('Fill with land')
                    poly_add_sea_btn = gr.Button('Fill with water')
                    shape_poly_regen_area_btn = gr.Button('Generative infill')
            edit_height_tab = gr.Tab("Height map editor")
            with edit_height_tab:
                with gr.Row():                  
                    close_btn = gr.Button('Close Polygon') # Changes to Open Polygon when closed
                    clear_btn = gr.Button('Clear Points')
                with gr.Row():
                    height_poly_regen_area_btn = gr.Button('Generative infill')
            edit_town_tab = gr.Tab("Town editor")
            with edit_town_tab:
                town_to_move = gr.Dropdown([], label='Select a town and click on the canvas to move it.')
                @towns_state.change(
                    inputs=[towns_state],
                    outputs=[town_to_move])
                def _dd(towns):
                    towns = [town['town_name'] for town in towns]
                    return gr.update(choices=towns)
                

    # ---------------------------------------------------------------------------- #
    #                                     utils                                    #
    # ---------------------------------------------------------------------------- #
    def scale_value(value, min_value, max_value, dtype: Literal['float', 'int']='float'):
        scaled_value = min_value + (value * (max_value - min_value))
        if dtype == 'int':
            return int(scaled_value)
        return scaled_value
    
    @town_updater.change(
        inputs=[shape_state, height_map_state, towns_state, roads_state],
        outputs=[output_image, current_map_image])
    def plot_map_after_town_update(shape, height_map, towns, roads):
        if roads != RoadGraph.empty():
            return_step = 'roads'
        else:
            return_step = 'towns'
        chart = plot_map(shape, height_map, towns, roads, return_step=return_step, resolution=RESOLUTION)
        return chart, chart

    # ---------------------------------------------------------------------------- #
    #                             Canvas Interventions                             #
    # ---------------------------------------------------------------------------- #
    edit_modes =['shape', 'height', 'town']
    @edit_shape_tab.select(
        outputs=[canvas_edit_mode])
    @edit_height_tab.select(
        outputs=[canvas_edit_mode])
    @edit_town_tab.select(
        outputs=[canvas_edit_mode])
    def change_edit_mode(evt: gr.SelectData):
        return edit_modes[evt.index]

    
    @output_image.select(
        inputs=[canvas_edit_mode,
                shape_state,
                poly_points_state, 
                poly_closed_state, 
                town_to_move, 
                towns_state,
                height_map_state, 
                town_updater],
        outputs=[poly_points_state, towns_state, town_updater])
    def canvas_interaction(edit_mode: str,
                           shape: np.ndarray,
                           poly_points: list, 
                           poly_closed: bool, 
                           town_to_move_name: str, 
                           towns: list, 
                           height_map: np.ndarray,
                           town_updater, 
                           evt: gr.SelectData):
        if shape is not None:            
            if edit_mode == 'shape' or edit_mode == 'height':
                # Note: poly points aren't stored in downscaled form
                poly_points = add_point(poly_points, poly_closed, evt.index)
            elif edit_mode == 'town' and height_map is not None:
                scaled_click_point_xy = (int(evt.index[0] * shape.shape[1] / RESOLUTION), 
                                        int(evt.index[1] * shape.shape[0] / RESOLUTION))
                for town in towns:
                    if town['town_name'] == town_to_move_name:
                        town['xyz'] = [scaled_click_point_xy[0], 
                                        scaled_click_point_xy[1], 
                                        height_map[[scaled_click_point_xy[1]], [scaled_click_point_xy[0]]]]
                        town_updater *= -1
                        break
        return poly_points, towns, town_updater
    # ---------------------------------------------------------------------------- #
    #                              Polygon Edit Shape                              #
    # ---------------------------------------------------------------------------- #
    # ------------------------------- Edit polygon ------------------------------- #
    def add_point(poly_points: list, poly_closed: bool, coordinates: tuple):
        if poly_closed:
            gr.Warning("Polygon is closed, to add more points open the polygon!")
            return poly_points
        
        poly_points.append(coordinates)
        return poly_points

    @close_btn.click(
        inputs=[poly_closed_state],
        outputs=[poly_closed_state])
    def close_polygon(poly_closed):
        if poly_closed:
            poly_closed = False
        else:
            poly_closed = True
        return poly_closed

    @poly_closed_state.change(
        inputs=[poly_closed_state],
        outputs=[close_btn])
    def update_close_btn_text(poly_closed):
        return 'Open Polygon' if poly_closed else 'Close Polygon'
    
    # ------------------------------- Clear points ------------------------------- #
    @current_map_image.change(
        inputs=[poly_points_state],
        outputs=[poly_points_state, poly_closed_state, close_btn])
    @clear_btn.click(
        inputs=[poly_points_state],
        outputs=[poly_points_state, poly_closed_state, close_btn])
    @output_image.clear(
        inputs=[poly_points_state],
        outputs=[poly_points_state, poly_closed_state, close_btn])
    def clear_points(points: list):
        points.clear()
        return points, False, 'Close Polygon'
    # ------------------------------ Add/remove area ----------------------------- #
    # TODO: If time allows it, make sure this function only gets called once per change,
    # right now it can be called multiple times but it doesn't take much time so it's not a big issue
    @poly_add_land_btn.click(
        inputs=[shape_state,
                height_map_state,
                towns_state,
                towns_to_connect_state,
                roads_state,
                poly_points_state, 
                poly_closed_state,
                current_map_image,
                output_image,
                gr.State('add')],
        outputs=[shape_state,
                 height_map_state,
                 towns_state,
                 towns_to_connect_state,
                 roads_state,
                 poly_points_state, 
                 poly_closed_state,
                 current_map_image,
                 output_image])
    @poly_add_sea_btn.click(
        inputs=[shape_state,
                height_map_state,
                towns_state,
                towns_to_connect_state,
                roads_state,
                poly_points_state, 
                poly_closed_state,
                current_map_image,
                output_image,
                gr.State('subtract')],
        outputs=[shape_state,
                 height_map_state,
                 towns_state,
                 towns_to_connect_state,
                 roads_state,
                 poly_points_state, 
                 poly_closed_state,
                 current_map_image,
                 output_image])
    def add_subtract_area(shape_mask: np.ndarray,
                          height_map: np.ndarray,
                          towns: list,
                          towns_to_connect: list,
                          roads: RoadGraph,
                          poly_points: list, 
                          poly_closed: bool,
                          high_res_image: np.ndarray, 
                          display_image: np.ndarray,
                          mode: Literal['add', 'subtract']):
        if not poly_closed:
            gr.Warning('Polygon is not closed, please close the polygon first!')
            return shape_mask, height_map, towns, towns_to_connect, roads, poly_points, poly_closed, high_res_image, display_image
        # TODO: We can keep it in one resolution instead of this up and down scaling
        scale_factor =  shape_mask.shape[0] / high_res_image.shape[0]
        scaled_poly_points = np.array(poly_points) * scale_factor
        
        scaled_poly_points = np.array(scaled_poly_points, dtype=np.int32).reshape((-1, 1, 2))
        
        # Create a mask for the polygon
        polygon_mask = np.zeros_like(shape_mask, dtype=np.uint8)
        # Fill the polygon on the mask
        polygon_mask = cv2.fillPoly(polygon_mask, [scaled_poly_points], 1)
        
        if mode == 'add':
            shape_mask[polygon_mask == 1] = 1
        elif mode == 'subtract':
            shape_mask[polygon_mask == 1] = 0
        chart = plot_map(shape=shape_mask, height_map=None, towns=None, roads=None, return_step='shape', resolution=RESOLUTION)
        
        # Reset town and roads
        return shape_mask, None, list(), list(), RoadGraph.empty(), [], False, chart, chart

    @shape_poly_regen_area_btn.click(
        inputs=[],
        outputs=[])
    @height_poly_regen_area_btn.click(
        inputs=[],
        outputs=[])
    def regen_area():
        # Should be two separate functions
        gr.Warning('Not implemented yet, coming soon!')
    
    # ------------------------------ Update display ------------------------------ #
    @poly_points_state.change(
        inputs=[
            poly_points_state,
            poly_closed_state,
            current_map_image],
        outputs=[
            output_image])
    @poly_closed_state.change(
        inputs=[
            poly_points_state,
            poly_closed_state,
            current_map_image],
        outputs=[
            output_image])
    def change_shape_display(
        poly_points: List[List[float]],
        poly_closed: bool,
        generated_map: Optional[np.ndarray] = None
    ) -> dict:
        polygon_dict = {'poly_points': poly_points, 'poly_closed': poly_closed}
        chart = plot_map(shape=None, height_map=None, towns=None, roads=None, return_step='change_poly_shape', polygon_dict=polygon_dict, current_map=generated_map, resolution=RESOLUTION)

        return chart
    
    # ---------------------------------------------------------------------------- #
    #                             Generation functions                             #
    # ---------------------------------------------------------------------------- #    
    # ----------------------------------- Shape ---------------------------------- #
    @run_generate_shape_button.click(
        inputs=[
            shape_size_slider,
            shape_quality_slider,
            shape_state],
        outputs=[
            shape_state,
            current_map_image,
            output_image,
            example_data_nr])
    def generate_shape(
        size: float,
        quality: float,
        shape: Optional[np.ndarray]
    ) -> dict:
        example_data_nr = None

        shape_size = scale_value(size, SHAPE_SIZE_MIN, SHAPE_SIZE_MAX)
        ddim_steps = scale_value(quality, DDIM_STEPS_MIN, DDIM_STEPS_MAX, 'int')
        
        if USE_EXAMPLE_DATA:
            files = os.listdir(os.path.join(EXAMPLE_DATA_PATH, 'shapes'))
            example_data_nr = random.randint(0, len(files) - 1)
            shape_path = os.path.join(EXAMPLE_DATA_PATH, 'shapes', f'shape_{example_data_nr}.npy')
            shape = np.load(shape_path)
        else:
            shape = height_shape_generator.generate_shape(shape_size, ddim_steps)
            
        chart = plot_map(shape=shape, height_map=None, towns=None, roads=None, return_step='shape', resolution=RESOLUTION)
        return shape, chart, chart, example_data_nr

    # --------------------------- Height map generation -------------------------- #
    @run_generate_height_map_button.click(
        inputs=[
            shape_state,
            height_quality_slider,
            example_data_nr],
        outputs=[
            height_map_state,
            current_map_image,
            output_image])
    def generate_height_map(
        shape: np.ndarray,
        quality: float,
        example_data_nr: int,
        *,
        only_generate: bool = False
    ) -> dict:
        def uniform_to_height(u: np.ndarray, *, mean: float = 300.0) -> np.ndarray:
            """
            Transform uniform samples to height samples
            u ~ U(0, 1)
            height ~ Exp(mean)
            """
            return - mean * np.log(1 - u)
        if shape is None:
            shape = generate_shape
        ddim_steps = scale_value(quality, DDIM_STEPS_MIN, DDIM_STEPS_MAX, 'int')
        if USE_EXAMPLE_DATA:
            height_map = np.load(os.path.join(EXAMPLE_DATA_PATH, 'heights', f'height_{example_data_nr}.npy'))
        else:
            height_map = height_shape_generator.generate_height(shape, ddim_steps)
        height_map = uniform_to_height(height_map)
        height_map[shape == 0] = -1

        if only_generate:
            return height_map
        chart = plot_map(shape, height_map, towns=None, roads=None, return_step='height_map', resolution=RESOLUTION)

        return height_map, chart, chart

    # ---------------------------------- 3D mesh --------------------------------- #
    @generate_mesh_button.click(
        inputs=[height_map_state],
        outputs=[output_mesh_hm])
    def generate_mesh(
        height_map: np.ndarray
    ) -> str:
        if height_map is None:
            gr.Warning('No height map generated yet, please generate a height map first!')
            return None
        output_path = 'assets/defaults/terrain_mesh.obj'
        output_path = heightmap_to_3d_mesh(height_map, output_path=output_path)
        return output_path

    # ----------------------------------- Towns ---------------------------------- #
    @run_generate_town_button.click(
        inputs=[
            shape_state,
            height_map_state,
            towns_state,
            roads_state,
            town_name_sampler,
            town_config_random_num_town_slider,
            town_config_custom_town_type_radio,
            town_config_custom_town_feature_radio],
        outputs=[
            towns_state,
            town_name_sampler,
            current_map_image,
            output_image])
    def generate_town(
        shape: Optional[shapely.MultiPolygon],
        height_map: Optional[np.ndarray],
        towns: List[dict],
        roads: RoadGraph,
        name_sampler: TownNameSampler,
        num_towns: int,
        town_type: TOWN_TYPE,
        town_config: List[str]
    ) -> dict:
        if height_map is None:
            gr.Warning('No height map generated yet, please generate a height map first!')
            
        possible_choices = np.where(height_map > 0)
        num_possible = len(possible_choices[0])
            
        if town_type == 'random':
            town_types = np.random.choice(TOWN_TYPES, num_towns, replace=True)
        else:
            town_types = [town_type] * num_towns
        
        if 'random' in town_config:
            is_coastals = np.random.choice([True, False], num_towns, replace=True)
        elif 'coastal' in town_config:
            is_coastals = [True] * num_towns
        else:
            is_coastals = [False] * num_towns
            
        new_names = [name_sampler.pop(town_type, is_coastal) for town_type, is_coastal in zip(town_types, is_coastals)]
        
        #TODO: INSERT TOWN GEN HERE 
        # towns = town_generator.generate(height_map, towns, new_names, town_types, is_coastals)
        for new_name, town_type, is_coastal in zip(new_names, town_types, is_coastals):
            idx = np.random.choice(num_possible)
            x = possible_choices[1][idx]
            y = possible_choices[0][idx]
            z = height_map[y, x]
            towns.append(
                Town(
                    town_type = town_type,
                    is_coastal = is_coastal,
                    xyz = [x,y,z],
                    town_name = new_name))
        
        chart = plot_map(shape, height_map, towns, roads, return_step='roads' if roads['edges'] else 'towns', resolution=RESOLUTION)

        return towns, name_sampler, chart, chart

    # ----------------------------------- Roads ---------------------------------- #
    @run_generate_road_button.click(
        inputs=[
            shape_state,
            height_map_state,
            towns_state,
            roads_state,
            road_config_road_cost,
            road_config_slope_factor, 
            road_config_bridge_factor,
            towns_to_connect_state],
        outputs=[
            roads_state,
            current_map_image,
            output_image])
    def generate_road(
        shape: Optional[shapely.MultiPolygon],
        height_map: Optional[np.ndarray],
        towns: List[dict],
        roads: RoadGraph,
        road_config_road_cost: float,
        road_config_slope_factor: float,
        road_config_bridge_factor: float,
        towns_to_connect: List[str]
    ):
        # Generate RoadGraph for everythiiiiing #TODO: Use the road_config to config which towns to connect
        nodes = {}
        for town in towns:
            if town['town_name'] in towns_to_connect:
                nodes[town['town_name']] = RoadNode(is_city=True, xyz=town['xyz'])
        
        heuristic = BridgeEuclideanHeuristic(height_map, 
                                             0.5,  
                                             cost_per_km=road_config_road_cost, 
                                             max_slope_cost_factor=road_config_slope_factor, 
                                             bridge_cost_factor=road_config_bridge_factor)
        astar = AStar(height_map, heuristic, max_runtime=10)
        greedy = GreedyBestFirst(height_map, heuristic)
        edges = {}
        for town1, town2 in itertools.combinations(nodes, 2):
            path, _ = astar.find_path(nodes[town1]['xyz'][:2], nodes[town2]['xyz'][:2])
            if path == 'timeout':
                path, _ = greedy.find_path(nodes[town1]['xyz'][:2], nodes[town2]['xyz'][:2])
            
            if path:
                edges[f'{town1}-{town2}'] = RoadEdge(line=shapely.LineString(path), connected_nodes=(town1, town2))
        
        road_graph = RoadGraph(nodes=nodes, edges=edges)
        chart = plot_map(shape, height_map, towns, road_graph, return_step='roads', resolution=RESOLUTION)
        return road_graph, chart, chart

    # ---------------------------------------------------------------------------- #
    #                                Reset Functions                               #
    # ---------------------------------------------------------------------------- #
    # ----------------------------------- Towns ---------------------------------- #
    @reset_town_button.click(
        inputs=[
            shape_state,
            height_map_state],
        outputs=[
            town_name_sampler,
            towns_state,
            roads_state,
            current_map_image,
            output_image])
    def reset_town(shape, height_map):
        chart = plot_map(shape, height_map, None, None, return_step='height_map', resolution=RESOLUTION)
        return TownNameSampler(), list(), RoadGraph.empty(), chart, chart

    # ----------------------------------- Roads ---------------------------------- #
    @reset_roads_button.click(
        inputs=[
            shape_state,
            height_map_state,
            towns_state],
        outputs=[
            roads_state,
            current_map_image,
            output_image])
    def reset_roads(shape, height_map, towns):
        chart = plot_map(shape, height_map, towns, None, return_step='towns', resolution=RESOLUTION)
        return RoadGraph.empty(), chart, chart


if __name__ == '__main__':
    demo.launch()
