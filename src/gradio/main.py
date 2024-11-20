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

# Used to store downloadable files, Note! This won't work with multiple users
GRADIO_TMP_DATA_FOLDER='data/gradio_tmp'
os.makedirs(GRADIO_TMP_DATA_FOLDER, exist_ok=True)

RESOLUTION=2000

# Run app with example data (True) or not (False)
USE_EXAMPLE_DATA = False
EXAMPLE_DATA_PATH = 'assets/defaults/example_data'

# Query parameters
SHAPE_SIZE_MIN = 0.0
SHAPE_SIZE_MAX = 0.5
DDIM_STEPS_MIN = 10
DDIM_STEPS_MAX = 500
    
with gr.Blocks() as demo:

    # used to generate names for towns
    town_name_sampler = gr.State(TownNameSampler)
    # state to keep track of the generation method
    town_generation_method = gr.State('Random')

    if not USE_EXAMPLE_DATA:
        height_shape_generator = HeightShapeGenerator()
        # TODO: Town generator object
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
                        with gr.Row():
                            manual_shape_size = gr.Checkbox(value=False, label='Manually query shape size', interactive=True)
                            shape_size_slider = gr.Slider(value=0.5, label='Island Size', info='Choose a higher number for larger', minimum=0.0, maximum=1.0, step=0.01, visible=False, interactive=False)
                            @manual_shape_size.input(inputs=[manual_shape_size, shape_size_slider], outputs=[shape_size_slider])
                            def _enable_shape_slider(checked: bool, value: float):
                                if checked:
                                    return gr.Slider(value=value, label='Island Size', info='Choose a higher number for larger', minimum=0.0, maximum=1.0, step=0.01, interactive=True, visible=True)
                                else:
                                    return gr.Slider(value=value, label='Island Size', info='Choose a higher number for larger', minimum=0.0, maximum=1.0, step=0.01, visible=False, interactive=False)
                                    
                        shape_quality_slider = gr.Slider(value=1.0, label='Generation Quality', info='Higher value means better quality but slower generation.', minimum=0.0, maximum=1.0, step=0.01, interactive=True)
                
                # --------------------------- Height map generation -------------------------- #
                
                with gr.Column():
                    run_generate_height_map_button = gr.Button('Generate Height Map')
                    with gr.Accordion('Advanced height map config', open=True):
                        height_quality_slider = gr.Slider(value=1.0, label='Generation Quality', info='Higher value means better quality but slower generation.', minimum=0.0, maximum=1.0, step=0.01, interactive=True)

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
                    def download_map(current_map_image):
                        if current_map_image is None:
                            current_map_image = plot_map(return_step='empty', resolution=RESOLUTION)
                        save_path = os.path.join(GRADIO_TMP_DATA_FOLDER,'gradio_tmp.png')
                        bgr_im = cv2.cvtColor(current_map_image, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(save_path, bgr_im)
                        return save_path
                    gr.DownloadButton('Download PNG', value=download_map, inputs=current_map_image)
                
            with gr.Tab('3d Mesh'):
                output_mesh_hm = gr.Model3D(
                        value=None,
                        interactive=False
                    )
                generate_mesh_button = gr.Button('Generate 3D Mesh')
                mesh_download_btn = gr.DownloadButton('Download 3D Mesh')
                @mesh_download_btn.click(
                    inputs=[output_mesh_hm],
                    outputs=[mesh_download_btn])
                def download_mesh(output_mesh_path):
                    if output_mesh_path is None: 
                        gr.Warning('No mesh generated yet, please generate before downloading')
                    return output_mesh_path
                
            # ----------------------------- Intervention Tabs ---------------------------- #
            gr.HTML("<H2>Custom editor</H2><p>Click on the map to edit the landscape</p>")
            edit_shape_tab = gr.Tab('Outline editor')
            with edit_shape_tab:
                gr.HTML("<p> Click to draw an area that should be changed\n <b>WARNING!</b> <i>This will completely reset the map to only the outline.</i></p>")
                with gr.Row():                  
                    close_btn = gr.Button('Close Polygon') # Changes to Open Polygon when closed
                    clear_btn = gr.Button('Clear Points')
                with gr.Row():
                    poly_add_land_btn = gr.Button('Fill with land')
                    poly_add_sea_btn = gr.Button('Fill with water')
                    shape_poly_regen_area_btn = gr.Button('Regenerate outline in the area')
            edit_height_tab = gr.Tab("Height map editor")
            with edit_height_tab:
                with gr.Row():                  
                    h_close_btn = gr.Button('Close Polygon') # Changes to Open Polygon when closed
                    h_clear_btn = gr.Button('Clear Points')
                with gr.Row():
                    height_poly_regen_area_btn = gr.Button('Regenerate only heights in the area')
                    shape_height_poly_regen_area_btn = gr.Button('Regenerate outline and heights in the area')
                    
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
    
    def scale_points_canvas_to_model_size(points_xy:np.ndarray, model_img_shape_hw:tuple):
        """
        points_xy: np.ndarray, shape=(n, 2)
        """
        w,h = model_img_shape_hw
        transformed_points = np.array([
            points_xy[:, 0] * w / RESOLUTION,
            points_xy[:, 1] * h / RESOLUTION
        ]).T.astype(int)
        return transformed_points
    
    def create_polygon_mask(unscaled_poly_points, shape: np.ndarray):
        scaled_poly_points = scale_points_canvas_to_model_size(np.array(unscaled_poly_points), shape.shape)
        scaled_poly_points = scaled_poly_points.reshape((-1, 1, 2))
        polygon_mask = np.zeros_like(shape, dtype=np.uint8)
        polygon_mask = cv2.fillPoly(polygon_mask, [scaled_poly_points], 1)
        return polygon_mask
    
    def uniform_to_height(u: np.ndarray, *, mean: float = 300.0) -> np.ndarray:
        """
        Transform uniform samples to height samples
        u ~ U(0, 1)
        height ~ Exp(mean)
        """
        return - mean * np.log(1 - u)
    
    def height_to_uniform(height: np.ndarray, *, mean: float = 300.0) -> np.ndarray:
        """
        Transform height samples to uniform samples
        height ~ Exp(mean)
        u ~ U(0, 1)
        """
        return 1 - np.exp(-height / mean)
    
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
                scaled_click_point_xy = scale_points_canvas_to_model_size(np.array(evt.index)[None], shape.shape)[0]
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

    @h_close_btn.click(
        inputs=[poly_closed_state, shape_state],
        outputs=[poly_closed_state])
    @close_btn.click(
        inputs=[poly_closed_state, shape_state],
        outputs=[poly_closed_state])
    def close_polygon(poly_closed, shape):
        if shape is not None:
            if poly_closed:
                poly_closed = False
            else:
                poly_closed = True
        else:
            gr.Warning('No shape generated yet, please generate a shape first!')
        return poly_closed

    @poly_closed_state.change(
        inputs=[poly_closed_state],
        outputs=[close_btn, h_close_btn])
    def update_close_btn_text(poly_closed):
        return 2 * ['Open Polygon' if poly_closed else 'Close Polygon']
    
    # ------------------------------- Clear points ------------------------------- #
    @h_clear_btn.click(
        inputs=[poly_points_state],
        outputs=[poly_points_state, poly_closed_state])
    @current_map_image.change(
        inputs=[poly_points_state],
        outputs=[poly_points_state, poly_closed_state])
    @clear_btn.click(
        inputs=[poly_points_state],
        outputs=[poly_points_state, poly_closed_state])
    @output_image.clear(
        inputs=[poly_points_state],
        outputs=[poly_points_state, poly_closed_state])
    def clear_points(points: list):
        points.clear()
        return points, False, 'Close Polygon'
    
    # ------------------------------ Add/remove area ----------------------------- #
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
        if len(poly_points) < 3:
            gr.Warning('Polygon needs at least 3 points!')
            return shape_mask, height_map, towns, towns_to_connect, roads, poly_points, poly_closed, high_res_image, display_image
    
        polygon_mask = create_polygon_mask(poly_points, shape_mask)
        
        if mode == 'add':
            shape_mask[polygon_mask == 1] = 1
        elif mode == 'subtract':
            shape_mask[polygon_mask == 1] = 0
        chart = plot_map(shape=shape_mask, height_map=None, towns=None, roads=None, return_step='shape', resolution=RESOLUTION)
        
        # Reset town and roads
        return shape_mask, None, list(), list(), RoadGraph.empty(), [], False, chart, chart

        # ------------------------------ Regenerate area ----------------------------- #     
    @shape_poly_regen_area_btn.click(
        inputs=[shape_state,
                height_map_state,
                towns_state,
                towns_to_connect_state,
                roads_state,
                poly_points_state, 
                poly_closed_state, 
                current_map_image, 
                output_image],
        outputs=[shape_state,
                 height_map_state,
                 towns_state,
                 towns_to_connect_state,
                 roads_state,
                 current_map_image, 
                 output_image])
    def infill_shape(shape:np.ndarray, 
                     height_map: Optional[np.ndarray],
                     towns_state: Optional[list],
                     towns_to_connect: Optional[list],
                     roads: Optional[RoadGraph],
                     poly_points: list, 
                     poly_closed: bool, 
                     generated_map: np.ndarray, 
                     display_map: np.ndarray):
        if USE_EXAMPLE_DATA:
            gr.Warning('Example data not supported for this feature!')
            return shape, height_map, towns_state, towns_to_connect, roads, generated_map, display_map
        if not poly_closed or len(poly_points) < 3:
            gr.Warning('Polygon needs to be closed and have at least 3 points!')
            return shape, height_map, towns_state, towns_to_connect, roads, generated_map, display_map
        
        polygon_mask = np.logical_not(create_polygon_mask(poly_points, shape)).astype(np.uint8)
        regenerated_shape = height_shape_generator.regenerate_shape(shape, polygon_mask, batch_size=1).squeeze(0)
        
        chart = plot_map(shape=regenerated_shape, height_map=None, towns=None, roads=None, return_step='shape', resolution=RESOLUTION)
        return regenerated_shape, None, list(), list(), RoadGraph.empty(), chart, chart


    @height_poly_regen_area_btn.click(
        inputs=[shape_state,
                height_map_state,
                towns_state,
                towns_to_connect_state,
                roads_state,
                poly_points_state, 
                poly_closed_state, 
                current_map_image, 
                output_image],
        outputs=[shape_state,
                 height_map_state,
                 towns_state,
                 towns_to_connect_state,
                 roads_state,
                 current_map_image, 
                 output_image])
    def infill_height(shape:np.ndarray, 
                     height_map: Optional[np.ndarray],
                     towns_state: Optional[list],
                     towns_to_connect: Optional[list],
                     roads: Optional[RoadGraph],
                     poly_points: list, 
                     poly_closed: bool, 
                     generated_map: np.ndarray, 
                     display_map: np.ndarray):
        if USE_EXAMPLE_DATA:
            gr.Warning('Example data not supported for this feature!')
            return shape, height_map, towns_state, towns_to_connect, roads, generated_map, display_map
        if not poly_closed or len(poly_points) < 3:
            gr.Warning('Polygon needs to be closed and have at least 3 points!')
            return shape, height_map, towns_state, towns_to_connect, roads, generated_map, display_map
        
        polygon_mask = np.logical_not(create_polygon_mask(poly_points, shape)).astype(np.uint8)
        regenerated_height = height_shape_generator.regenerate_height(height_to_uniform(height_map), shape, polygon_mask, batch_size=1).squeeze(0)
        regenerated_height = uniform_to_height(regenerated_height)
        #TODO: Should be masked?
        chart = plot_map(shape=shape, height_map=regenerated_height, towns=None, roads=None, return_step='height_map', resolution=RESOLUTION)
        return shape, regenerated_height, list(), list(), RoadGraph.empty(), chart, chart
    
    
    @shape_height_poly_regen_area_btn.click(
        inputs=[shape_state,
                height_map_state,
                towns_state,
                towns_to_connect_state,
                roads_state,
                poly_points_state, 
                poly_closed_state, 
                current_map_image, 
                output_image],
        outputs=[shape_state,
                 height_map_state,
                 towns_state,
                 towns_to_connect_state,
                 roads_state,
                 current_map_image, 
                 output_image])
    def infill_shape_and_height(shape:np.ndarray, 
                     height_map: Optional[np.ndarray],
                     towns_state: Optional[list],
                     towns_to_connect: Optional[list],
                     roads: Optional[RoadGraph],
                     poly_points: list, 
                     poly_closed: bool, 
                     generated_map: np.ndarray, 
                     display_map: np.ndarray):
        if USE_EXAMPLE_DATA:
            gr.Warning('Example data not supported for this feature!')
            return shape, height_map, towns_state, towns_to_connect, roads, generated_map, display_map
        if not poly_closed or len(poly_points) < 3:
            gr.Warning('Polygon needs to be closed and have at least 3 points!')
            return shape, height_map, towns_state, towns_to_connect, roads, generated_map, display_map
        
        polygon_mask = np.logical_not(create_polygon_mask(poly_points, shape)).astype(np.uint8)
        regenerated_shape, regenerated_height = height_shape_generator.regenerate_shape_and_height(shape, height_to_uniform(height_map), polygon_mask, batch_size=1)
        regenerated_shape = regenerated_shape.squeeze(0)
        regenerated_height = regenerated_height.squeeze(0)
        regenerated_height = uniform_to_height(regenerated_height)
        chart = plot_map(shape=regenerated_shape, height_map=regenerated_height, towns=None, roads=None, return_step='shape', resolution=RESOLUTION)
        
        return regenerated_shape, regenerated_height, list(), list(), RoadGraph.empty(), chart, chart
        
    
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
            manual_shape_size,
            shape_quality_slider],
        outputs=[
            shape_state,
            height_map_state,
            towns_state,
            towns_to_connect_state,
            roads_state,
            current_map_image,
            output_image,
            example_data_nr])
    def generate_shape(
        size: float,
        manual_size: bool,
        quality: float,
        *,
        only_generate: bool = False 
        ) -> dict:
        example_data_nr = None
        if manual_size:
            shape_size = scale_value(size, SHAPE_SIZE_MIN, SHAPE_SIZE_MAX)
        else:
            shape_size = None
        ddim_steps = scale_value(quality, DDIM_STEPS_MIN, DDIM_STEPS_MAX, 'int')
        
        if USE_EXAMPLE_DATA:
            files = os.listdir(os.path.join(EXAMPLE_DATA_PATH, 'shapes'))
            example_data_nr = random.randint(0, len(files) - 1)
            shape_path = os.path.join(EXAMPLE_DATA_PATH, 'shapes', f'shape_{example_data_nr}.npy')
            shape = np.load(shape_path)
        else:
            shape = height_shape_generator.generate_shape(shape_size, ddim_steps, batch_size=1).squeeze(0)
        
        if not only_generate:
            chart = plot_map(shape=shape, height_map=None, towns=None, roads=None, return_step='shape', resolution=RESOLUTION)
            return shape, None, list(), list(), RoadGraph.empty(), chart, chart, example_data_nr
        else:
            return shape, None, None, None, None, None, None, None

    # --------------------------- Height map generation -------------------------- #
    @run_generate_height_map_button.click(
        inputs=[
            shape_state,
            manual_shape_size,
            shape_quality_slider,
            height_quality_slider,
            example_data_nr],
        outputs=[
            shape_state,
            height_map_state,
            towns_state,
            towns_to_connect_state,
            roads_state,
            current_map_image,
            output_image])
    def generate_height_map(
        shape: np.ndarray,
        manual_shape_size: bool,
        size: float,
        quality: float,
        example_data_nr: int,
        *,
        only_generate: bool = False
    ) -> dict:
        if shape is None:
            shape = generate_shape(manual_shape_size, size, quality, only_generate=True)[0]
        ddim_steps = scale_value(quality, DDIM_STEPS_MIN, DDIM_STEPS_MAX, 'int')
        if USE_EXAMPLE_DATA:
            height_map = np.load(os.path.join(EXAMPLE_DATA_PATH, 'heights', f'height_{example_data_nr}.npy'))
        else:
            height_map = height_shape_generator.generate_height(shape, ddim_steps, batch_size=1).squeeze(0)
        height_map = uniform_to_height(height_map)
        height_map[shape == 0] = -1

        if only_generate:
            return height_map, None, None, None, None, None, None
        chart = plot_map(shape, height_map, towns=None, roads=None, return_step='height_map', resolution=RESOLUTION)

        return shape, height_map, list(), list(), RoadGraph.empty(), chart, chart

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
        output_path = os.path.join(GRADIO_TMP_DATA_FOLDER, 'terrain_mesh.obj')
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
            town_config_custom_town_feature_radio,
            current_map_image,
            output_image],
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
        town_config: List[str],
        current_map: np.ndarray,
        display_map: np.ndarray
    ) -> dict:
        if height_map is None:
            gr.Warning('No height map generated yet, please generate a height map first!')
            return towns, name_sampler, current_map, display_map
            
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
        
        # TODO: Uncomment if-else when town gen is up and running
        # if USE_EXAMPLE_DATA:
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
        # else:
        #   towns = town_generator.generate(height_map, towns, new_names, town_types, is_coastals)

        
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
            towns_to_connect_state, 
            current_map_image,
            output_image],
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
        towns_to_connect: List[str],
        current_map: np.ndarray,
        display_map: np.ndarray
    ):
        if len(towns) == 0 or len(towns_to_connect) < 2:
            gr.Warning('Need at least 2 towns to connect!')
            return roads, current_map, display_map
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
