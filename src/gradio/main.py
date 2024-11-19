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

from src.gradio.town_generator import TownGenerator


# Run app with example data (True) or not (False)
USE_EXAMPLE_DATA = True
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
    
    # Town generator object
    town_generator = TownGenerator(config_path = "out/city_gen/config.yaml",
                                   model_config_path="out/city_gen/model_config.yaml",
                                   checkpoint_path="out/city_gen/checkpoint.ckpt")

    # current states
    shape_state = gr.State(None)
    height_map_state = gr.State(None)
    towns_state = gr.State(list)
    roads_state = gr.State(RoadGraph.empty)
    current_map_image = gr.State(None) # TODO: Cache the map wo polygon plotted a better way
    
    poly_points_state = gr.State(list)
    poly_closed_state = gr.State(False)
    
    example_data_nr = gr.State(None)

    title = gr.HTML("<h1>Fantasy map generator</h1>")
    description = gr.HTML("""
        <p>Generate a fantasy map with islands, towns, and roads.</p>
        <p>Generate elements in the the order: shape -> heights -> towns -> roads</p>
        <p>In each iteration process click on the map to manually augment the map.</p>
    """)

    with gr.Row():

        # config column
        with gr.Column(scale = 2):
            with gr.Row():
                # ----------------------------- Shape generation ----------------------------- #
                with gr.Column():
                    with gr.Accordion('Shape config', open=False):
                        shape_size_slider = gr.Slider(value=0.5, label='Island Size', minimum=0.0, maximum=1.0, step=0.01, interactive=True)
                        shape_quality_slider = gr.Slider(value=1.0, label='Generation Speed(0) vs Quality(1)', minimum=0.0, maximum=1.0, step=0.01, interactive=True)
                    run_generate_shape_button = gr.Button('Generate Shape')
                
                # --------------------------- Height map generation -------------------------- #
                
                with gr.Column():
                    with gr.Accordion('Height Map config', open=False):
                        height_quality_slider = gr.Slider(value=1.0, label='Generation Speed(0) vs Quality(1)', minimum=0.0, maximum=1.0, step=0.01, interactive=True)
                    run_generate_height_map_button = gr.Button('Generate Height Map')

                # ------------------------------ Town generation ----------------------------- #
                with gr.Column():
                    with gr.Accordion('Town config', open=False):
                        # with gr.Tab('Random') as town_config_random_tab:
                            # town_config_random_num_town_slider = gr.Slider(value=5, label='Number of Towns to Add', minimum=1, maximum=30, step=1, interactive=True)
                            # @town_config_random_tab.select(outputs=[town_generation_method])
                            # def _(): return 'Random'
                        # with gr.Tab('Custom') as town_config_custom_tab:
                        town_config_random_num_town_slider = gr.Slider(value=5, label='Number of Towns to Add', minimum=1, maximum=30, step=1, interactive=True)
                        town_config_custom_town_type_radio = gr.Radio([("Random", "random")] + [(c.capitalize(), c) for c in TOWN_TYPES], value='random', label='Town Type', interactive=True)
                        town_config_custom_town_feature_radio = gr.Radio([('Random','random'), ('Coastal','coastal'), ('Inland','inland')], value = 'random', label='Town Features', interactive = True)
                        # @town_config_custom_tab.select(outputs=[town_generation_method])
                        # def _(): return 'Custom'
                        reset_town_button = gr.Button('Reset Towns')
                    run_generate_town_button = gr.Button('Generate Town(s)')

                # ------------------------------ Road generation ----------------------------- #
                with gr.Column():
                    with gr.Accordion('Road config', open=False):
                        # which towns to connect
                        with gr.Accordion('Towns to connect'):
                            @gr.render(inputs=[towns_state], triggers=[towns_state.change])
                            def _(towns):
                                if len(towns) == 0:
                                    gr.Markdown('No towns to connect')
                                else:
                                    for i, town in enumerate(towns):
                                        # TODO: add checkbox to select town and figure out how dynamic inputs work
                                        # https://www.gradio.app/guides/dynamic-apps-with-render-decorator
                                        # https://www.gradio.app/docs/gradio/checkbox
                                        with gr.Row():
                                            checkbox = gr.Checkbox(value=False, label='', interactive=True, show_label=False, container=False, scale=1, min_width=100)
                                            text_box = gr.Textbox(value=town['town_name'], lines=1, max_lines=1, interactive=True, show_label=False, container=False, scale=5)

                                            # towns_state[i]['town_name'] = text_box.value
                                            # towns_state[i]['connect'] = checkbox.value
                                            #gr.ClearButton(text_box, value=None, size='sm', icon=CLOSE_ICON, scale=1, min_width=10)
                        reset_roads_button = gr.Button('Reset Roads')
                        
                        with gr.Accordion('Road Cost Config', open=False):
                            road_config_road_cost = gr.Number(1, label='Road Cost')
                            road_config_slope_factor = gr.Number(1, label='Max Slope Cost Factor')
                            road_config_bridge_factor = gr.Number(100, label='Bridge Cost Factor')
                    run_generate_road_button = gr.Button('Generate Road')
            # --------------------------- Polygon shape change --------------------------- #
            with gr.Accordion('Shape Editor    (Warning! Will change height and reset towns+roads)'):
                with gr.Row():                  
                    close_btn = gr.Button('Close Polygon') # Changes to Open Polygon when closed
                    clear_btn = gr.Button('Clear Points')
                with gr.Row():
                    poly_add_land_btn = gr.Button('Add Land')
                    poly_add_sea_btn = gr.Button('Add Sea')
                    poly_regen_area_btn = gr.Button('Regenerate Area')
                    
        # -------------------------------- Display map ------------------------------- #
        with gr.Column(scale = 3):
            with gr.Tab('Map'):
                output_image = gr.Image(
                    value=plot_map(return_step='empty'),
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
    
    # ---------------------------------------------------------------------------- #
    #                                     utils                                    #
    # ---------------------------------------------------------------------------- #
    def scale_value(value, min_value, max_value, dtype: Literal['float', 'int']='float'):
        scaled_value = (value - min_value) / (max_value - min_value)
        if dtype == 'int':
            return int(scaled_value)
        return scaled_value
            
    # ---------------------------------------------------------------------------- #
    #                              Polygon Edit Shape                              #
    # ---------------------------------------------------------------------------- #
    # ------------------------------- Edit polygon ------------------------------- #
    @output_image.select(
        inputs=[poly_points_state, poly_closed_state],
        outputs=[poly_points_state])
    def add_point(poly_points: list, poly_closed: bool, evt: gr.SelectData):
        if poly_closed:
            gr.Warning("Polygon is closed, to add more points open the polygon!")
            return poly_points
        
        poly_points.append(evt.index)
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
    # TODO THIS FUNCTION IS SUPER BAD; A LOT GETS RERUN MULTIPLE TIMES BUT I NEED SLEEP OKAY
    # THIS FUNCTION WILL BE REVAMPED WHEN MODELS ARE IMPLEMENTED
    @poly_add_land_btn.click(
        inputs=[shape_state,
                height_map_state,
                towns_state,
                roads_state,
                poly_points_state, 
                poly_closed_state,
                current_map_image,
                output_image,
                gr.State('add')], # TODO: Hotfix for now
        outputs=[shape_state,
                 height_map_state,
                 towns_state,
                 roads_state,
                 poly_points_state, 
                 poly_closed_state,
                 current_map_image,
                 output_image])
    @poly_add_sea_btn.click(
        inputs=[shape_state,
                height_map_state,
                towns_state,
                roads_state,
                poly_points_state, 
                poly_closed_state,
                current_map_image,
                output_image,
                gr.State('subtract')],
        outputs=[shape_state,
                 height_map_state,
                 towns_state,
                 roads_state,
                 poly_points_state, 
                 poly_closed_state,
                 current_map_image,
                 output_image])
    def add_subtract_area(shape_mask: np.ndarray,
                          height_map: np.ndarray,
                          towns: list,
                          roads: RoadGraph,
                          poly_points: list, 
                          poly_closed: bool,
                          high_res_image: np.ndarray, 
                          display_image: np.ndarray,
                          mode: Literal['add', 'subtract']):
        if not poly_closed:
            gr.Warning('Polygon is not closed, please close the polygon first!')
            return shape_mask, height_map, towns, roads, poly_points, poly_closed, high_res_image, display_image
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
        chart = plot_map(shape=shape_mask, height_map=None, towns=None, roads=None, return_step='shape')
        
        # Reset town and roads
        return shape_mask, None, list(), RoadGraph.empty(), [], False, chart, chart
    
    @poly_regen_area_btn.click(
        inputs=[],
        outputs=[])
    def regen_area():
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
        chart = plot_map(shape=None, height_map=None, towns=None, roads=None, return_step='change_poly_shape', polygon_dict=polygon_dict, current_map=generated_map)

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

        if USE_EXAMPLE_DATA:
            files = os.listdir(os.path.join(EXAMPLE_DATA_PATH, 'shapes'))
            example_data_nr = random.randint(0, len(files) - 1)
            shape_path = os.path.join(EXAMPLE_DATA_PATH, 'shapes', f'shape_{example_data_nr}.npy')
            shape = np.load(shape_path)
            
        shape_size = scale_value(size, SHAPE_SIZE_MIN, SHAPE_SIZE_MAX)
        ddim_steps = scale_value(quality, DDIM_STEPS_MIN, DDIM_STEPS_MAX)
        chart = plot_map(shape=shape, height_map=None, towns=None, roads=None, return_step='shape')
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
        if USE_EXAMPLE_DATA:
            height_map = np.load(os.path.join(EXAMPLE_DATA_PATH, 'heights', f'height_{example_data_nr}.npy'))
            height_map = uniform_to_height(height_map)
            height_map[shape == 0] = -1
            
        ddim_steps = scale_value(quality, DDIM_STEPS_MIN, DDIM_STEPS_MAX)
        if only_generate:
            return height_map
        chart = plot_map(shape, height_map, towns=None, roads=None, return_step='height_map')

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
            # town_generation_method,
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
        # generation_method: Literal['Random', 'Custom'],
        num_towns: int,
        town_type: TOWN_TYPE,
        town_config: List[str]
    ) -> dict:
        possible_choices = np.where(height_map > 0)
        num_possible = len(possible_choices[0])

        if height_map is None:
            height_map = generate_height_map

        # if generation_method == 'Random':
            # for _ in range(num_towns):
            #     idx = np.random.choice(num_possible)
            #     z = height_map[possible_choices[0][idx], possible_choices[1][idx]]
            #     is_coastal = np.random.choice([True, False])
            #     town_type = np.random.choice(TOWN_TYPES)
                
            #     towns.append(
            #         Town(
            #             town_type=town_type,
            #             is_coastal=is_coastal,
            #             xyz=[possible_choices[1][idx], possible_choices[0][idx], z],
            #             town_name=name_sampler.pop(town_type, is_coastal)))

        # elif generation_method == 'Custom':
            # import pdb; pdb.set_trace()
            # is_coastal = 'is_coastal' in town_config
            # idx = np.random.choice(num_possible)
            # z = height_map[possible_choices[0][idx], possible_choices[1][idx]]
            
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
            
        # new_types = [town_type] * 10
        # new_is_coastals = ['is_coastal' in town_config] * 10
        new_names = [name_sampler.pop(town_type, is_coastal) for town_type, is_coastal in zip(town_types, is_coastals)]
        
        towns = town_generator.generate(height_map, towns, new_names, town_types, is_coastals)
            # towns.append(
            #     Town(
            #         town_type = town_type,
            #         is_coastal = is_coastal,
            #         xyz = [possible_choices[1][idx], possible_choices[0][idx], z],
            #         town_name = name_sampler.pop(town_type, is_coastal)))
        
        chart = plot_map(shape, height_map, towns, roads, return_step='roads' if roads['edges'] else 'towns')

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
            road_config_bridge_factor],
        outputs=[
            roads_state,
            output_image])
    def generate_road(
        shape: Optional[shapely.MultiPolygon],
        height_map: Optional[np.ndarray],
        towns: List[dict],
        roads: RoadGraph,
        road_config_road_cost: float,
        road_config_slope_factor: float,
        road_config_bridge_factor: float
    ):
        # Generate RoadGraph for everythiiiiing #TODO: Use the road_config to config which towns to connect
        nodes = {}
        for town in towns:
            if town["connect"]:
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
        return road_graph, plot_map(shape, height_map, towns, road_graph, return_step='roads')

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
            output_image])
    def reset_town(shape, height_map):
        chart = plot_map(shape, height_map, None, None, return_step='height_map')
        return TownNameSampler(), list(), RoadGraph.empty(), chart

    # ----------------------------------- Roads ---------------------------------- #
    @reset_roads_button.click(
        inputs=[
            shape_state,
            height_map_state,
            towns_state],
        outputs=[
            roads_state,
            output_image])
    def reset_roads(shape, height_map, towns):
        chart = plot_map(shape, height_map, towns, None, return_step='towns')
        return RoadGraph.empty(), chart


if __name__ == '__main__':
    demo.launch()
