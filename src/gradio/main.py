from typing import Literal, Optional, List, Union

import cv2
import numpy as np
import gradio as gr
import shapely
from src.gradio.mesh_map import heightmap_to_3d_mesh
from src.gradio.demo_types import TOWN_TYPE, TOWN_TYPES, Town, RoadGraph
from src.gradio.display_map import plot_map
from src.gradio.gradio_utils import TownNameSampler
from src.gradio.gradio_configs import CLOSE_ICON


with gr.Blocks() as demo:

    # used to generate names for towns
    town_name_sampler = gr.State(TownNameSampler)
    # state to keep track of the generation method
    town_generation_method = gr.State('Random')

    # current states
    shape_state = gr.State(None)
    height_map_state = gr.State(None)
    towns_state = gr.State(list)
    roads_state = gr.State(RoadGraph.empty)
    
    poly_points_state = gr.State(list)
    poly_closed_state = gr.State(False)

    with gr.Row():

        # config column
        with gr.Column():
            with gr.Row():

                # shape generation
                with gr.Column():
                    with gr.Accordion('Shape config', open=False):
                        shape_config_num_islands = gr.Slider(value=1, label='Number of Islands', minimum=1, maximum=10, step=1, interactive=True)
                    run_generate_shape_button = gr.Button('Generate Shape')
                
                # height map generation
                with gr.Column():
                    with gr.Accordion('Height Map config', open=False):
                        height_map_config_terrain_roughness = gr.Slider(value=0.2, label='Terrain Roughness', minimum=0, maximum=1, step=0.001, interactive=True)
                        height_map_config_max_height = gr.Slider(value=1000, label='Max Island Height [m]', minimum=1, maximum=8000, step=1, interactive=True)
                    run_generate_height_map_button = gr.Button('Generate Height Map')

                # town generation
                with gr.Column():
                    with gr.Accordion('Town config', open=False):
                        reset_town_button = gr.Button('Reset Towns')
                        with gr.Tab('Random') as town_config_random_tab:
                            town_config_random_num_town_slider = gr.Slider(value=5, label='Number of Towns', minimum=1, maximum=30, step=1, interactive=True)
                            @town_config_random_tab.select(outputs=[town_generation_method])
                            def _(): return 'Random'
                        with gr.Tab('Custom') as town_config_custom_tab:
                            town_config_custom_town_type_radio = gr.Radio([(c.capitalize(), c) for c in TOWN_TYPES], value=TOWN_TYPES[0], label='Town Type', interactive=True)
                            town_config_custom_town_feature_checkboxgroup = gr.CheckboxGroup([('Coastal', 'is_coastal')], label='Town Features')
                            @town_config_custom_tab.select(outputs=[town_generation_method])
                            def _(): return 'Custom'
                    run_generate_town_button = gr.Button('Generate Town(s)')

                # road generation
                with gr.Column():
                    with gr.Accordion('Road config', open=False):
                        reset_roads_button = gr.Button('Reset Roads')
                        
                        # which towns to connect
                        with gr.Accordion('Towns to connect'):
                            @gr.render(inputs=[towns_state], triggers=[towns_state.change])
                            def _(towns):
                                if not towns:
                                    gr.Markdown('No towns to connect')
                                for town in towns:
                                    # TODO: add checkbox to select town and figure out how dynamic inputs work
                                    # https://www.gradio.app/guides/dynamic-apps-with-render-decorator
                                    # https://www.gradio.app/docs/gradio/checkbox
                                    with gr.Row():
                                        gr.Checkbox(value=False, label='', interactive=True, show_label=False, container=False, scale=1, min_width=10)
                                        text_box = gr.Textbox(value=town['town_name'], lines=1, max_lines=1, interactive=True, show_label=False, container=False, scale=5)
                                        gr.ClearButton(text_box, value=None, size='sm', icon=CLOSE_ICON, scale=1, min_width=10)

                        road_config_road_cost = gr.Slider(value=0.5, label='Road Cost', minimum=0.1, maximum=1, step=0.1, interactive=True)
                        road_config_slope_cost = gr.Slider(value=0.5, label='Slope Cost Factor', minimum=0.1, maximum=1, step=0.1, interactive=True)
                        road_config_curvature_cost = gr.Slider(value=0.5, label='Curvature Cost Factor', minimum=0.1, maximum=1, step=0.1, interactive=True)
                    run_generate_road_button = gr.Button('Generate Road')

        # map column
        with gr.Column():
            with gr.Tab('Map'):
                output_image = gr.Image(
                    value=plot_map(return_step='empty'),
                    label='Map',
                    interactive=False,
                    image_mode='RGB',
                    show_download_button=False)
                
                change_shape_mode_btn = gr.Button('Change Shape Mode')
                
                with gr.Row():                    
                    close_btn = gr.Button('Close Polygon') # Changes to Open Polygon when closed
                    clear_btn = gr.Button('Clear Points')
                with gr.Row():
                    poly_add_land_btn = gr.Button('Add Land')
                    poly_add_sea_btn = gr.Button('Add Sea')
                    poly_regen_area_btn = gr.Button('Regenerate Area')
                
                with gr.Row():
                    gr.DownloadButton('Download PNG')
                    gr.DownloadButton('Download JSON')
                
            with gr.Tab('3d Mesh'):
                output_mesh_hm = gr.Model3D(
                        value=None,
                        interactive=True
                    )
                with gr.Row():
                    generate_mesh_button = gr.Button('Generate 3D Mesh')
            
            
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
    @clear_btn.click(
        inputs=[poly_points_state],
        outputs=[poly_points_state, poly_closed_state, close_btn])
    @output_image.clear(
        inputs=[poly_points_state],
        outputs=[poly_points_state, poly_closed_state, close_btn])
    def clear_points(points: list):
                        points.clear()
                        return points, False, 'Open Polygon'
    # ------------------------------ Add/remove area ----------------------------- #
    @poly_add_land_btn.click(
        inputs=[shape_state, 
                poly_points_state, 
                poly_closed_state,
                output_image, 
                gr.State('add')], # TODO: Hotfix for now
        outputs=[shape_state, 
                 poly_points_state, 
                 poly_closed_state])
    @poly_add_sea_btn.click(
        inputs=[shape_state, 
                poly_points_state, 
                poly_closed_state,
                output_image,
                gr.State('subtract')],
        outputs=[shape_state, 
                 poly_points_state, 
                 poly_closed_state])
    def add_subtract_area(shape_mask: np.ndarray, 
                          poly_points: list, 
                          poly_closed: bool,
                          high_res_image: np.ndarray, 
                          mode: Literal['add', 'subtract']):
        if not poly_closed:
            gr.Warning('Polygon is not closed, please close the polygon first!')
            return shape_mask, poly_points, poly_closed
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
        
        import matplotlib.pyplot as plt
        plt.imshow(shape_mask)
        return shape_mask, [], False
    
    @poly_regen_area_btn.click(
        inputs=[],
        outputs=[])
    def regen_area():
        gr.Warning('Not implemented yet, coming soon!')
    
    # ---------------------------------------------------------------------------- #
    #                             Generation functions                             #
    # ---------------------------------------------------------------------------- #
    # -------------------------------- Poly Change ------------------------------- #
    @change_shape_mode_btn.click(
        inputs=[
            shape_state,
            poly_points_state,
            poly_closed_state],
        outputs=[
            shape_state,
            output_image])
    @poly_points_state.change(
        inputs=[
            shape_state,
            poly_points_state,
            poly_closed_state],
        outputs=[
            shape_state,
            output_image])
    @poly_closed_state.change(
        inputs=[
            shape_state,
            poly_points_state,
            poly_closed_state],
        outputs=[
            shape_state,
            output_image])
    def change_shape_display(
        shape: Optional[np.ndarray],
        poly_points: List[List[float]],
        poly_closed: bool
    ) -> dict:
        polygon_dict = {'poly_points': poly_points, 'poly_closed': poly_closed}
        if shape is None:
            shape = np.load('assets/defaults/shape_0.npy')
        chart = plot_map(shape=shape, height_map=None, towns=None, roads=None, return_step='change_poly_shape', polygon_dict=polygon_dict)

        return shape, chart
    
    # ----------------------------------- Shape ---------------------------------- #
    @run_generate_shape_button.click(
        inputs=[
            shape_config_num_islands,
            poly_points_state,
            poly_closed_state],
        outputs=[
            shape_state,
            output_image])
    def generate_shape(
        shape_config_num_islands: int,
        poly_points: List[List[float]],
        poly_closed: bool
    ) -> dict:
        chart = plot_map(shape=None, height_map=None, towns=None, roads=None, return_step='shape')
        return None, chart

    # height map generation
    @run_generate_height_map_button.click(
        inputs=[
            shape_state,
            height_map_config_terrain_roughness,
            height_map_config_max_height],
        outputs=[
            height_map_state,
            output_image])
    def generate_height_map(
        shape: shapely.MultiPolygon,
        roughness: float,
        max_height: float,
        *,
        only_generate: bool = False
    ) -> dict:

        height_map = np.load('assets/defaults/height_map.npy')

        if only_generate:
            return height_map

        chart = plot_map(shape, height_map, towns=None, roads=None, return_step='height_map')

        return height_map, chart

    # 3D mesh generation
    @generate_mesh_button.click(
        inputs=[height_map_state],
        outputs=[output_mesh_hm])
    def generate_mesh(
        height_map: np.ndarray
    ) -> str:
        output_path = 'assets/defaults/terrain_mesh.obj'
        output_path = heightmap_to_3d_mesh(height_map, output_path=output_path)
        return output_path

    # town generation
    @run_generate_town_button.click(
        inputs=[
            shape_state,
            height_map_state,
            towns_state,
            roads_state,
            town_name_sampler,
            town_generation_method,
            town_config_random_num_town_slider,
            town_config_custom_town_type_radio,
            town_config_custom_town_feature_checkboxgroup],
        outputs=[
            towns_state,
            town_name_sampler,
            output_image])
    
    def generate_town(
        shape: Optional[shapely.MultiPolygon],
        height_map: Optional[np.ndarray],
        towns: List[dict],
        roads: RoadGraph,
        name_sampler: TownNameSampler,
        generation_method: Literal['Random', 'Custom'],
        num_towns: int,
        town_type: TOWN_TYPE,
        town_config: List[str]
    ) -> dict:
        possible_choices = np.where(height_map > 0)
        num_possible = len(possible_choices[0])

        if height_map is None:
            height_map = generate_height_map

        if generation_method == 'Random':
            for _ in range(num_towns):
                idx = np.random.choice(num_possible)
                z = height_map[possible_choices[0][idx], possible_choices[1][idx]]
                is_coastal = np.random.choice([True, False])
                town_type = np.random.choice(TOWN_TYPES)
                
                towns.append(
                    Town(
                        town_type=town_type,
                        is_coastal=is_coastal,
                        xyz=[possible_choices[1][idx], possible_choices[0][idx], z],
                        town_name=name_sampler.pop(town_type, is_coastal)))

        elif generation_method == 'Custom':
            # import pdb; pdb.set_trace()
            is_coastal = 'is_coastal' in town_config
            idx = np.random.choice(num_possible)
            z = height_map[possible_choices[0][idx], possible_choices[1][idx]]
            towns.append(
                Town(
                    town_type = town_type,
                    is_coastal = is_coastal,
                    xyz = [possible_choices[1][idx], possible_choices[0][idx], z],
                    town_name = name_sampler.pop(town_type, is_coastal)))
        
        chart = plot_map(shape, height_map, towns, roads, return_step='roads' if roads['edges'] else 'towns')

        return towns, name_sampler, chart

    # road generation
    @run_generate_road_button.click(
        inputs=[
            shape_state,
            height_map_state,
            towns_state,
            roads_state,
            road_config_road_cost,
            road_config_slope_cost,
            road_config_curvature_cost],
        outputs=[
            roads_state,
            output_image])
    def generate_road(*args):
        return roads_state, plot_map(shape_state, height_map_state, towns_state, roads_state, return_step='roads')

    # reset towns
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

    # reset roads
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
