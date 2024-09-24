import copy
import json
from typing import Literal

import cv2
import numpy as np
import shapely
import gradio as gr
from src.gradio.demo_types import TOWN_TYPE, TOWN_TYPES, Town, RoadGraph
from src.gradio.gradio_utils import TownNameSampler
from src.gradio.display_map import plot_map









# components


# generation configs

# town generation
town_config_custom_town_type_radio = gr.Radio([(c.capitalize(), c) for c in TOWN_TYPES], value=TOWN_TYPES[0], label='Town Type', interactive=True, render=False)
town_config_custom_town_feature_checkboxgroup = gr.CheckboxGroup([('Coastal', 'is_coastal')], label='Town Features', render=False)
town_config_random_num_town_slider = gr.Slider(value=5, label='Number of Towns', minimum=1, maximum=30, step=1, interactive=True, render=False)


with gr.Blocks() as demo:

    # used to generate names for towns
    town_name_sampler = gr.State(TownNameSampler)
    # state to keep track of the generation method
    town_generation_method = gr.State('Random')

    # current states
    shape_state = gr.State(shapely.MultiPolygon)
    height_map_state = gr.State(lambda: np.ndarray((0, 0), dtype=np.float32))
    towns_state = gr.State(list)
    roads_state = gr.State(RoadGraph.empty)

    with gr.Row():

        # config column
        with gr.Column():
            with gr.Row():

                # shape generation
                with gr.Column():
                    run_generate_shape_button = gr.Button('Generate Shape')
                    with gr.Accordion('Shape config', open=False):
                        shape_config_num_islands = gr.Slider(value=1, label='Number of Islands', minimum=1, maximum=10, step=1, interactive=True)
                
                # height map generation
                with gr.Column():
                    run_generate_height_map_button = gr.Button('Generate Height Map')
                    with gr.Accordion('Height Map config', open=False):
                        height_map_config_terrain_roughness = gr.Slider(value=0.2, label='Terrain Roughness', minimum=0, maximum=1, step=0.001, interactive=True)
                        height_map_config_max_height = gr.Slider(value=1000, label='Max Island Height [m]', minimum=1, maximum=8000, step=1, interactive=True)

                # town generation
                with gr.Column():
                    run_generate_town_button = gr.Button('Generate Town(s)')
                    with gr.Accordion('Town config', open=False):
                        reset_town_button = gr.Button('Reset Towns')
                        with gr.Tab('Random') as town_config_random_tab:
                            town_config_random_num_town_slider.render()
                            @town_config_random_tab.select(outputs=[town_generation_method])
                            def _(): return 'Random'
                        with gr.Tab('Custom') as town_config_custom_tab:
                            town_config_custom_town_type_radio.render()
                            town_config_custom_town_feature_checkboxgroup.render()
                            @town_config_custom_tab.select(outputs=[town_generation_method])
                            def _(): return 'Custom'

                # road generation
                with gr.Column():
                    run_generate_road_button = gr.Button('Generate Road')
                    with gr.Accordion('Road config', open=False):
                        reset_roads_button = gr.Button('Reset Roads')
                        
                        # which towns to connect
                        with gr.Column():
                            @gr.render(inputs=[towns_state], triggers=[towns_state.change])
                            def _(towns):
                                if not towns:
                                    gr.Markdown('No towns to connect')
                                    return
                                for town in towns:
                                    # TODO: add checkbox to select town and figure out how dynamic inputs work
                                    # https://www.gradio.app/guides/dynamic-apps-with-render-decorator
                                    # https://www.gradio.app/docs/gradio/checkbox
                                    gr.Checkbox(label=town['town_name'], value=False, interactive=True)

                        road_config_road_cost = gr.Slider(value=0.5, label='Road Cost', minimum=0.1, maximum=1, step=0.1, interactive=True)
                        road_config_slope_cost = gr.Slider(value=0.5, label='Slope Cost Factor', minimum=0.1, maximum=1, step=0.1, interactive=True)
                        road_config_curvature_cost = gr.Slider(value=0.5, label='Curvature Cost Factor', minimum=0.1, maximum=1, step=0.1, interactive=True)

        # map column
        with gr.Column():
            output_image = gr.Image(
                value=plot_map(return_step='empty'),
                label='Map',
                interactive=False,
                image_mode='RGB',
                show_download_button=False)
            with gr.Row():
                gr.Button('Download PNG')
                gr.Button('Download SVG')
                gr.Button('Download JSON')


    # generation functions

    # shape generation
    @run_generate_shape_button.click(
        inputs=[
            shape_config_num_islands],
        outputs=[
            shape_state,
            output_image])
    def generate_shape(
        shape_config_num_islands: int
    ) -> dict:

        shape = list(np.load('assets/defaults/shape.npz').values())
        chart = plot_map(shape, height_map=None, towns=None, roads=None, return_step='shape')

        return shape, chart

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
        max_height: float
    ) -> dict:

        height_map = np.load('assets/defaults/height_map.npy')
        chart = plot_map(shape, height_map, towns=None, roads=None, return_step='height_map')

        return height_map, chart

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
        shape: shapely.MultiPolygon,
        height_map: np.ndarray,
        towns: list[dict],
        roads: RoadGraph,
        name_sampler: TownNameSampler,
        generation_method: Literal['Random', 'Custom'],
        num_towns: int,
        town_type: TOWN_TYPE,
        town_config: list[str]
    ) -> dict:


        possible_choices = np.where(height_map > 0)
        num_possible = len(possible_choices[0])

        if generation_method == 'Random':
            for _ in range(num_towns):
                idx = np.random.choice(num_possible)
                xy = (
                    possible_choices[1][idx],
                    possible_choices[0][idx])
                z = height_map[*xy[::-1]]
                is_coastal = np.random.choice([True, False])
                town_type = np.random.choice(TOWN_TYPES)
                
                towns.append(
                    Town(
                        town_type=town_type,
                        is_coastal=is_coastal,
                        xyz=[*xy, z],
                        town_name=name_sampler.pop(town_type, is_coastal)))

        elif generation_method == 'Custom':
            # import pdb; pdb.set_trace()
            is_coastal = 'is_coastal' in town_config
            idx = np.random.choice(num_possible)
            xy = (
                possible_choices[1][idx],
                possible_choices[0][idx])
            z = height_map[*xy[::-1]]
            towns.append(
                Town(
                    town_type = town_type,
                    is_coastal = is_coastal,
                    xyz = [*xy, z],
                    town_name = name_sampler.pop(town_type, is_coastal)))
        
        chart = plot_map(shape, height_map, towns, roads, return_step='roads' if roads['edges'] else 'towns')

        return towns, name_sampler, chart

    # reset town names
    @reset_town_button.click(outputs=[town_name_sampler, towns_state, roads_state])
    def reset_town_names():
        return TownNameSampler(), list(), RoadGraph.empty()

    # reset roads
    @reset_roads_button.click(outputs=[roads_state])
    def reset_roads():
        return RoadGraph.empty()

if __name__ == '__main__':
    demo.launch()
