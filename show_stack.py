# %%
# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ipywidgets import GridBox, Checkbox
import ipywidgets as widgets
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation
from IPython.display import display, HTML
from aicsimageio import AICSImage
from preprocess import preprocess
from lif_metadata import LIFProcessor

# %%
# %%
class SceneSelector:
    def __init__(self, lif_file):
        stack = AICSImage(lif_file)
        self.labels = [scene.split('/')[-1] for scene in stack.scenes]
        self.checkboxes = [Checkbox(value=False, description=label) for label in self.labels]
        for i, checkbox in enumerate(self.checkboxes):
            checkbox.index = i  # assign index to each checkbox
            checkbox.observe(self.update_checked, names='value')
        self.grid = GridBox(self.checkboxes, layout=widgets.Layout(grid_template_columns="repeat(3, 120px)"))
        self.dict_checked = {}

    def update_checked(self, change):
        checkbox = change['owner']
        if checkbox.value:
            self.dict_checked[checkbox.index] = checkbox.description  # add to dict if checked
        else:
            self.dict_checked.pop(checkbox.index, None)  # remove from dict if unchecked

    def get_checked_dict(self):
        return self.dict_checked

    def display(self):
        display(self.grid)

# %%
class ImageViewer:
    def __init__(self, lif_file: str, selector: SceneSelector):
        self.lif_file = lif_file
        self.lif_stack = AICSImage(lif_file)
        self.dT = LIFProcessor(self.lif_file).get_time_interval()
        self.frames = range(self.lif_stack.dims.T)
        self.processed_dir, self.input_dir, self.output_dir, self.plots_dir, self.images_dir, self.condition_info_file, self.ch_info_file = preprocess(lif_file)
        self.selector = selector
        self.dict_checked = self.selector.get_checked_dict()
        self.dict_checked = dict(sorted(self.dict_checked.items(), key=lambda x:x[0]))
        self.scale_unit = 'um'
        self.scale_location = 'lower right'
        self.scale_length_fraction = 0.1
        self.bit_depth = self.lif_stack.data.dtype

    def set_condition(self, condition_info_file: str) -> dict:
        condition_info = pd.read_csv(condition_info_file)
        condition_dict = dict(zip(condition_info.scene, condition_info.condition))
        return condition_dict
    
    def get_color(self, ch_info_file: str):
        l_color = pd.read_csv(ch_info_file, index_col=0)['color'] 
        return l_color

    def active_ch_str(self, show_chs):
        ch_num = self.lif_stack.dims.C
        ch_active_dict = {i: int(i in show_chs) for i in range(ch_num)}
        ch_active_str = ''.join(str(ch_active_dict[i]) for i in range(ch_num))
        return ch_active_str
    
    def process_lut(self, file_path):
        try: # Try reading the file as TSV
            # Read as TSV
            df_lut = pd.read_csv(file_path, sep='\t', index_col=0)
            return df_lut
        
        except Exception as e:
            with open(file_path, "rb") as file:
                lut_data = file.read()
            rgb_values = [(lut_data[i], lut_data[i + 256], lut_data[i + 512]) for i in range(256)]
            df_lut = pd.DataFrame(rgb_values, columns=["Red", "Green", "Blue"])
            return df_lut

    def set_cmap(self, lut_name: str, lut_dir='luts'):
        lut_name = lut_name + '.lut'
        df = self.process_lut(os.path.join(lut_dir, lut_name))
        custom_cmap_data = df[['Red', 'Green', 'Blue']].values / 255 # Normalizing the RGB values to the range [0, 1]
        custom_cmap = ListedColormap(custom_cmap_data) # Creating a colormap using the normalized RGB values
        return custom_cmap

    def apply_cmap(self, img, cmap_name, is_custom=True, lut_dir='luts'):
        '''チャンネルの画像を指定のカラーマップでマッピング'''
        if is_custom:
            cmap = self.set_cmap(cmap_name, lut_dir)
        else:
            cmap = plt.cm.get_cmap(cmap_name)
        rgba_img = cmap(img)
        return (rgba_img[:, :, :3] * 255).astype(np.uint8)

    def setup_figure(self, num_subplots):
        fig, axes = plt.subplots(1, num_subplots, figsize=(3.2 * num_subplots, 4))
        fig.tight_layout()
        fig.subplots_adjust(top=0.9, wspace=0.05)
        return fig, axes

    def show_subplots(self, frame, show_chs, set_colors=None, show_position=True, saveimg=True, axes=None, alphas=None):
        try:
            artists = []
            condition_dict = self.set_condition(self.condition_info_file)
            dict_checked = self.dict_checked
            if axes is None:
                fig, axes = self.setup_figure(len(dict_checked))
            else:
                fig = axes[0].figure # Use the passed axes

            # Update suptitle with current frame
            if self.dT is not None:
                plt.gcf().suptitle(f'Time: {frame * self.dT} min')
            for idx, scene_key in enumerate(dict_checked):
                self.lif_stack.set_scene(scene_key)
                condition = condition_dict[dict_checked[scene_key]]
                title = condition
                if show_position:
                    title = f'{condition} \n({dict_checked[scene_key]})'
                img_chs = self.lif_stack.get_image_data("CYX", Z=0, T=frame)
                if alphas is not None:
                    # check if alphas is a list of floats and its length is equal to the number of channels
                    if isinstance(alphas, list) and len(alphas) == len(show_chs):
                        # zip show_chs and alphas by list comprehension
                        for ch_num, alpha in zip(show_chs, alphas):
                            # max of self.bit_depth, using np.iinfo(self.bit_depth).max
                            max = np.iinfo(self.bit_depth).max
                            img_chs[ch_num] = (img_chs[ch_num].astype(float) * alpha / max).astype(img_chs[ch_num].dtype)

                    else:
                        raise ValueError("alphas must be a list of floats and its length must be equal to the number of channels.")
                l_overlay = [self.apply_cmap(img_chs[ch_num], self.get_color(self.ch_info_file)[ch_num]) for ch_num in show_chs]
                if set_colors is not None:
                    l_overlay = [self.apply_cmap(img_chs[ch_num], color) for ch_num, color in zip(show_chs, set_colors)]
                showing_img = np.sum(l_overlay, axis=0)
                #showing_img = np.clip(showing_img, 0, 255).astype(np.uint8)
                ax = plt.subplot(1, len(dict_checked), idx + 1)
                ax.set_title(title)
                ax.set_xticks([]) # Hide x-axis ticks
                ax.set_yticks([]) # Hide y-axis ticks
                if idx == 0:
                # show scale bar 
                    scalebar = ScaleBar(self.lif_stack.physical_pixel_sizes.X, 'um', length_fraction=0.1, location='lower right', color='white', frameon=False)
                    ax.add_artist(scalebar)
                ax.imshow(showing_img)
                artists.append(ax) # append axes to the list
    
            if saveimg:
                # set name of the output file
                frame_code = str(frame).zfill(3) # frameを3桁のゼロ埋め文字列に変換
                ch_code = self.active_ch_str(show_chs)
                image_name = f't{frame_code}_ch{ch_code}.png'
                image_path = os.path.join(self.images_dir, image_name)
                # save image
                plt.savefig(image_path, dpi=200, bbox_inches='tight')
            #plt.close(fig)  # このfigureをクローズ
            #return artists
        
        except Exception as e:
            print(f"An error occurred while showing the merged image: {e}")

    def show_ani(self, show_chs, set_colors=None, frames=None, show_position=True, saveani=False, displayani=False):
        frames = self.frames if frames is None else frames     
        fig, axes = self.setup_figure(len(self.dict_checked))
        ani = FuncAnimation(fig, self.show_subplots, frames=frames, fargs=(show_chs, set_colors, show_position, False, axes), interval=100)
        if displayani:
            display(HTML(ani.to_jshtml()))
        if saveani:
            ch_code = self.active_ch_str(show_chs)
            if set_colors is None:
                mov_name = f'ch{ch_code}.mov'
            else:
                # open up elements in set_colors and print them out. '-' is the separator
                set_colors_str = '-'.join(set_colors)
                mov_name = f'ch{ch_code}_{set_colors_str}.mov'
            mov_path = os.path.join(self.images_dir, mov_name)
            ani.save(mov_path, writer='ffmpeg', fps=10, dpi=200)
            #return mov_path
        plt.close('all')
