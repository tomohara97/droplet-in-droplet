'''
変更点[2024/02/22]
- core-shell3相系を前提。2相系には非対応
- 各cellでshellとcoreを紐づけ (compute_core_shell)
- binarizeの機能をget_roiから独立させた
- ROI_confirmのみの表示。途中経過の表示は削除（sure_fgなど）
- plotのclassの削除
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt
from aicsimageio import AICSImage
import pandas as pd
import os
import seaborn as sns
from matplotlib.animation import FuncAnimation
from IPython.display import display, HTML
from lif_metadata import LIFProcessor
from preprocess import preprocess

class core_shell_analysis:
    def __init__(self, lif_file: str, 
                 cell_ROIch: int, cell_isInverted: bool, 
                 core_ROIch: int, core_isInverted: bool,
                 positions_for_test: list=None,
                 frames_for_test: list=None,
                 measure_chs=None, 
                 interval_min=None,
                 position_name_column: bool=False,
                 save_markers: bool=False):
        self.lif_file = lif_file
        self.lif_stack = AICSImage(self.lif_file)
        self.pixelarea_to_um2 = self.lif_stack.physical_pixel_sizes.X * self.lif_stack.physical_pixel_sizes.Y
        self.bit_depth = self.lif_stack.data.dtype
        self.bit_depth_max = np.iinfo(self.bit_depth).max
        self.n_frame, self.n_channel, self.n_slice, self.Y, self.X = self.lif_stack.shape
        self.n_position = len(self.lif_stack.scenes)
        self.processed_dir, self.input_dir, self.output_dir, self.plots_dir, self.images_dir, self.condition_info_file, self.ch_info_file = preprocess(self.lif_file)
        self.positions = range(self.n_position) if positions_for_test is None else positions_for_test
        self.frames = range(self.n_frame) if frames_for_test is None else frames_for_test
        self.cell_ROIch = cell_ROIch
        self.cell_isInverted = cell_isInverted
        self.core_ROIch = core_ROIch
        self.core_isInverted = core_isInverted
        self.measure_chs = [ch for ch in range(self.n_channel) if ch not in [self.core_ROIch, self.cell_ROIch]] if measure_chs is None else measure_chs
        self.interval_min = interval_min if interval_min is not None else LIFProcessor(self.lif_file).get_time_interval()
        self.position_name_column = position_name_column
        self.save_markers = save_markers

    def sampling_conditions(self, sampling_num=3):
        # Select three unique indices from both Positions and Frames, then sort them
        # randomly choose from self.positions and self.frames
        positions_for_test = np.random.choice(self.positions, sampling_num, replace=False)
        frames_for_test = np.random.choice(self.frames, sampling_num, replace=False)        
        l_test = sorted(list(zip(positions_for_test, frames_for_test)))
        return l_test

    def remove_small_ROIs(self, markers, min_area=10):
        unique_markers, counts = np.unique(markers, return_counts=True)
        for i, marker in enumerate(unique_markers):
            if counts[i] < min_area:
                markers[markers == marker] = 0
        return markers

    def binarize(self, src_img, isInverted: bool, manual_thresh: int=None, kernel_size_for_opening=2, opening_iterations=1, kernel_size_for_diaero=3, erosion_iterations=0, dialte_iterations=0):
        if src_img.dtype == np.uint16:
            src_img = (src_img / 256).astype(np.uint8)

        if manual_thresh is None:
            # Use Otsu's method if manual_thresh is not provided
            if isInverted:
                thresh, bin_img = cv2.threshold(src_img, 0, self.bit_depth_max, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            else:
                thresh, bin_img = cv2.threshold(src_img, 0, self.bit_depth_max, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            #print('threshold:', thresh)
        else:
            if isInverted:
                thresh, bin_img = cv2.threshold(src_img, manual_thresh, self.bit_depth_max, cv2.THRESH_BINARY_INV)
            else:
                thresh, bin_img = cv2.threshold(src_img, manual_thresh, self.bit_depth_max, cv2.THRESH_BINARY)
        if erosion_iterations > 0: # erosion for shrinking ROIs
            kernel = np.ones((kernel_size_for_diaero,kernel_size_for_diaero),np.uint8)
            bin_img = cv2.erode(bin_img, kernel, iterations = erosion_iterations)
        if dialte_iterations > 0:
            kernel = np.ones((kernel_size_for_diaero,kernel_size_for_diaero),np.uint8)
            bin_img = cv2.dilate(bin_img, kernel, iterations = dialte_iterations)
        if opening_iterations > 0: # opening for noise removal
            kernel = np.ones((kernel_size_for_opening,kernel_size_for_opening),np.uint8)
            bin_img = cv2.morphologyEx(bin_img,cv2.MORPH_OPEN,kernel,iterations = opening_iterations)
        return bin_img

    def get_roi(self, src_img, isInverted: bool, kernel_size_for_opening, opening_iterations, kernel_size_for_diaero, erosion_iterations, dialte_iterations, dist_threshold=0.02):
        img_BGR = cv2.cvtColor(src_img, cv2.COLOR_GRAY2BGR)
        bin_img = self.binarize(src_img, isInverted, kernel_size_for_opening=kernel_size_for_opening, opening_iterations=opening_iterations, kernel_size_for_diaero=kernel_size_for_diaero, erosion_iterations=erosion_iterations, dialte_iterations=dialte_iterations)
        
        # sure background & sure foreground frmo bin_img
        kernel2 = np.ones((3,3),np.uint8)
        sure_bg = cv2.dilate(bin_img,kernel2,iterations=2) # dilation for sure background        
        dist_transform = cv2.distanceTransform(bin_img,cv2.DIST_L2,5) # distance transform
        ret, sure_fg = cv2.threshold(dist_transform,dist_threshold*dist_transform.max(),self.bit_depth_max,0) # threshold for sure foreground
        sure_fg = np.uint8(sure_fg)      
        #print('閾値（距離変換で得られた値の最大値×', dist_threshold, '）:',ret)
        boundary = cv2.subtract(sure_bg,sure_fg) # detect boundary region

        # watershed
        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers+1
        markers[boundary==self.bit_depth_max] = 0
        markers = cv2.watershed(img_BGR,markers) # markersでは、境界領域: -1, 背景: 1, 前景: 2, 3, ... 
        edge = np.where(markers==-1, 1, 0).astype(np.uint8) # 境界領域を抽出
        markers[markers < 2] = 0 # remove background and boundary
        markers = self.remove_small_ROIs(markers)
        unique_markers = np.unique(markers[markers != 0]) # unique_markers = np.unique(markers) but over 0
        markers = np.searchsorted(unique_markers, markers) # renumbering markers from 1

        # show ROIs
        ROI_confirm = img_BGR.copy()
        ROI_confirm[edge == 1] = [0,255,0] # 境界の領域を塗る
        ROI_confirm = cv2.cvtColor(ROI_confirm, cv2.COLOR_BGR2RGB)

        return ROI_confirm, markers

    def compute_core_shell(self, cell_markers, core_bin_img):
        '''各ユニークなラベルを持つ細胞に対して、コアとシェルの領域および面積情報を計算する'''
        unique_labels = np.unique(cell_markers[cell_markers != 0]) # ユニークなラベルを取得し、背景（0）を除去
        core_markers, shell_markers = np.zeros_like(cell_markers, dtype=np.int32), np.zeros_like(cell_markers, dtype=np.int32)
        dict_area_infos = {}
        for label in unique_labels: # define core and shell, calculate areas
            #if label == 0: continue # ラベルが0（背景）の場合、処理をスキップする
            cell_region = (cell_markers == label)
            core_region, shell_region = cell_region & (core_bin_img != 0), cell_region & (core_bin_img == 0)
            core_markers[core_region], shell_markers[shell_region] = label, label
            cell_area, core_area, shell_area = np.sum(cell_region), np.sum(core_region), np.sum(shell_region)
            dict_area_infos[label] = [cell_area, core_area, shell_area, core_area / cell_area]
        return unique_labels, core_markers, shell_markers, dict_area_infos

    def measure_rois(self, cell_markers, core_bin_img, dict_img_ch_measure, frame):
        unique_labels, core_markers, shell_markers, dict_area_infos = self.compute_core_shell(cell_markers, core_bin_img)
        dict_ch_intensities = {measuring_ch: self.calculate_mean_intensities(measuring_img[frame], core_markers, shell_markers, unique_labels) for measuring_ch, measuring_img in dict_img_ch_measure.items()}
        return unique_labels, dict_area_infos, dict_ch_intensities, core_markers, shell_markers

    def calculate_mean_intensities(self, measuring_img, core_markers, shell_markers, unique_labels):
        return {label: [np.mean(measuring_img[core_markers == label]) if np.sum(core_markers == label) > 0 else np.nan,
                        np.mean(measuring_img[shell_markers == label]) if np.sum(shell_markers == label) > 0 else np.nan] 
                for label in unique_labels}
    
    def roi_results(self, cell_markers, core_bin_img, dict_img_ch_measure, frame, position):
        unique_labels, dict_area_infos, dict_ch_intensities, core_markers, shell_markers = self.measure_rois(cell_markers, core_bin_img, dict_img_ch_measure, frame)
        df = self.create_dataframe(unique_labels, dict_area_infos, dict_ch_intensities, position, frame)
        return df, core_markers, shell_markers

    def create_dataframe(self, unique_labels, dict_area_infos, dict_ch_intensities, position, frame):
        area_info_values = list(zip(*dict_area_infos.values()))
        df = pd.DataFrame({
            'Position': position, 
            'Frame': frame, 
            'ROI': unique_labels, 
            'cell_area (um2)': [value * self.pixelarea_to_um2 for value in area_info_values[0]],
            'core_area (um2)': [value * self.pixelarea_to_um2 for value in area_info_values[1]], 
            'shell_area (um2)': [value * self.pixelarea_to_um2 for value in area_info_values[2]], 
            'core_ratio': area_info_values[3]
        })
        for measuring_ch, dict_mean_intensities in dict_ch_intensities.items():
            intensity_values = list(zip(*dict_mean_intensities.values()))
            df[f'Mean_int.(Ch{measuring_ch})_core'] = intensity_values[0]
            df[f'Mean_int.(Ch{measuring_ch})_shell'] = intensity_values[1]
        return df

    def get_result_df(self, sampling=True, sampling_num=3, 
                      kernel_size_for_opening_cell=2, opening_iterations_cell=1, kernel_size_for_diaero_cell=3, erosion_iterations_cell=0, dialte_iterations_cell=0,
                      kernel_size_for_opening_core=3, opening_iterations_core=1,
                      erosion_iterations_for_core=0, dialte_iterations_for_core=0, kernel_size_for_diaero_core=1, manual_thresh_core=None):
        l_df = []
        sampling_conditions = self.sampling_conditions(sampling_num) if sampling else None

        for position in self.positions:
            self.lif_stack.set_scene(position)
            img_cell_ROIch, img_core_ROIch, *img_measure_chs = self.lif_stack.get_image_data("CTYX", C=[self.cell_ROIch, self.core_ROIch] + self.measure_chs)
            if self.bit_depth == np.uint16:
                img_cell_ROIch = (img_cell_ROIch/256).astype(np.uint8)
                img_core_ROIch = (img_core_ROIch/256).astype(np.uint8)
            dict_img_ch_measure = dict(zip(self.measure_chs, img_measure_chs))

            for frame in self.frames:
                ROI_confirm, markers = self.get_roi(src_img=img_cell_ROIch[frame], isInverted=self.cell_isInverted, 
                                                    kernel_size_for_opening=kernel_size_for_opening_cell, opening_iterations=opening_iterations_cell, kernel_size_for_diaero=kernel_size_for_diaero_cell, erosion_iterations=erosion_iterations_cell, dialte_iterations=dialte_iterations_cell)
                core_bin_img = self.binarize(img_core_ROIch[frame], self.core_isInverted, 
                                             kernel_size_for_opening=kernel_size_for_opening_core, opening_iterations=opening_iterations_core,
                                             kernel_size_for_diaero=kernel_size_for_diaero_core, erosion_iterations=erosion_iterations_for_core, dialte_iterations=dialte_iterations_for_core, 
                                             manual_thresh=manual_thresh_core)
                df, core_markers, shell_markers = self.roi_results(markers, core_bin_img, dict_img_ch_measure, frame, position)
                l_df.append(df)

                if self.save_markers:
                    np.save(f'{self.images_dir}/pos{position+1}_frame{frame+1}_cell_markers.npy', markers)
                    np.save(f'{self.images_dir}/pos{position+1}_frame{frame+1}_core_markers.npy', core_markers)
                    np.save(f'{self.images_dir}/pos{position+1}_frame{frame+1}_shell_markers.npy', shell_markers)
            
                if sampling and (position, frame) in sampling_conditions:
                    print(position, frame)
                    self.generate_ROI_process_img(ROI_confirm, core_bin_img, core_markers, shell_markers, position, frame, savefig=True)
                    continue

        result_df = pd.concat(l_df, ignore_index=True) # concatenate all dataframes in l_df
        if self.position_name_column:
            result_df['Position_name'] = result_df['Position'].apply(lambda x: self.lif_stack.scenes[x]) 
        result_df['Time (min)'] = result_df['Frame'] * self.interval_min if self.interval_min is not None else result_df['Frame']
        condition_info = pd.read_csv(self.condition_info_file)
        for col in condition_info.columns:
            if col.startswith('condition'):
                result_df[col] = result_df['Position'].apply(lambda x: condition_info[col][x])
        result_df.to_csv(f'{self.output_dir}/result_df.csv', index=False)
        return result_df

    def generate_ROI_process_img(self, ROI_confirm, core_bin_img, core_markers, shell_markers, position, frame, savefig=True):
        plt.figure(figsize=(20,20))  # figure sizeを調整
        plt.subplot(2, 2, 1)  # 1行2列のsubplotの1番目に描画
        plt.imshow(ROI_confirm)
        plt.title('cell-ROI')

        plt.subplot(2, 2, 2)  # 1行2列のsubplotの2番目に描画
        plt.imshow(core_bin_img, cmap='gray')
        plt.title('core_bin_img')

        plt.subplot(2, 2, 3)  # 1行2列のsubplotの3番目に描画
        plt.imshow(core_markers)
        plt.title('core_markers')

        plt.subplot(2, 2, 4)  # 1行2列のsubplotの4番目に描画
        plt.imshow(shell_markers)
        plt.title('shell_markers')

        fig_title = f'pos{position+1}_frame{frame+1}' #set title for the figure
        plt.suptitle(fig_title, fontsize=20)  # 全体のタイトルを設定

        if savefig:
            plt.savefig(f'{self.images_dir}/{fig_title}.png') # save figure
            plt.close()
        else:
            plt.show()
    
# %% test
'''from set_roi import core_shell_analysis
#lif_file = '/Users/ktomo/Library/CloudStorage/OneDrive-TheUniversityofTokyo/nojilab_storage/tomohara/Result/Leica/240217/a/a1-1_a6-3_xy-end2.lif'
#lif_file = '/Users/ktomo/Library/CloudStorage/OneDrive-TheUniversityofTokyo/nojilab_storage/tomohara/Result/Leica/240219/a1-1_a6-3_xyt.lif'
lif_file = '/Users/ktomo/Library/CloudStorage/OneDrive-TheUniversityofTokyo/nojilab_storage/tomohara/Result/Leica/240216/a1-1_a6-3_xyt.lif'
setroi_class = core_shell_analysis(lif_file, 
                                   cell_ROIch=1, cell_isInverted=True,
                                   core_ROIch=0, core_isInverted=False,
                                   positions_for_test=[13], 
                                   frames_for_test=[13],
                                   save_markers=True
                      )
#result_df = setroi_class.get_result_df(sampling=True, sampling_num=1)
result_df = setroi_class.get_result_df(sampling=False)
result_df'''


# %%
