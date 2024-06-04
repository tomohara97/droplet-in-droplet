# %%
import os
from aicsimageio import AICSImage

# %%
class LIFProcessor:
    def __init__(self, lif_file):
        self.lif_file = lif_file
        self.lif_stack = AICSImage(lif_file)
        
    def dict_extraction(self, keys2show, src_dict):
        dst_dict = {}
        for key in keys2show:
            if key in src_dict:
                dst_dict[key] = src_dict[key]
        return dst_dict

    def search_seq_info(self):
        metadata = self.lif_stack.metadata
        seq_info = None
        for elem in metadata.iter():
            if 'LDM_Block_Sequential' in elem.tag:
                seq_info = elem
                break
        return seq_info

    def confocal_settings(self):
        # 関数内のlif_stackをインスタンス変数に変更
        seq_info = self.search_seq_info()
        if seq_info is None:
            return None
        else:
            seq_master = seq_info[0]
            confocal_settings = seq_master[0].attrib
            confocal_settings2show = ['ScanMode', 'ScanSpeed', 'Zoom', 'ObjectiveName', 'PinholeAiry']
            confocal_settings_dict = self.dict_extraction(confocal_settings2show, confocal_settings)
            confocal_settings_dict['PinholeAiry (AU)'] = confocal_settings_dict.pop('PinholeAiry')
            confocal_settings_dict['PinholeAiry (AU)'] = round(float(confocal_settings_dict['PinholeAiry (AU)']), 3)

            if self.lif_stack.dims.T != 1:
                time_interval_min = float(confocal_settings['CycleTime']) / 60
                time_interval_min = round(time_interval_min, 1)
                confocal_settings_dict['Interval (min)'] = time_interval_min
            else:
                time_interval_min = None
            confocal_settings_dict_str = '\n'.join([f'{key}: {value}' for key, value in confocal_settings_dict.items()])
            return confocal_settings_dict_str, time_interval_min

    def process_seq_list(self):
        seq_info = self.search_seq_info()
        seq_list = seq_info[1]
        result = ['Sequential settings:']
        for seq_num in range(len(seq_list)):
            UserSettingName = seq_list[seq_num].attrib['UserSettingName']
            result.append(f'{UserSettingName}:')

            # Scan information
            l_aveaccu = ['LineAverage', 'Line_Accumulation', 'FrameAverage', 'FrameAccumulation']
            for aveaccu_name in l_aveaccu:
                if aveaccu_name in seq_list[seq_num].attrib:
                    aveaccu_count = seq_list[seq_num].attrib[aveaccu_name]
                    if aveaccu_count != '1':
                        result.append(f'\t{aveaccu_name}: {aveaccu_count}')
            # Laser information
            for aotf in seq_list[seq_num].iter('Aotf'):
                light_source_name = aotf.get('LightSourceName')  # 'LightSourceName'属性の取得
                if light_source_name == 'SuperContVisible':
                    light_source_name = 'WLL'
                # 'LaserLineSetting'タグを持つ全ての要素を検索
                for setting in aotf.iter('LaserLineSetting'):
                    intensity_dev = float(setting.get('IntensityDev'))  # 'IntensityDev'属性の取得
                    if intensity_dev > 0:  # 'IntensityDev'が0より大きい場合
                        intensity_dev = round(intensity_dev, 2) # intensity_devを小数点以下4桁で表示
                        laser_line = setting.get('LaserLine')  # 'LaserLine'属性の取得
                        result.append(f'\tExcitation: {light_source_name}({laser_line}nm) {intensity_dev}%')

            # Detector information
            detector_dict = {}
            for detector in seq_list[seq_num].iter('Detector'):
                channel = detector.get('Channel')
                IsActive = detector.get('IsActive')
                detector_name = detector.get('Name')
                gain = detector.get('Gain')
                #gain = round(float(gain), 1)
                try:
                    gain = round(float(gain), 1)
                except TypeError:
                    gain = None

                if detector.get('IsTimeGateActivated') == '1':
                    TimeGatePulseStart = detector.get('TimeGatePulseStart')
                    TimeGatePulseStart = float(TimeGatePulseStart) / 1e3
                    TimeGatePulseEnd = float(detector.get('TimeGatePulseEnd')) /1e3
                    TimeGateWavelength = detector.get('TimeGateWavelength')
                    TimeGate_str = f'TimeGate: {TimeGatePulseStart}-{TimeGatePulseEnd}ns for {TimeGateWavelength}nm'
                else:
                    TimeGate_str = 'TimeGate: OFF'
                detector_dict[channel] = [IsActive, detector_name, gain, TimeGate_str]

            for multiband in seq_list[seq_num].iter('MultiBand'):
                channel = multiband.get('Channel')
                TargetWaveLengthBegin = round(float(multiband.get('TargetWaveLengthBegin')), 1)
                TargetWaveLengthEnd = round(float(multiband.get('TargetWaveLengthEnd')), 1)
                detection_range = f'{TargetWaveLengthBegin}-{TargetWaveLengthEnd}nm'
                detector_dict[channel].append(detection_range)

            for value in detector_dict.values():
                if value[0] == '1':
                    try:
                        result.append(f'\tDetection: {value[1]} ({value[4]}) {value[2]}%, {value[3]}')
                    except IndexError:
                        raise ValueError('Detector information is not complete')

        return result

    def compile_file_info(self):
        file_name = os.path.basename(self.lif_file)
        dimensions = self.lif_stack.dims
        confocal_settings = self.confocal_settings()[0] if self.confocal_settings() else 'No confocal settings'
        sequential_settings = self.process_seq_list() if self.search_seq_info() else ['No sequential settings']
        return [file_name, dimensions, confocal_settings, sequential_settings]
    
    def show_metadata(self):
        file_name, dimensions, confocal_settings, sequential_settings = self.compile_file_info()
        print(f'File name: {file_name}')
        print(f'{dimensions}')
        print()
        print(confocal_settings)
        print('\n'.join(sequential_settings))

    def get_time_interval(self):
        '''return time interval in minutes'''
        time_interval_min = self.confocal_settings()[1] if self.confocal_settings() is not None else None
        return time_interval_min
# %%
#lif_file = '/Users/ktomo/Library/CloudStorage/OneDrive-TheUniversityofTokyo/nojilab_storage/tomohara/Result/Leica/230719/b/b1-1_b2-2_xyt_obj63x.lif'
#processor = LIFProcessor(lif_file)
#processor.process_seq_list()

# %% [markdown]
# # memo
# - positionごと（視野ごと）に撮影条件が定義されている。１つ目の視野の撮影条件を出力することにした。全視野で同じ撮影条件という前提
# - ひとつのlif fileにxzとxyなど複数のstackが入っている場合の場合分け←難しそう、保存時に分けて保存するのが手っ取り早い？

