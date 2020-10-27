import os
import cv2
import torch
import numpy as np
import pandas as pd
from nvidia.dali import ops
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator

from Countor_NN import Countor_NN
from Calibration_loader import Calibration_loader
from Countor import Countor
from config import paths, config

import numpy as np


def main():
    # create Neural network
    countor_net = Countor_NN()
    countor_net.eval()
    countor_net.cuda()
    countor_net.load_state_dict(
        torch.load(paths['path_faster_RCNN_weigths'], map_location=lambda storage, loc: storage))

    # input video file
    input_video_path = './unbox_test/input/MVI_40855.mp4'

    # output images folder
    output_images_folder = './unbox_test/output'

    # create video loader
    pipeline = VideoPipe(batch_size=config['batch_loader_size'],
                         num_threads=config['num_threads'],
                         device_id=0,
                         data=input_video_path)

    pipeline.build()

    dali_iter = DALIGenericIterator(pipelines=[pipeline],
                                    output_map=['image'],
                                    size=pipeline.epoch_size("Reader"),
                                    fill_last_batch=False,
                                    last_batch_padded=True)

    # load the calibration for this video
    calibration = Calibration_loader('16/')

    # create counter object
    counter_obj = Countor(countor_net, calibration, 0)

    # detect each frame
    for frame_index, video_data in enumerate(dali_iter):
        print('detecting:', frame_index)

        for image in video_data[0]['image']:
            image = image.permute((0, 3, 1, 2)).contiguous().float().div(255)
            counter_obj.step(image)
            pass
        pass
    pass

    # load the results as a DataFrame
    reform = {(outerKey, innerKey): values for outerKey, innerDict in counter_obj.results.items() for innerKey, values
              in innerDict.items()}
    pd_data = pd.DataFrame(reform, index=['left', 'top', 'x2', 'y2', 'conf']).T
    pd_data.reset_index(inplace=True)
    pd_data.rename(columns={"level_0": "ID", "level_1": "frame"}, inplace=True)

    # correct data to original
    pd_data['width'] = pd_data['x2'] - pd_data['left']
    pd_data['height'] = pd_data['y2'] - pd_data['top']

    # put in format the restuls
    counts = pd.DataFrame.from_dict(counter_obj.countor_restults, orient='index',
                                    columns=['frame_id', 'movement_id', 'x1', 'y1', 'x2', 'y2', 'length'])
    counts.reset_index(inplace=True)
    counts.rename(columns={'index': 'ID'}, inplace=True)
    counts['video_id'] = 0

    results_in_format = pd_data.loc[:, ['frame', 'ID', 'left', 'top', 'width', 'height', 'conf']]

    # prepare color to visualise
    color_manager = ColorManager()

    # load video to get each image of frames
    cap = cv2.VideoCapture(input_video_path)
    count_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # loop flag
    flag = 0

    # track points in dictionary format of each car
    dict_track_pts = {}

    # loop each frame of video
    while cap.isOpened() and flag < count_frames:
        output_image_path = output_images_folder + os.path.sep + str(flag) + '.png'
        # set position of video frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, flag)
        # read one frame, read origin image
        _, image_obj = cap.read()

        # draw rect bbox and car id
        # filter results with frame index
        result_temp = results_in_format.loc[results_in_format['frame'] == flag]
        count = len(result_temp)

        # loop bbox rect and car id of one frame image
        for index in range(0, count):
            car_id = int(result_temp.iloc[index]['ID'])
            x1 = int(result_temp.iloc[index]['left'])
            y1 = int(result_temp.iloc[index]['top'])
            w1 = int(result_temp.iloc[index]['width'])
            h1 = int(result_temp.iloc[index]['height'])
            det_conf = result_temp.iloc[index]['conf']

            # left top point
            pt1 = (x1, y1)
            # right bottom point
            pt2 = (x1 + w1, y1 + h1)

            # get multiple color from different car id
            color1 = color_manager.get_color(car_id)
            # draw bbox rect on origin image
            image_obj = cv2.rectangle(image_obj, pt1, pt2, color1, 2)

            # draw car id on image
            font = cv2.FONT_HERSHEY_SIMPLEX
            image_obj = cv2.putText(
                img=image_obj, text='Car-' + str(car_id) + ' ' + str(int(det_conf * 100)) + '%',
                org=(x1, y1 + 12), fontFace=font,
                fontScale=0.4, color=color1, thickness=1)

            # get center point of one bbox
            center = int(x1 + (w1 / 2)), int(y1 + (h1 / 2))

            # one car one dictionary object
            # append all of center point into list object
            # save the center point list into dictionary object
            if str(car_id) in dict_track_pts:
                list_temp = dict_track_pts.get(str(car_id))
                list_temp.append(center)
                pass
            else:
                list_temp = [center, ]
                dict_track_pts[str(car_id)] = list_temp
                pass
            pass
        pass

        # loop dictionary to get center point
        for id_key, list_temp in dict_track_pts.items():
            # draw one polyline when got more than 2 points
            if len(list_temp) > 2:
                # get line color by car id
                color1 = color_manager.get_color(int(id_key))
                # convert list to point array
                array_pts = [np.array(list_temp, np.int32)]
                # draw track polyline on image
                image_obj = cv2.polylines(img=image_obj, pts=array_pts, isClosed=False, color=color1, thickness=1)
            pass
        pass

        # save image to disk
        cv2.imwrite(output_image_path, image_obj)
        print('writing:', flag)

        flag += 1
        pass

    print("DONE")


class ColorManager(object):

    def __init__(self):
        self.PALETTE_HEX = [
            "#000000", "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
            "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
            "#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
            "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100",
            "#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F",
            "#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
            "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
            "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",
            "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
            "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00",
            "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
            "#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
            "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C",
            "#83AB58", "#001C1E", "#D1F7CE", "#004B28", "#C8D0F6", "#A3A489", "#806C66", "#222800",
            "#BF5650", "#E83000", "#66796D", "#DA007C", "#FF1A59", "#8ADBB4", "#1E0200", "#5B4E51",
            "#C895C5", "#320033", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94",
            "#7ED379", "#012C58"]

        self.COLORS = [*map(self._parse_hex_color, self.PALETTE_HEX)]
        self.PALETTE_RGB = np.asarray(self.COLORS)
        self.COLOR_DIFF_WEIGHT = np.asarray((3, 4, 2))
        self.color_index_scores = np.zeros(len(self.COLORS))
        self.color_map = {}

    @staticmethod
    def _parse_hex_color(s):
        r = int(s[1:3], 16)
        g = int(s[3:5], 16)
        b = int(s[5:7], 16)
        # r = int(s[1:3], 16) / 256
        # g = int(s[3:5], 16) / 256
        # b = int(s[5:7], 16) / 256
        return r, g, b

    def get_color(self, object_id, roi=None):
        if object_id not in self.color_map:
            if roi is None:
                color_id = self.color_index_scores.argmax()
                self.color_map[object_id] = self.COLORS[color_id]
                self.color_index_scores[color_id] -= 1
            else:
                mean_color = roi.mean(axis=(0, 1)) / 256
                color_scores = (np.square(self.PALETTE_RGB - mean_color) *
                                self.COLOR_DIFF_WEIGHT).sum(axis=1)
                color_id = (color_scores + self.color_index_scores).argmax()
                self.color_index_scores[color_id] -= 1
                self.color_map[object_id] = self.COLORS[color_id]
        return self.color_map[object_id]


class VideoPipe(Pipeline):

    def __init__(self, batch_size, num_threads, device_id, data):
        super(VideoPipe, self).__init__(batch_size, num_threads, device_id, seed=120)
        self.input = ops.VideoReader(device="gpu", filenames=data, sequence_length=1,
                                     shard_id=0, num_shards=1, random_shuffle=False, initial_fill=10)

    def define_graph(self):
        output = self.input(name="Reader")
        return output


if __name__ == "__main__":
    main()
    pass
