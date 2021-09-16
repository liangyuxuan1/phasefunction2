import torch
import numpy as np
import pickle
import os

# https://github.com/aws-samples/sagemaker-gpu-performance-io-deeplearning

class DataPreprocessor:
    """
    Preprocess the raw data to a temp data that is pickled
    """
    def dump(self, img_labels, images_base_directory, destination_pickle_path, destination_pickle_file_name, preprocessing_transformer):
        """
        Pickles the raw images
        :param parts: Divides the pickle into n parts, with number of images divided almost equally per part
        :param images_base_directory: Expected format file/dir name convention, e.g 001.ak47/001_0001.jpg
        :param destination_pickle_file_name: destination pickle file name, e.g train.pkl
        :param destination_pickle_path:
        """
        # Load images_base_directory
        # files_per_pickle = len(img_labels) // parts
        # pickle_part_num = 1
        result = None
        images = []
        for i in range(len(img_labels)):
            file_name = os.path.join(images_base_directory, img_labels.iloc[i,0]) + '.npy'
            image = np.load(file_name)
            h, w = image.shape
            image = torch.from_numpy(image).reshape(1, h, w)
            image = image.float()

            # apply preprocessing
            image = preprocessing_transformer(image)
            images.append(image)
            result = {
                "images": images
            }

        # Save final remaining parts
        self._save_part(destination_pickle_path, destination_pickle_file_name, result)

    def _save_part(self, destination_pickle_path, destination_pickle_file_name, result_obj_to_pickle):
        if result_obj_to_pickle is None: return

        pickle_file_path = os.path.join(destination_pickle_path, destination_pickle_file_name)
        with open(pickle_file_path, "wb") as f:
            pickle.dump(result_obj_to_pickle, f)

    def load(self, pickle_path):
        """
        Load pickled file
        :param pickle_path:
        :return:
        """
        with open(pickle_path, "rb") as f:
            obj = pickle.load(f)

        return obj["images"]