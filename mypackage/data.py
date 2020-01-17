import numpy as np
from PIL import Image
import glob, os

class DataSet:
    layers = {"highImg": 0, "lowImg": 1, "plastImg": 2, "alImg": 3,
                "lowImgReg": 4, "highImgReg": 5}

    @staticmethod
    def tiff_to_np(img, channels_to_use=[], channel_index_last=True):
        """ Reads the desired layers as a multi channel/layer tiff image
        img           - Full file path to the image
        channels_to_use - The images/layers/channels to use,
                            numbered in a list like [1, 3, 5].
                            Empty list means all channels.
        """
        images = []
        if len(channels_to_use) != 0:
            for ch in channels_to_use:
                img.seek(ch)
                slice_ = np.array(img)
                if len(slice_.shape) == 3:
                    slice_ = slice_[:, :, 0] # This was done for the chicken laser label image... should not be needed
                images.append(slice_)
        else:
            i = 0
            while True:
                try:
                    img.seek(i)
                    slice_ = np.array(img)
                    images.append(slice_)
                    i += 1
                except EOFError:
                    break

        if channel_index_last:
            return np.transpose(np.array(images), (1, 2, 0))
        else:
            return np.array(images)

    @staticmethod
    def read_tiff(imageName, channels_to_use=[], channel_index_last=True):
        img = Image.open(imageName)
        return DataSet.tiff_to_np(img, channels_to_use, channel_index_last)

    @staticmethod
    def load(dataSetFolder, channels_to_use=[]):
        info = dataSetFolder.split("/")[-1]

        X = []
        for infile in glob.glob(dataSetFolder + "/*.tif"):
            X.append(DataSet.read_tiff(infile, channels_to_use))

        Y = []
        for infile in glob.glob(dataSetFolder + "/labels/*.tif"):
            y = DataSet.read_tiff(infile, channels_to_use=[0])
            # This is done for the chicken laser label image... should not be needed
            # The labels should be in the correct format in the stored label file
            y[np.nonzero(y > 100)] = 0
            for idx, i in enumerate(np.unique(y)):
                y[np.nonzero(y == i)] = idx

            Y.append(y)

        return X, Y, info


if __name__ == "__main__":
    X, Y, info = DataSet.load("/home/thor/HI/Lokaverkefni/Code/data/sample", channels_to_use=[1,2,6])


    print(np.unique(Y))


    