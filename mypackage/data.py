import numpy as np
from PIL import Image
import glob, os
from sklearn.decomposition import PCA

class Dataset:
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
    def read_image(imageName, channels_to_use=[], channel_index_last=True):
        img = Image.open(imageName)
        return Dataset.tiff_to_np(img, channels_to_use, channel_index_last)

    @staticmethod
    def reset_label_values(imageName, newImageName, plot=False):
        '''This method assumes that the background is 65535 and that no label has a value less than the number of labeled classes
            Chicken Labels (Done with GIMP, saved as png [Compression=0, 16bpc Gray]):
                - Fat - Blue (0, 0, 50)
                - Meat - Red (150, 0, 0)
                - Wood - Green (0, 50, 0)
                - Cable Insulator - Green (0, 100, 0)
                - Belt - Green (0, 150, 0)
                - POM - Green (0, 200, 0)
        '''
        y = np.squeeze(Dataset.read_image(imageName, channel_index_last=False))

        print(f"Working on {imageName}")
        print(np.unique(y))
        y[y == 65535] = 0 # 65535 Is white background in 16bpc Gray
        y[y == 18932] = 1 # 18932 Is Red (150, 0, 0) in 16bpc Gray
        y[y == 1637]  = 2 #  1637 Is Fat (0, 0, 50) in 16bpc Gray
        for idx, value in enumerate(np.unique(y)):
            y[y == value] = idx
        image = Image.fromarray(y)

        if plot:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            im = plt.imshow(y)

            # https://stackoverflow.com/questions/25482876/how-to-add-legend-to-imshow-in-matplotlib
            values = np.unique(y)
            colors = [ im.cmap(im.norm(value)) for value in values]
            patches = [ mpatches.Patch(color=colors[i], label=f"Value {i}") for i in values ]
            plt.legend(handles=patches)

            plt.show()
            
        # Save the image after the plot to allow for cancelation
        image.save(newImageName)

    @staticmethod
    def __list_files_by_file_type(path, types):
        fileList = []
        for extension in types:
            fileList += glob.glob(os.path.join(path, f'*.{extension}'))
        return fileList

    @staticmethod
    def load(datasetFolder, channels_to_use=[]):
        info = datasetFolder.split("/")[-1]

        Y = []
        fileNames = []
        for infile in Dataset.__list_files_by_file_type(os.path.join(datasetFolder, "labels"), ['tif', 'png']):
            fileNames.append(infile.split('/')[-1].split('.')[0])
            y = Dataset.read_image(infile)
            Y.append(y)
            
        X = []
        for infile in fileNames:
            X.append(Dataset.read_image(f"{datasetFolder}/{infile}.tif", channels_to_use))

        # Ensuring only one contaminant type
        Y = np.array(Y)
        for i in np.unique(Y):
            if i > 2:
                Y[Y == i] = 2
        Y += 1
                
        return np.array(X), Y, info
    
    @staticmethod
    def PCA(X_train, X_test=None, n_components=3, plot=False):
        def stack(X):
            return np.resize(X, (n_items*n*m, k))
        def un_stack(X):
            return np.resize(principalComponents, (n_items, n, m, n_components))
        
        X_train = np.array(X_train)
        n_items, n, m, k = X_train.shape
        X_train = stack(X_train)
        pca = PCA(n_components=n_components)
        principalComponents = pca.fit_transform(X_train)
        X_train = un_stack(principalComponents)
        print("pca.explained_variance_ratio_")
        print(pca.explained_variance_ratio_)
        
        if X_test != None:
            print("This probably does not work without stacking and unstacking")
            X_test = pca.transform(X_test)
            
            return X_train, X_test
        
        if plot:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.imshow(X_train[0])
        
        return X_train
    
    @staticmethod
    def reset_all_label_values_in_folder(dataSetFolder, plot=True):
        
        for infile in glob.glob(os.path.join(dataSetFolder, "*.png")):
            print(infile)
            Dataset.reset_label_values(infile, infile.split(".png")[0]+"_out.png", plot)


if __name__ == "__main__":
    Dataset.reset_all_label_values_in_folder("/home/thor/HI/Lokaverkefni/Code/data/tomra/labels/tmp")
#     X, Y, info = Dataset.load("../data/tomra") #, channels_to_use=[1,2,6])
#     print(len(X), len(Y))
#     print(X[0].shape, Y[0].shape)
#     print(np.unique(Y))

    # y = Dataset.read_image('/home/thor/HI/Lokaverkefni/Code/data/tmp/chicken_fm_RGB_labels.png')
    # print(y)
    # print(y.shape)
    # print(np.unique(y))

#     Dataset.reset_label_values('/home/thor/HI/Lokaverkefni/Code/data/tomra/labels/20200213_120308_FM_fillet_repeat_sample_B_36.png', '/home/thor/HI/Lokaverkefni/Code/data/tomra/labels/20200213_120308_FM_fillet_repeat_sample_B_36_out.png', True)
#     Dataset.reset_label_values('/home/thor/HI/Lokaverkefni/Code/data/tomra/labels/20200213_120339_FM_fillet_repeat_sample_B_37.png', '/home/thor/HI/Lokaverkefni/Code/data/tomra/labels/20200213_120339_FM_fillet_repeat_sample_B_37_out.png', True)
#     Dataset.reset_label_values('/home/thor/HI/Lokaverkefni/Code/data/tomra/labels/20200213_120359_FM_fillet_repeat_sample_B_38.png', '/home/thor/HI/Lokaverkefni/Code/data/tomra/labels/20200213_120359_FM_fillet_repeat_sample_B_38_out.png', True)
