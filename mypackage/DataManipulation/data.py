import numpy as np
from PIL import Image
import glob, os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter

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
    def __add_legend_to_image(y, img, legend=None):
        # https://stackoverflow.com/questions/25482876/how-to-add-legend-to-imshow-in-matplotlib
        values = np.unique(y)
        colors = [ img.cmap(img.norm(value)) for value in values]
        if not legend:
            legend = values
        patches = [ mpatches.Patch(color=colors[i], label=f"[{i}] {name}") for i, name in enumerate(legend) ]
        plt.legend(handles=patches)

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
            img = plt.imshow(y)
            Dataset.__add_legend_to_image(y, img)
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
    def __only_one_contaminant(Y):
        for i in np.unique(Y):
            if i > 2:
                Y[Y == i] = 2
        return Y

    @staticmethod
    def load(dataset_folder, only_with_contaminant=False, only_one_contaminant_type=True, load_rest=False):
        info = []

        Y = []
        file_names = []
        for infile in Dataset.__list_files_by_file_type(os.path.join(dataset_folder, "labels"), ['tif', 'png']):
            file_name = infile.split('/')[-1].split('.')[0]
            y = Dataset.read_image(infile)
            if (not only_with_contaminant) or (np.max(np.unique(y)) > 1):
                                        # Here we assume that indexes 0 and 1 are belt and chicken meat
                Y.append(y)
                info.append(file_name)
                file_names.append(file_name)
            
            
        X = []
        for infile in file_names:
            X.append(Dataset.read_image(f"{dataset_folder}/{infile}.tif"))

        Y = np.array(Y)
        if only_one_contaminant_type:
            # Ensuring only one contaminant type
            Dataset.__only_one_contaminant(Y)
        Y += 1 # Have the indexes not as zero-indexed
        
        if load_rest:
            ###############################
            # Loading the unlabled images #
            X_rest = []
            for infile in Dataset.__list_files_by_file_type(dataset_folder, ['tif']):
                file_name = infile.split('/')[-1].split('.')[0]
                if file_name not in file_names:
                    print(f"Loading {file_name}")
                    x = Dataset.read_image(infile)
                    X_rest.append(x)
            return np.array(X), Y, info, np.array(X_rest)
        
        return np.array(X), Y, info
    
    @staticmethod
    def load_files(file_list, dataset_folder, only_one_contaminant_type=True):
        info = file_list

        Y = []
        X = []
        for file_name in file_list:
            Y.append(Dataset.read_image(f"{dataset_folder}/labels/{file_name}.png"))
            X.append(Dataset.read_image(f"{dataset_folder}/{file_name}.tif"))
        Y = np.array(Y)
        X = np.array(X)
        
        if only_one_contaminant_type:
            # Ensuring only one contaminant type
            Dataset.__only_one_contaminant(Y)
        Y += 1 # Have the indexes not as zero-indexed
        
        return X, Y, info
    
    @staticmethod
    def scale(X_test, X_train, scale='GlobalCenterting'):
        # Wording borrowed from here:
        #     https://machinelearningmastery.com/how-to-manually-scale-image-pixel-data-for-deep-learning/
        
        # TODO: Scale the data and compaire the "Variance Explained" difference
        #     https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py


        if scale == 'GlobalCenterting':
            # The maximum value observed is 75, thus 80 is more than the max
            max_pix_val = 80.0
            X_train /= max_pix_val
            if type(X_test) is np.ndarray: # Checking for "if not None:"
                X_test  /= max_pix_val
            
        elif scale == 'GlobalStandardization':
            # global standardization of pixels
            train = StackTransform(X_train)
            scaler = preprocessing.StandardScaler()
            scaler.fit(train.X_stack())
            X_train = train.Unstack(scaler.transform(train.X_stack()))
            if type(X_test) is np.ndarray:
                test = StackTransform(X_test)
                X_test = test.Unstack(scaler.transform(test.X_stack()))
            
        elif scale == 'RemoveTrend':
            average_spectra = np.squeeze(np.average(X_train, axis=(0, 1, 2)))
            x = np.arange(len(average_spectra))
            z = np.polyfit(x, average_spectra, 1)
            p = np.poly1d(z)
            
            X_train = X_train - p(x)
            if type(X_test) is np.ndarray:
                X_test  = X_test - p(x)
            
        elif scale == '1st_derivative':
            w, p = 21, 6 # See here Code/Scripts/SpectralDimensionReduction.ipynb
                         #   and here https://nirpyresearch.com/savitzky-golay-smoothing-method/
            X_train = savgol_filter(X_train, w, polyorder = p, deriv=1, axis=-1)
            X_train = X_train[:, :, :, 10:-10] # 10:-10 removes the noise at the ends # Interesting indexes are: (59, 67, 84)
            if type(X_test) is np.ndarray:
                X_test  = X_test[:, :, :, 10:-10]
                X_test  = savgol_filter(X_test,  w, polyorder = p, deriv=1, axis=-1)
            
            
        return X_test, X_train
    
    @staticmethod
    def PCA(X_train, X_test, n_components=3, plot=False, whiten=False):
        '''Takes as input:
               X_train - (n_items, n, m, k)
               X_test  - (m_items, n, m, k)
               Optionally:
               n_components=3
               plot=False
               whiten=False     # https://en.wikipedia.org/wiki/Whitening_transformation
           Outputs:
               X_train, X_test '''
        train = StackTransform(X_train)
        pca = PCA(n_components=n_components, whiten=whiten)
        principalComponents = pca.fit_transform(train.X_stack())
        X_train = train.Unstack(principalComponents, k=n_components)

        if type(X_test) is np.ndarray:
            test = StackTransform(X_test)
            X_test_stacked = pca.transform(test.X_stack())
            X_test = test.Unstack(X_test_stacked, k=n_components)
        
        if plot:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.title("Variance Explained")
            plt.xlabel("Principal Component")
            plt.ylabel("Proportion")
            plt.plot(np.arange(1, n_components+1), np.cumsum(pca.explained_variance_ratio_), "*")
            plt.plot(np.arange(1, n_components+1), [1]*n_components, "r--")
            plt.ylim(pca.explained_variance_ratio_[0]*0.8, 1.2)
            
#             n_items, n, m, k = X_test.shape
#             df = pd.DataFrame(X_test_stacked, columns=['PCA_1', 'PCA_2', 'PCA_3'])
#             df['Labels'] = Y_test.reshape((n_items*n*m, 1)).copy()
#             items = ['Belt', 'Meat', 'Contaminant 1', 'Contaminant 2', 'Contaminant 3', 'Contaminant 4']
#             for i, item in enumerate(items):
#                 df['Labels'][df['Labels'] == i+1] = item
#             sns.pairplot(df, hue='Labels', height=3)
#             plt.show()
        
        return X_train, X_test
    
    @staticmethod
    def train_test_split(X, Y, testRatio, randomState=345):
        if testRatio > 0:
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=testRatio, random_state=randomState)
        else:
            X_train, X_test, Y_train, Y_test = X, None, Y, None
        
        return X_train, X_test, Y_train, Y_test
    
    @staticmethod
    def reset_all_label_values_in_folder(dataSetFolder, plot=True):
        
        for infile in glob.glob(os.path.join(dataSetFolder, "*.png")):
            print(infile)
            Dataset.reset_label_values(infile, infile, plot)

class StackTransform():
    def __init__(self, X, Y=None):
        self.X = X
        self.Y = Y
        
    def __check_dimensions(self, data, n_items, n, m, k):
        if len(data.flatten()) != n_items*n*m*k:
            raise ValueError("The dimensions do not match")
        
    def X_stack(self):
        n_items, n, m, k = self.X.shape
        return np.resize(self.X, (n_items*n*m, k))
    
    def Y_stack(self):
        n_items, n, m, k = self.Y.shape
        return np.resize(self.Y, (n_items*n*m, k))
    
    def Unstack(self, Z, **kwargs):
        n_items, n, m, k = self.X.shape
        for key, value in kwargs.items():
            if key == 'k':
                k = value
            elif key == 'n_items':
                n_items = value
        self.__check_dimensions(Z, n_items, n, m, k)
        return np.resize(Z, (n_items, n, m, k))
            

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