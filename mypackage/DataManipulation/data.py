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
    TOMRA_WAVELENGTHS = np.array([928, 932, 935, 939, 942, 946, 950, 953, 957, 960, 964, 968, 971, 975, 978, 982, 986, 989, 993, 997, 1000, 1004, 1007, 1011, 1015, 1018, 1022, 1025, 1029, 1033, 1036, 1040, 1043, 1047, 1051, 1054, 1058, 1061, 1065, 1069, 1072, 1076, 1079, 1083, 1087, 1090, 1094, 1097, 1101, 1105, 1108, 1112, 1115, 1119, 1123, 1126, 1130, 1134, 1137, 1141, 1144, 1148, 1152, 1155, 1159, 1162, 1166, 1170, 1173, 1177, 1180, 1184, 1188, 1191, 1195, 1198, 1202, 1206, 1209, 1213, 1216, 1220, 1224, 1227, 1231, 1234, 1238, 1242, 1245, 1249, 1252, 1256, 1260, 1263, 1267, 1271, 1274, 1278, 1281, 1285, 1289, 1292, 1296, 1299, 1303, 1307, 1310, 1314, 1317, 1321, 1325, 1328, 1332, 1335, 1339, 1343, 1346, 1350, 1353, 1357, 1361, 1364, 1368, 1371, 1375, 1379, 1382, 1386, 1390, 1393, 1397, 1400, 1404, 1408, 1411, 1415, 1418, 1422, 1426, 1429, 1433, 1436, 1440, 1444, 1447, 1451, 1454, 1458, 1462, 1465, 1469, 1472, 1476, 1480, 1483, 1487, 1490, 1494, 1498, 1501, 1505, 1508, 1512, 1516, 1519, 1523, 1527, 1530, 1534, 1537, 1541, 1545, 1548, 1552, 1555, 1559, 1563, 1566, 1570, 1573, 1577, 1581, 1584, 1588, 1591, 1595, 1599, 1602, 1606, 1609, 1613, 1617, 1620, 1624, 1627, 1631, 1635, 1638, 1642, 1645, 1649, 1653, 1656, 1660, 1664, 1667, 1671, 1674])
    TOMRA_OBVIOUS_PLASTICS = ["20200213_120044_FM_fillet_repeat_sample_B_32", "20200213_120111_FM_fillet_repeat_sample_B_33", "20200213_120158_FM_fillet_repeat_sample_B_34", "20200213_120308_FM_fillet_repeat_sample_B_36", "20200213_120339_FM_fillet_repeat_sample_B_37", "20200213_120359_FM_fillet_repeat_sample_B_38"]

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
    def reset_label_values(imageName, newImageName, plot=False, weak_labels=False):
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

        if weak_labels:
            # This assumes the labels 0 and 1
            y[y == 1] = 2
        image = Image.fromarray(y)

        if plot:
            plt.figure()
            img = plt.imshow(y)
            Dataset.__add_legend_to_image(y, img)
            plt.show()
            
        # Save the image after the plot to allow for cancelation
        image.save(newImageName)

    @staticmethod
    def reset_all_label_values_in_folder(dataSetFolder, plot=True, weak_labels=False):

        for infile in glob.glob(os.path.join(dataSetFolder, "*.png")):
            print(infile)
            Dataset.reset_label_values(infile, infile, plot, weak_labels)

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
#         Y += 1 # Have the indexes not as zero-indexed
        
        if load_rest:
            ###############################
            # Loading the unlabled images #
            X_rest = []
            for infile in Dataset.__list_files_by_file_type(dataset_folder, ['tif']):
                file_name = infile.split('/')[-1].split('.')[0]
                if file_name not in file_names:
                    x = Dataset.read_image(infile)
                    X_rest.append(x)
            return np.array(X), Y, info, np.array(X_rest)
        
        return np.array(X), Y, info
    
    @staticmethod
    def load_files(file_list, dataset_folder, with_labels=True, only_one_contaminant_type=True):
        Y = []
        X = []
        for file_name in file_list:
            if with_labels:
                Y.append(Dataset.read_image(f"{dataset_folder}/labels/{file_name}.png"))
            X.append(Dataset.read_image(f"{dataset_folder}/{file_name}.tif"))
        X = np.array(X)
        if not with_labels:
            return X
        Y = np.array(Y)
        
        if only_one_contaminant_type:
            # Ensuring only one contaminant type
            Dataset.__only_one_contaminant(Y)
#         Y += 1 # Have the indexes not as zero-indexed
        
        return X, Y
    
    @staticmethod
    def scale(X_test, X_train, scaler='GlobalStandardization'):
        # Wording borrowed from here:
        #     https://machinelearningmastery.com/how-to-manually-scale-image-pixel-data-for-deep-learning/
        
        # TODO: Scale the data and compaire the "Variance Explained" difference
        #     https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py

#         if scale == 'GlobalCenterting':
#             # The maximum value observed is 75, thus 80 is more than the max
#             max_pix_val = 80.0
#             X_train /= max_pix_val
#             if type(X_test) is np.ndarray: # Checking for "if not None:"
#                 X_test  /= max_pix_val
            
        if type(scaler) == str:
            if scaler == 'GlobalStandardization':
                # global standardization of pixels
                train = StackTransform(X_train)
                scaler = preprocessing.StandardScaler(with_mean=True, with_std=True)
                scaler.fit(train.X_stack())
                X_train = train.Unstack(scaler.transform(train.X_stack()))
                if type(X_test) is np.ndarray:
                    test = StackTransform(X_test)
                    X_test = test.Unstack(scaler.transform(test.X_stack()))

            elif scaler == 'RemoveTrend':
                average_spectra = np.squeeze(np.average(X_train, axis=(0, 1, 2)))
                x = np.arange(len(average_spectra))
                z = np.polyfit(x, average_spectra, 1)
                p = np.poly1d(z)

                X_train = X_train - p(x)
                if type(X_test) is np.ndarray:
                    X_test  = X_test - p(x)

            elif scaler == '1st_derivative':
                w, p = 21, 6 # See here Code/Scripts/SpectralDimensionReduction.ipynb
                             #   and here https://nirpyresearch.com/savitzky-golay-smoothing-method/
                X_train = savgol_filter(X_train, w, polyorder = p, deriv=1, axis=-1)
                X_train = X_train[:, :, :, 10:-10] # 10:-10 removes the noise at the ends # Interesting indexes are: (59, 67, 84)
                if type(X_test) is np.ndarray:
                    X_test  = X_test[:, :, :, 10:-10]
                    X_test  = savgol_filter(X_test,  w, polyorder = p, deriv=1, axis=-1)
        else:
            if type(X_test) is np.ndarray:
                test = StackTransform(X_test)
                X_test = test.Unstack(scaler.transform(test.X_stack()))
            if type(X_train) is np.ndarray:
                train = StackTransform(X_train)
                X_train = train.Unstack(scaler.transform(train.X_stack()))
            
            
        return X_test, X_train, scaler
    
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
    def make_zoomed_in_dataset(X, Y, size=25, sample_multiplication=5, contaminant_type=2):
        count, n, m, k = X.shape
        output_count = count*sample_multiplication
        enlarged_X = np.zeros((output_count, size, size, k))
        enlarged_Y = np.zeros((output_count, size, size, 1))

        for i in range(output_count):
            choice = np.random.choice(count)
            x, y = Dataset.zoom_in_on_contaminant(X[choice], Y[choice], size=size, contaminant_type=contaminant_type)
            enlarged_X[i], enlarged_Y[i] = x, y

        return enlarged_X, enlarged_Y
    
    @staticmethod
    def get_max_min(center, size=32):
        MIN, MAX = 0, 100
        max_val = center + size // 2
        if max_val > MAX:
            max_val = MAX
        min_val = max_val - size
        if min_val < MIN:
            min_val = MIN
            max_val = min_val + size
        return min_val, max_val
            
    @staticmethod
    def zoom_in_on_contaminant(img, label, size=32, contaminant_type=2):
        MIN, MAX = 0, 100
        
        indices_x, indices_y, _ = np.nonzero(label == contaminant_type)

        if len(indices_x) != 0:
            random_center = np.random.choice(len(indices_x))
            start_x, end_x = Dataset.get_max_min(indices_x[random_center], size)
            start_y, end_y = Dataset.get_max_min(indices_y[random_center], size)
        else:
            # Select a random center with exponentially declining probability from the center pixels
            p = np.exp(-3.4551-np.linspace(0, 1, num=50))
            missing = 1 - np.sum(p) # The probability needs to sum to 1
            p[0] = p[0] + missing
            p = list(p[::-1]/2) + list(p/2)
            start_x, end_x = Dataset.get_max_min(np.random.choice(MAX, p=p), size)
            start_y, end_y = Dataset.get_max_min(np.random.choice(MAX, p=p), size)

        label = label[start_x:end_x, start_y:end_y, :].copy()
        img = img[start_x:end_x, start_y:end_y, :].copy()
        return img, label
    
    @staticmethod
    def zoom_in_on_center(img, x_center=50, y_center=50, size=32):
        start_x, end_x = Dataset.get_max_min(x_center, size)
        start_y, end_y = Dataset.get_max_min(y_center, size)
        return img[start_x:end_x, start_y:end_y, :]

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
    # Dataset.reset_all_label_values_in_folder("/home/thor/HI/Lokaverkefni/Code/data/tomra/labels/tmp")
    Dataset.reset_all_label_values_in_folder("/home/thor/HI/Lokaverkefni/Code/data/tomra_weak_labeling/labels/tmp", weak_labels=True)
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
