import scipy.ndimage as ndimage
from numba import jit
import numpy as np
import pandas as pd
import IPython
import mypackage
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_BASE_PATH = f"{DIR_PATH}/SpectralAbsorbtionDatabase.pkl"
TYPE_BACKGROUND  = 1
TYPE_CHICKEN     = 2
TYPE_CONTAMINANT = 3
    
class FakeDataset:
    
    def __init__(self, wavelengths="All", max_contaminants_per_piece=3, max_contaminant_classes=["HardPlastic", "Liner"], center_signals=False):
        f'''FakeDataset generates a HSI built on the database located at {DATA_BASE_PATH}
        
            max_contaminants_per_piece:  The maximum number of contaminants per piece
            max_contaminant_classes:     
        '''
        self.chicken_types = {0: "Fillet", 1: "Thigh"}
        self.database, self.plastic_types = self.__load_spectra_database()
        self.database_wavelengths = [col for col in self.database.columns if type(col) == float]
        
        if (type(wavelengths) == str) and (wavelengths == "All"):
            self.wavelengths = self.database_wavelengths
        else:
            self.wavelengths            = wavelengths
        self.max_contaminants_per_piece = max_contaminants_per_piece
        self.max_contaminant_classes    = max_contaminant_classes
        
        if center_signals:
#             for col in self.database_wavelengths:
#                 # Start all signals at zero at the first sampling wavelength
#                 self.database.loc[:, col] = self.database.loc[:, col] - self.database.iloc[:, 1]
#             for i, row in self.database.iterrows():
#                 row[1:] = row[1:] - row[1:].mean()
            self.database[self.database_wavelengths] = self.database[self.database_wavelengths].sub(self.database[self.database_wavelengths].mean(axis=1), axis=0)
            self.database[self.database_wavelengths] -= self.database[self.database_wavelengths].min().min()
        
    def __get_all_of_type(self, type_name):
        samples = self.database[self.database["Type"].str.contains(type_name)]
        return samples.loc[:, samples.columns != "Type"]
        
    def __load_spectra_database(self):
        def rename_database_group(substring, type_name):
            items = database[database["Type"].str.contains(substring)].index
            for i, idx in enumerate(items):
                database.loc[idx, "Type"] = f"{type_name}_{i}"

        database = pd.read_pickle(DATA_BASE_PATH)
        database.reset_index(inplace=True)
        database.rename(columns={"index": "Type"}, errors="raise", inplace=True)

        # Remove the black plastic glove
        database = database[database["Type"].str.contains("black_latex_glove") != True]
        # Remove all mixed materials
        database = database[database["Type"].str.contains("liner_on_") != True]
        # Here the order does matter, since some substrings used are contained in more than one groups
        rename_database_group("_fat_", "Fat")
        rename_database_group("chicken_not_as_red", "chicken_fillet_not_as_red")
        rename_database_group("chicken_fillet", "Fillet")
        rename_database_group("chicken_thigh", "Thigh")

#         rename_database_group("pu_belt", "Plastic_Pu_Belt")
#         rename_database_group("blue_belt_roll", "Plastic_Belt")
#         rename_database_group("tube", "Plastic_Tube")
#         rename_database_group("glove", "Plastic_Glove")
#         rename_database_group("liner", "Plastic_Liner")
#         rename_database_group("belt", "Plastic_Belt")
        types = database[database["Type"].str.contains("plastic")]["Type"].str.replace("[0-9()]+$", "")
        plastic_types = dict(enumerate(np.unique(types)))

        # plastics = database[database["Type"].str.contains("plastic") &  (database["Type"].str.contains("liner") != True)]
        
        return database, plastic_types
    
    def __resample_wavelengths(self, spectra_data, wavelengths):
        '''Returns One-dimensional linear interpolation (a.k.a a straight line in between sample points).'''
        n, m = spectra_data.shape
        resample = lambda spectrum: np.interp(wavelengths, self.database_wavelengths, spectrum)
        return np.apply_along_axis(resample, axis=1, arr=spectra_data) # Reduce spectral dimension down
    
    def __collect_samples(self, material_type, shape):
        def select_spectra(types):
            item_selected = np.random.choice(list(types.keys())) # Choose randomly one item
            type_selected = types[item_selected]
            samples = self.__get_all_of_type(type_selected)
            pixels_needed, spectral_dimension = shape
            # Then randomly select from the database with resampling
            idx_selected_spectrum = np.random.choice(len(samples), pixels_needed, replace=True)
            selected_spectrum = samples.iloc[idx_selected_spectrum].values
            # Resampling the wavebands to match the desired output
            return self.__resample_wavelengths(selected_spectrum, self.wavelengths)
            
        if   material_type == TYPE_BACKGROUND:
            return np.zeros(shape) # TODO: Add in the background
        elif material_type == TYPE_CHICKEN:
            return select_spectra(self.chicken_types)
        elif material_type == TYPE_CONTAMINANT:
            return select_spectra(self.plastic_types)
                    
    @staticmethod
    def transform_to_reflectance(data, C=1):
        '''The ransformation for absorbance to reflectance is the inverse of absorbance = log_10(C/reflectance).
            Thus the transformation form refelctance to absorbance is
                    reflectance = C/(10^absorbacne)'''
        return C / (10**data)
    
    def __add_noise(self, column, noise_variance=0.004):
        # https://stackoverflow.com/questions/14058340/adding-noise-to-a-signal-in-python
        # 0.004 is the average variance for the noise in the three black latex glove measurements
        return column + np.random.normal(0, noise_variance, len(column))
    
    def generate_image(self, base_label=None, size=64, fill_chicken=True, img=None):
        ''''Generates fake images based in the initialized FakeDataset parameters
            
            # Where x is the generated image and y is the corresponding label
            returns x, y'''
        x = np.zeros((size, size, len(self.wavelengths)))
        if type(base_label) is not np.ndarray:
            y = np.zeros((size, size, 1))
            # Set the whole image as belt background
            # Then place a resonable large oval chicken shape
            # Then place plastic contaminants as oval items on the chichen
        else:
            def get_max_min(center):
                MIN, MAX = 0, 100
                max_val = center + size // 2
                if max_val > MAX:
                    max_val = MAX
                min_val = max_val - size
                if min_val < MIN:
                    min_val = MIN
                    max_val = min_val + size
                return min_val, max_val
            indices_x, indices_y, _ = np.nonzero(base_label == TYPE_CONTAMINANT)
            random_center_x = np.random.choice(indices_x)
            random_center_y = np.random.choice(indices_y)
            start_x, end_x = get_max_min(random_center_x)
            start_y, end_y = get_max_min(random_center_y)
            
#             s = (100 - size) // 2
            y = base_label[start_x:end_x, start_y:end_y, :].copy()
            
        squeezed_y = np.squeeze(y)
        def fill_with_type(type_numb):
            shape = x[squeezed_y == type_numb].shape
            if shape[0] != 0:       # TODO: There should always be contaminants in all images
                if type_numb == TYPE_BACKGROUND:
                    x[squeezed_y == type_numb] = self.__collect_samples(TYPE_CHICKEN, shape)
                    y[y == TYPE_BACKGROUND] = TYPE_CHICKEN
                else:
                    x[squeezed_y == type_numb] = self.__collect_samples(type_numb, shape)
        fill_with_type(TYPE_CHICKEN)
        fill_with_type(TYPE_BACKGROUND)
        fill_with_type(TYPE_CONTAMINANT)
        y -= 1 # This is done since the TYPE_BACKGROUND is changed to chicken. To have the labels asa 1 and 2 not 2 and 3
        
        # TODO: Add a image transformation to add diversity to the images generated
        # TODO: Scale the label up from 64 to 100, then apply smaller gaussian_filter and then scale back down
        #           Compare the output with the obvious plastic items
        
        # Add noise and blur the image
        x = np.apply_along_axis(self.__add_noise, -1, x)
        x = ndimage.gaussian_filter(x, sigma=(1, 1, 0), order=0) # sigma is the standard deviation for Gaussian kernel per channel
        
        return x, y
    
    def get_images(self, numb_images, size=64):
        labels_path = f"{DIR_PATH}/../../data/tomra/"
        img, labels, _ = mypackage.Dataset.load(labels_path, only_with_contaminant=True)
        
        X, Y = [], []
        numb_available_labels = len(labels)
        selected_labels = np.random.choice(numb_available_labels, numb_images, replace=True)
        for i in selected_labels:
            x, y = self.generate_image(labels[i], size=size, img=img[i])
            X.append(x)
            Y.append(y)
        # IPython.display.clear_output(wait=True)   # TODO: Add in a progressbar/info log
        return np.array(X), np.array(Y)
        