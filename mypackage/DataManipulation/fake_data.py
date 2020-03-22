import numpy as np
import pandas as pd
import mypackage
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DATA_BASE_PATH = "SpectralAbsorbtionDatabase.pkl"
TYPE_BACKGROUND  = 1
TYPE_CHICKEN     = 2
TYPE_CONTAMINANT = 3
    
class FakeDataset:
    
    def __init__(self, wavelengths="All", max_contaminants_per_piece=3, max_contaminant_classes=["HardPlastic", "Liner"]):
        f'''FakeDataset generates a HSI built on the database located at {DATA_BASE_PATH}
        
            max_contaminants_per_piece:  The maximum number of contaminants per piece
            max_contaminant_classes:     
        '''
        self.chicken_types = {0: "Fillet", 1: "Thigh"}
        self.database, self.plastic_types = self.__load_spectra_database()
        self.database_wavelengths = [col for col in self.database.columns if type(col) == float]
        
        if wavelengths == "All":
            self.wavelengths = self.database_wavelengths
        else:
            self.wavelengths            = wavelengths
        self.max_contaminants_per_piece = max_contaminants_per_piece
        self.max_contaminant_classes    = max_contaminant_classes
        
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
        # Here the order does matter, since some substrings used are contained in more than one groups
        rename_database_group("_fat_", "Fat")
        rename_database_group("chicken_not_as_red", "chicken_fillet_not_as_red")
        rename_database_group("chicken_fillet", "Fillet")
        rename_database_group("chicken_thigh", "Thigh")

        rename_database_group("pu_belt", "Plastic_Pu_Belt")
        rename_database_group("blue_belt_roll", "Plastic_Belt")
        rename_database_group("tube", "Plastic_Tube")
        rename_database_group("glove", "Plastic_Glove")
        rename_database_group("liner", "Plastic_Liner")
        rename_database_group("belt", "Plastic_Belt")
        types = database[database["Type"].str.contains("Plastic")]["Type"].str.replace("[0-9()]+$", "")
        plastic_types = dict(enumerate(np.unique(types)))

        # plastics = database[database["Type"].str.contains("plastic") &  (database["Type"].str.contains("liner") != True)]
        
        return database, plastic_types
    
    def __resample_wavelengths(self, spectra_data, wavelengths):
        n, m = spectra_data.shape
        resample = lambda spectrum: np.interp(wavelengths, self.database_wavelengths, spectrum)
        return np.apply_along_axis(resample, axis=1, arr=spectra_data) # Reduce spectral dimension down
    
    def __collect_samples(self, material_type, shape):
        print(shape)
        print(material_type)
        def select_spectra(types):
            item_selected = np.random.choice(list(types.keys())) # Choose randomly one item
            type_selected = types[item_selected]
            samples = self.__get_all_of_type(type_selected)
            pixels_needed, spectral_dimension = shape
            # Then randomly select from the database with resampling
            idx_selected_spectrum = np.random.choice(len(samples), pixels_needed, replace=True)
            selected_spectrum = samples.iloc[idx_selected_spectrum].values
            # TODO: Before this I need to resample the wavebands
            print(f"Desired shape {shape}")
            print(f"selected_spectrum.shape {selected_spectrum.shape}")
            print(f"len(self.wavelengths) {len(self.wavelengths)}")
            return self.__resample_wavelengths(selected_spectrum, self.wavelengths)
            
        if   material_type == TYPE_BACKGROUND:
            return np.zeros(shape)
        elif material_type == TYPE_CHICKEN:
            return select_spectra(self.chicken_types)
        elif material_type == TYPE_CONTAMINANT:
            return select_spectra(self.plastic_types)
                    
    def transform_to_reflectance(self, data):
        pass
    
    def generate_image(self, base_label=None):
        ''''Generates fake images based in the initialized FakeDataset parameters
            
            # Where x is the generated image and y is the corresponding label
            returns x, y'''
        x = np.zeros((100, 100, len(self.wavelengths)))
        if type(base_label) is not np.ndarray:
            y = np.zeros((100, 100, 1))
            # Set the whole image as belt backgorund
            # Then place resonable large oval chicken shape
            # Then place plastic contaminants as oval items on the chichen
        else:
            y = base_label
            
        squeezed_y = np.squeeze(y)
        def fill_with_type(type_numb):
            x[squeezed_y == type_numb]  = self.__collect_samples(type_numb,  x[squeezed_y==type_numb].shape)
        fill_with_type(TYPE_BACKGROUND)
        fill_with_type(TYPE_CHICKEN)
        fill_with_type(TYPE_CONTAMINANT)
#         x[y == TYPE_BACKGROUND]  = self.__collect_samples(TYPE_BACKGROUND,  x[y==TYPE_BACKGROUND].shape)
#         x[y == TYPE_CHICKEN]     = self.__collect_samples(TYPE_CHICKEN,     x[y==TYPE_CHICKEN].shape)
#         x[y == TYPE_CONTAMINANT] = self.__collect_samples(TYPE_CONTAMINANT, x[y==TYPE_CONTAMINANT].shape)
        
        return x, y
    
    def get_images(self, numb_images):
        labels_path = "../data/tomra/"
        _, labels, _ = mypackage.Dataset.load(labels_path, only_with_contaminant=True)
        
        X, Y = [], []
        numb_available_labels = len(labels)
        selected_labels = np.random.choice(numb_available_labels, numb_images, replace=False)
        for i in selected_labels:
            x, y = self.generate_image(labels[i])
            X.append(x)
            Y.append(y)
        return np.array(X), np.array(Y)
        