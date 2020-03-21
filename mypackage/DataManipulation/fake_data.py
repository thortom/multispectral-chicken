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
        self.db, self.plastic_types = self.__load_spectra_db()
        
        if wavelengths == "All":
            self.wavelengths = [col for col in self.db.columns if type(col) == float]
        else:
            self.wavelengths            = wavelengths
        self.max_contaminants_per_piece = max_contaminants_per_piece
        self.max_contaminant_classes    = max_contaminant_classes
        
    def __load_spectra_db(self):
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
        rename_database_group("chicken_", "Chicken")

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
    
    def __collect_samples(self, material_type, shape):
        print(shape)
        print(material_type)
        pass
                    
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
        x[y == TYPE_BACKGROUND] = self.__collect_samples(TYPE_BACKGROUND, y[y==TYPE_BACKGROUND].shape)
        
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
        