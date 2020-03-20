import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class FakeDataset:
    DataBasePath = ""
    
    def __init__(self, wavelengths, max_contaminants_per_piece=3, max_contaminant_classes=["HardPlastic", "Liner"]):
        f'''FakeDataset generates a HSI built on the database located at {DataBasePath}
        
            max_contaminants_per_piece:  The maximum number of contaminants per piece
            max_contaminant_classes:     
        '''
        self.wavelengths                = wavelengths
        self.max_contaminants_per_piece = max_contaminants_per_piece
        self.max_contaminant_classes    = max_contaminant_classes
        
        self.db = self.__load_spectra_db()
        
    def __load_spectra_db(self):
        data = None

        # Read all file in folder
        folder = 'SpectralReflectanceData'
        # Try this to query for substring - https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.query.html
        database_name = "SpectralAbsorbtionDatabase"
        database = pd.read_pickle(database_name)
        # TODO: Group together the material types
        
        return database
                    
    def transform_to_reflectance(self, data):
        pass
    
    def generate_image(self):
        ''''Generates fake images based in the initialized FakeDataset parameters
            
            # Where x is the generated image and y is the corresponding label
            returns x, y'''
        image = np.zeros((100, 100, 1))
        return image, label
    
    def get_images(self, numb_images):
        X, Y = [], []
        for i in range(numb_images):
            x, y = self.generate_image()
            X.append(x)
            Y.append(y)
        return np.array(X), np.array(Y)
        