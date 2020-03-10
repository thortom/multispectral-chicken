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
        for fileName in os.listdir(folder):
            if '.csv' in fileName:

                colName = fileName[16:-4]
                tmp = pd.read_csv(f'{folder}/{fileName}', sep=";", index_col=0, names=[colName])

                # Merge the file to one bigger array with correctly labeled coloms
                if data is None:
                    data = tmp
                else:
                    data[colName] = tmp[colName]
    
    def generate_image(self):
        ''''Generates fake images based in the initialized FakeDataset parameters
            
            # Where x is the generated image and y is the corresponding label
            returns x, y'''
        return x, y
    
    def get_images(self, numb_images):
        X, Y = [], []
        for i in range(numb_images):
            x, y = self.generate_image()
            X.append(x)
            Y.append(y)
        return np.array(X), np.array(Y)
        