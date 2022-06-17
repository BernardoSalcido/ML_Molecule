# -*- coding: utf-8 -*-
"""
Created on Fri May 13 13:27:48 2022

@author: Alumnos
"""
from ML_Molecule import *

from Simulate_Spectra import *


class Predictor_ML_Molecule:
    
    
    def __init__(self,DataSet):
        
        self.solvent = False
        self.DataSet = DataSet
        ### Import Models 
        print('#-------------------Importing the models and data sets-------------------#')
        if DataSet == 'Chromophores_20000':
            self.Chromophore_Data_abs = ML_Data(Data_df_name = 'DataBase_20000/DataFrame/DataFrame_Absorption max (nm)_row_17276_col_3651.csv')    
            self.sc_abs = Open('DataBase_20000/SC/StandarScalarAbsorption max (nm)_row17276_col3651.sc')
            self.model_abs = LoadModel('DataBase_20000/Models/_RandomForestRegressor()_Absorption max (nm)_DimRed_FalseModel')
            
            self.Chromophore_Data_em = ML_Data(Data_df_name = 'DataBase_20000/DataFrame/DataFrame_emi FWHM (nm)_row_7198_col_3651.csv')    
            self.sc_em = Open('DataBase_20000/SC/StandarScalarEmission max (nm)_row18140_col3651.sc')
            self.model_em = LoadModel('DataBase_20000/Models/_RandomForestRegressor()_Emission max (nm)_DimRed_FalseModel')
        
        elif DataSet == 'Chromophore_500':
            self.Chromophore_Data_abs = ML_Data(Data_df_name = 'DataBase_500/DataFrame/DataFrame_Energy S1 (Abs) [eV]_row_1258_col_3013.csv')    
            self.sc_abs = Open('DataBase_500/SC/StandarScalarEnergy S1 (Abs) [eV]_row1258_col3013.sc')
            self.model_abs = LoadModel('DataBase_500/Models/_RandomForestRegressor()_Energy S1 (Abs) [eV]_DimRed_500Model')
            
            self.Chromophore_Data_em = ML_Data(Data_df_name = 'DataBase_500/DataFrame/DataFrame_Energy S1 (Em) [eV]_row_1213_col_3013.csv')    
            self.sc_em = Open('DataBase_500/SC/StandarScalarEnergy S1 (Em) [eV]_row1213_col3013.sc')
            self.model_em = LoadModel('DataBase_500/Models/_RandomForestRegressor()_Energy S1 (Em) [eV]_DimRed_500Model')
    
    def Calculate(self,mol,solvent,nm = False):
        
        if isinstance(mol, list):
            
            X_abs = list()
            X_em  = list()
            ind = 0
            self.mol = list()
            self.solvent = solvent
            self.islistmol = True
            for mol_i in mol:
                # instnaciate the class
                self.Mol_i = MolData_(mol_i,solvent)
                
                # Optimiza the 3D structure
                print('#-------------------Optimizing strucure-------------------#')
                self.Mol_i.Structure_generator()
                self.Mol_i.Structure_generator_solv()
                
                self.Model_abs = self.Mol_i
                self.Model_em = self.Mol_i
                
            
                ## Calulate the descripotrs
                ## Abs
                print('#-------------------Calculating descripotrs-------------------#')
                self.Model_abs.Descriptor_calculator(self.Chromophore_Data_abs.Descriptors_Name_list)
                self. Model_abs.Descriptor_calculator_solv(self.Chromophore_Data_abs.Descriptors_Name_list)
                ## Em
                self.Model_em.Descriptor_calculator(self.Chromophore_Data_em.Descriptors_Name_list)
                self. Model_em.Descriptor_calculator_solv(self.Chromophore_Data_em.Descriptors_Name_list)
                
                ## Preprosesing
                print('#-------------------Preprosesing-------------------#')
                self.Model_abs.Preprosesing(Standar_Scalar= self.sc_abs ,Imputer='Simple_Imputer')
                self.Model_em.Preprosesing(Standar_Scalar= self.sc_em ,Imputer='Simple_Imputer')
                
                if ind ==0:
                    X_abs = self.Model_abs.X
                    X_em = self.Model_em.X
                    
                else:    
                    X_abs = np.vstack([X_abs, self.Model_abs.X])
                    X_em = np.vstack([X_em, self.Model_em.X]) 
                
                self.mol.append(self.Mol_i)
                
                ind += 1 
                
            print('#-------------------Predict-------------------#')
            
            X_abs = np.array(X_abs)
            X_em = np.array(X_em)
            
            self.model_abs.predict(X_abs,nm = nm)
            self.model_em.predict(X_em,nm = nm)
            
        else:
            self.islistmol = False
            # instnaciate the class
            self.Mol_i = MolData_(mol,solvent)
            # Optimiza the 3D structure
            print('#-------------------Optimizing strucure-------------------#')
            self.Mol_i.Structure_generator()
            self.Mol_i.Structure_generator_solv()
            
            self.Model_abs = self.Mol_i
            self.Model_em = self.Mol_i
            self.mol = list()
            self.mol.append(self.Mol_i)
            ## Calulate the descripotrs
            ## Abs
            print('#-------------------Calculating descripotrs-------------------#')
            self.Model_abs.Descriptor_calculator(self.Chromophore_Data_abs.Descriptors_Name_list)
            self. Model_abs.Descriptor_calculator_solv(self.Chromophore_Data_abs.Descriptors_Name_list)
            ## Em
            self.Model_em.Descriptor_calculator(self.Chromophore_Data_em.Descriptors_Name_list)
            self. Model_em.Descriptor_calculator_solv(self.Chromophore_Data_em.Descriptors_Name_list)
            
            ## Preprosesing
            print('#-------------------Preprosesing-------------------#')
            self.Model_abs.Preprosesing(Standar_Scalar= self.sc_abs ,Imputer='Simple_Imputer')
            self.Model_em.Preprosesing(Standar_Scalar= self.sc_em ,Imputer='Simple_Imputer')
            
            print('#-------------------Predict-------------------#')
            self.model_abs.predict(self.Model_abs.X,nm = nm)
            self.model_em.predict(self.Model_em.X,nm = nm)
            
        
    # Predict using the 
    def PredictSpectra(self,FWHM_abs,FWHM_em,w_min = False, w_max=False):  
        
        self.abs_list = self.model_abs.Predict_list
        self.em_list = self.model_em.Predict_list
        print(self.abs_list)
        print(self.em_list)
        #### 
        for i in range(len(self.abs_list)):
            
            mol_i = self.mol[i]
            
            abs_i = self.abs_list[i]
            em_i = self.em_list[i]    
                
            SimulateSpectra(abs_i,em_i,FWHM_abs,FWHM_em,w_min,w_max,mol_i,solvent = self.solvent)













