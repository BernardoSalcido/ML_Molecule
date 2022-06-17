#------------- Required Library --------------#
# importing sys
import sys
# adding Libraries_ to the system path (specify the path where the "Libraries_" folder is located)
sys.path.insert(0, 'C:/Users/berna/Google Drive/ML_Molecule/Libraries_')
# ML_classes  contain all the classes and functions required regarding to the Machine Learning methodology
from ML_Molecule import *
from Simulate_Spectra import *

#%%--------------------------------- Creating the [X] [Y] dataframe
#------------- Step 1 ():
# instanciate the ML_Data class using the spectrocopy information      
spect_data_name ='DataBase_/Spectroscopy_Data_Example_1.csv'

solv_data_name = 'DataBase_/Solvent_Example_1.csv'             #File with the name and the smile strucure of each Solvent used in spectrocopiy data
chromphore_data_name = 'DataBase_/Chromophores_Example_1.csv'  #File with the name and the smile strucure of each molecule used in spectrocopiy data

Chromophore_Data = ML_Data(spect_data_name= spect_data_name,solv_data_name = solv_data_name,chromophore_data_name =chromphore_data_name )  
#---------- Step 2:
# Append the descripotors files previously calculated (Example 1) files 
molecule_descriptor_list = ['Descripotors/Structural_Descriptor_Num_Desc_1565.csv'] 
Chromophore_Data.Open_descriptor_mol_List(molecule_descriptor_list)

# list of solvent descripotrs files that you whant to append
Solvent_descriptor_list = ['Descripotors/Structural_Descriptor_Solv_Num_Desc_1443.csv']
Chromophore_Data.Open_descriptor_solv_List(Solvent_descriptor_list)
#---------- Step 3:
Property_list = ['Energy S1 (Em) [eV]'] # Select the name of the propieite as is in the principal data 

### For this dataset we have these option
## 'Wavelength S1 (Abs) [nm]'   'Energy S1 (Abs) [eV]'
## 'Wavelength S1 (Em) [nm]'    'Energy S1 (Em) [eV]' 
## 'Delta_Stocks [eV]'
## 'Tau_Averge [ns]'            '(1/Tau_Averge)' [1/ns]

Limit_list = [[0,4]] # This contain the max and minum limis for a specific propitie, the unities are in the same scale than the propietie
Solvent_list =['all'] # also you can only slect some solvent, and the metod will filter only that you ad in the list, this list have to be with the smile strucure of a solvent
Type_molecule_list =['all'] # Also you can filter only a specific kind of molecules, to select all kind of molecules is with the key 'all'

Chromophore_Data.Filter_Property(Property_list,Limit_list,Solvent_list,Type_molecule_list,nick_search_solv = 'Name') ## Method that make de filtering
## 'nick_search_solv' is used to specify if the metod goig to serch the solvent descripotrs usin the solvent name,
##  in this case that in the data set the solven is defined usin the name we going to use nick_search_solvent = 'Name'                                            
df_Data = Chromophore_Data.Data # this variable is to visalisate the corresponding data that contain all 
                                # the information redy to start the ML metodology
Chromophore_Data.Save_df() ##
#################################################################################################################
#%% ---------------------------------- Training models
#----------Absortion 
#----------- Step 1:
# If we arredy do the previous steps we can used the file saved in DataFrames_Data to the begining and avoid the previous 7 Steps:
Chromophore_Data = ML_Data(Data_df_name = 'DataFrames_ML_Data/DataFrame_Energy S1 (Abs) [eV]_row_1258_col_3013.csv')
Chromophore_Data.Preprosesing(Scaling=True,Imputer='Simple_Imputer')
# --------- Step 2:
# Create the model
model = Model_(Data = Chromophore_Data, Model = RandomForestRegressor(), Num_red = 500, cv = 3)
# ---------- Step 3 (Optional)
# Train and plot the validation results of that model
model.Plot_Results(r_round = 3, mae_round = 2,mse_round = 2) #(Plot the performance of the model)
# ---------- Step 4:
# Fit the model with all the data 
model.fit()
# Save the model in a file named Models Files, it creat a folde for each model with the model vaiables required
model.Save()
#----------Emission 
#----------- Step 1:
# If we arredy do the previous steps we can used the file saved in DataFrames_Data to the begining and avoid the previous 7 Steps:
Chromophore_Data_Em = ML_Data(Data_df_name = 'DataFrames_ML_Data/DataFrame_Energy S1 (Em) [eV]_row_1213_col_3013.csv')
Chromophore_Data_Em.Preprosesing(Scaling=True,Imputer='Simple_Imputer')
# --------- Step 2:
# Create the model
model = Model_(Data = Chromophore_Data_Em, Model = RandomForestRegressor(), Num_red = 500, cv = 3)
# ---------- Step 3 (Optional)
# Train and plot the validation results of that model
model.Plot_Results(r_round = 3, mae_round = 2,mse_round = 2) #(Plot the performance of the model)
# ---------- Step 4:
# Fit the model with all the data 
model.fit()
# Save the model in a file named Models Files, it creat a folde for each model with the model vaiables required
model.Save()
#################################################################################################################
#%% ------------------------------ Predict for a new molecule
#---------- Step 1:    
# Load Data frames, the standar sclar and the models to use for predict  
sc_abs = Open('StandarScalar/StandarScalarEnergy S1 (Abs) [eV]_row1258_col3013.sc')
sc_em = Open('StandarScalar/StandarScalarEnergy S1 (Em) [eV]_row1213_col3013.sc')

model_Abs = LoadModel('ModelsFiles/_RandomForestRegressor()_Energy S1 (Abs) [eV]_DimRed_500Model')
model_Em = LoadModel('ModelsFiles/_RandomForestRegressor()_Energy S1 (Em) [eV]_DimRed_500Model')

Data_df_name ='DataFrames_ML_Data/DataFrame_Energy S1 (Abs) [eV]_row_1258_col_3013.csv'
Chromophore_Data_Abs = ML_Data(Data_df_name = Data_df_name)

Data_df_name ='DataFrames_ML_Data/DataFrame_Energy S1 (Em) [eV]_row_1213_col_3013.csv'
Chromophore_Data_Em = ML_Data(Data_df_name = Data_df_name)
#%%
# ---------- Step 2:
# Set the molecule strurue and the solvent strucure    
# mol ='CC1(C)\C(NC2=C1C1=CC=CC=C1C=C2)=C\C=C/C1=[N+](CCCCCC(=O)ON2C(=O)CCC2=O)C2=C(C3=C(C=CC=C3)C=C2)C1(C)C'
mol = 'CCN(CC)C1=CC2=C(C=C1)C(=C3C=CC(=[N+](CC)CC)C=C3O2)C4=CC=CC=C4C(=O)O' # Rodamine B  
#solvent = 'CN(C)C(C)=O' #DMF
solvent = 'CCO' # ethanol 

# instnaciate the class with the chromophore and the solvent 
Mol_i = MolData_(mol,solvent)

# Optimiza the 3D structure 
Mol_i.Structure_generator()
Mol_i.Structure_generator_solv()

## insatciate one class from absortion and one for emition
Mol_Abs = Mol_i
Mol_Em = Mol_i 

## 
descripot_list_Abs = Chromophore_Data_Abs.Descriptors_Name_list
descripot_list_Em = Chromophore_Data_Em.Descriptors_Name_list

## Calulate the descripotrs using a list of names descripotrs
Mol_Abs.Descriptor_calculator(descripot_list_Abs) # 
Mol_Abs.Descriptor_calculator_solv(descripot_list_Abs)

Mol_Em.Descriptor_calculator(descripot_list_Em)
Mol_Em.Descriptor_calculator_solv(descripot_list_Em)

## Preprosesing 
Mol_Abs.Preprosesing(Standar_Scalar= sc_abs ,Imputer='Simple_Imputer')
Mol_Em.Preprosesing(Standar_Scalar= sc_em ,Imputer='Simple_Imputer')

## Predict  
Abs = model_Abs.predict(Mol_Abs.X,nm = True)
Em = model_Em.predict(Mol_Em.X,nm = True)
#%%
# ---------- Step 3: Simulate spectra 
FWHM_Abs = 40 #
FWHM_EM = 50 # 
Min = 450 #
Max = 800 #

SimulateSpectra(Abs,Em,FWHM_Abs,FWHM_EM,Min,Max)

#%%
mol = 'CC1(C)\C(NC2=C1C1=CC=CC=C1C=C2)=C\C=C/C1=[N+](CCCCCC(=O)ON2C(=O)CCC2=O)C2=C(C3=C(C=CC=C3)C=C2)C1(C)C'
solvent = 'CCO' # ethanol 
Mol_i = MolData_(mol,solvent)
Mol_i.Structure_generator()
Mol_i.Structure_generator_solv()

SimulateSpectra(511,544,45,50,450,800,Mol_i.mol2D)
