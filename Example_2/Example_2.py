#------------- Required Library --------------#
# importing sys
import sys
# adding Libraries_ to the system path (specify the path where the "Libraries_" folder is located)
sys.path.insert(0, 'C:/Users/berna/Google Drive/ML_Molecule/Libraries_') # Speficy your path  
# ML_classes  contain all the classes and functions required regarding to the Machine Learning methodology
from ML_Molecule import *

#------------------- Example 2 -----------------------#
#%%----------- Step 1: 
# Open the molecular property files (these should be in a folder Data)
# This file has to follow the same structure as the example file “Spectroscopy_Data_Coumarins_Example_2.csv”  in order to work properly.

spect_data_name ='DataBase_Example_2/Spectroscopy_Data_Coumarins_Example_2.csv'

chromphore_data_name = 'DataBase_Example_2/Coumarins_Molecules_Example_2.csv'             #File with the name and the smile strucure of each Solvent used in spectrocopiy data
solv_data_name = 'DataBase_Example_2/Coumarins_Solvents_Example_2.csv'                    #File with the name and the smile strucure of each molecule used in spectrocopiy data

Coumarine_MLData = ML_Data(spect_data_name = spect_data_name,solv_data_name = solv_data_name,chromophore_data_name =chromphore_data_name )  

# In this case we already have de structural data for the solvents you can use
# In addition the progrma crated a file with the smile structure of the chromophores in the same folder Data finishing as _Molecules
# this filed with the Solvent file going to be used to generate the optimize molecular strucure 
# WARNING: be sure that there are any solvents named different in this file and in the principal data file
#%%---------- Step 4:
# In this step we want to append the descripotrs, for this case we just have some strucural and quantum descripors:
# For the cromophore descripotrs we have Strucural descripotrs calulated with the sofware CODESSA,
# quantum descripotrs obtained using GAUSSIAN and CODESSA, and some QTAIM descripotrs
# For the solvent we going to use QTAIM descripotrs and a data set of empirial descripotr.  
# WARNING: be shure that each descripotr file that you append have the same size that all of them an the index of eaqch cromophore is correct in order to avoid errors

# list of chromophore descripotrs files that you whant to append
molecule_descriptor_list = ['Descriptors/Descriptor_Coumarine_Strucuture.csv','Descriptors/Descriptor_Coumarine_wavefunction.csv','Descriptors/Descriptors_Coumarine_QTAIM_Groups.csv'] 
Coumarine_MLData.Open_descriptor_mol_List(molecule_descriptor_list)

# list of solvent descripotrs files that you whant to append
Solvent_descriptor_list = ['Descriptors/Descriptors_Cumarine_Solvent_Emp.csv','Descriptors/Descriptors_Coumarinas_Solvent_QTAIM_Groups.csv']
Coumarine_MLData.Open_descriptor_solv_List(Solvent_descriptor_list)
#%%---------- Step 5:
# In this step we will filter the corresponding propitie that we want to study, and also there are some meny filtering options
## This proses is somehow computational time consumeing and also it consume maney RAM memory, that is specialy a problen in big datasets 

Property_list = ['Energy S1 (Abs) [eV]'] # Select the name of the propieite as is in the principal data 
## For this example we have these option
## 'Wavelength S1 (Abs) [nm]'   'Energy S1 (Abs) [eV]'
## 'Wavelength S1 (Em) [nm]'    'Energy S1 (Em) [eV]' 
## 'Delta_Stocks [eV]'

Limit_list = [[0,10]] # This contain the max and minum limis for a specific propitie, the unities are in the same scale than the propietie
 
Solvent_list =['all'] # also you can only slect some solvent, and the metod will filter only that you ad in the list, this list have to be with the smile strucure of a solvent
 
Type_molecule_list =['all'] # Also you can filter only a specific kind of molecules, to select all kind of molecules is with the key 'all'

Coumarine_MLData.Filter_Property(Property_list,Limit_list,Solvent_list,Type_molecule_list,nick_search_solv = 'Name',nick_search_chromophore ='Smile') ## Method that make de filtering

## 'nick_search_solv' is used to specify if the metod goig to serch the solvent descripotrs usin the solvent name,
##  in this case that in the data set the solven is defined usin the name we going to use nick_search_solvent = 'Name'                                            

df_Data = Coumarine_MLData.Data # this variable is to visalisate the corresponding data that contain all 
                                # the information redy to start the ML metodology
Coumarine_MLData.Save_df() ##
#%%------------- Step 6: (Optional) 
# If you alredy do the previous steps and you have the File 'Spectroscopy_Data_Example_1_df_old.csv' in your Data/Example_1/ folder
# you can also creat the ML_data starting with this filed and avoid repeat the first 5 steps
# I strongly recomend change the name of this file to avoid overwriting when you will creat diferents data using diferetns descriptors or also weh you want to stody diferetns propieties

Data_df_name ='Data/Spectroscopy_Data_Coumarins_Example_2_df.csv'
Coumarine_MLData = ML_Data(Data_df_name = Data_df_name)

#%% ---------- Step 7:
# This are the first step rewardin with ML metodology, all previos step are pre-ML 
# This is the preprosesign of the data, you can mak or not a standarization process (strongly recomeneded) and imputer data using diferents metod 
# this only incluid Simple_Imputer and Nearest_neighbors but is easy to add more metods if is necesary

Coumarine_MLData.Preprosesing(Scaling=True,Imputer='Nearest_neighbors')
# In additon this metod plots the histogram from the training set and test set (this are only ploted in the console)
#%% ----------- Step7: (Optional )
# This method make some useful plots in order to understand the data distibution from you data set
Coumarine_MLData.Data_Distribution()
#%% --------- Step 7:
# First we going to creat a model with the aditional parameteres requirede 
# Data is the class the we creat previously which have all the information needed to train the model
# Model is the model from sklern, see https://scikit-learn.org/stable/supervised_learning.html to know all the modeles that you can use
# Num_red is the dimentional reduction, if you doent need to use that set as False or do no incluid the variable 
# cv is the iteration from the Cross validation prosses
model = Model_(Data = Coumarine_MLData, Model = GradientBoostingRegressor(), Num_red = 300, cv = 5)
# We can validate our model this going to print in the console the results of the validation but also creat a atributes that 
# you could be asses from then after the validation prosses such as: print(model.cv_r2_mean_test) or  print(model.cv_mean_absolute_error_mean_test)
model.validation()
# This is the final step in our ML journey with this metod ge going to creat a plot tha show us how well our model do with the data set.
# the Parameters r_round, mae_round and mse_round correspound to the nomber of desimals that you gant that the table have for each metric     
model.Plot_Results(r_round = 3, mae_round = 2,mse_round = 2)
# These plots were saved in the Resultados folder in .png file and also in .svg file to extra editinf if is nesesary
#%% ----------Step 8:
# ----------- Feature importance 
# With this metod we going to plot the importance of each feature from the model
# in some cases we only want to know the importance form the most importan features in this case we could set de number of plots 
# each plot have maximus 20, for example if we eant to plot only the 60th best features we ned to set the num_plot as 3
model.Feature_importances(num_plots= 3)
# The metod make 2 plots one with error bars and another without
# As the other metods it also creat a .png file ans a .svg file     

#%% --------- Extra: 
#-------------- Using Model Cells
# This is an extra functionality that is somwaht powerfull, that is using or treingn more than one model at the same time  
# The first Parameter 'Model_names' contain the name of the models that you want to add this vertion only incluid: 
# 'LinearRegression'.'SVR','DecisionTreeRegressor','BayesianRidge','SGDRegressor','GradientBoostingRegressor','GaussianProcessRegressor'
# 'KernelRidge','MLPRegressor','RandomForestRegressor', 'LogisticRegression'
Model_names =['RandomForestRegressor','GradientBoostingRegressor']

# 'Dic_parameters_list' this parameter is a list of dictionare of parameters for each kind of model ther have to be a dictioanry per kind of model
# if you add more kind of model be sgure to add another dictionary of parameters and append to the list of parametres 
# the dictionary have to follow the next strucure: name of the parameter with '' and then a list of the diferent values for the parameter                 
# you have to refer to the sklearn documentation of each model to know the available parametrs tha you can use in the diferetn kind of models
Dic_parameters_list = list()           

Dic_parameters = {}
Dic_parameters_list.append(Dic_parameters)

Dic_parameters = {}
Dic_parameters_list.append(Dic_parameters)

#------------- Dimension Reduction 
# the Parameter Num_Red_list refer to the diferetns dimentional reduction values 
Num_Red_list=[200,250,300,350]
# The model going to train all the feasibles convination of all this parameters in this case we goint o train 
# 2 (number of parametres Random fores) * 2 (number of parametres Gradient boosting) * 5 (number of dinamntional reductions elements)  = 20 diferents model
# as you see if you use too man models an parametres it goin to take too much time
# I recomend to set the cors validation at the minimum 3 and the wehn you find a good model you can return to the previos seccion an train the model with more cv accurate   
Coumarine_model_Cell= ModelCell(Coumarine_MLData,Model_names,Dic_parameters_list,Num_Red_list,3)

# This metod creat a cross validation frame with all the models trained and some of them results 
Coumarine_model_Cell.CrossValidation()
df_validationCell = Coumarine_model_Cell.Validation.df_validation
# This metod creat a Plots that show the distribution of the scores from the number of reduction 
# the width of the fill refer to the number of models that get this score from this redcution number specificaly
# the max and the minimum of each violin also are importan because refert o the best or the worst model per each dimension reduction value(depend if tis the r2 score or mse)     
Coumarine_model_Cell.PlotValidation()
# These plots were saved in the Graficas_Validacion folder in .png file and also in .svg file to extra editinf if is nesesary 
Coumarine_model_Cell.PlotResult(r_round = 3, mae_round = 2,mse_round = 2)
# Agaian you can plot all the Feature importance plot to each model in the Model cell with the metod 
Coumarine_model_Cell.PlotFeatureImportance(num_plots= 3)


