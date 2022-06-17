# -*- coding: utf-8 -*-

"""
Created on Thu Aug 12 12:40:50 2021

@author: Bernardo Arturo Salcido Santacruz 
"""
######################################################## Librerias ######################################################

#---------- Rdkit
from rdkit.Chem import AllChem as Chem
from rdkit.Chem.Descriptors import ExactMolWt
#---------- Mordred
from mordred import Calculator, descriptors
#----------  plots
import matplotlib as mpl  
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
# ---------- Maths
import math
import numpy as np
import pandas as pd
from collections import Counter
#----------- ML
from sklearn.model_selection import ParameterGrid
import sklearn.utils
# --------- Preprosesign 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer

#---------- Regression models
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression 
from sklearn.svm import SVR  
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge 
from sklearn.ensemble import VotingRegressor 

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LogisticRegression
#------------ Dimension reduction
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

# -------- Scores 
import sklearn.metrics
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate 


#-------- Extras 
import copy
from time import time
import numpy as np
import pickle
import dill
import os 

#--------------------------------------------- Manage file functions ----------------------------------- ### 
def get_filepaths(directory):
    """
    This function will generate the file names in a directory 
    tree by walking the tree either top-down or bottom-up. For each 
    directory in the tree rooted at directory top (including top itself), 
    it yields a 3-tuple (dirpath, dirnames, filenames).
    """
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.

    return file_paths  # Self-explanatory.


def CheckDir(dirName):
    
    if not os.path.exists(dirName):
        os.makedirs(dirName)
        print("Directory " , dirName ,  " Created ")
    else:    
        print("Directory " , dirName ,  " already exists")  

def Open(file):
    f = open(file,'rb')
    instanciate_class = dill.load(f)
    f.close()        
    return instanciate_class
#--------------------------------------------------------------------------------------------------------------------##


#------------------------------------------------------- Data Class -------------------------------------------------------#

#----------------------------- Required functions for this class ------------------------------# 

def Data_frame_descriptor_generator(Descriptors,Data_mol, is_solvent=False, NaValue = 0):
    
    # Description: Function that creat the data frame with all descriptor of the molecules
    # Imputs
    #   % Descriptors: tuple of descriptors to calculate using the Mordred library
    #   % Data_mol:    file which cointein the strucural information of molecule
    #   % is_solvent: if is True meaning that the Data_mol is a data base of solvents 
    #   % NaValue: the defalt value if the descripotr could not be calcualted         
    # Outputs 
    #   % df_descriptors: a dataframe of pandas which contain the values of descriptors per each molecule
    #   % index_descriptrs: list that contain if a descripotrs is apenede or not
    #------------------------------------- Beginning of the function ---------------------------
    if is_solvent == True:
        name_is_solvent = 'Solv_' # Add the label Sovent at the name of descriptor if is a solvent
    else: 
        name_is_solvent = '' # Add nothing if is not a solvent
        
    df_descriptors = pd.DataFrame() # It creates the data frame in which going to contain the descriptor values      
    
    #------------ Calculation of the descriptors
    INDX_Desc = 0 # Descriptor index  
    
    for Descriptor in Descriptors: # Iterate over the descriptors 
        Name_Descriptor = name_is_solvent + str(Descriptor)
        Descriptor_list =[]
        INDX_Desc = INDX_Desc + 1
        INDX_Mol = 0
        for mol in Data_mol: # Iterate over the molecules
            INDX_Mol = INDX_Mol + 1
            print('Molecula: /t'+ str(mol))
            try:
                Descriptor_Value = Descriptor(mol) # Function that calculates the descriptor using mol(Rdkit molecule) and Descriptor(Mordred)

            except:
                Descriptor_Value = NaValue # If it can not calculate the default value is defined by the NaValue input

            Descriptor_list.append(Descriptor_Value) # add de descriptor value 
            print('Index_Decriptor: \t ',INDX_Desc, '\t',Name_Descriptor)
            print('Index_Mol: \t ',INDX_Mol)
    
        df_descriptors[Name_Descriptor] = Descriptor_list # It adds a descriptro for all molecules in de df 

    return df_descriptors

    #------------------------------------- End of the function ---------------------------

def Save_data_descriptors(df_descriptors,index_descriptrs,descripotrs_incluided, is_solvent = False):
    # Description: Function that save the df_descriptor in .csv format  
    # Imputs 
    #   % df_descriptors: a dataframe of pandas which contain the values of descriptors per each molecule (obtained from Data_frame_descriptor_generator function)
    #   % is_solvent: if is True meaning that the df_descriptros is a data base of solvents
    
    #------------------------------------- Beginning of the function ---------------------------
    if is_solvent == True:
        name_is_solvent = 'Solv_' # Add the label Sovent at the name of the file if is a solvent
    else: 
        name_is_solvent = '' # Add nothing if is not a solvent

    Num_Descriptors = str(len(list(df_descriptors.columns)))
    
    dirfile = 'Descriptors/'
    
    CheckDir(dirfile) # 
    
    Name = dirfile +'Structural_Descriptor_'+name_is_solvent +'Num_Desc_'+ Num_Descriptors +'.csv'
    Name_indx = dirfile +'Structural_Descriptor_'+name_is_solvent +'Num_Desc_'+ Num_Descriptors +'_Meta'+'.csv' 
    
        
    meta_descripotr_df = pd.DataFrame()

    
    meta_descripotr_df['index_descriptrs'] = index_descriptrs
    
    
    df_descriptors.to_csv(Name, index=False) # Save the datafram 
    
    meta_descripotr_df.to_csv(Name_indx, index=False) # Save the datafram 
    #------------------------------------- End of the function ---------------------------
        
def List_Descripotr_genarate(Interval_list = list()):
    # Description: Function that generate the tuple of descriptor (Mordred)
    # Inputs
    #   % Interval_list = is the interval of descriptros to add into the tuple
    #   % it have to be a lis of list in wich each elelemn of the second anidate lis contai9n the minimum and maximus interval
    #   % Example Interval_list  = [[1,10],[20,50],...]
    # output 
    # % Descriptors: tuple of descriptors to calculate using the Mordred library
    #------------------------------------- Beginning of the function ---------------------------  
    Descriptors_Cell = list()
    for Intervals in Interval_list:
        Descriptors_Cell.append(Calculator(descriptors).descriptors[Intervals[0]:Intervals[1]]) ## all descriptors 
    Descriptors = tuple() # tuple of descriptors 
    for Descriptors_list in Descriptors_Cell:
        Descriptors = Descriptors +  Descriptors_list
    return Descriptors
    #------------------------------------- End of the function ---------------------------
    
def List_Descriptor_all():    
    # Description: Function that generate the tuple of descriptor (Mordred)
    # output 
    #   % Descriptors: tuple of descriptors to calculate using the Mordred library
    
    #------------------------------------- Beginning of the function ---------------------------
    Descriptors_Cell = list()
    Descriptors_Cell.append(Calculator(descriptors).descriptors[0:1825]) ## all descriptors 
    Descriptors = tuple() # tuple of descriptors 
    for Descriptors_list in Descriptors_Cell:
        Descriptors = Descriptors +  Descriptors_list
    return Descriptors
    #------------------------------------- End of the function ---------------------------

def Hist(Y_test,Y_train):
    # Description: Function that creat a histogram of the test data and train data. 
    # input
    #   % Y_test: test data    
    #   % Y_train: train data 

    #------------------------------------- Beginning of the function ---------------------------
    pd_Y_test = pd.DataFrame(Y_test)
    pd_Y_train = pd.DataFrame(Y_train)
    Keys = pd_Y_test.keys()
    Propietie = list(Keys)
    
    #------Train Histogram   
    plt.figure(figsize=(8, 6), dpi=500) # Crate the figure 
    
    pd_Y_train.hist(bins=20,rwidth=0.9,grid=False) # Creat the histogram
    
    plt.title('Training set')
    plt.xlabel(Propietie[0]) 
    plt.ylabel('Count')
    plt.show() 
    
    #------Test Histogram 
    plt.figure(figsize=(8, 6), dpi=500) # Crate the figure 
    
    pd_Y_test.hist(bins=20,rwidth=0.9,grid=False) # Creat the histogram
    
    plt.title('Test set')
    plt.xlabel(Propietie[0]) 
    plt.ylabel('Count')
    plt.show()
    #------------------------------------- End of the function ---------------------------

def Get_Mol_Smile(df,nick_mol='Solvent'):
       # Description: Function that creat a df with the structure (smile) of all the molecules (not repeated) starting from the original data set
        # Inputs
        #   % df: The original data frame, it has to contain a column that cointain the structure of the solvents
        #   % nick_mol: Is the name of the column that contain the structural information of the molecules
        # Output 
        #   % 
        
        #------------------------------------- Beginning of the function --------------------------
        # Obtain the Smile with out duplicates
        df_mol_smile = df[nick_mol]
        smile = list(df_mol_smile.drop_duplicates())
        
        # Creat a name index to each solvent 
        name_mol = list()
        for i in range(len(smile)): # 
            name_i = str(nick_mol)+'_'+str(i)
            name_mol.append(name_i)
        
        # Creat and Join the elements to the df
        df_Solv = pd.DataFrame()
        df_Solv['Name'] = name_mol 
        df_Solv['Smile'] = smile 
        
        return df_Solv
     #------------------------------------- End of the function ---------------------------

def is_trash(df):
    # Description: Function that analyzes a column (features) from a data frame and check if is relevant, 
    # if all the elements are the same means that this column is not necessary therefore the function returns true   
        # Inputs
        #   % df: a colum from a dataframe 
        # Output 
        #   % state: true if the column is not relevant and false if is relevant 
    #------------------------------------- Beginning of the function --------------------------
    elements = list(df)
    state = True # set the inicial state 
    element_0 = elements[0] # save the first element 
    nan = np.isnan(element_0)
    
    for i in range(len(elements)):
        element_i = elements[i]
        
        if nan == True: # in the case that the value is nan 
            if np.isnan(element_i) == False: # If one of the element is different from the first therefore the column is relevant and change the value of the state
                state = False
                break
            
        elif (element_i is not element_0): # If one of the element is different from the first therefore the column is relevant and change the value of the state
                state = False
                break    
    return state
    #------------------------------------- End of the function ---------------------------

def Delate_Colmns(df):
    # Description: Function that delate the columns that are irrelevant from becasue have all the elemenst equal
        # Inputs
        #   % df: the dataframe
        # Output
        #   % df
        #   % Descripotrs that are not trash
    #------------------------------------- Beginning of the function --------------------------
    columns = list(df.columns)
    index_descriptrs =list()
    for colum_i in columns: 
        if (is_trash(df[colum_i])):
            df = df.drop(colum_i,1)
            index_descriptrs.append(False) # set false to the descripo
        else:
             index_descriptrs.append(True)
    return df, index_descriptrs
    #------------------------------------- End of the function ---------------------------
def PlotDistribution(X,namefile = 'Distribution',mean_bool = True, std_bool= True,bins = 30,root = 'Distribucion/',mean_round = 2,distance_box = (1/3)):
    
    columns = X.columns
    
    for name in columns:
    
        X = np.array(X[name])
        
        ## Get parameters
        mean = X.mean()
        median = np.median(X)
        std = X.std()
        
        ## Plots 
        fig, ax = plt.subplots(dpi = 500)
        
        
        ax.hist(X,bins=bins,facecolor='royalblue', edgecolor='navy')
        ax.set_xlabel(name)
        ax.set_ylabel('Cuentas')
        
        # Stats box  
        props = dict(boxstyle='round', facecolor='darkorange', alpha=0.1,edgecolor='darkorange')
        higt_label = ax.get_ylim()[1]*(3/4)
        
        if std_bool ==True:
            wheig_label = mean+std+(std*(distance_box))
        else:
            wheig_label = mean+(std*(distance_box))
         
    
        text_stats = '\n'.join((
        r'$\mu=%.2f$' % (mean, ),
        r'$\sigma=%.2f$' % (std, )))
        
        ax.text(wheig_label,higt_label,text_stats,fontsize='xx-large',bbox=props)
        
        # Mean
        if mean_bool == True:
            ax.axvline(x=mean,color ='darkred')
        
            mean_text = r'$\mu$ = '+ str(np.round(mean,mean_round))
              
        # Std
        if std_bool == True:
            higt_rectangle = ax.get_ylim()
            ax.add_patch(Rectangle((mean, 0), std, higt_rectangle[1],facecolor='darkorange',alpha=0.2))
            ax.add_patch(Rectangle((mean-std, 0), std, higt_rectangle[1],facecolor='darkorange',alpha=0.2))
            ax.axvline(x=mean+std,color ='darkorange')
            ax.axvline(x=mean-std,color ='darkorange')
        
        ##
        dirfile = 'Distribution_Plots/'
        
        CheckDir(dirfile) # 
        
        tag_name = '_'+str(name)
        
        
        File  = dirfile+ namefile + tag_name 
        ##Save 
        plt.savefig(File,transparent=False)
        plt.savefig(File+'.svg',transparent=True)
    
        plt.show() 


def Search_descriptor(descriptor_check,descriptor_list):
    check = False 
    for descriptor in descriptor_list:
        if str(descriptor) == str(descriptor_check):
            check = True
            break
    return check
    
################################## Definition calsses  ##################################

class ML_Data:
    # Description: class which contain the information of the data, all molecule and solvent  
    # Atributes:
    #   %
    #   %
 
    #------------------------------------- Beginning of class ---------------------------
    def __init__(self,spect_data_name = False,solv_data_name = False,chromophore_data_name = False, Data_df_name = False):
        # Description: a function that imports the spectroscopy data and solvent data
        # Input
        #   % spect_data_name: name of the .csv file that contains the spectroscopi data   
        #   % solv_data_name: name of the .csv file that contains the solvent data
        #   % chromophore_data_name:  name of the .csv file that contain the molecule data
        #   % Data_df_name: name of the .csv file that contains a data frame with the [Y] and [X]

        #------------------------------------- Beginning of the function ---------------------------
        if Data_df_name == False: 
            
            self.list_df_descriptors_mol = list()  # the list that going to contain the df molecule descriptors
            self.list_df_descriptors_solv = list() # the list that going to contain the df solvent  descriptors
            
            
            self.spect_data = pd.read_csv(spect_data_name) # df that contain the spectroscopi data
            
            self.name_without_extention = os.path.splitext(spect_data_name)[0] # Core name of the original file 
            
            if solv_data_name == False: # if does not exist a solfent data frame
                 self.solv_data = Get_Mol_Smile(self.spect_data,nick_mol='Solvent')  # df that contain the solvent data
                 # Save solvent df
                 File = self.name_without_extention +'_Solvent.csv'
                 self.solv_data.to_csv(File,index=False) 
                 self.solv_data_name = File
            else: 
                 self.solv_data = pd.read_csv(solv_data_name) # import the df that contain the solvent data
                 
            if chromophore_data_name ==False: # if does not exist a molecular data frame
                
                self.df_molecules_smile = Get_Mol_Smile(self.spect_data,nick_mol='Smile')   
                # Save solvent df
                File = self.name_without_extention +'_Molecules.csv'
                self.df_molecules_smile.to_csv(File,index=False)
            
            else:
                self.df_molecules_smile = pd.read_csv(chromophore_data_name)  # import the df that contain the molecule data
                    
        else:  ## Init the class only with the a data frame 
        
            self.Data = pd.read_csv(Data_df_name)
            
            columns = self.Data.columns
            self.Descriptors_Name_list = columns[5:len(columns)] # Creat a list of descripotrs name:
            Property = columns[4]
            self.Property_list = [Property]
        #------------------------------------- End of the function ---------------------------
    
        
    def Structure_generator(self,Name,is_solvent = False):
           
        # Description:Function which generate the 2D and 3D strucure of each molecule and save a .sdf file with all molecules in .mol format
            # input
            #   % Name: name of the .sdf file
            #   % is_solvent: if is True meaning that the Data_mol is a data base of solvents
        
        #------------------ Previous functions ------------------------- 
        def smiletocxsmile (cxsmile):
            # Description: transform cxsmile into normal smile 
            # Inputs
            #   % cxsmile
            # Output 
            #   % smile
            #------------ Beginning of the function -----------
            smile = "" 
            for i in range(len(cxsmile)):
                char_i = cxsmile[i]
                if char_i == "|":
                    break
                smile += char_i
            return smile
            #---------- End of the function --------------------
            
        #------------------ End previous functions------------------------
        
        #------------------------------------- Beginning of the function ---------------------------
        
        dirfile = 'Molecule_Structure/'
        
        CheckDir(dirfile) # 
        
        File_name_2D = dirfile + Name+'_2D.sdf' # creat the 2D name file   
        File_name_3D = dirfile + Name+'_3D.sdf' # creat the 3D name file
        
        
        ## Crear los archivos .sdf 
        w2D = Chem.SDWriter(File_name_2D) # creat the 2D file (RDTkid)
        w3D = Chem.SDWriter(File_name_3D) # creat the 3D file (RDTkid)
        
        
        if is_solvent == True:
            
            df = self.solv_data
            is_solvent_name = 'Solvent_' # Add the label Sovent at the name of file if is a solvent
        else:
            ##df = self.spect_data
            df = self.df_molecules_smile
            is_solvent_name = ''
        
        
        for i in range(len(df['Smile'])): # Iterate over all molecules 
              
            print('Indice mol:\t'+str(i))
            
            Name =  is_solvent_name + df['Name'][i]  #  name molecule
            print('Name momlecule:\t'+str(Name))
            cxsmile = df['Smile'][i] # pick the smile from the df 
            Smile = smiletocxsmile(cxsmile) # transform  cxsmile to smile  
            File_Name = (str(i)+'_'+ '.mol')
            File_Name ='Molecule_Structure/'+ File_Name # creat the name of individual file  
            print('Smile:\t'+str(Smile))
            
            
            #-------Generate the structure  
            mol2D = Chem.MolFromSmiles(Smile) # crate the 2D strucure 
            mol2D.SetProp("_Name",Name) # add the name of molecule 
            
            mol3D = Chem.AddHs(mol2D) # crate the 3D structure 
            
            params = Chem.ETKDG() # optimization parameter
            params.useRandomCoords=True 
            
            Chem.EmbedMolecule(mol3D,params) # optimization 
            
            w2D.write(mol2D) # save the .sdf files
            w3D.write(mol3D) 
            
            print('Number Molecule: ',i)
            
            #print(Chem.MolToMolBlock(mol3D),file=open(File_Name,'w+')) # save the individual .mol files 
            
        w2D.close() # close the files  
        w3D.close() 
        #------------------------------------- End of the function ---------------------------
    
    def Descriptor_generator(self,File_Data_mol,is_solvent = False,descripotrs_incluided = 'all',Na_value = 0,FilterData = True):
        
        # Description: Function that creat and save the df_descriptor (using Data_frame_descriptor_generator function) in a .csv file 
            # input
            #   % File_Data_mol: name of the .sdf file whith the strucural information of all molecules
            #   % is_solvent: if is True meaning that the Data_mol is a database of solvents
            #   % descripotrs_incluided: is the interval of descriptors to add into the tuple
            #     it have to be a list of list in which each element of the second nesting is containg the minimum and maximum interval
            #     Example descripotrs_incluided  = [[1,10],[20,50],...] Check (https://mordred-descriptor.github.io/documentation/master/descriptors.html) 
            #     to know all the descriptors that you can add
            #   % Na_value: The value that the function going to add if the descriptor could not be calculated
            #   % FilterData: if is true delate the columns in the data frame that gat in all the cases the same value 
    
        #------------------------------------- Beginning of the function ---------------------------
        self.Data_mol = Chem.SDMolSupplier(File_Data_mol,removeHs = False) # import the strucural .sdf file 
        
        if descripotrs_incluided == 'all':
            Descriptors = List_Descriptor_all()  # Creat the descriptor tuple 
        else:
            Descriptors = List_Descripotr_genarate(descripotrs_incluided)
        
        
        self.df_descriptors = Data_frame_descriptor_generator(Descriptors,self.Data_mol,is_solvent=is_solvent,NaValue = Na_value) # creat the df descriptor 
        

        if FilterData== True:  
            self.df_descriptors, self.index_descriptrs = Delate_Colmns(self.df_descriptors) # Delate trash descriptors
        
        #### Guarda un archivo con los Datos
        Save_data_descriptors(self.df_descriptors,self.index_descriptrs,descripotrs_incluided,is_solvent = is_solvent)  # save the file 
        #------------------------------------- End of the function ---------------------------
    
    
    def Open_descriptor_mol_List(self,listNames,sep = False):
        # Description: Import the molecular df_descriptor and save it 
            # Input 
            #   % listNames: a list of .csv name file descriptros 
            #   % sep: sep is the kind of separation of the file if is false the default is a .csv format
            
        #------------------------------------- Beginning of the function ---------------------------
        self.molecule_descriptor_list = listNames
        for name in listNames: #iter over each df_descriptor fild 
            
            if sep == False:
                df_descriptors = pd.read_csv(name) ### open the .csv file     
            elif sep== 'tab':
                df_descriptors = pd.read_csv(name, sep='\t') ### open the .txt file
            
            #df_descriptors = df_descriptors.dropna(axis='columns') ## option to filter descriptors
            
            self.list_df_descriptors_mol.append(df_descriptors) # add de df to a list of df
            
        #------------------------------------- End of the function ---------------------------
    
    def Open_descriptor_solv_List(self,listNames,sep = False):
        
        # Description: Import the solvent df_descriptor and save it 
        # Input 
        #   % listNames: a list of .csv names files descriptro 
        #   % sep: sep is the kind of separation of the file if is false the default is a .csv format
        #------------------------------------- Beginning of the function ---------------------------
        self.Solvent_descriptor_list = listNames
        for name in listNames: #iter over each df_descriptor fild 
            
            if sep == False:
                
                df_descriptors = pd.read_csv(name) ### open the .csv file     
            elif sep== 'tab':
                df_descriptors = pd.read_csv(name, sep='\t') ### open the .txt file
            
            #df_descriptors = df_descriptors.dropna(axis='columns') ## option to filter descriptors
            
            self.list_df_descriptors_solv.append(df_descriptors)
         #------------------------------------- End of the function ---------------------------

    
    def Filter_Property(self,Property_list,Limit_list,Solvent_list,Type_molecule_list,nick_search_chromophore = 'Smile',nick_search_solv = 'Smile'):
       
        # Description: This function groups the properties to study with the corresponding molecular and solvent descriptors previously imported by
        # Open_descriptor_mol_List and Open_descriptor_solv_List. In addition, the method includes a filtering option
        # Input 
        #   % Property_list: list of the properties name to group (only one) 
        #   % Limit_list: list of the limits of the properties which the program going to work (only one) 
        #   % Solvent_list: list of the solvents which the program going to work 
        #   % Type_molecule_list: list of the type of molecules which the program going to work

        
        #------------------------------------- Previous functions --------------------------------------------------------# 
        
        def Append_descriptor(Data,df_descriptors,df_molecule,Descriptors_Name_list,df,nick_search = 'Smile'): #### Add de values 
            
            # Description: Function that searches a molecule in the spectroscopy data frame and adds the value of the corresponding descriptor into a new data frame 
            # Input 
            #   % Data: new data frame in which the descriptors going to be appended  
            #   % df_descriptors: the descriptors that contain the data frame values
            #   % df_molecule: data frame that contains the information of the molecule
            #   % Descriptors_Name_list: a list of the name of the descriptors 
            #   % nick_search: the name of the column to make the search

            #------------------------------------- Beginning of the function ---------------------------
            Num_des = len(Descriptors_Name_list) 
            Indx_mol_list = list() # lita que va a guardar lo indice de las moleculas para previas iteraciones
            for i in range(Num_des): 
                Name_Descriptor = Descriptors_Name_list[i]
                Descriptor_list =[]
                print('Descriptor: \t',i,'\t', Name_Descriptor)
                indice = 0 ## indice que cuenta en que dato se encuentra  
                for numdata in Index:
                    #print('data: \t',numdata)
                    molecule = df[indice] ##
                    print('Molecule:\t',molecule)
                    if i == 0:
                        Indx_mol = df_molecule[df_molecule[nick_search]==molecule].index.values ## Busca la molecula
                        print('Index molecule:\t',Indx_mol)
                        Indx_mol = Indx_mol[0] ### obtiene el valor del indice de la molecula
                        
                        Indx_mol_list.append(Indx_mol)
                    else:
                        Indx_mol = Indx_mol_list[indice] ## Selecion el inidce de la molecula
                    
                    print('Index molecule:\t',Indx_mol)
                    Descriptor_Value = df_descriptors.iloc[Indx_mol,i]
                    
                    Descriptor_list.append(Descriptor_Value)
                    indice = indice + 1 
                #print('Index molecule_list:   ', Indx_mol_list)
                Data[Name_Descriptor] = Descriptor_list
            return Data
    
        #---------------------------- End previous functions-----------------------------------------------------------------------------#
        
        #------------------------------------- Beginning of the function -----------------------------#
        df = self.spect_data # select df of the class
        
        self.Solvent_list = Solvent_list
        self.Property_list = Property_list
        
        #---- Filtering
        Column = ['#','Solvent','Smile','Dye type']
        i = 0
        ##### Filtrar propiedades
        for Propiedad in Property_list:
            Limit = Limit_list[i]
            df = df.dropna(subset=[Propiedad]) ### Filtra los datos que no tienene la propiedad selecionada 
            df = df[df[Propiedad] > Limit[0]] ### Filtra limite inferior
            df = df[df[Propiedad] < Limit[1]] ### Filtra limite superior 
            Column.append(Propiedad)
            i = i + 1 
            
        Data = pd.DataFrame.from_dict(df) ### Genera el frame de pandas
        
        ##### Filtrar solvente
        if Solvent_list[0] != 'all':
            Solvent = Data.Solvent.isin(Solvent_list)
            Data= Data[Solvent] ## Filtra el solvente
        
        ##### Filtrar tipo de molecula
        if Type_molecule_list[0] != 'all':
            
            Molecule_list = Data['Dye type'].isin(Type_molecule_list)
            Data= Data[Molecule_list] ## Filtra el tipo de molecula 
        
        
        Data = Data[Column] 
        
        ##### Genera el indice de moleculas sobre el cual va a iterar
        
        Index = Data['#'].values.tolist() 
        df_solvents = Data['Solvent'].values.tolist()
        
        if nick_search_chromophore == 'Name': ## 
            
            df_molecules = Data['Chromophore Name'].values.tolist()
        else:
            df_molecules = Data['Smile'].values.tolist()
        
        
        self.Data = Data 
                        
        #----- Add the descriptors to the data set 
        
        #--- Molecular descripotrs
        for df_descriptor in self.list_df_descriptors_mol:
            df_descriptor_Names_list = list(df_descriptor.columns)
            df_mol_chromofore = self.df_molecules_smile
            
            self.Data = Append_descriptor(self.Data,df_descriptor,df_mol_chromofore,df_descriptor_Names_list,df_molecules,nick_search=nick_search_chromophore)
           
        #--- Solvent descripotrs 
        for df_descriptor in self.list_df_descriptors_solv:
            df_descriptor_Names_list = list(df_descriptor.columns)
            df_mol_solvent = self.solv_data ### data frame de estructra de solventes
            
            self.Data = Append_descriptor(self.Data,df_descriptor,df_mol_solvent,df_descriptor_Names_list,df_solvents, nick_search=nick_search_solv)  ### añade al set de datos
            
        #---  Building the list of the descriptors name
        Column = self.Data.columns 
        self.Descriptors_Name_list = Column[5:len(Column)] 
        #------------------------------------- End of the function ---------------------------
            
    def Filter_substructure(self,Substructure_list):
        
        # Description: Function that filter the  spectroscopy dat (created by Filter_Property) it only keep the molecules with a specific substructure and delete de data provided by molecules that deas not contain this substructure   
        # input 
        #   % Substructure_list: a list of substructures that you want that the molecules in the data_ML class keep contain.
        #------------------------------------- Beginning of the function -----------------------------#
        Sub_list = list()
        for Smile in self.Data.Smile:
                   
            for Substructure in Substructure_list:
                mol_sub = Chem.MolFromSmiles(Substructure)
                mol =  Chem.MolFromSmiles(Smile)
                Bool = mol.HasSubstructMatch(mol_sub)
                if Bool == True:
                    break
            Sub_list.append(Bool)
        self.Data['Substructure']  = Sub_list   
        self.Data = self.Data[self.Data['Substructure'] == True] 
        #------------------------------------- End of the function -----------------------------------#
        
    def Filter_Descriptor(self,Descriptor_list,Limit_list):
        
        # Description: function that eliminated all the descripitrs data that there are not in the specific list with the specific limit 
        # input 
        #   % Descriptor_list: a list of descriptors that you want to maintain  
        #   % Limit_list: the list of the limits per descriptor

        #------------------------------------- Beginning of the function -----------------------------#
        for Descriptor, Limit in zip(Descriptor_list,Limit_list):
            
            self.Data = self.Data[self.Data[Descriptor] >= Limit[0] ]
            self.Data = self.Data[self.Data[Descriptor] <= Limit[1] ]
        #------------------------------------- End of the function -----------------------------------#        
        
    def Discriminate_substructure(self,Substructure_list):
        
       # Description: Function which that  the spectroscopy data by delating that data provided by a molecule that contains a specific substructure 
        # input 
        #   % Substructure_list: a list of substructures that you want to eliminate from the data set

        #------------------------------------- Beginning of the function -----------------------------#
        Sub_list = list()
        for Smile in self.Data.Smile:
                   
            for Substructure in Substructure_list:
                mol_sub = Chem.MolFromSmiles(Substructure)
                mol =  Chem.MolFromSmiles(Smile)    
                Sub_list.append(mol.HasSubstructMatch(mol_sub))
                
                
        self.Data['Substructure']  = Sub_list   
        self.Data = self.Data[self.Data['Substructure'] == False] 
        #------------------------------------- End of the function -----------------------------------#
    
    def Save_df(self,namefile = 'DataFrame'):
        # Descripotos: Function that save the df created in Preporsesign methond and save in memory,
        # In orde to open a Datas_Ml class only with this crated file 
        #------------------------------------- Beginning of the function -----------------------------#
        
        dirfile = 'DataFrames_ML_Data/'
        CheckDir(dirfile) # 
        
        
        Shape = self.Data.shape
        row = Shape[0]
        col = Shape[1]
        prop = self.Property_list[0] 
        
        tagfile = '_'+str(prop)+'_row_' +str(row)+'_col_'+ str(col)+ '.csv'
        
        File = dirfile + namefile + tagfile
        print('The File "'+ File +'" have been saved')
        
        
        self.Data.to_csv(File,index = False) 
        #------------------------------------- End of the function -----------------------------------#
        
    def Preprosesing(self,Solv_Category = False,Scaling = True,Imputer = False,star_data = 5):
        # Descripción: Function that use a df that conatin the descripores and the spectroscopi properties and get the X and Y matrix ready to use in ML algorithm. 
        # In addition, it contains some preprocessing features such as scaling or imputer data        
        # input 
        #   % Solv_Category: if it is true it going to implement solvent categorization, that means add columns, the sated value from this columns going to be 1 if the data contain the specific solvent and 0 if no, it could be useful if you have less than 10 different solvents in your data set.
        #   % Scaling: make a scaling process in each feature
        #   % Imputer: it defines the imputer metod
        #   % star_data: it number specifies the column in which the descriptors data start
        # output
        #   %
        #------------------------------------- Beginning of the function -----------------------------#
        Data = self.Data
        prop = self.Property_list[0] 
        #---- Get the independent variables 
        Num_Descriptors = len(self.Data.columns) - star_data
    
        Num_Propieties = 1
        
      
     
        Y = Data.iloc[:,star_data-1] # dependent variables
        X = Data.iloc[:,star_data-1+Num_Propieties:] #independent variables  
        
        
        
        Y = Y.to_numpy()
        X = X.to_numpy()
        
        #### Missing Data
        if Imputer != False:
            
            imputer = SimpleImputer(strategy = 'constant', fill_value = 0)
            X = imputer.fit_transform(X)
            
            if Imputer == 'Nearest_neighbors':
                imp = KNNImputer(n_neighbors=2, weights="uniform")
                X = imp.fit_transform(X)
            elif Imputer == 'Simple_Imputer':
                imp = SimpleImputer(missing_values=np.nan, strategy='mean')
                imp.fit(X)
                X = imp.transform(X)
                      
                
        
        ## Feature Scaling 
        if Scaling == True:
            
            sc = StandardScaler() # standarization object
            X = sc.fit_transform(X)
            self.Standar_Scalar = sc
        else:
            self.Standar_Scalar = False
            
        ## Splitting into Training set and Test set 
        X_sh,Y_sh = sklearn.utils.shuffle(X,Y) ## Revuelve los datos 
        X_train, X_test, Y_train, Y_test = train_test_split(X_sh, Y_sh, test_size = 0.2, random_state = 0)
        
        
        #--- print the relaive sice from traing-Validation set and test set
        total = X_train.shape[0] + X_test.shape[0]
        print('Train:', X_train.shape[0]/total)
        print('Test:', X_test.shape[0]/total)    

        ### Plots 
        df_Y = pd.DataFrame()
        df_Y_train = pd.DataFrame()
        df_Y_test = pd.DataFrame()
        
        df_Y[str(prop)] = Y
        df_Y_train[str(prop)+'_Train'] = Y_train
        df_Y_test[prop +'_Test'] = Y_test
    
        PlotDistribution(df_Y,std_bool= False)
        PlotDistribution(df_Y_train,std_bool= False)
        PlotDistribution(df_Y_test,std_bool= False)

        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.X = X
        self.Y = Y
        #------------------------- Save SC 
        Shape = self.Data.shape
        row = Shape[0]
        col = Shape[1]
        
        nametag = 'StandarScalar' +str(prop)+'_row'+str(row)+ '_col' +str(col)
        
        dirfile = 'StandarScalar/'
        
        CheckDir(dirfile) # 
        
        namefile_sc   = dirfile +nametag+'.sc'
        
        ### Write data
        f = open(namefile_sc,'wb')
        dill.dump(self.Standar_Scalar,f)
        f.close()
        
        #------------------------------------- End of the function -----------------------------------#
    
        
    def Plot_Prop_Desc(self,Name_desc):
        # Descripción: Function that plot in the X axes a specific descriptor and in the y axes the specifcated propitie  
        # input 
        #   % Name_desc: the descriptor name that you want to plot  
        
        #------------------------------------- Beginning of the function -----------------------------#
        Data = self.Data
        Prop = Data.iloc[:,4]
        Desc = Data[Name_desc]
        Prop_name =self.Data.columns[4]
        Prop_name = str(Prop_name)
        
        ##plot
        plt.figure(figsize=(8, 6), dpi=500)
        plt.scatter(Desc,Prop)
        plt.xlabel(Name_desc) 
        plt.ylabel(Prop_name)
        plt.show()
        #------------------------------------- End of the function -----------------------------------#
    
    def Data_Distribution(self):
        
        # Description: Function that plot some specic information useful from the data set analizis.
        # it plots  the properties as a function of the index of the data and the molecular weight, 
        # it also plots the properties as a function of the molecular weight,
        # and a histogram of the solvent and  from the kind of molecules

        #------------------------------------- Beginning of the function -----------------------------#
        Index = self.Data['#']
        Smile = self.Data['Smile']
        Kind_dye = self.Data['Dye type']
        Solvent = self.Data['Solvent']
        Prop = self.Data.iloc[:,4]
        Prop_name =self.Data.columns[4]
        Prop_name = str(Prop_name)
        
    
        #### Pesos Moleculares
        MolWt = list()
        for mol in Smile:
           MolWt.append(ExactMolWt(Chem.MolFromSmiles(mol)))
        
        #### Graficas
        # Peso molecular (Indice) 
        plt.figure(figsize=(8, 6), dpi=500)
        plt.scatter(Index,MolWt)
        plt.xlabel('Index') 
        plt.ylabel('Molecular weight')
        plt.show()
        # Propiedad (Indice)
        plt.figure(figsize=(8, 6), dpi=500)
        plt.scatter(Index,Prop)
        plt.xlabel('Index') 
        plt.ylabel(Prop_name)
        plt.show()
        
        # Propiedad (Peso molecular)
        plt.figure(figsize=(8, 6), dpi=500)
        plt.scatter(MolWt,Prop)
        plt.xlabel('Molecular weight') 
        plt.ylabel(Prop_name)
        plt.show()
        
        ##
        plt.figure(figsize=(8, 6), dpi=500)
        plt.hist(MolWt, bins=20,rwidth=0.9)
        plt.xlabel('Molecular weight') 
        plt.ylabel('Count')
        plt.show()
        ###
        ##Solventes
        Count = Counter(Solvent)
        Num_solv = len(Count.keys())
        
        plt.figure(figsize=(8, 6), dpi=500)
        plt.hist(Solvent,bins=Num_solv,rwidth=0.9,align = 'mid')
        plt.xticks(rotation=90)
        plt.xlabel('Solvent') 
        plt.ylabel('Count')
        plt.show()
        
        plt.figure(figsize=(8, 6), dpi=500)
        plt.hist(Prop,bins=20,rwidth=0.9)
        plt.xlabel(Prop_name) 
        plt.ylabel('Count')
        plt.show()
        
        ### Tipo de molecula
        print('Plot kind molecule')
        Count = Counter(Kind_dye)
        Num_Kind_dye = len(Count.keys())
        bins = np.arange(Num_Kind_dye) - 0.5
        plt.figure(figsize=(8, 6), dpi=500)
        
        plt.hist(Kind_dye, bins=bins,rwidth=0.9,align = 'mid')
        bins=Num_Kind_dye
        #ticks = [j*(1/bins) + (1/bins)/2 for j in range(bins)]
        #plt.xticks(ticks,rotation=90)
        plt.xticks(rotation=90)
        plt.xlabel('Kind_dye') 
        plt.ylabel('Count')
        plt.show()
        #------------------------------------- End of the function -----------------------------------#

#------------------------------------- End of the class ---------------------------

#----------------------------------------- Model Class -------------------------------------------------#

#---------- Required functions for this class -------------------------

def LoadModel(dirfile):
    
    ### files    
    files = get_filepaths(dirfile)
    
    for file in files:
            
        if file.endswith('.Meta'):
            
            f = open(file,'rb')
            meta_list = dill.load(f)
            f.close()
            
        elif file.endswith('.Model'):
            
            f = open(file,'rb')
            model = dill.load(f)
            f.close()
            
        elif file.endswith('.DimRed'):
            
            f = open(file,'rb')
            DimRedMethod = dill.load(f)
            f.close()
            
        elif file.endswith('.crossVal'): 
            
            f = open(file,'rb')
            crossVal = dill.load(f)
            f.close()
    
    # load model
    Open_Model = Model_(False,model,meta_list[1],meta_list[2])
    Open_Model.Propriety = meta_list[0]
    Open_Model.CrossVal = crossVal
    
    if meta_list[1]!= False:
        
        Open_Model.Skbest = DimRedMethod
    
    return Open_Model
    

#------------------------------------- Beginning of the class ---------------------------
class Model_:
    # Description: Class that keeps the the atributes with the model based in sklearn library ()  
    # Atributes:
    #   %
    #   %
    #   %
    def __init__(self,Data,Model,Num_red = False,cv = 5):
        
        # Description: Function to iniciate the class.   
        # Input:
        #   % Data: Variable that keeps the the database with which the model going to train
        #   % Model: The model (from the sk-learn library)
        #   % Num_red: the number features after the dimentional reduction 
        #   % cv: the iteration in the cross validation algorithm
        
        #------------------------------------- Beginning of the function -----------------------------#
        # Defined the atributes of the class
        self.Data = Data
        if Data == False:
            self.Propriety = False
        else:
            self.Propriety = Data.Property_list[0]
        
        self.Num_red = Num_red
        self.cv = cv 
        
        
        self.Model = Model
        #------------------------------------- end of the function -----------------------------#
        # Select the dimencion reduction method  
        if Num_red != False:
            self.Skbest =SelectKBest(f_regression, k=Num_red) # dim reduction
    
    def Save(self,namemodel = 'Model'):
        # Description: Function That save the class in a binary file using the library drill 
        # Input: 
        #   % file: the file in with the class going to saved 
        #------------------------------------- Beginning of the function -----------------------------#
        meta_list = list()
        
        meta_list.append(self.Propriety)
        meta_list.append(self.Num_red)
        meta_list.append(self.cv)    
        
        model = self.Model
        
        #------ Dimencional reduction 
        if self.Num_red != False:    
            DimRedMethod = self.Skbest
        else:
            DimRedMethod = False
        
        crossVal = self.CrossVal
        
        nametag = '_' +str(self.Model)+'_'+str(self.Propriety)+ '_DimRed_' +str(self.Num_red)
        
        dirfile = 'ModelsFiles/'+nametag+ namemodel +'/'
        
        CheckDir(dirfile) # 
        
        namefile_meta   = dirfile + namemodel +nametag+'.Meta'
        namefile_model  = dirfile + namemodel +nametag+'.Model'
        namefile_dimRed = dirfile + namemodel +nametag+'.DimRed'
        namefile_crossval = dirfile + namemodel +nametag+'.crossVal'
        

        ### Write data
        f = open(namefile_meta,'wb')
        dill.dump(meta_list,f)
        f.close()
        
        f = open(namefile_model,'wb')
        dill.dump(model,f)
        f.close()
        
        
        f = open(namefile_dimRed,'wb')
        dill.dump(DimRedMethod,f)
        f.close()
        
        f = open(namefile_crossval,'wb')
        dill.dump(crossVal,f)
        f.close()
        
        #------------------------------------- End of the function -----------------------------#
    def validation(self):
        
        # Description: Function that train and validate the model.   
        # Input:
        
        #------------------------------------- Beginning of the function -----------------------------#    
        Data = self.Data
    
        X_train = Data.X_train
        X_test = Data.X_test
        Y_train = Data.Y_train
        Y_test = Data.Y_test
        
    
        #------ Dimencional reduction 
        if self.Num_red != False:
            
            X_train = self.Skbest.fit_transform(X_train,Y_train)
            X_test = self.Skbest.transform(X_test)
        
        #------------- Training -----------------------------# 
        print('---------------------Training\t'+str(self.Propriety)+'---------------------')
        t = time()
        self.Model.fit(X_train, Y_train)
        DT = time() -t # Tiempo de entrenamiento
        self.DT = DT 
        print('Time:\t'+str(DT)) ### Añade el tiempo en que tarda en entrenar el modelo
        
        
        #-------------- Validation --------------------------#
        print('---------------------Cross validation\t'+str(self.Propriety)+'---------------------')
        print('Iteration:\t'+str(self.cv))        
        scores = cross_validate(self.Model, X_train, Y_train, cv=self.cv,scoring=('r2','neg_mean_absolute_error','neg_root_mean_squared_error'),return_train_score=True)   
        self.CrossVal = scores
        
        #------------ Get the parameters of the validation --------------# 
        # Train

        cv_r2_mean = scores['train_r2'].mean()
        cv_r2_std = scores['train_r2'].std()*2
        
        self.cv_r2_mean_train = cv_r2_mean
        self.cv_r2_std_train = cv_r2_std
        
        cv_mean_absolute_error_mean = np.absolute(scores['train_neg_mean_absolute_error'].mean())
        cv_mean_absolute_error_std = scores['train_neg_mean_absolute_error'].std()*2
        
        self.cv_mean_absolute_error_std_train = cv_mean_absolute_error_std
        self.cv_mean_absolute_error_std_train = cv_mean_absolute_error_std
        
        cv_mean_squared_error_mean = np.absolute(scores['train_neg_root_mean_squared_error'].mean())
        cv_mean_squared_error_std = scores['train_neg_root_mean_squared_error'].std()*2
        
        self.cv_mean_squared_error_mean_train = cv_mean_squared_error_mean
        self.cv_mean_squared_error_std_train = cv_mean_squared_error_std
        
        print('---------------------Train Results---------------------')
        print('Train score r2: {:.6f} +- {:.6f}'.format(cv_r2_mean,cv_r2_std))
        print('Train score max error: {:.6f} +- {:.6f}'.format(cv_mean_absolute_error_mean,cv_mean_absolute_error_std))
        print('Train score absolute error: {:.6f} +- {:.6f}'.format(cv_mean_squared_error_mean,cv_mean_squared_error_std))
        print()
        
        ## Test
        cv_r2_mean = scores['test_r2'].mean()
        cv_r2_std = scores['test_r2'].std()*2
        
        self.cv_r2_mean_test = cv_r2_mean
        self.cv_r2_std_test = cv_r2_std
        
        cv_mean_absolute_error_mean = np.absolute(scores['test_neg_mean_absolute_error'].mean())
        cv_mean_absolute_error_std = scores['test_neg_mean_absolute_error'].std()*2
        
        self.cv_mean_absolute_error_mean_test = cv_mean_absolute_error_mean
        self.cv_mean_absolute_error_std_test = cv_mean_absolute_error_std
        
        cv_mean_squared_error_mean = np.absolute(scores['test_neg_root_mean_squared_error'].mean())
        cv_mean_squared_error_std = scores['test_neg_root_mean_squared_error'].std()*2
        
        self.cv_mean_squared_error_mean_test = cv_mean_squared_error_mean
        self.cv_mean_squared_error_std_test = cv_mean_squared_error_std
        
        print('---------------------Validation Results---------------------')  
        print('Test score r2: {:.6f} +- {:.6f}'.format(cv_r2_mean,cv_r2_std))
        print('Test score max error: {:.6f} +- {:.6f}'.format(cv_mean_absolute_error_mean,cv_mean_absolute_error_std))
        print('Test score absolute error: {:.6f} +- {:.6f}'.format(cv_mean_squared_error_mean,cv_mean_squared_error_std))
        print()
        #------------------------------------- End of the function -----------------------------#
        
    def fit(self):
        # Description: Function that fit an validate the model using al the data from the data set, the is used after the model evaluation
        #              It's fucntion is used to traina a model used to predict new values.   
        # Input:
            
        #------------------------------------- Beginning of the function -----------------------------#    
        Data = self.Data 
        X = Data.X
        Y = Data.Y
        
        if self.Num_red != False:
            X = self.Skbest.fit_transform(X,Y)
        self.Model.fit(X,Y)
        
        
        #------------------- Get and prints the scores of the training ----------------------------#
        #  validation 
        print('---------------------Cross validation\t'+str(self.Propriety)+'---------------------')
        scores = cross_validate(self.Model,X, Y, cv=self.cv,scoring=('r2','neg_mean_absolute_error','neg_root_mean_squared_error'),return_train_score=True)
        self.CrossVal = scores
        
        cv_r2_mean = scores['test_r2'].mean()
        cv_r2_std = scores['test_r2'].std()*2
        
        print('r2(cv):\t'+str(cv_r2_mean)+' +/ '+ str(cv_r2_std))
        
        cv_mean_absolute_error_mean = np.absolute(scores['test_neg_mean_absolute_error'].mean())
        cv_mean_absolute_error_std = scores['test_neg_mean_absolute_error'].std()*2
        
        print('mae(cv):\t'+str(cv_mean_absolute_error_mean)+' +/ '+ str(cv_mean_absolute_error_std))
        
        cv_mean_squared_error_mean = np.absolute(scores['test_neg_root_mean_squared_error'].mean())
        cv_mean_squared_error_std = scores['test_neg_root_mean_squared_error'].std()*2
        
        print('mse(cv):\t'+str(cv_mean_squared_error_mean)+' +/ '+ str(cv_mean_squared_error_std))
        
        ### score 
        score = self.Model.score(X,Y)
        print('r2:\t' + str(score))
        #------------------------------------- End of the function -----------------------------#
        
    def Plot_Results(self, r_round = 3, mae_round = 1,mse_round = 0):
        # Description: Function that plots the results from the final evaluation of the model.   
        # Input:
        #   % r_round: the number of desimals to print in the final for the r2 metric
        #   % mae_round: the number of desimals to print in the final for the mae metric
        #   % mse_round: the number of desimals to print in the final for the mse metric
        
        #------------------------------------- Beginning of the function -----------------------------#
        Data = self.Data
        X_train = Data.X_train
        X_test = Data.X_test
        Y_train = Data.Y_train
        Y_test = Data.Y_test
        Model = self.Model
        #----------------------- Dimencional reduction --------------------------------#
        if self.Num_red != False:
            
            X_train = self.Skbest.fit_transform(X_train,Y_train)
            X_test = self.Skbest.transform(X_test)
            # Training 
            Model.fit(X_train,Y_train)
    
        X_train_r = X_train
        X_test_r = X_test  
        
        # Training 
        Model.fit(X_train_r,Y_train)
        
        # Results
        Test_predict = Model.predict(X_test_r)
        
        r2 = sklearn.metrics.r2_score(Y_test,Test_predict)
        r2 = np.round(r2,r_round)
        mae = sklearn.metrics.mean_absolute_error(Y_test,Test_predict) 
        mae = np.round(mae,mae_round)
        mse = sklearn.metrics.mean_squared_error(Y_test,Test_predict) 
        mse = mse**(1/2) 
        mse = np.round(mse,mse_round)
    
        # Predicted 
        Y_train = pd.DataFrame(Y_train)
        
        Y_predicted = Model.predict(X_test_r)
        
        predicted = cross_val_predict(Model, X_train_r,Y_train, cv=self.cv)
        
        #-------------------- Validation ----------------------#        
    
        scores = cross_validate(Model, X_train_r, Y_train, cv=self.cv ,scoring=('r2','neg_mean_absolute_error','neg_root_mean_squared_error'),return_train_score=True,return_estimator =True)   
        self.CrossVal = scores 
        # Test
        cv_r2_mean = scores['test_r2'].mean()
        cv_r2_std = scores['test_r2'].std()*2
        cv_r2_str =  str(np.round(cv_r2_mean,r_round))+' +/- '+str(np.round(cv_r2_std,r_round))
        
        cv_mean_absolute_error_mean = scores['test_neg_mean_absolute_error'].mean()
        cv_mean_absolute_error_mean = np.absolute(cv_mean_absolute_error_mean)
        cv_mean_absolute_error_std = scores['test_neg_mean_absolute_error'].std()*2
        cv_mean_absolute_error_str = str(np.round(cv_mean_absolute_error_mean,mae_round)) +' +/- '+ str(np.round(cv_mean_absolute_error_std,mae_round))
        
        cv_mean_squared_error_mean = scores['test_neg_root_mean_squared_error'].mean()
        cv_mean_squared_error_mean = np.absolute(cv_mean_squared_error_mean)
        cv_mean_squared_error_std = scores['test_neg_root_mean_squared_error'].std()*2
        cv_mean_squared_error_str = str(np.round(cv_mean_squared_error_mean,mse_round))+' +/- '+str(np.round(cv_mean_squared_error_std,mse_round))
        
        y = Y_train
        
        # Grafica 
        plt.figure(figsize=(8, 6), dpi=500)
        
        Title = str(Model) + 'Num_Dim_Red = '+ str(self.Num_red)
        Title_name = str(Model) + 'Num_Dim_Red = '+ str(self.Num_red)  
        
        
        Name_fig = 'Result_'+str(self.Propriety)+'_'+Title_name.replace('.','_') + '_num_data_' +str(len(Y_test)+len(Y_train))  
        a_table  = plt.subplot2grid((40,1),(0,0),colspan = 1,rowspan = 1)
        a_plot = plt.subplot2grid((40,1),(6,0),colspan = 1,rowspan = 34)
        
    
    
        a_plot.scatter(Y_train, predicted,c='royalblue', edgecolors='navy',alpha=0.6,label='Training data')
        a_plot.scatter(Y_test, Y_predicted,c='red', edgecolors='darkred',alpha=0.5,label='Test data')
        a_plot.plot([y.min(), y.max()], [y.min(), y.max()], c='k',ls='--', lw=3)
        a_plot.set_xlabel(self.Propriety)
        a_plot.set_ylabel('Predicted')
        a_plot.legend()
        
        #-------------------- Validation ----------------------# 
       
        Results = [ [cv_r2_str,cv_mean_absolute_error_str,cv_mean_squared_error_str],[r2,mae,mse]]
        Results_labels =['r2','Mean absolute error ','Mean squared error']
        Row_labels = ['Cross val','Test']
        
        a_table.table(cellText=Results, colLabels = Results_labels, rowLabels = Row_labels , cellLoc='center',rowLoc='center',colLoc='center', loc='upper center')
        a_table.set_xticks = ([])
        a_table.set_yticks = ([])
        a_table.axis('off')
        
        a_table.set_title(Title)
        
        dirfile = 'Results/Predict_Experimental/'
        CheckDir(dirfile) #
        
        plt.savefig(dirfile+Name_fig,transparent=True)
        plt.savefig(dirfile+Name_fig+'.svg',transparent=True)
        
        plt.show()
        #------------------------------------- End of the function -----------------------------#
    
    def Feature_importances(self,num_plots):
        # Description: Function that plots the importance of each feature.   
        # Input:
        #   % num_plots: the number of plots, each plot can have no more the 20 descripotrs
        
        #------------------------------------- Previous functions --------------------------------------------------------# 
        def split_Data_Frame_Row(df, wanted_parts=1):
            # Description: Function that split the df  
            # Input:
            #   % wanted_parts: the nomber of dataframes after the spliting
            
            #------------------------- Beginning of the function --------------------#
            num_row = len(df.iloc[:,0])
            Num_row_df =  np.rint(num_row/wanted_parts)
            df_list = list()
            df_i =pd.DataFrame()
            count = 0
            for i in range(num_row):
                df_i = df_i.append(df.iloc[i,:])
                count = count+1
                if count == Num_row_df :
                    df_list.append(df_i)
                    df_i = pd.DataFrame()
                    count = 0
                elif i == num_row :
                    df_list.append(df_i)
            return df_list
            #------------------------- End of the function --------------------#
            
        #------------------------------------- End previous functions --------------------------------------------------------#
         
        #------------------------------------- Beginning of the function -----------------------------#
        ### Directory 
        dirfile = 'Results/Feature_Importance/'
        CheckDir(dirfile) #
        ###
        
        X = self.Data.X
        Y = self.Data.Y
        Descriptors_Name_list = self.Data.Descriptors_Name_list
        
        
        Model = self.Model
        
        
        if self.Num_red != False: ## dimencional reduccion to the name of the descriptors 
            
        
            dim_red_index_descripotr = self.Skbest.get_support(indices=True) 
            Descriptors_Name_list = list(Descriptors_Name_list[dim_red_index_descripotr])
        
            
            #row = len(X[:,1])
            #column = len(X[1,:])
            
            #Arrey_Descriptors =  np.empty((row,column),dtype = 'object')
            
            #Arrey_Descriptors_r = .Skbest(Arrey_Descriptors)
            #Descriptors_Name_list = list(Arrey_Descriptors_r[0,:])
            
        if len(Descriptors_Name_list) < 20:
            num_barras = self.Num_red
        else:
            num_barras = 20 ### numero de barras por grafica
            print('num_barras'+str(num_barras))
        
        
        output = self.CrossVal 
        
        ### 
        for idx,estimator in enumerate(output['estimator']):
            if idx == 0:   
                feature_importances = pd.DataFrame(estimator.feature_importances_,
                                               index = Descriptors_Name_list,
                                                columns=[str(idx)]) #.sort_values('importance', ascending=False)
            else:
                feature_importances[str(idx)] = estimator.feature_importances_
              
        ###
        importances_arrey = np.array(feature_importances)
        
        importances = np.mean(importances_arrey, axis=1)
        print(importances.shape)
        importances_std = np.std(importances_arrey, axis=1)*2
        print(importances_std.shape)    
        #### Crea el data frame
        Max = np.max(importances+importances_std)
        Max_2 = np.max(importances)
        Min = np.min(importances-importances_std)
        importances = list(importances)
        importances_std = list(importances_std)
        
        
        df_Coefs = pd.DataFrame({'Descriptor_name':Descriptors_Name_list,'Importances':importances,'sigma':importances_std})
        Name_reduction_inf = '_Num_red_'+str(self.Num_red)+'_'
        NameTable = dirfile + 'FI_'+str(self.Propriety)+str(Model)+ Name_reduction_inf +'.txt'
        
        wanted_parts = math.ceil(len(df_Coefs.iloc[:,:])/num_barras) ## number of plots
        ####
        df_Coefs.sort_values('Importances',ascending = False,inplace=True)
        df_Coefs.to_csv(NameTable) ## save the table 
        
        df_list = split_Data_Frame_Row(df_Coefs, wanted_parts)
        #print(df_list)
        numplot = 0
        
        
        
        current_plot = 0 
        for df_coefs in df_list:
            current_plot += 1
            if num_plots - current_plot > 0:  
                  Coefs = df_coefs.Importances
                  Coefs = Coefs.values.tolist()
                  
                  Sigma = df_coefs.sigma
                  
                  Name = df_coefs.Descriptor_name
                  x = np.arange(len(Coefs))
                  x = x.tolist()
                  ######## Grafica
                  plt.figure(dpi= 500)
                  plt.bar(x,Coefs,yerr=Sigma)
                  
                  Name_plt = 'Feature importances \n '+self.Propriety +'\n'+str(Model)+'\n '+ Name_reduction_inf
                  Name_fig = 'FI_Error'+self.Propriety +str(numplot)+'_'
                  plt.title(Name_plt)
                  plt.xlabel('Descriptor')
                  plt.ylabel('Feature importances')
                  plt.xticks(range(len(x)), Name, rotation=90)
                  plt.ylim(Min,Max)
                  plt.xlim(-1,num_barras)
                  plt.savefig(dirfile+Name_fig+str(Model)+Name_reduction_inf,transparent=True)
                  plt.savefig(dirfile+Name_fig+str(Model)+Name_reduction_inf+'.svg',transparent=True)
                  plt.show()
                  numplot += 1
                  ######### Grafica 2
                  plt.figure(dpi= 500)
                  plt.bar(x,Coefs)
                  
                  Name_plt = 'Feature importances \n '+self.Propriety +'   \n'+str(Model)+'\n '+ Name_reduction_inf
                  Name_fig = 'FI_'+self.Propriety +str(numplot)+'_'
                  plt.title(Name_plt)
                  plt.xlabel('Descriptor')
                  plt.ylabel('Feature importances')
                  plt.xticks(range(len(x)), Name, rotation=90)
                  plt.ylim(0,Max_2)
                  plt.xlim(-1,num_barras)
                  plt.savefig(dirfile+Name_fig+str(Model)+Name_reduction_inf,transparent=True)
                  plt.savefig(dirfile+Name_fig+str(Model)+Name_reduction_inf+'.svg',transparent=True)
                  plt.show()
                  
         #------------------------------------- End of the function -----------------------------#
    
    def predict(self,X,nm = False):
        # Description: Function that predict the values using all the data set (without sopliting). In order to use this function the .fit metoda have to ben used previosuly  
            # Input:
            #   % X: featueres for an example this have to be an a 1D arrey with the total number of features with which the model trained previously  
        
        #------------------------- Beginning of the function --------------------#
        Model_treined = self.Model 
        
        if self.Num_red != False:
           
            X = self.Skbest.transform(X)
                   
        self.Predict_list = Model_treined.predict(X)
        
        index = 1
        for Predict_val in self.Predict_list:
            
            print('#' + str(index))
            print(str(self.Propriety)+':\t'+str(Predict_val))
            print('[nm]'+':\t'+str(1240/Predict_val))
            
            index += 1 
        if nm == True:
            Predict_val = 1240/Predict_val
            self.Predict_list = list(1240/np.array(self.Predict_list))
        
        return Predict_val
        #------------------------- End of the function --------------------#

        #------------------------------------- End of the function ---------------------------
    #------------------------------------- End of the class ---------------------------
    
        
##------------------- Model_Cell class ------------------------ #


#---------- Required functions for this class -------------------------#

def Plot_Violin(df,Values,key,Metric):
   
    
    Title = 'Violin_Plots'
    
    
    ### Directory 
    dirfile = 'Results/Validation_Plots/'
    CheckDir(dirfile) #
    ###
    
    Name = dirfile +'Violin_Plots' +'_'+ Metric+'_'+ key 

    Data_Violin = list()
    labels = ['']
    for Value in Values:
        df_v = df[df[key] == Value]
        
        y = np.array(df_v[Metric])
        Data_Violin.append(y)
        labels.append(str(Value))
    
    fig = plt.figure(dpi=500)
    
    parts = plt.violinplot(Data_Violin,showmeans=True)
    #### Formato 
    ### Relleno 
    Total = len(Data_Violin)
    Inc = round(300/Total)
    i = 0
    for pc in parts['bodies']:
        color_edge = mpl.cm.winter(i)
        color_L = list(color_edge)
        color_L[3] = 0.3 ### redefine el valor del alafa p
        color_fille = tuple(color_L)

        pc.set_facecolor(color_fille)
        pc.set_edgecolor(color_edge)  
        #pc.set_alpha(0.5)
        i = i + Inc
    ### barras
    pc = parts['cmins']
    pc.set_edgecolor('black')
    pc.set_linewidth(1.5)
    pc = parts['cmaxes']
    pc.set_edgecolor('black')
    pc.set_linewidth(1.5)
    pc = parts['cbars']
    pc.set_edgecolor('black')
    pc.set_linewidth(1.5)
    pc = parts['cmeans']
    pc.set_edgecolor('black')
    pc.set_linewidth(2.5)

    plt.title(Title)
    plt.ylabel(Metric)
    plt.xlabel(key)
    plt.xticks(np.arange(len(labels)),labels)
   
    plt.savefig(Name+'.png',transparent=True)
    plt.savefig(Name+'.svg',transparent=True)
    plt.show()



#### Model inf 
Dic_parameters_list = list()
Model_names =['RF','GB']

Dic_parameters = {}
#Dic_parameters = {}
Dic_parameters_list.append(Dic_parameters)

Dic_parameters = {}
#Dic_parameters = {}
Dic_parameters_list.append(Dic_parameters)



def str_to_model(Str_model,kwargs):
      
    if Str_model =='LinearRegression':
        Model = LinearRegression(**kwargs)
    elif Str_model =='SVR':
        Model = SVR(**kwargs)
    elif Str_model =='DecisionTreeRegressor':
        Model = DecisionTreeRegressor(**kwargs)    
    elif Str_model =='BayesianRidge':
        Model = BayesianRidge(**kwargs)
    elif Str_model =='SGDRegressor':
        Model = SGDRegressor(**kwargs)
    elif Str_model =='GradientBoostingRegressor':
        Model = GradientBoostingRegressor(**kwargs)
    elif Str_model =='GaussianProcessRegressor':
        Model = GaussianProcessRegressor(**kwargs)
    elif Str_model =='KernelRidge':
        Model = KernelRidge(**kwargs)
    elif Str_model =='MLPRegressor':
        Model = MLPRegressor(**kwargs)    
    elif Str_model == 'RandomForestRegressor':
        Model = RandomForestRegressor(**kwargs)
    elif Str_model == 'LogisticRegression':
        Model = LogisticRegression(**kwargs)
    
    return(Model)
#---------- End of required functions for this class ------------------------- 

#------------------------------------- Beginning of the class ---------------------------
class ModelCell():
   
    def __init__(self,Data,Type_Model_list,ParametersDic_list,Dim_Red_list,cv): 
        
        
        self.Type_Model_list = Type_Model_list   
        self.Dic_parameters_list = ParametersDic_list    
        ###
        self.Model_cell = list() ## Contain diferents types models 
        for i in range(len(self.Type_Model_list)):
            
            Model_name = self.Type_Model_list[i]
            Dic_parameters = self.Dic_parameters_list[i]
            
            Parameters = ParameterGrid(Dic_parameters)
            
            Model_list = list() ## contain differents hiperparameters  
            for kwargs in Parameters:
                Model_dr_list = list() 
                Model = str_to_model(Model_name,kwargs)
                
                for Dim_Red in Dim_Red_list: 
                    
                    model_dr = Model_(Data,Model,Dim_Red,cv)
                    Model_dr_list.append(model_dr)
                
                Model_list.append(Model_dr_list)
                
            self.Model_cell.append(Model_list)   
        
        self.Validation = Validation_cell_()
    
    def SaveClass(self,file):
        rute = 'ModelCell_Classl/'
        namefile = rute + file 
        ### Write data
        f = open(namefile,'wb')
        dill.dump(self,f)
        f.close()    
    
    def CrossValidation(self):
        
        print('-----------------Validacion-----------------')
        for model_list in self.Model_cell: 
            for model_dr_list in model_list:
                for model_dr in model_dr_list:
                    
                    print('Model:\t'+ str(model_dr.Model))
                    print('Num_dimetional reduction:\t'+ str(model_dr.Num_red))
                    print('Cross Validation Iters:\t'+str(model_dr.cv))
                    model_dr.validation()
            
                    self.Validation.appendResults(model_dr)
        self.Validation.Data_frame_validation_generator()
            
    def PlotValidation(self):
        df = self.Validation.df_validation
        
        key = 'Num dimecional reduction'
        Values = list(df[key].unique())
    
        ###r2
        Metric = 'Test r2'
        y_std = 'Test r2 std '
        
        Plot_Violin(df,Values,key,Metric)
        
        ###MAE
        Metric = 'Test mean absolute error'
        y_std = 'Test mean absolute error std '
        
        Plot_Violin(df,Values,key,Metric)
        ###RMSR
        Metric = 'Test mean squared error'
        y_std = 'Test mean squared error std '
        
        Plot_Violin(df,Values,key,Metric)
    
    def PlotResult(self,r_round = 3, mae_round = 1,mse_round = 0):
        print('-----------------Tests-----------------')
        for model_list in self.Model_cell: 
            for model_dr_list in model_list:
                for model_dr in model_dr_list:
                    
                    print('Model:\t'+ str(model_dr.Model))
                    print('Num_dimetional reduction:\t'+ str(model_dr.Num_red))
                    print('Cross Validation Iters:\t'+str(model_dr.cv))
                    model_dr.Plot_Results( r_round = 3, mae_round = 1,mse_round = 0)
    
    def PlotFeatureImportance(self,num_plots):
        print('-----------------Feature Importance-----------------')
        for model_list in self.Model_cell: 
            for model_dr_list in model_list:
                for model_dr in model_dr_list:
                    
                    print('Model:\t'+ str(model_dr.Model))
                    print('Num_dimetional reduction:\t'+ str(model_dr.Num_red))
                    print('Cross Validation Iters:\t'+str(model_dr.cv))
                    model_dr.Feature_importances(num_plots)
        
                
#------------------------------------- End of the class ---------------------------        
     
         
## ----------------------- Class validation----------------------------------------#

#---------- Required functions for this class -------------------------
def list_invert(Lista):
  ### Definition: Function that change the order from nested function   
  B = list()
  for b in range(len(Lista[0])): 
    A = list()  
    for a in range(len(Lista)):
      A.append(Lista[a][b])
    B.append(A)
  return B



#------------------------------------- Beginning of the class ---------------------------
class Validation_cell_():
    # Class that keep the validation results from each model with it's corresponding dimentional reduction

    def __init__(self):
        
        self.Names = list() ## model Name 
        self.Time = list()  ## Training time 
          
        self.Train_r2 = list() 
        self.Train_mean_absolute_error = list()
        self.Train_mean_squared_error = list()
        
        self.Train_r2_std = list() 
        self.Train_mean_absolute_error_std = list()
        self.Train_mean_squared_error_std = list()
        
        self.Test_r2 = list() 
        self.Test_mean_absolute_error = list()
        self.Test_mean_squared_error = list()
        
        self.Test_r2_std = list() 
        self.Test_mean_absolute_error_std = list()
        self.Test_mean_squared_error_std = list()
    
        self.Num_Dimens_Reduct = list()
        
    
    def appendResults(self,model):
        
        ### Parameters
        self.Names.append(str(model.Model))
        self.Time.append(model.DT)
        self.Num_Dimens_Reduct.append(model.Num_red)   
        
        
        ### Results 
        self.Train_r2.append(model.cv_r2_mean_train) 
        self.Train_r2_std.append(model.cv_r2_std_train) 
        
        self.Train_mean_absolute_error.append(model.cv_mean_absolute_error_std_train)
        self.Train_mean_absolute_error_std.append(model.cv_mean_absolute_error_std_train) 
        
        self.Train_mean_squared_error.append(model.cv_mean_squared_error_mean_train) 
        self.Train_mean_squared_error_std.append(model.cv_mean_squared_error_std_train) 
        
        self.Test_r2.append(model.cv_r2_mean_test) 
        self.Test_r2_std.append(model.cv_r2_std_test) 
        
        self.Test_mean_absolute_error.append(model.cv_mean_absolute_error_mean_test) 
        self.Test_mean_absolute_error_std.append(model.cv_mean_absolute_error_std_test) 
      
        self.Test_mean_squared_error.append(model.cv_mean_squared_error_mean_test) 
        self.Test_mean_squared_error_std.append(model.cv_mean_squared_error_std_test)
    
         
    def Data_frame_validation_generator(self):
        
        #--- Creat the dataframe with the corresponding validation metrics ---#
        df_list = list() 
        
        df_list.append(pd.DataFrame({'Model name':self.Names,'Trainer time [s]':self.Time}))
        df_list.append(pd.DataFrame({'Train r2':self.Train_r2,'Train r2 std ':self.Train_r2_std}))
        df_list.append(pd.DataFrame({'Train mean absolute error':self.Train_mean_absolute_error,'Train mean absolute error std ':self.Train_mean_absolute_error_std}))
        df_list.append(pd.DataFrame({'Train mean squared error':self.Train_mean_squared_error,'Train mean squared error std ':self.Train_mean_squared_error_std}))
        df_list.append(pd.DataFrame({'Test r2':self.Test_r2,'Test r2 std ':self.Test_r2_std}))
        df_list.append(pd.DataFrame({'Test mean absolute error':self.Test_mean_absolute_error,'Test mean absolute error std ':self.Test_mean_absolute_error_std}))
        df_list.append(pd.DataFrame({'Test mean squared error':self.Test_mean_squared_error,'Test mean squared error std ':self.Test_mean_squared_error_std}))
        df_list.append(pd.DataFrame({'Num dimecional reduction':self.Num_Dimens_Reduct}))
        
        
        self.df_validation = pd.DataFrame()
        for df in df_list :
            self.df_validation = pd.concat([self.df_validation,df], axis=1)

#------------------------------------- End of the class ---------------------------

#------------------------------ Mol Data class-----------------------------------------------------#

#------------------------------------- Beginning of the class ---------------------------
class MolData_:  # One molecule Data class
    
    def __init__(self,mol,solvent = False ,Indx = False):
        
        self.Indx = Indx
        
        self.mol = mol
        self.solvent = solvent
        
    def Structure_generator(self):
        
        ## Definicion de variable 
        Name = str(self.mol)   ## molecule name
        print('Molecule:\t'+str(Name))
        ## Generacion de la estrucutra
        self.mol2D = Chem.MolFromSmiles(self.mol) 
        
        self.mol2D.SetProp("_Name",Name) ## append the nombre
        
        mol3D = Chem.AddHs(self.mol2D) ## creat the 3D strucure
        
        params = Chem.ETKDG() ## optimizacion parameter
        params.useRandomCoords=True 
        
        self.mol3D = mol3D
        
        ## Solvent 
    def Structure_generator_solv(self,Solv= False):
        
        
        if Solv != False:
            self.solvent = Solv
        
        
        Name = str(self.solvent)   ## molecule name 
        print('Solvent:\t'+str(Name))
        
        ## Generacion de la estrucutra
        mol2D = Chem.MolFromSmiles(self.solvent) 
        mol2D.SetProp("_Name",Name) ##  append the nombre
        
        mol3D = Chem.AddHs(mol2D) ## Creat the 3D strucure
        
        params = Chem.ETKDG() ## Optimizacion parameter
        params.useRandomCoords=True 
        
        self.solv3D = mol3D
        
    def Descriptor_calculator(self,descriptor_list,descripotrs_incluided = 'all'):
        
        
        if descripotrs_incluided == 'all':
            Descriptors = List_Descriptor_all()  # Creat the descriptor tuple 
        else:
            Descriptors = List_Descripotr_genarate(descripotrs_incluided)
        
        mol = self.mol3D
        
        df = pd.DataFrame()
        
        #### Calculus of molecular descriptors 
        #print('---------------Calculing molecule descriptors ------------------')
        for Descriptor in Descriptors: ### itera sobre los descriptores 
            Name_Descriptor = str(Descriptor)
            if Search_descriptor(Descriptor,descriptor_list):
                try:
                    Descriptor_Value = Descriptor(mol) ### Calcula el valor del descriptor para cada molecula 
                except:
                    Descriptor_Value = np.nan ## si no lo puede calcular  nan
                
                df[Name_Descriptor] = [Descriptor_Value]
        
        self.df_descriptors_mol = df
        self.df_descriptors = df 
        
    def Descriptor_calculator_solv(self,descriptor_list,descripotrs_incluided = 'all'):
        
        
        if descripotrs_incluided == 'all':
            Descriptors = List_Descriptor_all()  # Creat the descriptor tuple 
        else:
            Descriptors = List_Descripotr_genarate(descripotrs_incluided)
        
        
        solvent = self.solv3D
        
        df = self.df_descriptors_mol
         
        #### descriptor solvent of solvent descriptors
        #print('---------------Calculing solvent descriptors ------------------') 
 
        for Descriptor in Descriptors: ### itera sobre los descriptores 
            Name_Descriptor = 'Solv_'+str(Descriptor)
           
            if Search_descriptor(Name_Descriptor,descriptor_list):
                try:
                    Descriptor_Value = Descriptor(solvent) ### Calcula el valor del descriptor para cada molecula 
                except:
                    Descriptor_Value = np.nan ## si no lo puede calcular  nan
                
                df[Name_Descriptor] = [Descriptor_Value]
                
        self.df_descriptors = df    
        
    def Preprosesing(self,Standar_Scalar = False, Imputer = False):
        
        X = self.df_descriptors
        #print('---------------Calculing solvent descriptors ------------------')
        
        
        imputer = SimpleImputer(strategy = 'constant', fill_value = 0)
        X = imputer.fit_transform(X)
        
        ### imputer
        if Imputer != False:
            if Imputer == 'Nearest_neighbors':
                imp = KNNImputer(n_neighbors=2, weights="uniform")
                X = imp.fit_transform(X)
            elif Imputer == 'Simple_Imputer':
                imp = SimpleImputer(missing_values=np.nan, strategy='mean')
                imp.fit(X)
                X = imp.transform(X)
                
         ## Scalar
        if Standar_Scalar != False:
            X = Standar_Scalar.transform(X)
       
            
        self.X = X
        
    def Check(self):
        X = self.X
        for i in range(len(X)):
            x_i = X[0,i]
            if np.isnan(x_i):
                print(x_i)
                X[0,i] = 0
            if np.isinf(x_i):
                print(x_i)
                X[0,i] = 0
        self.X = X

        
        