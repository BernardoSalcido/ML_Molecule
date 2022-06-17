# -*- coding: utf-8 -*-
"""
Created on Mon May 16 15:16:31 2022

@author: Alumnos
"""

import tkinter as tk 
from tkinder import *
from PIL import ImageTk, Image ## install pillow 
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, 
NavigationToolbar2Tk)
#%%
#------------------- Previosus function 
def files():
    root.filename = filedialog.askopenfilename(initialdir = 'C:/Users/Alumnos/Documents/',title ="Select a File",filetypes = (("Molecules database","*.sdf"),("Molecule","*.mol"),("all files",".**")))
    
def Calculate():
    
    Predictor = Predictor_ML_Molecule(DataSet = 'Chromophore_500')
    Predictor.Calculate(mol,solvent,nm = True)
    
    
    
    FWHM_abs = 45
    FWHM_em = 50
    w_min = 300
    w_max = 900
    
        
    abs_list = Predictor.model_abs.Predict_list
    em_list = Predictor.model_em.Predict_list
    

    for i in range(len(abs_list)):
        
        mol_i = Predictor.mol[i]
        abs_i = abs_list[i]
        em_i = em_list[i]    
    
    SimulateSpectraTk(abs_i,em_i,FWHM_abs,FWHM_em,w_min,w_max,mol_i,solvent = solvent)

def SimulateSpectraTk(w_abs,w_em,FWHM_abs,FWHM_em,w_min = False, w_max=False,mol= False,solvent = False, Save = False):
    
    wavelengt = np.linspace(200,1000,1000)
    
    a = 0.5/(FWHM_abs*2)
    
    f = 0.5 #
    ### Absortion
    I_abs = lineshape(wavelengt,FWHM_abs,w_abs,f,a) 
    I_abs = I_abs /np.max(I_abs) ## Normalize 
    ### Emission
    
    a = 0.5/(FWHM_em*2)
    I_em = lineshape(wavelengt,FWHM_em,w_em,f,-a) 
    I_em = I_em /np.max(I_em) ## Normalize 
    ### Figure
    fig, ax = plt.subplots(dpi = 300)
    ax.plot(wavelengt, I_abs,label='Absortion  ('+str(round(w_abs))+' nm)')
    ax.plot(wavelengt, I_em,label='Emission  ('+str(round(w_em))+' nm)')
    
    if w_max != False:
        
        ax.set_xlim((w_min,w_max))
    
    if mol != False:
        
        Draw.MolToFile(mol.mol2D,'TEMP/image_mol.png')  
        
        
        Molecule = plt.imread('TEMP/image_mol.png')
        
        
        imagebox = OffsetImage(Molecule,zoom=0.4)
        imagebox.image.axes = ax
        
        XY = [w_em+(FWHM_em*2.2),0.65]
        add_Molecule = AnnotationBbox(imagebox,XY,xybox = (30.,-30.),frameon =False,xycoords='data',boxcoords = "offset points",pad=0.5)
        ax.add_artist(add_Molecule)
    

    ax.set_xlabel('Wavelength [nm]')
    ax.set_ylabel('Intencity [a.u.]')
    plt.title('Absoriton an emission spectra in '+str(solvent))
    ax.legend()
    
    # creating the Tkinter canvas
    # containing the Matplotlib figure
    canvas = FigureCanvasTkAgg(fig, master = frame_Plot)  
    canvas.draw()
  
    # placing the canvas on the Tkinter window
    canvas.get_tk_widget().pack()
  
    # creating the Matplotlib toolbar
    toolbar = NavigationToolbar2Tk(canvas,frame_Plot)
    toolbar.update()
  
    # placing the toolbar on the Tkinter window
    canvas.get_tk_widget().pack()
  
    
##-------------------------------------
root = tk.Tk()
root.title('Simulate absortiona and emission spectra \n using Machine Learning')
#root.iconbitmap('Icon.ico')
#root.geometry("500x500")
#root.rowconfigure(0, minsize=800, weight=1)
#root.columnconfigure(1, minsize=800, weight=1)

## ---------------------------- Grid 
frame_settings = tk.LabelFrame(root, text ="Settings...",relief=tk.RAISED, bd=1)
frame_settings.grid(row = 0,column=0, padx=50, pady=50)

frame_Plot = tk.LabelFrame(root,text ="Plots.....",relief=tk.RAISED, bd=1)
frame_Plot.grid(row = 0,column=1, padx=150, pady=50)

####-------------------------- Import


label_import_1 = tk.Label(frame_settings,text="Molecule structure")
label_import_1.pack()

entry_smile = tk.Entry(frame_settings,text ="SMILE")
entry_smile.pack()

mol = entry_smile.get() 

label_import_2 = tk.Label(frame_settings,text=" or \n open the '.sdf', '.mol' or '.csv' files ")
label_import_2.pack()

FileOpen_buttom = tk.Button(frame_settings,text="Open file",command = files)
FileOpen_buttom.pack()

label_import_3 = tk.Label(frame_settings,text="\n ")
label_import_3.pack()
### ----------------------------- Options 

label_Options_2 = tk.Label(frame_settings,text = "Solvent structure")
label_Options_2.pack()

entry_smile_solvent = tk.Entry(frame_settings,text ="Solvent SMILE")
entry_smile_solvent.pack()

label_Options_3 = tk.Label(frame_settings,text = "or \n Select Solvent")
label_Options_3.pack()

# Dropdown menu options
options = [
    "Methanol",
    "Acetone",
    "Ethanol",
    "Acetonitrile",
    "DMSO",
    "Water",
    "Cyclohexane"
]
  
# datatype of menu text
clicked = tk.StringVar()
  
# initial menu text
clicked.set( "Methanol" )
  
# Create Dropdown menu
drop = tk.OptionMenu(frame_settings , clicked , *options )
drop.pack()

solvent = entry_smile_solvent.get() 


### ----------------------------- Plots 

Calculate_buttom = tk.Button(frame_Plot,text="Calculate",command = Calculate)
Calculate_buttom.pack()

root.mainloop()

