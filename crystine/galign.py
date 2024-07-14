import matplotlib
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd

# Reading the data , invert = Yes or No
path=str(input("Enter the path of the excel file:\n"))
file_name=str(input("Enter name of Output file:\n"))
print("The Y-Axis shall be inverted in case of NHE plots or when CBM < VBM)")
invert=int(input("Invert Y-Axis? (1 or 0)"))

      # Converting the data into a dataframe
df = pd.read_excel(path) # can also index sheet by name or fetch all sheets
COF_names = df['NAME'].tolist()
COF_vbms = df['VBM'].tolist()
COF_cbms = df['CBM'].tolist()

vb_col="#b6c8ba"
cb_col="#b3cbd3"
inside_text_col="black"
width_bar=1
bar_font=0.7

if invert==1:
  ymax = max(COF_vbms)+2
  ymin = min(COF_cbms)-2
else:
  ymin = min(COF_vbms)-2
  ymax = max(COF_cbms)+2

fig = plt.figure() 
ax = fig.add_subplot(111) 

# plt.style.use('ggplot')

n=len(COF_names)
my_xticks=[]

for i in range(0,n):

  vbm= COF_vbms[i]
  cbm= COF_cbms[i]
  name= COF_names[i]
  mid=i-0.5
  
  if invert ==1:
    #upper rec     (VBM)
    edge_val=vbm
    x_loc=width_bar*i
    y_loc=edge_val
    ht=ymax-edge_val
    rect2 = matplotlib.patches.Rectangle((x_loc, y_loc), width_bar , ymax-edge_val,edgecolor='black',facecolor=vb_col, linewidth=1)  #vbm#3C5488
    plt.text(x_loc+0.5, edge_val +0.5, 'VBM='+str(round(vbm,2)), fontsize = 10*bar_font,horizontalalignment='center',c=inside_text_col) 

    #lower rectangle  (CBM) -> in case of invert == CBM (since will be inveted later , and will come on top)
    edge_val=cbm
    x_loc=width_bar*i
    y_loc=ymin
    ht=edge_val-ymin
    rect1 = matplotlib.patches.Rectangle((x_loc, y_loc), width_bar , ht  ,edgecolor='black',facecolor=cb_col, linewidth=1)  #vbm#3C5488  
    plt.text(x_loc+0.5, edge_val-0.5, 'CBM='+str(round(cbm,2)), fontsize = 10*bar_font,horizontalalignment='center',c=inside_text_col) 

    plt.text(width_bar*(i)+0.5, vbm+ (cbm-vbm)/2, 'Eg='+str(round(cbm-vbm, 2)), fontsize = 10*bar_font,horizontalalignment='center') 
  else:
    #upper rec     (CBM)
    edge_val=cbm
    x_loc=width_bar*i
    y_loc=edge_val
    ht=ymax-edge_val
    rect2 = matplotlib.patches.Rectangle((x_loc, y_loc), width_bar , ymax-edge_val,edgecolor='black',facecolor=cb_col, linewidth=1)  #vbm#3C5488
    plt.text(x_loc+0.5, edge_val+0.5, 'CBM='+str(round(cbm,2)), fontsize = 10*bar_font,horizontalalignment='center',c='white') 

    #lower rectangle  (VBM) 
    edge_val=vbm
    x_loc=width_bar*i
    y_loc=ymin
    ht=edge_val-ymin
    rect1 = matplotlib.patches.Rectangle((x_loc, y_loc), width_bar , ht  ,edgecolor='black',facecolor=vb_col, linewidth=1)  #vbm#3C5488  
    plt.text(x_loc+0.5, edge_val -0.5, 'VBM='+str(round(vbm,2)), fontsize = 10*bar_font,horizontalalignment='center',c='white') 

    plt.text(width_bar*(i)+0.5, vbm+ (cbm-vbm)/2, 'Eg='+str(round(cbm-vbm, 2)), fontsize = 10*bar_font,horizontalalignment='center') 

  ax.add_patch(rect1) 
  ax.add_patch(rect2)  

  my_xticks.append(name)

x=np.arange(width_bar*0.5,n*width_bar,1*width_bar, dtype=float)

plt.axhline(y = 1, color = 'mediumvioletred', linestyle = '--') 
plt.xticks(x, my_xticks,fontsize=8)
plt.yticks(fontsize=10)
plt.ylabel('E vs $E_{vaccum}$ (eV)')

plt.xlim([0, n*width_bar]) 
plt.ylim([ymin, ymax]) 
  
hori_line=['HER','OER']
if invert==1:
    hori_val=[-4.44,-5.67]
else:
    hori_val=[-4.44,-1.23]

# offset  for lines
y_offset=0.02
for j in range(0,len(hori_line)):
    plt.axhline(y=hori_val[j],alpha=1,linestyle='dashed',linewidth=1)
    # plt.text(width_bar*(i)+width_bar+0.4, hori_val[j] + y_offset ,  hori_line[j] +  '(' + str(hori_val[j]) + ')', fontsize =f,horizontalalignment='center')
    plt.text(width_bar*(i)+width_bar+0.4, hori_val[j] + y_offset ,  hori_line[j] +  '(' + str(hori_val[j]) + ')', fontsize=10*bar_font  ,horizontalalignment='center')

if invert==1:
  plt.gca().invert_yaxis()

plt.tight_layout()
plt.savefig(file_name,transparent=False,dpi=600)
plt.show()