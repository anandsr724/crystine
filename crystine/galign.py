import matplotlib
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import re
import sys
import argparse


def get_text_color(bg_color):
    r, g, b = matplotlib.colors.to_rgb(bg_color)
    luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b)
    return 'black' if luminance > 0.5 else 'white'


def process_file(input_file_path , 
                 file_type,   output_file_path,
                 vb_color, cb_color,
                 width , height):
  # Reading file names and type 
  # path=str(input("Enter the path of the excel file:\n"))
  # file_name=str(input("Enter name of Output file:\n"))
  # file_type = 'excel'  # excel or txt or dat

  # Converting the data into a dataframe
  if file_type == 'excel':
    df = pd.read_excel(input_file_path) 
  else:
    df = pd.read_csv(input_file_path, sep=r'\s+')

  df.columns = df.columns.str.upper()   # convert column to all caps

  COF_names = df['NAME'].tolist()
  COF_vbms = df['VBM'].tolist()
  COF_cbms = df['CBM'].tolist()

  if COF_cbms[0]   >  COF_vbms[0]:   # normal case 
    invert = 0
  else:
    invert = 1                      # invert the plot , i.e. negative y if upside

  inside_text_col="black"

  vb_text_color=get_text_color(vb_color)
  cb_text_color=get_text_color(cb_color)

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
      rect2 = matplotlib.patches.Rectangle((x_loc, y_loc), width_bar , ymax-edge_val,edgecolor='black',facecolor=vb_color, linewidth=1)  #vbm#3C5488
      plt.text(x_loc+0.5, edge_val +0.5, 'VBM='+str(round(vbm,2)), fontsize = 10*bar_font,horizontalalignment='center',c=vb_text_color) 

      #lower rectangle  (CBM) -> in case of invert == CBM (since will be inveted later , and will come on top)
      edge_val=cbm
      x_loc=width_bar*i
      y_loc=ymin
      ht=edge_val-ymin
      rect1 = matplotlib.patches.Rectangle((x_loc, y_loc), width_bar , ht  ,edgecolor='black',facecolor=cb_color, linewidth=1)  #vbm#3C5488  
      plt.text(x_loc+0.5, edge_val-0.5, 'CBM='+str(round(cbm,2)), fontsize = 10*bar_font,horizontalalignment='center',c=cb_text_color) 

      plt.text(width_bar*(i)+0.5, vbm+ (cbm-vbm)/2, 'Eg='+str(round(cbm-vbm, 2)), fontsize = 10*bar_font,horizontalalignment='center') 
    else:
      #upper rec     (CBM)
      edge_val=cbm
      x_loc=width_bar*i
      y_loc=edge_val
      ht=ymax-edge_val
      rect2 = matplotlib.patches.Rectangle((x_loc, y_loc), width_bar , ymax-edge_val,edgecolor='black',facecolor=cb_color, linewidth=1)  #vbm#3C5488
      plt.text(x_loc+0.5, edge_val+0.5, 'CBM='+str(round(cbm,2)), fontsize = 10*bar_font,horizontalalignment='center',c=cb_text_color) 

      #lower rectangle  (VBM) 
      edge_val=vbm
      x_loc=width_bar*i
      y_loc=ymin
      ht=edge_val-ymin
      rect1 = matplotlib.patches.Rectangle((x_loc, y_loc), width_bar , ht  ,edgecolor='black',facecolor=vb_color, linewidth=1)  #vbm#3C5488  
      plt.text(x_loc+0.5, edge_val -0.5, 'VBM='+str(round(vbm,2)), fontsize = 10*bar_font,horizontalalignment='center',c=vb_text_color) 

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
  plt.savefig(output_file_path,transparent=False,dpi=600)
  # plt.show()

def ret_parser():
    parser = argparse.ArgumentParser(
        description="Extracts info from your OUTCAR file")
    parser.add_argument(
        "--file", type=str, default='txt', help="excel / dat / txt , input file type"
    )
    parser.add_argument(
        "--path", type=str, help="input file path , containing name , and energy levels of VBM and CBM"
    )
    parser.add_argument(
        "--output", type=str, default='galign' , help="output file name"
    )
    parser.add_argument(
        "--vbcol", type=str, default="#b6c8ba" , help="color of the valence band"
    )
    parser.add_argument(
        "--cbcol", type=str, default="#b3cbd3" , help="color of the conduction band"
    )
    parser.add_argument(
        "--w", type=int, default=8, help="width of the plot (df=8)"
    )
    parser.add_argument(
        "--h", type=int, default=5 , help="height of the plot (df=5)"
    )
    return parser

def main():
    #main starts from here 

    print("___________________________________")
    print("|     Hey there, I'm Crystine     |")
    print("___________________________________")

    # output_file_path = "galign"
    args = ret_parser().parse_args()

    process_file(input_file_path = args.path, 
                 file_type=args.file,   output_file_path = args.output,
                 vb_color = args.vbcol  , cb_color=args.cbcol,
                 width = args.w , height = args.h)

if __name__ == "__main__":
    main()