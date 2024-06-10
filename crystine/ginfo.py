import pandas as pd
import re
import numpy as np
import sys
import argparse


def ret_parser():
    parser = argparse.ArgumentParser(
        description="Extracts info from your OUTCAR file")
    parser.add_argument(
        "--excel", type=int, default=1, help="1 if you want to write data of VBM CBM of all K-Points in in excel file"
    )
    return parser

def process_file(input_file, output_file,excel_gen):
    # Read the text file into a list of lines

    #change this according to your OUTCAR
    # ideal_VB_occupation= int(input('What is the ideal occupation of Valence Band in your system:\n')) 
    # ideal_VB_occupation= 2

    file = open(input_file, 'r')

    lines = iter(file)

    # for line in lines:
    #     print (line)

    # with open(input_file, 'r') as file:
    #     lines = file.readlines()

    # Initialize lists to store relevant data
    k_points = np.array([])
    band_numbers = np.array([])
    band_energies = np.array([])
    occupations = np.array([])
    start_now=0
    
    # Loop through the lines and extract data
    print("Reading OUTCAR file..")
    
    for line in lines:

        if (start_now==0):
            '''
            this part runs till the E-Fermi line is found
            E-fermi :  -1.7372     XC(G=0):  -3.5696     alpha+bet : -2.8600

            '''
            if (' E-fermi : ' in line):
                start_now=1
	        #extract all the numbers from the string and convert them into a list
             
                s1 = [float(s) for s in re.findall(r'-?\d+\.?\d*',line)]
                #CHECK HOW AND THIS WORK OR BETTER WAY TO DO IT   

                print("Fermi Energy = ",s1[0]) 
                print("Reading K-Points..")

                while('k-point' not in line):   
                    # to jump to the K-Points line               
                    line = next(lines)

            else:
                # Since E-Fermi still not found continue
                continue

        if 'k-point' in line:
            # fetch info from k-points line
            #  k-point     1 :       0.0000    0.0000    0.0000
            k_point = line.split(':')[0].split()[-1]
            sys.stdout.flush()
            sys.stdout.write('\rK-Points Read ' + k_point)
            sys.stdout.flush()

        elif ('band' in line) or (line.split()==[]):
            # print(line.split())
            #skip the band ... line 
            # line = next(lines)
            continue 
        elif '-------------------' in line:
            # got to the end
            break
        else:
            values = line.split()
            k_points= np.append(k_points,k_point)
            band_numbers=np.append(band_numbers,values[0])
            band_energies=np.append(band_energies,values[1])
            occupations=np.append(occupations,values[2])


    # Create a DataFrame from the extracted data
    df = pd.DataFrame({
        'K-Point': k_points,
        'Band Number': band_numbers,
        'Band Energy': band_energies,
        'Occupation': occupations
    })


    # this dataframe enteries of all the bands of all the k-ponints 

    # Convert columns to numeric type
    df[['K-Point', 'Band Number', 'Band Energy', 'Occupation']] = df[['K-Point', 'Band Number', 'Band Energy', 'Occupation']].apply(pd.to_numeric, errors='coerce')
    ideal_VB_occupation= int(df.loc[:,"Occupation"].max())

    df.to_csv('doexcel_files.csv',  index=False,header=True)

    # now we need to extarct only those data which belong to VBM , CBM of that particular K-Point

    filtered_df = df[ (( (df['Occupation']==ideal_VB_occupation     )   &    (df['Occupation'].shift(-1) != ideal_VB_occupation))  |  ( (df['Occupation']==0   )   &    (df['Occupation'].shift(1) != 0))) ].copy()


    band_gap = np.array([])

    counting=0
    
    #-ve=VBM 
    
    VB_Max=-999999
    CB_Min=999999



# K-Point	Band Number	Band Energy	Occupation	Band Gap
    for i in range(filtered_df.shape[0]) :


        if filtered_df.iloc[i].at["Occupation"]==ideal_VB_occupation:
            VB=filtered_df.iloc[i].at["Band Energy"]
            CB=filtered_df.iloc[i+1].at["Band Energy"]

            band_gap_temp=CB-VB

            if(VB_Max<VB):
                VB_Max=VB
                VBM_KP=int(filtered_df.iloc[i].at["K-Point"])

            if(CB_Min>CB):
                CB_Min=CB
                CBM_KP=int(filtered_df.iloc[i+1].at["K-Point"])



            band_gap=np.append(band_gap,band_gap_temp)
        else:
            band_gap=np.append(band_gap,None)
        counting+=1

    filtered_df['Band Gap'] = band_gap

    if( CBM_KP == VBM_KP ):
        typee="Direct"
    else:
        typee="Indirect"


    onee= 'Band Gap = ' +  str(round(CB_Min-VB_Max,4)) 
    twoo='(CBM at '+ str(CBM_KP) + ') CBM = '+ str(CB_Min) + " eV"
    threee='(VBM at '+ str(VBM_KP)+ ') VBM = '+str(VB_Max) + " eV"

    data = {'K-Point': ['---'], 'Band Number': [twoo], 'Band Energy': [threee], 'Occupation': [typee], 'Band Gap': [onee]}
    new_df = pd.DataFrame(data)

    # simply concatenate both dataframes
    filtered_df = pd.concat([new_df, filtered_df]).reset_index(drop = True)

    result = pd.concat([new_df,filtered_df])
    # print(result)	

    print("")
    print("___________________________")
    print(twoo)
    print(threee)
    print("Band Gap Type : ",typee)
    print(onee)
    
    # excel_gen = int(input('Should I write data of VBM CBM of all K-Points in in excel file (1 or 0)?\n'))
    if(excel_gen==1):
        filtered_df.to_excel(output_file+".xlsx", index=False, engine='openpyxl')
        print(output_file+".xlsx"+" file written.")


    file = open(output_file+".log","w")
    file.write(   str( "Band Gap Type : "+typee +'\n' + onee +'\n'+ twoo +'\n'+ threee    )  )
    file.close()
    print(output_file+".log file written.")


def main():
    #main starts from here 

    print("___________________________")
    print("|  Welcome, I'm Crystine   |")
    print("___________________________")


    input_file_path = './OUTCAR'  # Replace with your actual file path
    #later rename the above file to OUTCAR
    output_file_path = "ginfo"
    args = ret_parser().parse_args()
    process_file(input_file_path, output_file_path,excel_gen=args.excel)

if __name__ == "__main__":
    main()