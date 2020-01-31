from __future__ import print_function
import pickle
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

import sys
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import imghdr
#import tensorflow as tf

#%%
#If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']

SPREADSHEET_ID = '1BxlhiJCKj-LllOOIQ3ML0Vsl9umivXPttpBEQ-b__cI'
RANGE_NAME = 'data!A1:L1000'


def getGoogleSheet():

    """Shows basic usage of the Sheets API.
    Prints values from a sample spreadsheet.
    """
    creds = None
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    service = build('sheets', 'v4', credentials=creds)

    # Call the Sheets API
    sheet = service.spreadsheets()
    result = sheet.values().get(spreadsheetId=SPREADSHEET_ID,
                                range=RANGE_NAME).execute()
    values = result.get('values', [])

    '''
    if not values:
        print('No data found.')
    else:
        print('Name, Major:')
        for row in values:
            # Print columns A and E, which correspond to indices 0 and 4.
            print('%s, %s' % (row[0], row[1]))
    ''' 
    return values
   
#%% Renaming only for valid Imgs
DISCREPANT_IMG = 0
ANIMATED_IMG = 0
DECODEERROR_IMG = 0
'''
def checkValidImg(img):
    try:
        with Image.open(img):
            return False
    except:
        return True

def checkAnimated(img):
    with Image.open(img) as f:
        try:
            f.seek(1)
        except EOFError:
            return False
        else: #animated
            return True

def decode_img(ID,IMG_WIDTH,IMG_HEIGHT):
    try:
        img = tf.io.read_file(ID)
        # convert the compressed string to a 3D uint8 tensor       
        img = tf.image.decode_image(img, channels=3)  
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)
        tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])
        return False
    except:
        return True
'''
            
def rename(ret,filename,values):
    '''
    if checkValidImg(filename):
        global DISCREPANT_IMG  
        DISCREPANT_IMG+=1
        ret.append('CANNOT OPEN ' + str(filename))
        return
    if checkAnimated(filename):
        global ANIMATED_IMG
        ret.append('ANIMATED ' + str(filename))
        ANIMATED_IMG+=1
        return
    if decode_img(filename,32,32):
        global DECODEERROR_IMG #not jpg png bmp gif
        ret.append('DECODEERROR ' + str(filename))
        DECODEERROR_IMG+=1
        return
    '''
    values = values[1:]    
    pokename = filename.split('/')[-1]
    pokefoldername = filename.split('/')[-2].replace('/','').lower()
    pokename = pokename.split('.')[0]
    pokename = pokename.split('-')[0]
    pokename = pokename.lower()

    
    for row in values:
        number = int(row[0])
        kor_name = row[1]
        eng_name = row[3].lower()
        
        if pokefoldername in kor_name or pokefoldername in eng_name:                            
            #print(number,kor_name,pokefoldername)
       
            re_pokename = row[0].zfill(3) + '_' + row[1]
            re_pokename.replace(' ','')
          
            return re_pokename#, pokename
   

#%%
def SearchDirectory(ret,directory,filename):
    for t in os.listdir(directory):
        if os.path.isdir(os.path.join(directory,t)):
            SearchDirectory(ret,os.path.join(directory,t),filename)
        else:
            if filename in t :
                ret.append(os.path.join(directory,t))
    return

def IMGtoJPEG(source,outpath):
    with Image.open(source) as im:
        rgb_im = im.convert('RGB')
        rgb_im.save(outpath)
        print(outpath)

def main(argv):
    values = getGoogleSheet() 
    path = argv[1].replace('/','')
    imglist = []
    SearchDirectory(imglist,path,'.png')
    SearchDirectory(imglist,path,'.jpg')
    SearchDirectory(imglist,path,'.gif')

    savepath = path+'_rename/'
    if not os.path.exists(savepath):
            os.mkdir(savepath) 
            
    cnt = 2439 #kaggle data is 2338
    #cnt = 1
    unusable = []
    usable = []
    
    for img in imglist:
        re_pokename = rename(unusable,img,values)
        if re_pokename:
            re_pokename = re_pokename+'_'+ str(cnt).zfill(5)+'.jpeg'
            #IMGtoJPEG(img,os.path.join(savepath,re_pokename))
               
            try:
                IMGtoJPEG(img,os.path.join(savepath,re_pokename))
                usable.append(img)

            except:
                unusable.append('CANNOT CONVERT TO JPEG '+ img)
                print('CANNOT CONVERT TO JPEG ' + str(img))
               
   
            #shutil.copy(img,os.path.join(savepath,re_pokename)) 
            cnt+=1
        else:
            unusable.append('UNVALID POKEMON NAME' + str(img))
                        
                     


    print('total data:',len(imglist),'usable data:',len(usable))
    print('unusable data:', len(unusable))

if __name__ == '__main__':
    if(len(sys.argv) < 1):
        print("Usage: python sourceFolder recFolder recCh")
    elif(len(sys.argv) >= 1):
        main(sys.argv)

