import os
import numpy as np                                                  # Import libraries      date: 12/07/22
from PIL import Image,ImageDraw,ImageFont
import pypylon.pylon as py
import pickle
import h5py
import time as tm
import glob
import matplotlib.pyplot as plt
from PIL import Image 
import time
import sys
import snap7
from snap7 import util
from scipy import signal
from keras.models import load_model
from keras import layers
from keras.layers import MaxPooling2D

str1 = "\nParameters\n ------------- \n Path:\n"
str2 = "Model\n "
str3 = "Geometric:\n"
str4 = "Control:\n"
head = 'Supervised classification'
head1= '\nNeuroWood running\n'
strP = '"C:\\Program Files\\Basler\\pylon 6\\Applications\\x64\\bin\\pylonviewer.exe"'
strS = "C:/Users/SISTEMA DE VISAO/Documents/NW-main/Image_classify/sample"

class neurowood():                                                  # Main class

    def Extract5(X,args):                                           # Feature Extraction
      num,_,_,_ = X.shape
      Xf = np.zeros((num,180)) 
      for i in range(0,num):
        Xf[i] = FilterDim5(X[i],args)
      return Xf

    def FilterDim5(im0,args):                                       # Feature Extraction
      s = np.zeros(180)
      im = im0[:,:,0] + im0[:,:,1]
      f1 = np.array([[1,4,6,4,1], [4,16,24,16,4],[6,24,476,24,6],
                        [4,16,24,16,4],[1,4,6,4,1] ])
      #th   = .8      
      out = signal.convolve2d(im,f1*(1/256))  
      med = np.average(out)
      for i in range(0,464):
        for j in range(0,216):
          if out[j,i]>med*args.THF:
            out[j,i]=0
      for j in range(0,20):  
        for i in range(0,9): 
          s[9*j+i] = sum(sum(out[24*i:24*(i+1),23*j:23*(j+1)])/(24*23))       
      return s 	

    def FilterDim8(im0,args):                                       # Feature Extraction
      s = np.zeros(3312)
      im = im0[:,:,0] + im0[:,:,1]
      f1 = np.array([[1,4,6,4,1], [4,16,24,16,4],[6,24,476,24,6],
                        [4,16,24,16,4],[1,4,6,4,1] ])
      #th   = .9      
      out = signal.convolve2d(im,f1*(1/256))  
      med = np.average(out)
      for i in range(0,464):
        for j in range(0,216):
          #if out[j,i]>med*args.THF:
          if  out[j,i]<med*args.THF*.95 or out[j,i]>med*args.THF*1.05:
            out[j,i]=0
      for j in range(0,92):  
        for i in range(0,36): 
          s[36*j+i] = sum(sum(out[6*i:6*(i+1),5*j:5*(j+1)])/(6*5))       
      return s 	

    def Parameters(args):                                           # Show Parameters
      print(str1,args.Dir,"\n",args.DirClass,"\n")
      print(str2,args.NameModel,"\n")
      print(str3,"th   ",args.th," th2  ",args.th2,"   edge ",args.edge,
            "\n rx   ",args.rx," ","  ry   ",args.ry," "," r0y  ",args.r0y,
            "\n delt",args.delt," "," xc  ",args.xc," ","yc  ",args.yc,
            "\n Dx  ",args.Dx," "," Dy  ",args.Dy," "," X1  ",args.X1," ",
            "\n Y1  ",args.Y1," "," FrX ",args.FrX," "," FrY ",args.FrY,"\n")
      print(str4,"trigger   ",args.trigger," delay  ",args.delay,"   THF ",args.THF)
        
    def LabelRun(args):                                            # Create image dataset
      run,trigger,count = True,False,0
      camera            = neurowood.OpenBasler()              
      print(head)
      while(run):
        count             += 1
        im                 = neurowood.baslerFast(camera)
        trigger,minX,minY  = neurowood.Wall(im,args) 
        if trigger:
          im1    = neurowood.enhance(im,args,minX,minY) 
          im1.save(args.Dir+"Sample"+str(count)+args.Type) 
          print("Sample:",str(count)," class:",args.Class)
          tm.sleep(args.delay) 
      camera.Close()

    def basler():                                                 # Open camera and grab image
        
      device = py.TlFactory.GetInstance().CreateFirstDevice()
      camera = py.InstantCamera(device)
      camera.Open()
      camera.PixelFormat = "RGB8"
      grab  = camera.GrabOne(400)
      image = grab.Array
      image = Image.fromarray(image)
      camera.Close()
      return image

    def baslerFast(camera):                                      # Grab image from camera       
      grab  = camera.GrabOne(400)
      image = grab.Array
      image = Image.fromarray(image) 
      return image

    def LoadModel(args):                                         # Load matematical model
      Path       = args.DirClass + args.NameModel + ".h5"
      model      = load_model(Path)
      return model

    def ClassifyFast(model,args):                                # Run classification 
      run         = True
      trigger     = False
      run1,count0,count1        = 0,0,0
      ply0,ply1   = 0, 0
      camera      = neurowood.OpenBasler()                              # call OpenBasler
      if args.Siemens == "on":  
        plc         = neurowood.OpenPLC('10.10.0.30',0,1)               # call OpenPCL
        neurowood.Sincronize(7,plc,1,0) 
        #sign = int.from_bytes(plc.db_read(1,4,2), byteorder='big')
        #sign = plc.db_read(db_number: int, start: int, size: int) 
        #print(sign)
      neurowood.Parameters(args)  
      print(head1) 
        
      while(run):       
        tac = tm.time() 
        run1 += 1
        im                 = neurowood.baslerFast(camera)               # call baslerFast
        trigger,minX,minY  = neurowood.Wall(im,args)	                # call Wall
#        trigger = int.from_bytes(plc.db_read(1,4,2), byteorder='big')
        if trigger:
          cl  = neurowood.CNN(im,model,args,minX,minY,run1)            # call CNN
          Class = 0
          if cl == 0 : Class = 1
          print('Class:',Class) 
          if Class == 0: count0+= 1
          if Class == 1: count1+= 1
          if (count0+count1)%10==0: print("Bad: ",count0," Good: ",count1)  
          if args.Siemens == "on":  
            neurowood.Sincronize(9,plc,1,1)                             # call sincronize   
            neurowood.Sincronize(Class,plc,1,2)     
            neurowood.Sincronize(0,plc,1,3)    
            neurowood.Sincronize(8,plc,1,3)      
            neurowood.Sincronize(0,plc,1,3)     
            neurowood.Sincronize(0,plc,1,1) 
            
      camera.Close()

    def enhance(im,args,minX,minY):                              # Enhance image
      frame   =  (args.FrX,args.FrY)
      CROP0   =  [minX,minY,minX+args.X1,minY+args.Y1]
      CROP1   =  [0,0,args.Dx,args.Dy]
      im      =  im.crop(CROP0)
      im      =  im.resize(frame)
      im      =  im.crop(CROP1)
      return im

    def Wall(im,args):                                           # Get Trigger and corners
      im0                     = np.array(im)/256
      rx,ry,yc,xc,edge        = args.rx,args.ry,args.yc,args.xc,args.edge
      trigger                 = False
      channel,minX,minY       = 0, 0, 0
      wall0,wall1,wall2,quide = 0, 0, 0, 0
      r0y                     = args.r0y
      delt                    = args.delt
      wall0  = sum(im0[r0y:ry          ,xc     ,channel]) /(ry-r0y)
      wall2  = sum(im0[r0y+delt:ry+delt,int(xc/5+args.trigger),channel]) /(ry-r0y) 
      if wall2>args.th2 and wall0>args.th:
        trigger = True
        minX,minY = neurowood.corner(im0,args,channel)                 # call corner
      return trigger,minX,minY

    def OpenImage(im,name):                                     # Show image
      plt.figure(figsize=(8,8))
      plt.title(name)
      plt.imshow(im)
      plt.show()

    def OpenPylon():                                            # Open app camera pylon
      os.system(strP)


    def showEnhance(im0,args,minX,minY):                        # show image bondary
      d1 = ImageDraw.Draw(im0)
      font = ImageFont.truetype("arial.ttf", 25)
      d1.text((minX, minY), "corner", font = font)
      d1.text((args.xc/5+args.trigger,args.r0y+args.delt), "trigger", font = font)    
      d1.text((10,10), "trigger: "+str(args.trigger)+" pix", font = font) 
      d1.text((10,50), "delay: "+str(args.delay)+" milisec", font = font) 
      d1.text((10,90), "threshold2: "+str(args.th2), font = font)
      d1.text((10,130), "edge : "+str(args.edge ), font = font)  
      d1.text((10,170), "quality index: "+str(args.THF ), font = font)   
      im                                = np.array(im0)/256  
      channel                           = 0  
      wall0  = sum(im[args.r0y:args.ry          ,args.xc     ,channel]) /(args.ry-args.r0y)
      wall2  = sum(im[args.r0y+args.delt:args.ry+args.delt,int(args.xc/5+args.trigger),channel]) /(args.ry-args.r0y)
      im[args.r0y+args.delt:args.ry+args.delt,int(args.xc/5)-2+args.trigger:int(args.xc/5)+2+args.trigger,0] = 1  # trigger     
      im[args.r0y:args.ry,args.xc-1:args.xc+1,0:2]        = .5                            # corner y        
      im[args.yc-1:args.yc+1,0:args.rx       ,0:2]        = .5                            # corner x
      im[minY-1:minY+1,minX:args.X1                    ,1]= .5                            # superior line 
      im[minY+args.delt-1:minY+args.delt+1,minX:args.X1,1]= .5                            # inferior line
      im[minY:minY+args.delt,minX-1:minX+1             ,1]= .5                            # left line                     
      im[minY-8:minY+8,minX-8:minX+8,:]                   = 1                             # corner
      print("wall0 :",wall0," th:",args.th, "wall2 :",wall2," th2:",args.th2)
      neurowood.OpenImage(im,"enhanced image")
       

    def CNN(im,model,args,minX,minY,run):                       # Aply feature and classify sample
#      xf5    = np.zeros((2,180))
      img      =np.zeros((2,53,115,3))
      img1     =np.zeros((2,215,460,3))
      Class = 1
      xf8    = np.zeros((2,3312))  
      im1    = neurowood.enhance(im,args,minX,minY)
      if args.Save == 'on':
       if run%5 == 0: im1.save(strS+str(tm.time()) +".bmp")  #im1.save(strS+str(tm.time()) +".bmp")    
#      if args.Filter == '180':   
#        xf5[0]  = neurowood.FilterDim5(np.array(im1)/256,args)                        # call FilterDim5
#        Class   = np.argmax(model.predict(xf5[0:1], verbose=0)) 
      if args.Filter == '3312':  
        xf8[0]  = neurowood.FilterDim8(np.array(im1)/256,args)                        # call FilterDim8 
        #pred    = model.predict(xf8[0:1], verbose=0)
        #if pred[0,0]>.7: Class = 0
        Class   = np.argmax(model.predict(xf8[0:1], verbose=0)) 
      if args.Filter == "full":
        img1[0]  = (np.array(im1)/256)
        img      = MaxPooling2D((4,4), name = "max0")(img1)
        Class    = np.argmax(model.predict(img[0:1], verbose=0))
      tm.sleep(args.delay)
      return Class

    def OpenBasler():                                           # Open camera
      device  = py.TlFactory.GetInstance().CreateFirstDevice()
      camera  = py.InstantCamera(device)
      camera.Open()
      camera.PixelFormat = "RGB8"
      return camera

    def corner(im0,args,channel):                               # Find image corners
      rx,ry,r0y,yc,xc,edge     = args.rx,args.ry,args.r0y,args.yc,args.xc,args.edge   
      my,mx,Max                = 0,0,0 
      minY,minX                = args.r0y,0

      for i in range(0,rx):
        mx = (im0[yc,i,channel]-im0[yc,i+1,channel])**2
        if mx >= Max:
          Max  = mx
          minX0 = i
      minX0 = minX0+edge
      my,mx,Max                = 0,0,0
      for i in range(r0y,ry):
        my = (im0[i,xc,channel]-im0[i+1,xc,channel])**2
        if my>= Max:
          Max = my
          minY1     = i
      minY1 = minY1 + edge     
      minY = minY1 
      minX = minX0  
      return minX,minY

    def OpenPLC(IP,rack,slot):                             # Open PLC Siemens
      plc = snap7.client.Client()
      plc.connect(IP,rack,slot)
      print('PLC siemens:',plc.get_connected())
      return plc  

    def Sincronize(Class,plc,DB,start):                    # Write to PLC Siemens
      plc.db_write(DB,start,bytes([Class]))

