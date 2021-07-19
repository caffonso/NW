import numpy as np
from PIL import Image 
# import cv2
# import pypylon.pylon as py
import pickle
import h5py
import time as tm
import os
import glob
import matplotlib.pyplot as plt
from PIL import Image 
from keras.models import load_model
import time
import logging
import sys
import snap7
from snap7 import util
from scipy import signal
from keras import layers 

class neurowood():
    def sobel(img):
      #Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
      Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)  
      #Ix = ndimage.filters.convolve(img, Kx)
      Iy = ndimage.filters.convolve(img, Ky) 
      #G = np.hypot(Ix, Iy)
      #G = G / G.max() * 1
      #theta = np.arctan2(Iy, Ix)

      return (Iy)


    def FaberNWH(img,args):
      th     = .325
      count  = 0 
      Class  = 1 
      im     = img[:,:,0]
      for i in range(0,args.Dy):
        for j in range(0,args.Dx):
          if im[i,j]>th:
            count += 1 
      if count == 0:
        Class = 0

      return Class

    def Extract5(X,args): 
      num,_,_,_ = X.shape
      Xf = np.zeros((num,180)) 
      for i in range(0,num):
        Xf[i] = FilterDim5(X[i],args)

      return Xf

    def RGBData(args):
      Path     = "/home/neurowood/backup/ImageSourceTrain/A11/"
      NumTrain = 1107
      Dy,Dx,Type       = args.Dy,args.Dx,args.Type
      data_train       = np.zeros((NumTrain,Dy,Dx,3))
      #data_train       = np.zeros((NumTrain,54,115,3))
      os.chdir(Path)
      files=glob.glob('*.bmp') 
      for i in range(0,NumTrain):      
        pil_im                   = Image.open(files[i])
        #_,minX,minY              = Wall(pil_im,args)    
        #im                       = enhance(pil_im,args,minX,minY) 
        #data_train[i]            = np.array(im)/256
        data_train[i]            = np.array(pil_im)/256

      return data_train

    def savePickle(X_train):
      Path     = "/home/neurowood/backup/ImageSourceTrain/features/Faber_11_6_010.pckl"
      print(Path)
      f = open(Path, 'wb')
      pickle.dump([X_train], f)
      f.close() 

    def FilterDim(im0,args):
      im = im0[:,:,0]  
      f1 = np.array([[1,4,6,4,1], [4,16,24,16,4],[6,24,476,24,6],
                    [4,16,24,16,4],[1,4,6,4,1] ])
      th   = args.THF
    #  th   = .95 #.75            
      s = np.zeros(10)

      out = signal.convolve2d(im,f1*(1/256))
      med = np.average(out)
      for i in range(0,460):
        for j in range(0,215):
          if out[j,i]>med*th:
            out[j,i]=0

      for i in range(0,10): 
        s[i]  = sum(sum(out[21*i:21*(i+1),10:450]))/100

      return s 

    def FilterDim2(im0,args):
          im = im0[:,:,0]  
          f1 = np.array([[1,4,6,4,1], [4,16,24,16,4],[6,24,476,24,6],
                        [4,16,24,16,4],[1,4,6,4,1] ])
          #th   = .75
          s = np.zeros(50)

          out = signal.convolve2d(im,f1*(1/256))
          med = np.average(out)
          for i in range(0,460):
            for j in range(0,215):
              if out[j,i]>med*args.THF:
                out[j,i]=0

          for j in range(0,9):  
            for i in range(0,5): 
              s[10*i+j] = sum(sum(out[43*i:43*(i+1),46*j:46*(j+1)])/100)  

          return s	


    def FilterDim5(im0,args):
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

    def NetDim(im):
      Class,th   = 2, .1

      ply567  = FilterDim(im,105 ,170,450)
      scrap   = FilterDim(im, 55 ,105,450)  

      if ply567 > th:
        Class = 3    
      if scrap > th:
        Class = 4

      return Class


    def Filter(im0):
      im = im0[:,:,0]  
      f1 = np.array([[1,4,6,4,1], [4,16,24,16,4],[6,24,476,24,6],
                     [4,16,24,16,4],[1,4,6,4,1] ])
      Filter,Class,th   = 0,0,.75
      y0,y1,x0,x1,ye,xe = 10,213,10,450,100,400 

      out = signal.convolve2d(im,f1*(1/256))
      med = np.average(out)
      for i in range(0,460):
        for j in range(0,215):
          if out[j,i]>med*th:
            out[j,i]=0

      Filter = (sum(sum(out[y0:y1,x0:x1])) - sum(sum(out[y0:ye,xe:x1])))/100 
      #print(Filter) 
      if Filter > 0.5:
        Class = 1

      return Class

    def Parameters(args):
      print("Parameters\n",args) 

    def ClassLabel(args):
      im     = basler()
      h      = tm.localtime().tm_hour 
      m      = tm.localtime().tm_min  
      s      = tm.localtime().tm_sec  
      time   = str(h)+'_'+str(m)+'_'+str(s)

      im.save(args.Dir+args.Class+"/Sample"+time+args.Type) 
      print("sample:",time," class:",args.Class)

    def LabelRun(args):
      h                 = tm.localtime().tm_hour 
      m                 = tm.localtime().tm_min  
      s                 = tm.localtime().tm_sec  
      time              = str(h)+'_'+str(m)+'_'+str(s)
      run,trigger,count = True,False,0
      camera            = OpenBasler() 
      head              = 'Automatic classification'                 

      print(head)
      while(run):
        count             += 1
        im                 = baslerFast(camera)
        trigger,minX,minY  = LabelWall(im,args) #	 
        if trigger:
          im1    = enhance(im,args,minX,minY) # 
          im1.save(args.Dir+args.Class+"/Sample"+str(count)+args.Type) 
          print("Sample:",str(count)," class:",args.Class)
          tm.sleep(args.delay) 
      camera.Close()

    def basler():
      device = py.TlFactory.GetInstance().CreateFirstDevice()
      camera = py.InstantCamera(device)
      camera.Open()
      camera.PixelFormat = "RGB8"
      grab  = camera.GrabOne(400)
      image = grab.Array
      image = Image.fromarray(image)
      camera.Close()
      #image = Image.open("/home/neurowood/backup/ImageSourceTrain/B5/Sample610.bmp")

      return image

    def baslerFast(camera):
      grab  = camera.GrabOne(400)
      image = grab.Array
      image = Image.fromarray(image) 
      #image = Image.open("/home/neurowood/backup/ImageSourceTrain/A/Sample1562.bmp")   
      return image

    def LoadModel(args):
      Path       = args.DirClass + args.NameModel + ".h5"
      model      = load_model(Path)
      return model

    def LoadModelQBC(args):
      Path1       = args.DirClass + "FaberNWFt1" + ".h5"
      model1      = load_model(Path1)
      Path2       = args.DirClass + "FaberNWFt2" + ".h5"
      model2      = load_model(Path2)
      Path3       = args.DirClass + "FaberNWFt3" + ".h5"
      model3      = load_model(Path3)
      return model1,model2,model3


    def ClassifyFast(model,args):
      run         = True
      trigger     = False
      run1         = 0
      ply0,ply1   = 0, 0
      camera      = OpenBasler() 
      head        = 'NeuroWood running'                 
      plc         = OpenPLC('10.10.0.30',0,1) 
      Sincronize(7,plc,1,0) 
      #Path1       = args.DirClass + "Faber023" + ".h5"
      #model1      = load_model(Path1)
      xi     = np.zeros((2,54,115,3))
      xiii     = np.zeros((2,215,460,3))
      xii     = np.zeros((2,50))
      x0      = np.zeros((2,180))
      model1,model2,model3 =  LoadModelQBC(args)
      status = np.argmax(model.predict(x0[0:1])) 
      status = np.argmax(model1.predict(x0[0:1]))  
      status = np.argmax(model2.predict(x0[0:1])) 
      status = np.argmax(model3.predict(x0[0:1])) 
      #Class       = 0
      print(head)
      while(run):
        tac = tm.time() 
        run1 += 1
        im                 = baslerFast(camera)

        trigger,minX,minY  = Wall(im,args)	
        #print('time',tac - tm.time())
        if trigger:
          #run1 += 1
          Class  = CNN(im,model,args,minX,minY,run1,model1,model2,model3)
          print('Class:',Class) 
          #print('time',tac - tm.time())
          Sincronize(9,plc,1,1)     ##
          Sincronize(Class,plc,1,2)     ## Class
          Sincronize(0,plc,1,3)     ##
          Sincronize(8,plc,1,3)     ##
          Sincronize(0,plc,1,3)     ##
          Sincronize(0,plc,1,1)     ##

          #if Class == 4:
          #  ply0 += 1 
          #else:
          #  ply1 += 1  
          if args.SetUp=="setup":
            imE  = enhance(im,args,minX,minY)
            run = showStatus(imE,Class)
          #if run1 == 20:
          #  print("correct;",ply0,"wrong:",ply1) 
          #  QualityReport(args,run,ply0,ply1) 
          #  camera.Close()
          #  run1 = 0 
      #tic = tm.time()
      #print('time:',round(100*(tic-tac),2),'ms')  
      #QualityReport(args,run,ply0,ply1)
      camera.Close()

    def enhance(im,args,minX,minY):
      frame   =  (args.FrX,args.FrY)
      CROP0   =  [minX,minY,minX+args.X1,minY+args.Y1]
      CROP1   =  [0,0,args.Dx,args.Dy]
      im      =  im.crop(CROP0)
      im      =  im.resize(frame)
      im      =  im.crop(CROP1)

      return im

    def Wall(im,args):
      im0                     = np.array(im)/256
      rx,ry,yc,xc,edge        = args.rx,args.ry,args.yc,args.xc,args.edge
      trigger                 = False
      channel,minX,minY       = 0, 0, 0
      wall0,wall1,wall2,quide = 0, 0, 0, 0
      r0y                     = args.r0y
      delt                    = args.delt

      wall0  = sum(im0[r0y:ry          ,xc     ,channel]) /(ry-r0y)
      #wall1  = sum(im0[r0y+delt:ry+delt,int(xc/2),channel]) /(ry-r0y)
      #wall2  = sum(im0[r0y:ry          ,int(xc/5),channel]) /(ry-r0y)

      #wall1  = sum(im0[r0y:ry          ,int(xc/5),channel]) /(ry-r0y) # new
      wall2  = sum(im0[r0y+delt:ry+delt,int(xc/5+args.trigger),channel]) /(ry-r0y)  # new 65
      #guide  = sum(im0[yc              ,0:rx     ,channel]) /rx

      if (args.action=="enhance"):
        #print("wall0:" ,round(wall0 ,2),"wall1:" ,round(wall1 ,2),"wall2:" ,round(wall2 ,2),'quide',round(guide,2))
        print("wall0:" ,round(wall0 ,2),"wall2:" ,round(wall2 ,2))
        minX,minY = corner(im0,args,channel)
        #if guide > .54:
        #  minX = 0
        showEnhance(im0,args,minX,minY)  

      #if wall0>args.th: 
      if wall2>args.th2 and wall0>args.th:
        trigger = True
        minX,minY = corner(im0,args,channel)

      return trigger,minX,minY

    def LabelWall(im,args):
      im0                     = np.array(im)/256
      rx,ry,yc,xc,edge        = args.rx,args.ry,args.yc,args.xc,args.edge
      trigger                 = False
      channel,minX,minY       = 0, 0, 0
      wall0,wall1,wall2,quide = 0, 0, 0, 0
      r0y                     = args.r0y
      delt                    = args.delt

      wall0  = sum(im0[r0y:ry          ,xc       ,channel]) /(ry-r0y)
      #wall1  = sum(im0[r0y+delt:ry+delt,int(xc/5),channel]) /(ry-r0y)
      #wall2  = sum(im0[r0y:ry          ,int(xc/5),channel]) /(ry-r0y)  
      #guide  = sum(im0[yc              ,0:rx     ,channel]) /rx

      wall2  = sum(im0[r0y+delt:ry+delt,int(xc/5),channel]) /(ry-r0y)  # new

      #if wall>args.th: 
      if wall0>args.th and wall2>args.th2:
        trigger = True
        #minX,minY = corner(im0,args,channel)

      return trigger,minX,minY

    def OpenImage(im,name):
      plt.figure(figsize=(8,8))
      plt.title(name)
      plt.imshow(im)
      plt.show()

    def OpenPylon():
      os.system("/opt/pylon5/bin/./PylonViewerApp")

    def showEnhance(im,args,minX,minY):
      rx,ry,yc,xc                       = args.rx,args.ry,args.yc,args.xc 
      r0y                               = args.r0y
      delt                              = args.delt

      im[r0y:ry,xc-2-120:xc+2-120,0:2]                         = 1 # wall0  
      #im[r0y:ry,int(xc/5)-2:int(xc/5)+2,0:2]           = 1   
      im[r0y+delt:ry+delt,int(xc/5)-2+args.trigger:int(xc/5)+2+args.trigger,0:2] = 1 # wall2
      im[r0y+delt:ry+delt,int(xc)-2+80:int(xc)+2,0:2] = 1 # wall22
      im[yc-2:yc+2,0:rx,0:2]                           = 1
      #im[yc-2+200:yc+2+200,0:rx,0:2]                   = 1
      im[minY-2:minY+2,:,1]                            = 0
      im[minY+delt-2:minY+delt+2,:,1]                  = 0
      im[:,minX-2:minX+2,1]                            = 0
      im[:,minX+1050-2:minX+1050+2,1]                  = 0
      im[minY-5:minY+5,minX-5:minX+5,:]                = 1
      OpenImage(im,"enhanced image")



    def CNN(im,model,args,minX,minY,run,model1,model2,model3):
      x      = np.zeros((2,args.Dy,args.Dx,3))
      xf5    = np.zeros((2,180))
      im1    = enhance(im,args,minX,minY)     
      im1.save("/home/neurowood/backup/ImageSourceTrain/A1/Sample"+str(tm.time())+".bmp") 
      x[0]   = np.array(im1)/256

      if args.NameModel=="FaberNW0"  or args.NameModel =='FaberNW02':
        xf5[0]  = FilterDim5(x[0],args)    
        Class  = np.argmax(model.predict(xf5[0:1]))
        if Class == 1:
          Class = 0
        else:
          Class = 1
        tm.sleep(args.delay)

      if args.committee == "Histogram":
        Class  = 1
        ClassH = FaberNWH(x[0],args) 
        Class  = np.argmax(model.predict(x[0:1]))
        if ClassH+Class == 0:
          Class = 0
        tm.sleep(args.delay)

      if args.NameModel == "FaberNW3x" or args.NameModel =='FaberNW6x' or args.NameModel == 'FaberNW7x':
        Class  = np.argmax(model.predict(x[0:1]))
        tm.sleep(args.delay)

      if args.NameModel == "Faber022" or args.NameModel == "Faber023" or args.NameModel == "Faber025":
        xf[0]  = FilterDim(x[0],args)    
        Class  = np.argmax(model.predict(xf[0:1]))


      if args.committee == "yes" :
        Class = 1  
        xf5[0]  = FilterDim5(x[0],args) 
        cl   = np.argmax(model.predict(xf5[0:1]))
        cl1  = np.argmax(model1.predict(xf5[0:1]))
        #cl2  = np.argmax(model2.predict(x[0:1])) 
        #cl3  = np.argmax(model3.predict(x[0:1]))  
        tm.sleep(args.delay)

        if cl+cl1 < 1:
          Class = 0

      if args.NameModel=="FaberNWFt"  or args.NameModel=="FaberNWFt2" or args.NameModel == "FaberNWFt4":
        xf5[0]  = FilterDim5(x[0],args)    
        Class  = np.argmax(model.predict(xf5[0:1]))
        #if Class == 1:
        #Class = np.argmax(model1.predict(x[0:1]))
        tm.sleep(args.delay)

      return Class

    def OpenBasler():
      device  = py.TlFactory.GetInstance().CreateFirstDevice()
      camera  = py.InstantCamera(device)
      camera.Open()
      camera.PixelFormat = "RGB8"
      return camera

    def showStatus(im1,Class):
      run    = True
      status = 'repproved'
      if Class==0:
        status  = "approved"
      OpenImage(im1,"classified: "+status)
      run    = False
      return run

    def corner(im0,args,channel):
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

    def QualityReport(args,run,count0,count1):
      prod = np.zeros((8,2))
      addres = args.DirClass+ 'QualityReport_'+str(tm.time())+'_'+'.txt'
      str1 = '\nProduction report: '+str(tm.localtime().tm_mday)+'/'+str(tm.localtime().tm_mon)+'/'+str(tm.localtime().tm_year)
      str2 = '\ninit:'+str(args.tm_hour)+'h '+str(args.tm_min)+'min '+str(args.tm_sec)+'s'
      str3 = '\nend:' +str(tm.localtime().tm_hour)+'h '+str(tm.localtime().tm_min)+'min'+str(tm.localtime().tm_sec)+'s\n\n'
      str4 = 10*(' ')+'Classification\n[h]  #right   #wrong\n'
      str0 = str1 + str2 + str3 + str4 

      file = open(addres,'w') 
      file.write(str0)
      for i in range(0,1):
        #file.write(str(i+1)+5*(' ')+str(prod[i,0])+5*(' ')+str(prod[i,1])+'\n')
        file.write(str(i+1)+5*(' ')+str(count0)+5*(' ')+str(count1)+'\n')
      file.close

    def OpenPLC(IP,rack,slot):
      plc = snap7.client.Client()
      plc.connect(IP,rack,slot)
      print('PLC siemens:',plc.get_connected())
      return plc  

    def Sincronize(Class,plc,DB,start):       # DB,start
      #print('Class',Class)
      plc.db_write(DB,start,bytes([Class]))
 
  
