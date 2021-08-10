import numpy as np
from PIL import Image 
#import cv2                             #here
#import pypylon.pylon as py             #here
import pickle
import h5py
import time as tm
import os
import matplotlib.pyplot as plt
from PIL import Image 
from keras.models import load_model
import time
import logging
import sys
import snap7
from snap7 import util

class neurowood():

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
        trigger,_,_        = LabelWall(im,args)	 
        if trigger:
          im.save(args.Dir+args.Class+"/Sample"+str(count)+args.Type) 
          print("Sample:",str(count)," class:",args.Class)
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
      return image

    def baslerFast(camera):
      grab  = camera.GrabOne(400)
      image = grab.Array
      image = Image.fromarray(image)    
      return image

    def LoadModel(args):
      Path       = args.DirClass + args.NameModel + ".h5"
      model      = load_model(Path)
      return model

    def ClassifyFast(model,args):
      run,trigger = True,False
      ply0,ply1   = 0, 0
      camera      = OpenBasler() 
      head        = 'NeuroWood running'                 
    #  plc         = OpenPLC('10.10.0.30',0,1) ##
      Class       = 0
      print(head)
      while(run):
        im                 = baslerFast(camera)
        trigger,minX,minY  = Wall(im,args)	 
        if trigger:
          Class,im1  = CNN(im,model,args,minX,minY)
    #      Sincronize(Class,plc,1,0) ##
    #      Sincronize(9,plc,1,0)     ##
          if Class == 0:
            ply0 += 1 
          else:
            ply1 += 1  
          if args.SetUp=="setup":
            run = showStatus(im1,Class)

      QualityReport(args,run,ply0,ply1)
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
      im0                    = np.array(im)/256
      rx,ry,yc,xc,edge       = args.rx,args.ry,args.yc,args.xc,args.edge
      trigger                = False
      channel,minX,minY,wall = 0, 0, 0, 0
      r0y                    = args.r0y
      delt                   = args.delt
      wall  = sum(im0[r0y:ry,xc    ,channel])/ry
      midle = sum(im0[r0y+delt:ry+delt,xc-450,channel])/ry
      guide = sum(im0[yc,0:rx    ,channel])/rx

      if (args.action=="enhance"):
        print("wall:" ,round(wall ,2),'quide',round(guide,2),'midle',round(midle,2))
        minX,minY = corner(im0,args,channel)
        showEnhance(im0,args,minX,minY)  

      #if wall>args.th: 
      if wall>args.th and guide>args.thguide:# and midle>args.thmidle:
        trigger = True
        minX,minY = corner(im0,args,channel)

      return trigger,minX,minY

    def LabelWall(im,args):
      im0                    = np.array(im)/256
      rx,ry,yc,xc,edge       = args.rx,args.ry,args.yc,args.xc,args.edge
      trigger                = False
      channel,minX,minY,wall = 0, 0, 0, 0
      r0y                    = args.r0y
      delt                   = args.delt 
      wall  = sum(im0[r0y:ry,xc,channel])/ry 
      midle = sum(im0[r0y+delt:ry+delt,xc-450,channel])/ry
      guide = sum(im0[yc,0:rx,channel])/rx

      #if wall>args.th:
      if wall>args.th and guide>args.thguide:# and midle>args.thmidle:
        trigger = True
        minX,minY = corner(im0,args,channel)

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
      im[ry0:ry,xc-2:xc+2,0:2]          = 1   
      im[r0y+delt:ry+delt,(xc-450)-2:(xc-450)+2,0:2]          = 1
      im[yc-2:yc+2,0:rx,0:2]            = 1
      im[minY-2:minY+2,:,1]             = 0
      im[minY+550-2:minY+550+2,:,1]     = 0
      im[:,minX-2:minX+2,1]             = 0
      im[:,minX+1200-2:minX+1200+2,1]   = 0
      im[minY-5:minY+5,minX-5:minX+5,:] = 1
      OpenImage(im,"enhanced image")
      print(r0y)  

    def CNN(im,model,args,minX,minY):
      x       = np.zeros((2,args.Dy,args.Dx,3))
      im1    = enhance(im,args,minX,minY)     
      x[0]   = np.array(im1)/256
      Class  = np.argmax(model.predict(x[0:1]))
      print("class:",Class)
      return Class,im1

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
      my,mx                = np.zeros(ry),np.zeros(rx)  
      maximumX,maximumY    = 0, 0 


      for i in range(0,rx):
        mx[i] = (im0[yc,i,channel]-im0[yc,i+1,channel])**2
        if mx[i] > maximumX:
          maximumX = mx[i]
          minX = i
      minX = minX+edge
      for i in range(r0y,ry):
        my[i] = (im0[i,xc,channel]-im0[i+1,xc,channel])**2
        if my[i]> maximumY:
          maximumY = my[i]
          minY     = i
      minY = minY + edge
      return minX,minY

    def QualityReport(args,run,count0,count1):
      prod = np.zeros((8,2))
      addres = args.DirClass+ 'QualityReport_'+str(tm.localtime().tm_mday)+'_'+str(tm.localtime().tm_mon)+'.txt'
      str1 = '\nProduction report: '+str(tm.localtime().tm_mday)+'/'+str(tm.localtime().tm_mon)+'/'+str(tm.localtime().tm_year)
      str2 = '\ninit:'+str(args.tm_hour)+'h '+str(args.tm_min)+'min '+str(args.tm_sec)+'s'
      str3 = '\nend:' +str(tm.localtime().tm_hour)+'h '+str(tm.localtime().tm_min)+'min'+str(tm.localtime().tm_sec)+'s\n\n'
      str4 = 10*(' ')+'Classification\n[h]  #ply10   #others\n'
      str0 = str1 + str2 + str3 + str4 

      file = open(addres,'w') 
      file.write(str0)
      for i in range(0,8):
        file.write(str(i+1)+5*(' ')+str(prod[i,0])+5*(' ')+str(prod[i,1])+'\n')

      file.close

    def OpenPLC(IP,rack,slot):
      plc = snap7.client.Client()
      plc.connect(IP,rack,slot)
      print('PLC siemens:',plc.get_connected())
      return plc  

    def Sincronize(Class,plc,DB,start):       # DB,start
      print('Class',Class)
      plc.db_write(DB,start,bytes([Class]))
 
  
