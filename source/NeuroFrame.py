import argparse
from gooey import Gooey, GooeyParser

from NeuroLib0 import neurowood as nw

@Gooey(program_name='NeuroWood')
def gui_parser():
  parser = GooeyParser(description='Interface')
  subparser = parser.add_subparsers()

  parser_data = subparser.add_parser(name='Data')
  parser_data.set_defaults(wich='Data')
  parser_data.add_argument('action'    ,type=str,   help=' '           ,default=' ',    choices=['par','pylon','label','show','labelRun','savePickle', ' '])
  parser_data.add_argument('Class'     ,type=str,   help='Class'       ,default='A2', choices=['A2','A1','A4','A5','A6','A7','A8','A9','B10','B11','B12','B13','B14','B15','B6',' ']) 
  parser_data.add_argument('Type'      ,type=str,   help='type file'   ,default='.bmp', choices=['.bmp','.jpg','.png'            ])
  parser_data.add_argument('Dir'       ,type=str,   help='diretory'    ,default= "/home/neurowood/backup/ImageSourceTrain/"       )
  parser_data.add_argument('th'        ,type=float, help='threshold'   ,default=.01  )
  parser_data.add_argument('th2'       ,type=float, help='th2'         ,default=.2   )
  parser_data.add_argument('thmidle'   ,type=float, help='thmidle'     ,default=.25  )
  parser_data.add_argument('edge'      ,type=int,   help='crop/part'   ,default=1    )
  parser_data.add_argument('delay'     ,type=float, help='delay'       ,default=.04  )
  parser_data.add_argument('trigger'   ,type=int,   help='trigger'     ,default= 80  )
  parser_data.add_argument('rx'        ,type=int,   help='range x'     ,default=315  )    
  parser_data.add_argument('r0y'       ,type=int,   help='range0y'     ,default=280  ) # 160 240 300
  parser_data.add_argument('ry'        ,type=int,   help='range y'     ,default=380  ) # 350
  parser_data.add_argument('delt'      ,type=int,   help='delt '       ,default=420  )
  parser_data.add_argument('xc'        ,type=int,   help='center x'    ,default=1180 ) 
  parser_data.add_argument('yc'        ,type=int,   help='center y'    ,default=510  )
  parser_data.add_argument('Dx'        ,type=int,   help='crop in X'   ,default=460  ) #230 460   
  parser_data.add_argument('Dy'        ,type=int,   help='crop in y'   ,default=215  ) #108 215
  parser_data.add_argument('X1'        ,type=int,   help='total lenght',default=1280 )
  parser_data.add_argument('Y1'        ,type=int,   help='total width' ,default=720  ) # ply9 ply10
  parser_data.add_argument('FrX'       ,type=int,   help='Frame X'     ,default=630  ) #300 600 565 580 547  548
  parser_data.add_argument('FrY'       ,type=int,   help='Frame Y'     ,default=365  ) #155 310 307 342  307
  parser_data.add_argument('THF'       ,type=float, help='time'        ,default=.85  )

  parser_Class = subparser.add_parser(name='Classify')
  parser_Class.set_defaults(wich='Classify')
  parser_Class.add_argument('action'   ,type=str, help=' '  ,default='run',choices=['par','model','pylon','enhance','run'])
  parser_Class.add_argument('SetUp'    ,type=str, help=' '           ,default='production',choices=['setup','production'])
  parser_Class.add_argument('DirClass' ,type=str, help='Diretory'    ,default='/home/neurowood/backup/ImageSource/')
  parser_Class.add_argument('NameModel',type=str, help='CNN',default='FaberNWFt',choices=['FaberNW02','FaberNW0','FaberNWFt']) 
  parser_Class.add_argument('committee'   ,type=str, help='committee'  ,default='no',choices=['no','yes','Histogram'])
  parser_Class.add_argument('th'       ,type=float, help='threshold'   ,default=.01   ) #.01
  parser_Class.add_argument('th2'      ,type=float, help='th2'        ,default=.2 )    #.25 .11
  parser_Class.add_argument('thmidle'  ,type=float, help='thguide'   ,default=.05  )
  parser_Class.add_argument('edge'     ,type=int, help='crop/part'   ,default=1    )
  parser_Class.add_argument('rx'       ,type=int, help='range x'     ,default=315  )    
  parser_Class.add_argument('r0y'       ,type=int, help='range0y'    ,default=280 ) # 315 200 160 240 300
  parser_Class.add_argument('ry'       ,type=int, help='range y'     ,default=380  ) #415 300 350
  parser_Class.add_argument('delt'     ,type=int, help='delt '       ,default=420 ) #480
  parser_Class.add_argument('xc'       ,type=int, help='center x'    ,default=1180 ) 
  parser_Class.add_argument('yc'       ,type=int, help='center y'    ,default=510  )
  parser_Class.add_argument('Dx'       ,type=int, help='crop in X'   ,default=460  ) #230 460   
  parser_Class.add_argument('Dy'       ,type=int, help='crop in y'   ,default=215  ) #108 215
  parser_Class.add_argument('X1'       ,type=int, help='total lenght',default=1280 )
  parser_Class.add_argument('Y1'       ,type=int, help='total width' ,default=720  ) # ply9 ply10
  parser_Class.add_argument('FrX'      ,type=int, help='Frame X'     ,default=630  ) #600 300 600 565 580 547  548
  parser_Class.add_argument('FrY'      ,type=int, help='Frame Y'     ,default=365  ) #310 155 310 307 342  307	
  parser_Class.add_argument('Exposure' ,type=int, help='time'        ,default=800  )
  parser_Class.add_argument('delay'    ,type=float, help='delay'      ,default=.2 )
  parser_Class.add_argument('trigger'    ,type=int, help='trigger'  ,default= 80  )
  parser_Class.add_argument('THF' ,type=float, help='time'      ,default=.85  )
  args = parser.parse_args()
  
  return args

def main():
  args = gui_parser()
  
  # Data set
  if (args.wich =="Data"):
    if (args.action=="par"):
      nw.Parameters(args)
    if (args.action=="pylon"):
      OpenPylon()
    if (args.action=="label"):
        ClassLabel(args)
    if (args.action=="show"):
      OpenImage(basler(),"Sample data set")
    if (args.action=="labelRun"):
        LabelRun(args)
    if (args.action=="savePickle"):
        savePickle(Extract5(RGBData(args),args))

  # Classify 
  if (args.wich =="Classify"):
    if (args.action=="par"):
      Parameters(args)

    if (args.action=="model"):
      model = LoadModel(args)
      model.summary()

    if (args.action=="pylon"):
      OpenPylon()

    if (args.action=="enhance"):
      Wall(basler(),args)
      
    if (args.action=="run"):
      print('running neurowood')
      model       = LoadModel(args)
      #Path1       = args.DirClass + "Faber023" + ".h5"
      #model1      = LoadModelQBC
      ClassifyFast(model,args)

if __name__ == '__main__':
    main()
