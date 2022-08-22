import argparse                                                  # Import Libraries        
from gooey import Gooey, GooeyParser
from NeuroLib import neurowood as nw

@Gooey(program_name='NeuroWood', required_cols=3, clear_before_run = True)
#default_size=(300, 200)
def gui_parser():
  parser = GooeyParser(description='Computer Vision')
  subparser = parser.add_subparsers()
  act    = ['par','pylon','show','enhance','run','labelRun',' ']
  pathM  = 'C:/Users/SISTEMA DE VISAO/Documents/NW-main/Model/'  
  pathT  = 'C:/Users/SISTEMA DE VISAO/Documents/NW-main/Train/' 
  Class0 = ['A2','A1',' ']
  FileT  = ['.bmp','.jpg','.png']
  Models = ['ModelFaberBWFB247','ModelFaberBWFB245','ModelFaberBWFC07','ModelFaberBWFC06','ModelFaberBWFC05']
  fil    = ['180','3312',"full"]  
  Sim    = ['on','off']  
  warning= "This application does not requires GPU device "  
  print(warning)
  parser_data = subparser.add_parser(name='Data')              # Create Formular
  parser_data.set_defaults(wich='Data')
  parser_data.add_argument('action'    ,type=str,   help='NeuroWood'           ,default=' '          ,choices=     act)
  parser_data.add_argument('NameModel' ,type=str,   help='Neural Network'         ,default='ModelFaberBWFB245',choices=  Models)  
  parser_data.add_argument('Filter'    ,type=str,   help='Feature'     ,default='3312'       ,choices=     fil)     
#  parser_data.add_argument('Siemens'   ,type=str,   help='CHP'         ,default='on'        ,choices=     Sim)  
  parser_data.add_argument('Class'     ,type=str,   help='wood'         ,default='A1'         ,choices=  Class0) 
  parser_data.add_argument('Save'      ,type=str,   help='Image'        ,default='on'         ,choices=     Sim)   
#  parser_data.add_argument('Type'      ,type=str,   help='type file'   ,default='.bmp'       ,choices=  FileT ) 
#  parser_data.add_argument('DirClass'  ,type=str,   help='Diretory'    ,default                         =pathM)
#  parser_data.add_argument('Dir'       ,type=str,   help='Training'    ,default                         =pathT)
#  parser_data.add_argument('th'        ,type=float, help='threshold'   ,default=.01  )         # wall trigger threshold 
#  parser_data.add_argument('th2'       ,type=float, help='th2'         ,default=.2   )         # wall2 trigger threshold       
  parser_data.add_argument('edge'      ,type=int,   help='pixels'       ,default=30    )        #1  
  parser_data.add_argument('delay'     ,type=float, help='miliseconds'  ,default=.0   )         #.04
  parser_data.add_argument('trigger'   ,type=int,   help='pixels'       ,default= 120 )        # trigger ( advance ) 80
#  parser_data.add_argument('rx'        ,type=int,   help='range x'     ,default=315  )        # corner x 
#  parser_data.add_argument('r0y'       ,type=int,   help='range0y'     ,default=280  )        # Wall corner y0
#  parser_data.add_argument('ry'        ,type=int,   help='range y'     ,default=380  )        # Wall corner y
#  parser_data.add_argument('delt'      ,type=int,   help='delt '       ,default=420  )        # width part
  parser_data.add_argument('xc'        ,type=int,   help='pixel pos.'    ,default=1220 )       # 1180
#  parser_data.add_argument('yc'        ,type=int,   help='center y'    ,default=510  )
#  parser_data.add_argument('Dx'        ,type=int,   help='crop in X'   ,default=460  ) 
#  parser_data.add_argument('Dy'        ,type=int,   help='crop in y'   ,default=215  ) 
#  parser_data.add_argument('X1'        ,type=int,   help='total lenght',default=1280 )
#  parser_data.add_argument('Y1'        ,type=int,   help='total width' ,default=720  ) 
#  parser_data.add_argument('FrX'       ,type=int,   help='Frame X'     ,default=630  ) 
#  parser_data.add_argument('FrY'       ,type=int,   help='Frame Y'     ,default=365  ) 
  parser_data.add_argument('THF',type=float, help='quality level'       ,default=1  )  
  args = parser.parse_args()
  args.Siemens  = 'on'
#  args.Filter   = '3312'
  args.Type     = '.bmp'
  args.DirClass = pathM 
  args.Dir      = pathT
  args.th2      = .2 
  args.th       = .01
  args.rx       = 315
  args.r0y      = 280
  args.ry       = 380
  args.delt     = 420
  args.yc       = 510
  args.Dx       = 460
  args.Dy       = 215
  args.X1       = 1280
  args.Y1       = 720
  args.FrX      = 630
  args.FrY      = 365
  return args
 
def main():
  args = gui_parser()
  if (args.wich =="Data"):                             # Call functions 
    if (args.action=="par"):                                    # Show paramters and model layers
      nw.Parameters(args)
      model = nw.LoadModel(args)
      model.summary()
    if (args.action=="pylon"):                                  # Open camera
      nw.OpenPylon()
    if (args.action=="show"):                                   # Show image
      nw.OpenImage(nw.basler(),"Sample")
    if (args.action=="labelRun"):                               # Create dataset
        nw.LabelRun(args)
    if (args.action=="run"):                                    # Run Classification 
      print('running neurowood')
      model       = nw.LoadModel(args)
      nw.ClassifyFast(model,args) 
    if (args.action=="enhance"):                                # Show enhanced image
      im          = nw.basler()  
      _,minX,minY = nw.Wall(im,args)
      nw.showEnhance(im,args,minX,minY)
    
if __name__ == '__main__':
    main()