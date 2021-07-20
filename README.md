-----------
# NeuroWood
-----------
NeuroWood é um software para classificação de qualidade da madeira, voltado à soluções para a indústria madeireira e moveleira. Utiliza técnicas de aprendizado de máquina e visão computacional para classificar imagens do fluxo de processos na madeira em tempo real, permitindo tomada de decisão rápida e automática.

## Instalação

### Pré-Requisitos

#### Python 3

Certifique-se de que o Python versão 3 ou superior está disponível no sistema:
```
python3 --version
```
Se necessário, utilize o gerenciador de pacotes do seu sistema operacional ou obtenha a versão do Python mais recente em <https://www.python.org/downloads/>

#### Bibliotecas Python

Utilize o pip para instalar as dependências:
```
sudo python3 -m pip install numpy scipy sklearn scikit-image
sudo python3 -m pip install https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-16.04/wxPython-4.0.0b1-cp36-cp36m-linux_x86_64.whl phoenix gooey
```
Pode ser necessário instalar dependências para o módulo wxPython. Elas podem ser verificadas [aqui](https://github.com/wxWidgets/Phoenix/blob/master/README.rst#prerequisites)

Serão utilizadas seguintes bibliotecas python:

*  numpy 
*  PIL import Image 
*  cv2 
*  pypylon.pylon as py 
*  pickle 
*  h5py 
*  time as tm 
*  os 
*  glob 
*  matplotlib.pyplot as plt 
*  PIL import Image  
*  keras.models import load_model 
*  logging 
*  sys 
*  snap7 
*  snap7 import util 
*  from scipy import signal 
*  from keras import layers 

## Setup

### Câmera

A interface entre o *hardware* da câmera Basler (https://www.baslerweb.com/en/products/cameras/area-scan-cameras/ace/aca1300-200uc/) e a sistema de visão será realizado através do app **pypylon**.  

Maiores informações sobre os requisitos, setups e instalção do app pode ser obtidas através do link: https://github.com/basler/pypylon

Para instalação windows:
```
pip3 install pypylon
```

O acesso ao app **pypylon**, deve ser feito através do prompt através da método **OpenPylon()**:

```
    def OpenPylon():
      os.system("/opt/pylon5/bin/./PylonViewerApp")
```

Os parâmetros da câmera deve ser ajustado no app neurowood acessando <data><action><pylon>
    

![image](pylon.png)

 Aparecera a tela do app da camera onde deve ajustar:
*  Exposure ( default = 800 )
    
### Capitura das imagem

Incialmente deve-se ajustar as imagens estáticas, possicionando uma amostra de peca diretamente 
abaixo da câmera e ajustar os recortes através do método **Enhance**


        minX,minY = corner(im0,args,channel)
        showEnhance(im0,args,minX,minY)  

    
    
