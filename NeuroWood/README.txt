ALTERAÇÃO DA MEDIÇÃO DE LARGURA

Nova regra:
1. Para cada um dos 13 patches/frames, é feita UMA medição no centro horizontal.
2. Medições inválidas são ignoradas.
3. A largura final da tábua é a MAIOR medição válida.
4. A calibração continua 2 px = 1 mm.

Arquivos alterados:
- core/pipeline.py
- vision/piece_detection.py
- gui.py
- main.py

A GUI passa a mostrar "Largura máxima".
O resumo de produção continua mostrando "Largura média" quando se refere à média
das larguras finais de todas as tábuas processadas.
