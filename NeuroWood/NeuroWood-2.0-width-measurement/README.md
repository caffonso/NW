# NeuroWood 2.0 - medição de largura

Versão do pipeline com:

- diferença contra fundo fixo;
- detecção das bordas superior e inferior;
- medição somente no primeiro frame capturado;
- 5 pontos de medição;
- calibração de 2 pixels = 1 mm;
- largura final calculada pela mediana das 5 medições válidas.

## Instalação

```bash
python -m pip install -r requirements.txt
```

Ajuste `camera.image_path` em `config.yaml` e execute:

```bash
python main.py
```

## Resultado da largura

Após `pipeline.run_once()`:

```python
pipeline.last_piece_info["width_mm"]
pipeline.last_piece_info["width_samples_mm"]
pipeline.last_piece_info["width_measurements"]
```

A calibração atual é fixa no pipeline:

```python
pixels_per_mm=2.0
```
