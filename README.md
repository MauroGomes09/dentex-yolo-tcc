# Detecção de patologias em radiografias panorâmicas com YOLOv8 (base DENTEX)

Código e resultados agregados do Trabalho de Conclusão de Curso *Desenvolvimento e Avaliação de um Modelo de Deep Learning para Detecção de Patologias em Radiografias Panorâmicas* (Bacharelado em Ciência da Computação, Universidade Federal de Pelotas, 2026).

## Resumo

O trabalho aplica transferência de aprendizado com detectores de objetos YOLOv8, pré-treinados no COCO, para localizar quatro achados em radiografias panorâmicas da base pública DENTEX: **cárie**, **cárie profunda**, **lesão periapical** e **dente impactado**. Os notebooks reproduzem a análise exploratória, os experimentos de configuração (YOLOv8n e YOLOv8s, a 640 e a 1.024 pixels), o treinamento do modelo final, a escolha do limiar de confiança na validação e a avaliação em um teste interno de 106 imagens. A avaliação usa precisão, revocação (sensibilidade), F1, AP@0,5 por classe (pycocotools, o mesmo protocolo do desafio DENTEX) e matriz de confusão.

## Estrutura

```
.
├── notebooks/            experimentos, executados no Kaggle (GPU T4)
│   ├── 01_baseline_yolov8n.ipynb
│   ├── 02_experimentos_2a4_configuracao.ipynb
│   ├── 03_experimento_5_yolov8s_1024.ipynb
│   └── 04_modelo_final_teste_interno.ipynb
├── scripts/              análises sem GPU, sobre os artefatos do notebook 04
│   ├── analise_teste.py
│   └── contagem_treino_por_regiao.py
├── resultados/           métricas agregadas (CSV), sem imagens nem pesos
│   ├── configuracao/
│   └── modelo_final/
├── docs/REPRODUCAO.md    ambiente, ordem de execução e descrição dos arquivos
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

Os notebooks estão exatamente como foram executados; só foram removidos os metadados da plataforma. A lógica do código não foi alterada.

## Dados

A base é a [DENTEX](https://arxiv.org/abs/2305.19112) (MICCAI 2023), hospedada no Hugging Face (`ibrahimhamamci/DENTEX`). Os notebooks baixam apenas os arquivos de treinamento e de validação do subconjunto com diagnóstico, lendo os arquivos compactados remotamente. **Este repositório não redistribui imagens, anotações nem pesos treinados.** A base é distribuída sob CC BY-NC-SA 4.0 (README do Hugging Face e do GitHub); respeite essa licença ao usá-la.

## Protocolo

- Subconjunto totalmente anotado com diagnóstico: 705 imagens de treinamento, 50 de validação e 250 de teste oficial.
- O teste oficial **não foi usado**: seus rótulos estão em turco, em outro formato, e a correspondência com as quatro classes não está documentada. No lugar dele, 15% das 705 imagens de treinamento foram separadas como **teste interno**, de forma estratificada pela classe mais rara de cada imagem, com semente 42. Resultado: 599 imagens de treinamento, 50 de validação e 106 de teste.
- Modelo final: YOLOv8s, 1.024 pixels, 100 épocas, lote 16, semente 42, treinamento determinístico. O melhor ponto de parada (época 45) e o limiar de confiança (0,29, o que maximiza o F1 médio entre as quatro classes) foram escolhidos **na validação**, antes de olhar o teste.
- Uma detecção conta como acerto quando a classe está correta e a interseção sobre a união (IoU) com a anotação é de pelo menos 0,5. Especificidade e acurácia não são reportadas, porque não há exemplos negativos no nível da caixa.

| Experimento | Modelo | Resolução | Observação |
|---|---|---|---|
| 1 | YOLOv8n | 640 | linha de base, parada antecipada (paciência 20) |
| 2 | YOLOv8n | 640 | 100 épocas fixas |
| 3 | YOLOv8s | 640 | 100 épocas fixas |
| 4 | YOLOv8n | 1.024 | 100 épocas fixas |
| 5 | YOLOv8s | 1.024 | 100 épocas fixas (configuração escolhida) |
| Final | YOLOv8s | 1.024 | treino sem o teste interno; avaliação no teste interno |

## Resultados no teste interno (106 imagens, limiar 0,29)

| Classe | AP@0,5 | Precisão | Revocação | F1 |
|---|---|---|---|---|
| Dente impactado | 0,901 | 0,850 | 0,904 | 0,876 |
| Cárie | 0,463 | 0,492 | 0,471 | 0,481 |
| Cárie profunda | 0,424 | 0,627 | 0,483 | 0,545 |
| Lesão periapical | 0,225 | 0,400 | 0,182 | 0,250 |
| **Média das classes** | **0,503** | **0,592** | **0,510** | **0,538** |

Estes números vêm de um teste interno e **não são comparáveis diretamente** com o ranking do desafio DENTEX, que usa o teste oficial oculto. A lesão periapical tem só 22 anotações no teste, e por isso sua estimativa é instável. A discussão completa, incluindo a avaliação por um cirurgião-dentista e as limitações, está na monografia.

## Como reproduzir

Veja [docs/REPRODUCAO.md](docs/REPRODUCAO.md). Em resumo: importe um notebook no Kaggle, ative a GPU T4 e a Internet, e use *Save & Run All*. O treinamento é determinístico (semente 42), e a monografia reporta que execuções repetidas deram resultados idênticos no mesmo ambiente.

## Autoria e uso de inteligência artificial

Autor: Mauro Artur Gomes de Oliveira. O desenvolvimento do código contou com apoio de uma ferramenta de inteligência artificial generativa, conforme declarado na monografia. O autor revisou o código, executou todos os experimentos e responde por ele.

## Licença

O código deste repositório é distribuído sob a licença MIT (veja `LICENSE`). A base DENTEX (CC BY-NC-SA 4.0) e a biblioteca Ultralytics (AGPL-3.0) têm licenças próprias; consulte-as antes de reutilizar dados ou pesos.

## Como citar

Veja `CITATION.cff`. Referência da base: Hamamci, I. E. et al. *DENTEX: Dental Enumeration and Tooth Pathosis Detection Benchmark for Panoramic X-rays*. arXiv:2305.19112, 2023.
