# Guia de reprodução

## Ambiente

Todos os experimentos foram executados em notebooks do Kaggle, com a Internet ligada e GPU NVIDIA Tesla T4 (o treinamento usa uma só GPU, `device=0`).

| Componente | Versão |
|---|---|
| Python | 3.13.15 |
| PyTorch | 2.11.0 (CUDA 12.8) |
| Ultralytics | 8.4.183 (fixada na primeira célula de cada notebook) |
| Demais pacotes | `remotezip`, `huggingface_hub`, `pyyaml`, `pycocotools`, `openpyxl` (instalados pela primeira célula) |

O arquivo `requirements.txt` lista os pacotes para quem quiser rodar fora do Kaggle. Fora dele, os caminhos `/kaggle/working` dos notebooks precisam ser ajustados, e os resultados podem diferir se a GPU ou as versões forem outras.

## Ordem de execução

Os notebooks são independentes: cada um baixa os dados de que precisa.

| Notebook | O que faz | Tempo aproximado | Saída |
|---|---|---|---|
| `01_baseline_yolov8n` | Linha de base: YOLOv8n a 640 px, com parada antecipada | cerca de 0,5 h | `baseline_artefatos`, zip |
| `02_experimentos_2a4_configuracao` | Experimentos 2 a 4, com avaliação pelo Ultralytics e pelo pycocotools | 4 a 6 h | `artefatos_exp2a4.zip` |
| `03_experimento_5_yolov8s_1024` | Experimento 5 (YOLOv8s a 1.024 px) | 1,5 a 2,5 h | `artefatos_exp5.zip` |
| `04_modelo_final_teste_interno` | Divisão do teste interno, treino do modelo final, limiar, avaliação no teste e artefatos para a análise | 1 a 1,5 h | `artefatos_modelo_final.zip` e o pacote de imagens para avaliação clínica |

Como rodar cada um: *Settings* → acelerador **GPU T4 x2** e **Internet On** → *Save Version* → *Save & Run All (Commit)*. Ao parar a sessão, `/kaggle/working` é apagado: baixe os arquivos da aba **Output** da versão salva.

## Determinismo

Todos os treinamentos usam `seed=42` e `deterministic=True`. A monografia relata que duas execuções do experimento 1 deram métricas idênticas e que as 46 primeiras épocas do experimento 2 coincidem com as do experimento 1. Isso vale para o mesmo ambiente; outra GPU ou outras versões podem alterar os números.

## Notas sobre o notebook 04

- O parâmetro `PESOS_PRONTOS` permite pular o treino com pesos já treinados. Use só pesos treinados **com a mesma divisão** deste notebook: o modelo do experimento 5 viu as imagens do teste interno no treino e não serve para avaliar nele.
- `AVALIAR_TESTE = True` avalia o teste interno uma única vez, depois de o modelo e o limiar estarem fixos.
- Os pacotes gerados incluem as imagens da base DENTEX e, por isso, não estão neste repositório.

## Scripts

Ambos rodam sem GPU. As instruções de uso estão no cabeçalho de cada arquivo.

- `scripts/analise_teste.py`: lê o `teste_predicoes.json` gerado pelo notebook 04 e calcula a revocação por região da arcada (intervalo de Wilson de 95%), a matriz de confusão no limiar escolhido, a análise com cárie e cárie profunda unidas e as figuras de exemplos. Precisa dos artefatos do notebook 04.
- `scripts/contagem_treino_por_regiao.py`: conta as anotações do treinamento por classe e por região da arcada, baixando só o arquivo de anotações do DENTEX.

## Arquivos de resultados

Todos são métricas agregadas, sem imagens. Os arquivos gerados pelos scripts de análise usam ponto e vírgula como separador.

| Arquivo | Conteúdo |
|---|---|
| `configuracao/resumo_exp2a4.csv`, `resumo_exp5.csv` | Por experimento e por pesos (`best` e `last`): épocas, melhor época, P, R, mAP50, mAP50-95, AP50 por classe, pelo Ultralytics e pelo pycocotools |
| `modelo_final/divisao_resumo.csv` | Imagens e caixas por classe em cada conjunto (599, 50 e 106 imagens) |
| `modelo_final/teste_interno_imagens.csv` | Nomes das 106 imagens do teste interno |
| `modelo_final/args_treino.yaml` | Hiperparâmetros do treino do modelo final |
| `modelo_final/validacao_metricas.csv`, `validacao_no_limiar.csv` | Métricas na validação (pycocotools) e acertos por classe no limiar escolhido |
| `modelo_final/curva_limiar_validacao.csv` | F1 por classe e médio para cada limiar de confiança na validação |
| `modelo_final/teste_metricas.csv`, `teste_no_limiar.csv` | Métricas no teste interno e VP, FP, FN, precisão, revocação e F1 por classe |
| `modelo_final/matriz_confusao_teste.csv` | Matriz de confusão no limiar escolhido (anotação × predição, com a coluna "não detectada") |
| `modelo_final/revocacao_por_regiao.csv` | Revocação por classe, região e arcada, com IC de Wilson 95% |
| `modelo_final/carie_unificada.csv` | Comparação com cárie e cárie profunda como uma só classe |
| `modelo_final/contagem_treino_por_regiao.csv` | Caixas do treinamento por classe, região e arcada |
