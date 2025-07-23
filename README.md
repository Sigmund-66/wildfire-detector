# 🔥 🏞️ Modelo de detecção de fogo e fumaça (Wildfire Detect)
Modelo de detecção de fogo e fumaça em florestas e outros contextos.


# :clipboard:Sobre

Trata-se de um modelo de detecção de objetos que identica fogo e fumação em imagens. Foi desenvolvido para ser usado em contextos de monitoramento de florestas para procurar possíveis focos de incêndios ou situações que o fogo já está avançado. Para sua construção foi usado o modelo pré-treinado [Yolo da Ultralytics](https://docs.ultralytics.com/pt/models/yolo11/), já para obter os dados a serem usados no treinamento pegamos dois datasets de imagens públicos do site [Roboflow
Universe](https://universe.roboflow.com/) 

### Sobre o YOLO

YOLO **(You Only Look Once)**, é um modelo popular de detecção de objetos e segmentação de imagens desenvolvido por Joseph Redmon e Ali Farhadi na Universidade de Washington. Lançado em 2015, o YOLO ganhou popularidade por sua alta velocidade e precisão. Abaixo uma lista das versões já lançadas do YOLO e seu ano de lançamento.

+ O YOLOv2, lançado em 2016, aprimorou o modelo original incorporando normalização em lote, caixas delimitadoras (anchor boxes) e agrupamentos de dimensões.
+ YOLOv3, lançado em 2018, aprimorou ainda mais o desempenho do modelo usando uma rede de backbone mais eficiente, múltiplas âncoras e agrupamento de pirâmide espacial.
+ O YOLOv4 foi lançado em 2020, introduzindo inovações como aumento de dados Mosaic, um novo cabeçalho de detecção sem âncora e uma nova função de perda.
+ YOLOv5 melhorou ainda mais o desempenho do modelo e adicionou novos recursos, como otimização de hiperparâmetros, rastreamento de experimentos integrado e exportação automática para formatos de exportação populares.
+ YOLOv6 foi tornado de código aberto pela Meituan em 2022 e é usado em muitos dos robôs de entrega autônomos da empresa.
+ O YOLOv7 adicionou tarefas adicionais, como estimativa de pose no conjunto de dados de pontos-chave COCO.
+YOLOv8 lançado em 2023 pela Ultralytics, introduziu novos recursos e melhorias para desempenho, flexibilidade e eficiência aprimorados, oferecendo suporte a uma gama completa de tarefas de visão de IA.
+ O YOLOv9 introduz métodos inovadores como Programmable Gradient Information (PGI) e Generalized Efficient Layer Aggregation Network (GELAN).
+ O YOLOv10 criado por pesquisadores da Universidade de Tsinghua usando o pacote Python Ultralytics, fornece avanços de detecção de objetos em tempo real, introduzindo um cabeçalho End-to-End que elimina os requisitos de Supressão Não Máxima (NMS).
+ YOLO11 (mais novo): Os modelos YOLO mais recentes da Ultralytics oferecem desempenho de última geração (SOTA) em várias tarefas, incluindo detecção de objetos, segmentação, estimativa de pose, rastreamento e classificação, aproveitando os recursos em diversas aplicações e domínios de IA.

### Sobre o Roboflow

O site Roboflow Universe é uma plataforma colaborativa de datasets (conjuntos de dados) voltada para visão computacional. 
Trata-se de um repositório público onde pesquisadores, desenvolvedores, estudantes e empresas podem:

+ Compartilhar datasets anotados (com labels) para tarefas como detecção de objetos, segmentação, classificação, etc.

+ Explorar datasets criados por outros usuários.

+ Importar datasets diretamente para projetos de machine learning, especialmente com ferramentas como YOLO, TensorFlow, PyTorch, etc.

+ Visualizar e manipular datasets (ex: ver bounding boxes, tipos de classes, estatísticas de anotações, etc).

# :pushpin:Objetivo
Os incêndios florestais têm consequências devastadoras para o meio ambiente, a saúde humana e a economia. 
Sistemas baseados em visão computacional e inteligência artificial conseguem identificar sinais de fumaça 
ou chamas em tempo real, por meio de câmeras ou imagens de satélite. A resposta rápida evita que pequenos focos virem grandes incêndios.
Para isso a ideia de criar um modelo de detecção de objetos para monitorar florestas, detectar incêndios ou focos e adotar estratégias de combate e prevenção
o mais rapidamente.

# :question:Como usar


# :bar_chart:Resultados
Os resultados foram obtidos fazendo dois treinameinos distintos, sendo o segundo realizado com 
os pesos do primeiro utilizando a técnica do fine-tuning (treino contínuo). O segundo se saiu melhor do primeiro visto que além de utilizar os pesos do anterior também contou com um dataset mais robusto e diversificado. Ambos os treinos e testes foram feitos no ambiente do Google Colab.

## Treinamento 1
**🖥️Configuração da máquina:** 
+ Ambiente de execução Google Colab Tesla (T4) 
+ RAM do sistema 12.7GB 
+ RAM da GPU 15GB  
+ Disco 112.6GB
     
**Dataset utilizado:**  
:fire: **fire-smoke data Dataset** — por *me*, publicado no [Roboflow Universe](https://universe.roboflow.com/me-p4nto/fire-smoke-data), junho de 2025. Visitado em 18 de julho de 2025.  
**Modelo YOLOv11:** `yolo11s.pt`    
**Código do treino:**  

```
model = YOLO("yolo11s.pt")

resultados = model.train(
    data="/content/drive/MyDrive/fire-smoke_data.v3i.yolov11/data.yaml",\
    epochs=120,
    patience=40,
    imgsz=640,
    pretrained=True,
    batch=8,
    hsv_v=0.5,
    hsv_h=0.3,
    degrees=0.3,
    mosaic=0.5,
    mixup=0.0,
)
```
### Validação - Métricas  
| Class | Images | Instances  | Box(P) | R     | mAP50 | mAP50-95 |
|-------|-------:|-----------:|-------:|------:|------:|---------:|
| all   |  571   |   1540     | 0.726  | 0.737 | 0.777 | 0.489    |
| fire  |  317   |   912      | 0.688  | 0.734 | 0.747 | 0.405    |
| smoke |  447   |   628      | 0.764  | 0.739 | 0.806 | 0.573    |  

### Explicação das métricas
| Métrica | O que significa                                                                                         |
|---------|---------------------------------------------------------------------------------------------------------|
| Class   | Quantidade de classes que podme ser detectadas na imagem. Definidas no arquivo **data.yaml** do dataset.|
| Images  | Número de imagens usadas na validação (571 imagens).                                                    |
|Instances| Quantidade total de objetos rotulados no conjunto (1540 no total).                                      |
| Box(P)  | Precisão (Precision): proporção de predições corretas entre todas as predições feitas.                  |
| R       | Recall: proporção de objetos reais detectados corretamente pelo modelo.                                 |                            
| mAP50   | Média da precisão (AP) com IoU ≥ 0.50 — principal métrica usada para comparar modelos.                  |
| mAP50-95| Média da precisão em múltiplos thresholds de IoU (de 0.50 a 0.95 com passo de 0.05) —  métrica mais rigorosa e completa da COCO.|

**Todas as classes (desempenho geral):**
O modelo apresenta equilíbrio entre precisão e recall, e um **mAP@0.5 = 77.7%**, indicando boa qualidade nas detecções com sobreposição de pelo menos 50%. O **mAP@0.5:0.95** = 48.9% mostra que o desempenho diminui com limiares de IoU mais altos, o que é esperado.

**Classe fire:**
O modelo detecta bem o fogo, mas com precisão um pouco inferior à de smoke. Seu **mAP50-95 (40.5%)** e sugere que a detecção de fire é menos precisa nos limites de bounding box, ou tem variação maior nos dados.

**Classe smoke:**
A classe smoke tem o melhor desempenho geral. A **precisão** de **76.4%** e **mAP50** de **80.6%** indicam que o modelo é muito confiável para detectar fumaça.
O **mAP50-95** também é alto **(57.3%)**, o que mostra bom alinhamento das caixas preditas com as caixas reais mesmo em limiares mais rigorosos.


### Gráfico F1 confidence curve
<img width="2250" height="1500" alt="BoxF1_curve" src="https://github.com/user-attachments/assets/26528de3-fdc8-491e-bd4c-d58fb0865de1" />  

Esse gráfico é uma curva F1-Confidence gerado após o término do treinamento do modelo e localizada em `runs/detect/train`. Ele mostra como o valor F1-score varia em função do nível de confiança aplicado para filtrar as detecções do modelo.

+O eixo X (Confidence): representa o limiar de confiança utilizado nas predições. Valores variam de 0 a 1.Quanto mais alto o limiar, mais exigente o modelo é para 
  considerar uma detecção como válida.Quanto mais baixo, mais permissivo (aceita mais detecções, mesmo com baixa confiança).

+O eixo Y (F1): representa o valor do F1-score, que é a média harmônica entre precisão (precision) e revocação (recall).Valores mais altos indicam melhor equilíbrio 
  entre precisão e recall.

+O pico do F1-score geral (azul grosso) ocorre aproximadamente no ponto 0.317 de confiança, com um valor de F1 = 0.73.Isso significa que o melhor equilíbrio entre 
  precisão e recall é atingido quando o modelo considera apenas detecções com confiança acima de 31.7%.

+A curva da classe smoke (laranja) apresenta desempenho superior à de fire (azul claro), mantendo valores de F1 mais altos em toda a faixa de confiança.
+A curva fire tem uma queda mais acentuada em valores altos de confiança, indicando que o modelo se torna excessivamente seletivo e perde recall rapidamente nessa classe.


### Gráfico curve Precision x Recall
<img width="2250" height="1500" alt="BoxPR_curve" src="https://github.com/user-attachments/assets/9ce5c943-a25c-4b74-8374-71f89960c56e" />

Curva Precision-Recall (PR) gerada após a validação de um modelo YOLOv11 para detecção das classes fire e smoke. Ele fornece uma visão detalhada do equilíbrio entre precisão e revocação (recall) em diferentes thresholds de confiança.

+ **Classe smoke:** Melhor curva PR - modelo consegue manter alta precisão mesmo com recall elevado. Excelente desempenho.
+ **Classe fire:** Um pouco mais baixo → o modelo perde precisão mais rapidamente conforme tenta aumentar o recall. Isso pode indicar dificuldade de distinguir o fogo, ou caixas menos precisas.
+ **Desempenho geral:** mAP@0.5 = 0.777 mostra que o modelo tem bom desempenho global, compatível com aplicações reais.

### Matriz confusão normalizada (gráfico)
<img width="2250" height="1500" alt="confusion_matrix_normalized" src="https://github.com/user-attachments/assets/e113552a-4ba9-4c1f-9fee-6ab9ed38fadc" />

### Interpretação da matriz

+ Classe fire (linha 1):
81% das previsões como fire foram corretas. 2% das previsões como fire na verdade eram smoke.
74% das vezes que o modelo deveria prever background, ele errou e previu fire — indicando falsos positivos com fire.

+ Classe smoke (linha 2):
77% das previsões como smoke foram corretas.
3% na verdade eram fire. 26% das vezes que o modelo deveria prever background, ele previu smoke.

+ Classe background (linha 3):16% das previsões como background eram na verdade fire.
21% eram smoke e 63% {1 - (0.16 + 0.21)} eram de fato o background.  

O modelo está apresentando muitos falsos positivos para fire principalmente com o background (74%), apesar de ter uma boa precisão. A classe smoke teve o melhor desempenho, porém ainda se confunde um pouco com background(26%).


## Treinamento 2

**🖥️Configuração da máquina:** 
+ Ambiente de execução Google Colab Tesla (T4) 
+ RAM do sistema 12.7GB 
+ RAM da GPU 15GB  
+ Disco 112.6GB
     
**Dataset utilizado:**  
🚬 **smoke Dataset** — por *smoke*, publicado no [Roboflow Universe](https://universe.roboflow.com/smoke-3mr5d/smoke-thgo9), maio de 2025. Visitado em 18 de julho de 2025.  
**Modelo YOLOv11:** Modelo gerado no treinamento 1    
**Código do treino:**  

```

model = YOLO("best.pt")

resultados = model.train(
    data="/content/drive/MyDrive/smoke.v1i.yolov11/data.yaml",\
    epochs=70,
    patience=40,
    imgsz=640,
    hsv_v=0.6,
    hsv_h=0.015,
    hsv_s=0.7,
    degrees=5.0,
    shear=2.0,
    perspective=0.001,
    scale=0.5,
    mosaic=0.0,
    mixup=0.0,
)


```

### Validação - Métricas  
| Class | Images | Instances  | Box(P) | R     | mAP50 | mAP50-95 |
|-------|-------:|-----------:|-------:|------:|------:|---------:|
| all   |  357   |   581      | 0.917  | 0.868 | 0.929 | 0.628    |
| fire  |  141   |   223      | 0.877  | 0.857 | 0.909 | 0.57     |
| smoke |  321   |   358      | 0.958  | 0.880 | 0.950 | 0.686    |  

### Gráfico F1 confidence curve
<img width="2250" height="1500" alt="BoxF1_curve" src="https://github.com/user-attachments/assets/c5191c99-6c50-4d01-94b2-13d17409a59b" />

+O pico do F1-score geral (azul grosso) ocorre aproximadamente no ponto 0.317 de confiança, com um valor de F1 = 0.73.Isso significa que o melhor equilíbrio entre 
  precisão e recall é atingido quando o modelo considera apenas detecções com confiança acima de 31.7%.

+A curva da classe smoke (laranja) apresenta desempenho superior à de fire (azul claro), mantendo valores de F1 mais altos em toda a faixa de confiança.
+A curva fire tem uma queda mais acentuada em valores altos de confiança, indicando que o modelo se torna excessivamente seletivo e perde recall rapidamente nessa classe.


### Gráfico curve Precision x Recall
<img width="2250" height="1500" alt="BoxPR_curve" src="https://github.com/user-attachments/assets/0c4b5b65-b449-4e71-9cc4-33c36cac4357" />


+ **Classe smoke:** Melhor curva PR - modelo consegue manter alta precisão mesmo com recall elevado. Excelente desempenho.
+ **Classe fire:** Um pouco mais baixo → o modelo perde precisão mais rapidamente conforme tenta aumentar o recall. Isso pode indicar dificuldade de distinguir o fogo, ou caixas menos precisas.
+ **Desempenho geral:** mAP@0.5 = 0.777 mostra que o modelo tem bom desempenho global, compatível com aplicações reais.

### Matriz confusão normalizada (gráfico)
<img width="2250" height="1500" alt="confusion_matrix_normalized" src="https://github.com/user-attachments/assets/585c33f3-96b3-4b6d-9855-f0487a17da72" />


### Interpretação da matriz

+ Classe fire (linha 1):
81% das previsões como fire foram corretas. 2% das previsões como fire na verdade eram smoke.
74% das vezes que o modelo deveria prever background, ele errou e previu fire — indicando falsos positivos com fire.

+ Classe smoke (linha 2):
77% das previsões como smoke foram corretas.
3% na verdade eram fire. 26% das vezes que o modelo deveria prever background, ele previu smoke.

+ Classe background (linha 3):16% das previsões como background eram na verdade fire.
21% eram smoke e 63% {1 - (0.16 + 0.21)} eram de fato o background.  

O modelo está apresentando muitos falsos positivos para fire principalmente com o background (74%), apesar de ter uma boa precisão. A classe smoke teve o melhor desempenho, porém ainda se confunde um pouco com background(26%).



# :dart:Conclusão
A modelo do primeiro treinamento resultados inferiores ao segundo provavelmente em razão do seu dataset que possui muitas imagens duplicadas, proporção de imagens desiguais (mais imagens contendo a classe smoke do a classe fire) e muitos falsos positivos para fire que se confunde com o background. Mesmo assim, o modelo ainda pode ser útil principalmente para detectar fumaça, pois foi a classe que teve o melhor desempenho nas métricas. Sabendo dessas limitações e com alguns ajustes o modelo do treino 1 ainda pode ser bem utilizado.

Já o modelo do segundo treinamento apresentou um desempenho melhor em praticamente todas as métricas especialmente se tratando do classe smoke, o desempenho da classe fire também melhorou consideravelmente

