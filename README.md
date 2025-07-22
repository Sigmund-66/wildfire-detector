# 🔥 🏞️ Modelo de detecção de fogo e fumaça (Wildfire Detect)
Modelo de detecção de incêndios em florestas 



# :clipboard:Sobre

Trata-se de um modelo de detecção de objetos que identica fogo e fumação em imagens. Foi desenvolvido para ser usado em contextos de monitoramento de florestas para procurar possíveis focos de incêndios ou situações que o fogo já está avançado. Para sua construção foi usado o modelo pré-treinado [Yolo da Ultralytics](https://docs.ultralytics.com/pt/models/yolo11/), já para obter os dados a serem usados no treinamento pegamos dois datasets de imagens públicos do site [Roboflow
Universe](https://universe.roboflow.com/) 

### Sobre o YOLO

YOLO (You Only Look Once), é um modelo popular de detecção de objetos e segmentação de imagens desenvolvido por Joseph Redmon e Ali Farhadi na Universidade de Washington. Lançado em 2015, o YOLO ganhou popularidade por sua alta velocidade e precisão. Abaixo uma lista das versões já lançadas do YOLO e seu ano de lançamento.

+O YOLOv2, lançado em 2016, aprimorou o modelo original incorporando normalização em lote, caixas delimitadoras (anchor boxes) e agrupamentos de dimensões.
+YOLOv3, lançado em 2018, aprimorou ainda mais o desempenho do modelo usando uma rede de backbone mais eficiente, múltiplas âncoras e agrupamento de pirâmide espacial.
+O YOLOv4 foi lançado em 2020, introduzindo inovações como aumento de dados Mosaic, um novo cabeçalho de detecção sem âncora e uma nova função de perda.
+YOLOv5 melhorou ainda mais o desempenho do modelo e adicionou novos recursos, como otimização de hiperparâmetros, rastreamento de experimentos integrado e exportação automática para formatos de exportação populares.
+YOLOv6 foi tornado de código aberto pela Meituan em 2022 e é usado em muitos dos robôs de entrega autônomos da empresa.
+O YOLOv7 adicionou tarefas adicionais, como estimativa de pose no conjunto de dados de pontos-chave COCO.
+YOLOv8 lançado em 2023 pela Ultralytics, introduziu novos recursos e melhorias para desempenho, flexibilidade e eficiência aprimorados, oferecendo suporte a uma gama completa de tarefas de visão de IA.
+O YOLOv9 introduz métodos inovadores como Programmable Gradient Information (PGI) e Generalized Efficient Layer Aggregation Network (GELAN).
+O YOLOv10 criado por pesquisadores da Universidade de Tsinghua usando o pacote Python Ultralytics, fornece avanços de detecção de objetos em tempo real, introduzindo um cabeçalho End-to-End que elimina os requisitos de Supressão Não Máxima (NMS).
+YOLO11 (mais novo): Os modelos YOLO mais recentes da Ultralytics oferecem desempenho de última geração (SOTA) em várias tarefas, incluindo detecção de objetos, segmentação, estimativa de pose, rastreamento e classificação, aproveitando os recursos em diversas aplicações e domínios de IA.

## Sobre o Roboflow


# :pushpin:Objetivo
Monitorar florestas para detectar incêndios ou focos para adotar estratégias de combate e prevenção.

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

### Gráfico F1 confidence curve
<img width="2250" height="1500" alt="BoxF1_curve" src="https://github.com/user-attachments/assets/26528de3-fdc8-491e-bd4c-d58fb0865de1" />

### Gráfico curve Precision x Recall
<img width="2250" height="1500" alt="BoxPR_curve" src="https://github.com/user-attachments/assets/9ce5c943-a25c-4b74-8374-71f89960c56e" />

### Matriz confusão normalizada
<img width="2250" height="1500" alt="confusion_matrix_normalized" src="https://github.com/user-attachments/assets/e113552a-4ba9-4c1f-9fee-6ab9ed38fadc" />

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
    pretrained=True,
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



# :dart:Conclusão


