# Modelo de detecção de fogo e fumaça (Wildfire Detect)
Modelo de detecção de incêndios em florestas 



# :clipboard:Sobre


# :pushpin:Objetivo
Monitorar florestas para detectar incêndios ou focos para adotar estratégias de combate e prevenção.

# :question:Como usar

# :bar_chart:Resultados
Os resultados foram obtidos fazendo dois treinameinos distintos, sendo o segundo realizado com 
os pesos do primeiro utilizando a técnica do fine-tuning (treino contínuo). O segundo se saiu melhor do primeiro visto que além de utilizar os pesos do anterior também contou com um dataset mais robusto.

## Treinamento 1
**Configuração da máquina:**  
🖥️ Ambiente de execução Google Colab Tesla (T4) - RAM do sistema 12.7GB - RAM da GPU 15GB - Disco 112.6GB  
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



# :dart:Conclusão


