# Modelo de detecção de fogo e fumaça (Wildfire Detect)
Modelo de detecção de incêndios em florestas 

![train_batch27500](https://github.com/user-attachments/assets/6c5d7214-bdbc-40ae-90db-e294e35234c3)


# Objetivo
Monitorar florestas para detectar incêndios ou focos para adotar estratégias de combate e prevenção.

# Como usar

# Resultados
Os resultados foram obtidos fazendo dois treinameinos distintos, sendo o segundo realizado com 
os pesos do primeiro utilizando a técnica do fine-tuning (treino contínuo). O segundo se saiu melhor do primeiro visto que além de utilizar os pesos do anterior também contou com um dataset mais robusto.

## Treinamento 1
### Gráfico F1 confidence curve
<img width="2250" height="1500" alt="BoxF1_curve" src="https://github.com/user-attachments/assets/26528de3-fdc8-491e-bd4c-d58fb0865de1" />

### Gráfico curve Precision x Recall
<img width="2250" height="1500" alt="BoxPR_curve" src="https://github.com/user-attachments/assets/9ce5c943-a25c-4b74-8374-71f89960c56e" />

### Matriz confusão normalizada
<img width="3000" height="2250" alt="confusion_matrix_normalized" src="https://github.com/user-attachments/assets/e113552a-4ba9-4c1f-9fee-6ab9ed38fadc" />





# Conclusão


