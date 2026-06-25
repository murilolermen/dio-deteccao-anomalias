#%%

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import shap
#%%

url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"


# Carregando o dataset
df = pd.read_csv(url)

df.head()




#Verificando a distribuição das classes
df['Class'].value_counts(normalize=True)


#log transformação em Amount para deixar a distribuição mais próxima de uma distribuição normal
df['amount_log'] = np.log1p(df['Amount'])


scaler = StandardScaler()

#calculando a média e o desvio padrão da coluna Amount e aplicando a transformação
df['amount_scaled'] = scaler.fit_transform(df[['Amount']])


x =df.drop('Class', axis=1)
y = df['Class']

#separando os dados em treino e teste, com 30% dos dados para teste, e estratificando a variável alvo para manter a proporção das classes
x_train, x_test, y_train, y_test = train_test_split(
    
    x, y, test_size=0.3, random_state=42, stratify=y
    
    )

#prever se a transação é fraudulenta ou não, utilizando regressão logística
# %%
model = LogisticRegression(max_iter=1000)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)




# %%
#mostrar o relatório de classificação, que inclui precisão, recall e f1-score para cada classe
print(classification_report(y_test, y_pred))


# %%
# 1. Calcula as taxas de Falsos Positivos (fpr) e Verdadeiros Positivos (tpr)
y_probs = model.predict_proba(x_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

# 2. Plota a Curva ROC
plt.plot(fpr, tpr)
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

# 3. Calcula e imprime a métrica AUC (Área Sob a Curva)
print("AUC:", roc_auc_score(y_test, y_probs))

#%%
# Calcula a curva de precisão-recall
precision, recall, _ = precision_recall_curve(y_test, y_probs)

plt.plot(recall, precision)
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()
# %%

#executando o modelo Random Forest para comparar com a regressão logística

rf = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

rf.fit(x_train, y_train)

y_pred_rf = rf.predict(x_test)

print(classification_report(y_test, y_pred_rf))
# %%

# Inicializa o modelo XGBoost sem o parâmetro obsoleto
xgb = XGBClassifier(
    scale_pos_weight=10, # ajuda com desbalanceamento
    eval_metric="logloss"
)

# Treina o modelo
xgb.fit(x_train, y_train)

# Faz as previsões
y_pred_xgb = xgb.predict(x_test)

# Imprime as métricas de avaliação
print(classification_report(y_test, y_pred_xgb))
# %%

#mostra impacto de cada variável na decisão do modelo

explainer = shap.TreeExplainer(xgb)
shap_values = explainer(x_test)

shap.plots.bar(shap_values)
# %%
