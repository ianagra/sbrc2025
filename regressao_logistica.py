import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# Leitura do dataset
df = pd.read_csv('dataset_survival.csv')

# Seleção das features numéricas para padronização
numeric_features = [
    'throughput_download', 'rtt_download', 'throughput_upload', 'rtt_upload',
    'throughput_download_std', 'rtt_download_std', 'throughput_upload_std', 'rtt_upload_std'
]

# Seleção das features categóricas (one-hot encoding já aplicado)
client_features = [col for col in df.columns if col.startswith('client_')]
site_features = [col for col in df.columns if col.startswith('site_')]

# Combinando todas as features
X_features = numeric_features + client_features + site_features

# Preparando X e y
X = df[X_features].copy()
y = df['cluster']

# Padronização das features numéricas
scaler = StandardScaler()
X[numeric_features] = scaler.fit_transform(X[numeric_features])

# Treinamento do modelo de regressão logística
model = LogisticRegression(max_iter=5000)
model.fit(X, y)

# Análise dos coeficientes para o cluster 1
coef_dict = dict(zip(X_features, model.coef_[0]))

# Ordenando os coeficientes por valor absoluto
sorted_coef = sorted(coef_dict.items(), key=lambda x: abs(x[1]), reverse=True)

print("Análise dos coeficientes para o cluster 1:")
print("\nCoeficientes ordenados por importância (valor absoluto):")
print("-" * 60)
print(f"{'Feature':<40} {'Coeficiente':>15}")
print("-" * 60)
for feature, coef in sorted_coef:
    print(f"{feature:<40} {coef:>15.6f}")

# Análise adicional por grupo de features
print("\nAnálise por grupo de features:")
print("-" * 60)

# Métricas para features numéricas
numeric_coef = {k: v for k, v in coef_dict.items() if k in numeric_features}
print("\nFeatures numéricas:")
print(f"Média absoluta dos coeficientes: {np.mean(np.abs(list(numeric_coef.values()))):.6f}")
print("Top 3 features numéricas mais importantes:")
sorted_numeric = sorted(numeric_coef.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
for feature, coef in sorted_numeric:
    print(f"- {feature}: {coef:.6f}")

# Métricas para features de cliente
client_coef = {k: v for k, v in coef_dict.items() if k in client_features}
print("\nFeatures de cliente:")
print(f"Média absoluta dos coeficientes: {np.mean(np.abs(list(client_coef.values()))):.6f}")
print("Top 3 clientes mais impactantes:")
sorted_client = sorted(client_coef.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
for feature, coef in sorted_client:
    print(f"- {feature}: {coef:.6f}")

# Métricas para features de site
site_coef = {k: v for k, v in coef_dict.items() if k in site_features}
print("\nFeatures de site:")
print(f"Média absoluta dos coeficientes: {np.mean(np.abs(list(site_coef.values()))):.6f}")
print("Todos os sites ordenados por impacto:")
sorted_site = sorted(site_coef.items(), key=lambda x: abs(x[1]), reverse=True)
for feature, coef in sorted_site:
    print(f"- {feature}: {coef:.6f}")

# Salvando os coeficientes em um arquivo CSV
coef_df = pd.DataFrame(sorted_coef, columns=['Feature', 'Coefficient'])
coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()

# Adicionando uma coluna para categorizar as features
def categorize_feature(feature):
    if feature in numeric_features:
        return 'Numeric'
    elif feature in client_features:
        return 'Client'
    else:
        return 'Site'

coef_df['Feature_Type'] = coef_df['Feature'].apply(categorize_feature)

# Ordenando por valor absoluto do coeficiente
coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False)

# Salvando o DataFrame em um arquivo CSV
coef_df.to_csv('logistic_regression_coefficients.csv', index=False)
print("\nCoeficientes salvos no arquivo 'logistic_regression_coefficients.csv'")