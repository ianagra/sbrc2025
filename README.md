# Análise de Desempenho de Redes por Combinação de Pontos de Mudança e Modelo de Sobrevivência

[![Licença MIT](https://img.shields.io/badge/Licença-MIT-blue.svg)](LICENSE)

Repositório oficial da implementação referente ao artigo **"Análise de Desempenho de Redes pela Combinação de Pontos de Mudanças Estatísticas e Modelo de Sobrevivência"**, submetido ao **Simpósio Brasileiro de Redes de Computadores e Sistemas Distribuídos (SBRC) 2025**.

## Descrição

Este repositório contém o código-fonte, datasets e scripts necessários para reproduzir os experimentos do artigo, que propõe uma metodologia inovadora para análise de desempenho de redes combinando **detecção estatística de pontos de mudança** com **modelos de sobrevivência**.

## Estrutura do Repositório

```
.
├── 📂 datasets/ # Conjuntos de dados utilizados
│ ├── 📂 ts_ndt/ # Séries temporais para cada par cliente-servidor
│ ├── 📂 ts_ndt_cp/ # Séries temporais rotuladas com os pontos de mudança detectados pelo VWCD
│ ├── 📂 ts_ndt_results/ # Séries temporais rotuladas com os pontos de mudança, os clusters e as estatísticas locais
│ ├── 📜 dados_ndt.csv # Extrato do banco de dados de testes NDT para o período analisado
│ ├── 📜 dados_ndt.parquet # Extrato do banco de dados de testes NDT em formato parquet
│ ├── 📜 survival_ndt.parquet # Dataset de sobrevivência rotulado com os clusters e outras informações
│ └── 📜 ts_metadata_ndt.parquet # Informações das séries temporais
│
├── 📂 imgs/ # Imagens dos gráficos gerados
│
├── 📜 cluster_proportions_clients.csv # Tabela contendo as proporções de tempo em cada cluster para cada cliente
├── 📜 cluster_proportions_servers.csv # Tabela contendo as proporções de tempo em cada cluster para cada servidor
├── 📜 coefficients.csv # Tabela contendo os coeficientes da regressão logística associados ao cluster 1 de todas as features
├── 📜 coefficients.json # Tabela de coeficientes em formato JSON, passada para os LLMs
├── 📜 environment.yml # Arquivo de configuração do ambiente conda
├── 📜 survmixclust_thr.pkl # Modelo SurvMixClust treinado.
│
├── 📜 process_results.py # Script com as funções para processar os resultados e rotular os dados
├── 📜 SurvMixClust.py # Script com as funções que implementam o algoritmo SurvMixClust
├── 📜 SurvMixClust_utils.py # Script com as funções auxiliares do algoritmo SurvMixClust
├── 📜 timeseries_processing.py # Script com as funções de processamento das séries temporais
├── 📜 visual_analysis.py # Script com as funções de plotagem dos gráficos
├── 📜 VWCD.py # Script com as funções que implementam o algoritmo VWCD
│
├── 📜 use_example.ipynb # Jupyter notebook contendo a implementação da metodologia proposta
│
├── 📜 LICENSE # Arquivo contendo a licença de uso
└── 📜 README.md # Este arquivo
```

## Pré-requisitos

- **Gerenciador de Pacotes**: Conda (Miniconda ou Anaconda) instalado
- **Python**: Versão 3.7.x (gerenciada automaticamente pelo ambiente Conda)
- **Principais Bibliotecas**:
  - `lifelines` (modelos de sobrevivência)
  - `ruptures` (detecção de pontos de mudança)
  - `scikit-learn`, `pandas`, `numpy` (análise de dados)
  - `matplotlib`, `plotly` (visualização)
  - `rpy2` (integração Python-R)
  - `torch` (redes neurais)
- **R (4.2.2)**: Dependências R (e.g., `survival`, `ggplot2`, `dplyr`) são instaladas automaticamente via Conda
- **Jupyter Notebook**: Incluído no ambiente

## Instalação (Linux)

### Clone o repositório:

   ```bash
   git clone https://github.com/ianagra/sbrc2025.git
   cd sbrc2025
   ```

### Crie o ambiente Conda usando o arquivo `environment.yml`:

   ```bash
   conda env create -f environment.yml
   ```

### Ative o ambiente:

   ```bash
   conda activate sbrc2025
   ```

### Verifique a instalação:

   ```bash
   python -c "import lifelines, ruptures; print('Ambiente configurado!')"
   ```

O ambiente inclui 1.213 dependências (Python + R). A instalação pode levar 10-15 minutos dependendo da conexão.
Para uso com Jupyter Notebook, o kernel sbrc2025 estará automaticamente disponível.

## Contato

Para dúvidas, entre em contato:

    Ian Agra - ian@land.ufrj.br

Nota: Este trabalho está em revisão para o SBRC 2025. Os resultados podem sofrer ajustes antes da versão final.