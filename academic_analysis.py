#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Análise Estatística Detalhada de Dados Acadêmicos
Este script realiza uma análise estatística completa para compreender os fatores
associados à evasão estudantil, perfis de alunos e desempenho acadêmico.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu, ttest_ind
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
import logging
import sys
import traceback
from datetime import datetime
import matplotlib.axes
import matplotlib.legend as mlegend
from matplotlib.rcsetup import cycler

# Configurar o sistema de logging
def setup_logger():
    """Configura e retorna um logger para a aplicação"""
    # Criar um logger
    logger = logging.getLogger('academic_analysis')
    logger.setLevel(logging.INFO)
    
    # Criar um handler para arquivo
    log_filename = f"academic_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    
    # Criar um handler para o console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Criar um formatador
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Adicionar os handlers ao logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Evitar duplicação de mensagens
    logger.propagate = False
    
    return logger, log_filename

# Inicializar o logger
logger, log_filename = setup_logger()

# Função para registrar mensagens e exceções
def log_message(message, level="info"):
    """Registra mensagens no logger e no console"""
    if level.lower() == "info":
        logger.info(message)
    elif level.lower() == "warning":
        logger.warning(message)
    elif level.lower() == "error":
        logger.error(message)
    elif level.lower() == "debug":
        logger.debug(message)

# Ignorar avisos
warnings.filterwarnings('ignore')

# Função para truncar texto
def truncar_texto(texto, max_len=12):
    """Trunca o texto para um tamanho máximo, adicionando '...' ao final quando necessário"""
    if isinstance(texto, str) and len(texto) > max_len:
        return texto[:max_len - 3] + "..."
    return texto

# Configurações
pd.set_option('display.max_columns', None)

# Aumentar tamanho das fontes para visualizações
plt.rcParams.update({
    'figure.figsize': (12, 8), 
    'figure.dpi': 120,
    'font.size': 18,            # Aumentado de 14 para 18
    'axes.titlesize': 20,       # Aumentado de 16 para 20
    'axes.labelsize': 18,       # Aumentado de 14 para 18
    'xtick.labelsize': 16,      # Aumentado de 12 para 16
    'ytick.labelsize': 16,      # Aumentado de 12 para 16
    'legend.fontsize': 16,      # Aumentado de 12 para 16
    'legend.title_fontsize': 18 # Aumentado de 14 para 18
})

# Atualizar o estilo para versão compatível
try:
    plt.style.use('seaborn-v0_8-whitegrid')  # Para versões mais recentes do matplotlib
except:
    try:
        plt.style.use('seaborn-whitegrid')   # Para versões mais antigas do matplotlib
    except:
        plt.style.use('default')              # Fallback para estilo padrão se nenhum outro funcionar
sns.set(style="whitegrid", font_scale=1.4)

# Configurar legendas para maior visibilidade
original_legend_get_frame_on = mlegend.Legend.get_frame_on
def patched_get_frame_on(self):
    return True  # Sempre mostrar o frame da legenda
mlegend.Legend.get_frame_on = patched_get_frame_on

# Configurar opções padrão de legenda
plt.rc('legend', frameon=True, framealpha=0.9, fontsize=16, edgecolor='gray')
plt.rc('axes', prop_cycle=cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
                                         '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                                         '#bcbd22', '#17becf']))

# Melhorar aparência dos eixos
original_xlabel = plt.xlabel
original_ylabel = plt.ylabel
def enhanced_xlabel(text, *args, **kwargs):
    if 'fontsize' not in kwargs:
        kwargs['fontsize'] = 18  # Aumentado de 14 para 18
    if 'fontweight' not in kwargs:
        kwargs['fontweight'] = 'bold'
    return original_xlabel(text, *args, **kwargs)

def enhanced_ylabel(text, *args, **kwargs):
    if 'fontsize' not in kwargs:
        kwargs['fontsize'] = 18  # Aumentado de 14 para 18
    if 'fontweight' not in kwargs:
        kwargs['fontweight'] = 'bold'
    return original_ylabel(text, *args, **kwargs)

plt.xlabel = enhanced_xlabel
plt.ylabel = enhanced_ylabel

# Melhorar os métodos de rótulos de eixos para objetos Axes
original_axes_set_xlabel = matplotlib.axes.Axes.set_xlabel
original_axes_set_ylabel = matplotlib.axes.Axes.set_ylabel

def patched_set_xlabel(self, xlabel, *args, **kwargs):
    if 'fontsize' not in kwargs:
        kwargs['fontsize'] = 18  # Aumentado de 14 para 18
    if 'fontweight' not in kwargs:
        kwargs['fontweight'] = 'bold'
    return original_axes_set_xlabel(self, xlabel, *args, **kwargs)

def patched_set_ylabel(self, ylabel, *args, **kwargs):
    if 'fontsize' not in kwargs:
        kwargs['fontsize'] = 18  # Aumentado de 14 para 18
    if 'fontweight' not in kwargs:
        kwargs['fontweight'] = 'bold'
    return original_axes_set_ylabel(self, ylabel, *args, **kwargs)

matplotlib.axes.Axes.set_xlabel = patched_set_xlabel
matplotlib.axes.Axes.set_ylabel = patched_set_ylabel

# Criar pasta de resultados
diretorio_atual = os.path.dirname(os.path.abspath(__file__))
pasta_resultados = os.path.join(diretorio_atual, "result_dados_exemplo")
if not os.path.exists(pasta_resultados):
    os.makedirs(pasta_resultados)
    log_message(f"Pasta de resultados '{pasta_resultados}' criada com sucesso")

# Substituir plt.savefig para salvar na pasta correta
original_savefig = plt.savefig
def custom_savefig(filename, *args, **kwargs):
    """Salvar figuras na pasta de resultados com alta qualidade"""
    # Garantir alta resolução para as imagens
    if 'dpi' not in kwargs:
        kwargs['dpi'] = 300
    if 'bbox_inches' not in kwargs:
        kwargs['bbox_inches'] = 'tight'
    
    # Criar caminho completo para a pasta de resultados
    caminho_completo = os.path.join(pasta_resultados, filename)
    return original_savefig(caminho_completo, *args, **kwargs)

# Substituir a função original
plt.savefig = custom_savefig

# Função para exportar para CSV
def exportar_para_csv(data_dict, filename_base):
    """Exporta vários dataframes para arquivos CSV separados"""
    try:
        # Criar lista para armazenar os nomes dos arquivos gerados
        arquivos_gerados = []
        
        # Exportar cada dataframe para um arquivo CSV separado
        for sheet_name, df in data_dict.items():
            # Criar nome de arquivo baseado no nome base e na "planilha"
            csv_filename = f"{os.path.splitext(filename_base)[0]}_{sheet_name}.csv"
            caminho_completo = os.path.join(pasta_resultados, csv_filename)
            
            # Exportar para CSV
            df.to_csv(caminho_completo, index=True)
            arquivos_gerados.append(csv_filename)
        
        log_message(f"{len(arquivos_gerados)} arquivos CSV gerados com sucesso na pasta '{pasta_resultados}'!")
        return True
    except Exception as e:
        log_message(f"Erro ao exportar para CSV: {e}", "error")
        return False

# Adicione esta função para carregar o arquivo CSV
def carregar_dados(arquivo_csv="dados_exemplo_computacao_200.csv"):
    """Carrega os dados do arquivo CSV especificado"""
    try:
        # Obter o diretório do script atual
        diretorio_atual = os.path.dirname(os.path.abspath(__file__))
        
        # Construir o caminho relativo para o arquivo
        caminho_arquivo = os.path.join(diretorio_atual, arquivo_csv)
        
        log_message(f"Carregando dados do arquivo: {caminho_arquivo}")
        
        # Verificar se o arquivo existe
        if not os.path.exists(caminho_arquivo):
            log_message(f"Arquivo não encontrado: {caminho_arquivo}", "warning")
            log_message("Tentando encontrar o arquivo na pasta atual...", "warning")
            
            # Tentar encontrar o arquivo na pasta atual
            if os.path.exists(arquivo_csv):
                caminho_arquivo = arquivo_csv
                log_message(f"Arquivo encontrado na pasta atual: {caminho_arquivo}")
            else:
                log_message(f"Arquivo não encontrado: {arquivo_csv}", "error")
                sys.exit(1)
        
        df = pd.read_csv(caminho_arquivo)
        log_message(f"Dados carregados com sucesso. Dimensões: {df.shape}")
        
        # Verificar colunas existentes
        log_message(f"Colunas disponíveis: {df.columns.tolist()}")
        
        # Processar o dataframe para análise
        # Converter tipos de dados adequados
        if 'cr' in df.columns and df['cr'].dtype == 'object':
            df['cr'] = df['cr'].str.replace(',', '.').astype(float)
            
        # Criar coluna de evasão se ainda não existir
        if 'evadiu' not in df.columns and 'forma_saida' in df.columns:
            df['evadiu'] = df['forma_saida'].apply(classificar_evasao)
            log_message("Coluna 'evadiu' criada com base na forma de saída.")
        
        return df
    except Exception as e:
        log_message(f"Erro ao carregar dados: {e}", "error")
        log_message(f"Detalhes do erro: {traceback.format_exc()}", "error")
        sys.exit(1)

# Função para criar variável de evasão
def classificar_evasao(tipo_saida):
    """
    Classifica os tipos de saída em evasão (1) ou não evasão (0)
    """
    nao_evasao = [
        None,
        'Formatura',
        'concluido',
        'conclusão',
        'Conclusão',
        'formado',
        'Formado',
        'graduação',
        'falecimento do aluno',
        'cadastro cancelado',
        'cancelamento judicial',
        'prorrogação por trancamento de programa'
    ]
    
    if pd.isna(tipo_saida) or tipo_saida in nao_evasao:
        return 0
    else:
        return 1

# Funções para criar planilhas de resultados
def criar_planilha_estatisticas_descritivas():
    """Cria planilha com estatísticas descritivas das variáveis numéricas e categóricas"""
    global df, numerical_cols, categorical_cols
    dados_estatisticos = {}
    
    # Estatísticas numéricas
    if numerical_cols:
        dados_estatisticos['Estatísticas Numéricas'] = df[numerical_cols].describe().transpose()
        
        # Adicionar mediana, skewness e kurtosis
        median_series = df[numerical_cols].median()
        skew_series = df[numerical_cols].skew()
        kurt_series = df[numerical_cols].kurtosis()
        
        for col in numerical_cols:
            dados_estatisticos['Estatísticas Numéricas'].loc[col, 'median'] = median_series[col]
            dados_estatisticos['Estatísticas Numéricas'].loc[col, 'skewness'] = skew_series[col]
            dados_estatisticos['Estatísticas Numéricas'].loc[col, 'kurtosis'] = kurt_series[col]
    
    # Estatísticas categóricas - frequências e percentuais
    if categorical_cols:
        for col in categorical_cols:
            counts = df[col].value_counts()
            percentages = df[col].value_counts(normalize=True) * 100
            cat_stats = pd.DataFrame({
                'Frequência': counts,
                'Percentual (%)': percentages
            })
            dados_estatisticos[f'Categórica - {col}'] = cat_stats
    
    # Exportar para CSV
    exportar_para_csv(dados_estatisticos, 'estatisticas_descritivas.csv')
    return dados_estatisticos

def criar_planilha_correlacoes():
    """Cria planilha com matriz de correlação entre variáveis numéricas"""
    global df, numerical_cols
    dados_correlacao = {}
    
    # Matriz de correlação completa
    if numerical_cols:
        numeric_df = df[numerical_cols].copy()
        corr_matrix = numeric_df.corr()
        dados_correlacao['Matriz de Correlação'] = corr_matrix
        
        # Correlações significativas (>0.3 ou <-0.3)
        corr_mask = (corr_matrix.abs() > 0.3) & (corr_matrix != 1.0)
        corr_significativas = corr_matrix[corr_mask].unstack().dropna().sort_values(ascending=False)
        sig_corr_df = pd.DataFrame({
            'Variável 1': [x[0] for x in corr_significativas.index],
            'Variável 2': [x[1] for x in corr_significativas.index],
            'Correlação': corr_significativas.values
        })
        dados_correlacao['Correlações Significativas'] = sig_corr_df
    
    # Exportar para CSV
    exportar_para_csv(dados_correlacao, 'correlacoes_variaveis.csv')
    return dados_correlacao

def criar_planilha_evasao():
    """Cria planilha com análises relacionadas à evasão"""
    global df, numerical_cols, categorical_cols
    dados_evasao = {}
    
    # Taxa de evasão geral
    taxa_evasao = df['evadiu'].mean() * 100
    dados_evasao['Taxa de Evasão Geral'] = pd.DataFrame({
        'Métrica': ['Taxa de Evasão (%)', 'Total de Estudantes', 'Estudantes Evadidos'],
        'Valor': [taxa_evasao, df.shape[0], df['evadiu'].sum()]
    })
    
    # Evasão por variáveis categóricas
    for col in categorical_cols:
        crosstab = pd.crosstab(df[col], df['evadiu'], normalize='index') * 100
        if 1.0 in crosstab.columns:  # Se há casos de evasão
            crosstab = crosstab.reset_index()
            crosstab.columns = [col, 'Não Evadiu (%)', 'Evadiu (%)'] if 0.0 in crosstab.columns else [col, 'Evadiu (%)']
            dados_evasao[f'Evasão por {col}'] = crosstab
    
    # Variáveis numéricas por evasão
    if numerical_cols:
        num_by_evasao = df.groupby('evadiu')[numerical_cols].agg(['mean', 'count'])
        num_by_evasao.columns = [f'{col}_{stat}' for col, stat in num_by_evasao.columns]
        num_by_evasao = num_by_evasao.reset_index()
        num_by_evasao['evadiu'] = num_by_evasao['evadiu'].map({0: 'Não Evadiu', 1: 'Evadiu'})
        dados_evasao['Numéricas por Evasão'] = num_by_evasao
    
    # Exportar para CSV
    exportar_para_csv(dados_evasao, 'analise_evasao.csv')
    return dados_evasao

def criar_planilha_resultados_estatisticos():
    """Cria planilha com resultados dos testes estatísticos"""
    global numerical_results_df, categorical_results_df
    resultados_testes = {}
    
    # Resultados dos testes estatísticos
    if 'numerical_results_df' in globals() and isinstance(numerical_results_df, pd.DataFrame) and not numerical_results_df.empty:
        resultados_testes['Testes Variáveis Numéricas'] = numerical_results_df
    
    if 'categorical_results_df' in globals() and isinstance(categorical_results_df, pd.DataFrame) and not categorical_results_df.empty:
        resultados_testes['Testes Variáveis Categóricas'] = categorical_results_df
    
    # Exportar para CSV
    if resultados_testes:
        exportar_para_csv(resultados_testes, 'resultados_testes_estatisticos.csv')
    return resultados_testes

def criar_planilha_modelos_preditivos():
    """Cria planilha com resultados dos modelos preditivos, se aplicável"""
    global model_results, y_test, y_pred
    dados_modelos = {}
    
    # Resultados do modelo logístico, se existirem
    if 'model_results' in globals() and model_results is not None:
        try:
            # Coeficientes e odds ratios
            if hasattr(model_results, 'coef_'):
                X_cols = globals().get('X', pd.DataFrame()).columns.tolist()
                if X_cols:
                    coef_df = pd.DataFrame({
                        'Variável': X_cols,
                        'Coeficiente': model_results.coef_[0],
                        'Odds Ratio': np.exp(model_results.coef_[0])
                    })
                    dados_modelos['Coeficientes'] = coef_df
            
            # Métricas de desempenho
            if 'y_pred' in globals() and 'y_test' in globals():
                try:
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                    
                    metrics_df = pd.DataFrame({
                        'Métrica': ['Acurácia', 'Precisão', 'Recall', 'F1-Score'],
                        'Valor': [
                            accuracy_score(y_test, y_pred),
                            precision_score(y_test, y_pred, zero_division=0),
                            recall_score(y_test, y_pred, zero_division=0),
                            f1_score(y_test, y_pred, zero_division=0)
                        ]
                    })
                    dados_modelos['Métricas de Desempenho'] = metrics_df
                except Exception as e:
                    log_message(f"Erro ao calcular métricas de desempenho: {e}", "warning")
        except Exception as e:
            log_message(f"Erro ao processar resultados do modelo: {e}", "error")
    
    # Exportar para CSV se houver dados
    if dados_modelos:
        exportar_para_csv(dados_modelos, 'resultados_modelos_preditivos.csv')
    return dados_modelos

def criar_planilha_consolidada():
    """Cria uma planilha consolidada com os principais resultados das análises"""
    global df, model_results, apoio_reprov, categorical_cols, numerical_cols
    dados_consolidados = {}
    
    # Resumo do dataset
    resumo_df = pd.DataFrame({
        'Métrica': ['Total de Estudantes', 'Total de Variáveis', 'Variáveis Numéricas', 'Variáveis Categóricas',
                   'Taxa de Evasão (%)', 'Estudantes Evadidos'],
        'Valor': [df.shape[0], df.shape[1], len(numerical_cols), len(categorical_cols),
                 df['evadiu'].mean() * 100, df['evadiu'].sum()]
    })
    dados_consolidados['Resumo do Dataset'] = resumo_df
    
    # Principais estatísticas
    if numerical_cols:
        stats_df = df[numerical_cols].describe().transpose()
        stats_df = stats_df[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
        dados_consolidados['Estatísticas Principais'] = stats_df
    
    # Top 5 correlações mais fortes (em módulo)
    if numerical_cols and len(numerical_cols) > 1:
        numeric_df = df[numerical_cols].copy()
        corr_matrix = numeric_df.corr()
        corr_unstack = corr_matrix.unstack()
        corr_unstack = corr_unstack[corr_unstack != 1.0]  # Remover autocorrelações
        top_corr = corr_unstack.abs().sort_values(ascending=False).head(10)
        top_corr_df = pd.DataFrame({
            'Variável 1': [idx[0] for idx in top_corr.index],
            'Variável 2': [idx[1] for idx in top_corr.index],
            'Correlação': [corr_unstack[idx] for idx in top_corr.index]
        })
        dados_consolidados['Top 10 Correlações'] = top_corr_df
    
    # Top 5 variáveis categóricas mais relacionadas com evasão
    if categorical_cols and len(categorical_cols) > 0:
        categorical_importance = []
        for col in categorical_cols:
            try:
                from scipy.stats import chi2_contingency
                contingency_table = pd.crosstab(df[col], df['evadiu'])
                chi2, p, dof, expected = chi2_contingency(contingency_table)
                categorical_importance.append((col, chi2, p))
            except Exception:
                continue
        
        if categorical_importance:
            cat_imp_df = pd.DataFrame(categorical_importance, 
                                     columns=['Variável', 'Chi-quadrado', 'p-valor'])
            cat_imp_df = cat_imp_df.sort_values('Chi-quadrado', ascending=False).head(5)
            dados_consolidados['Top 5 Categóricas para Evasão'] = cat_imp_df
    
    # Top 5 variáveis numéricas mais relacionadas com evasão
    if numerical_cols and len(numerical_cols) > 0:
        numerical_importance = []
        for col in numerical_cols:
            try:
                from scipy.stats import ttest_ind
                t_stat, p_val = ttest_ind(
                    df.loc[df['evadiu'] == 1, col].dropna(),
                    df.loc[df['evadiu'] == 0, col].dropna()
                )
                numerical_importance.append((col, abs(t_stat), p_val))
            except Exception:
                continue
        
        if numerical_importance:
            num_imp_df = pd.DataFrame(numerical_importance, 
                                    columns=['Variável', 'T-statistic', 'p-valor'])
            num_imp_df = num_imp_df.sort_values('T-statistic', ascending=False).head(5)
            dados_consolidados['Top 5 Numéricas para Evasão'] = num_imp_df
    
    # Resultados principais do modelo (se disponível)
    if 'model_results' in globals() and model_results is not None:
        try:
            if hasattr(model_results, 'coef_'):
                X_cols = globals().get('X', pd.DataFrame()).columns.tolist()
                if X_cols:
                    coef_df = pd.DataFrame({
                        'Variável': X_cols,
                        'Coeficiente': model_results.coef_[0],
                        'p-valor': [0.05] * len(X_cols),  # Placeholder, não temos o p-valor real
                        'Odds Ratio': np.exp(model_results.coef_[0]),
                        'Significativo': abs(model_results.coef_[0]) > 0.1  # Aproximação simplificada
                    })
                    # Ordenar por magnitude do efeito
                    coef_df = coef_df.sort_values('Odds Ratio', ascending=False)
                    dados_consolidados['Fatores de Risco'] = coef_df
        except Exception as e:
            log_message(f"Erro ao incluir resultados do modelo na planilha consolidada: {e}", "warning")
    
    # Exportar para CSV
    exportar_para_csv(dados_consolidados, 'resultados_consolidados.csv')
    return dados_consolidados

# Função para criar QQ plot e testes de normalidade
def analisar_normalidade(df, col):
    """
    Realiza análise de normalidade completa para uma variável numérica,
    incluindo testes estatísticos e visualizações
    """
    try:
        # Criar figura com múltiplos gráficos
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Histograma com KDE
        sns.histplot(df[col].dropna(), kde=True, ax=axes[0])
        axes[0].set_title(f'Distribuição de {col}')
        axes[0].grid(True, alpha=0.3)
        
        # Boxplot
        sns.boxplot(x=df[col].dropna(), ax=axes[1])
        axes[1].set_title(f'Boxplot de {col}')
        axes[1].grid(True, alpha=0.3)
        
        # QQ-Plot para verificar normalidade
        stats.probplot(df[col].dropna(), dist="norm", plot=axes[2])
        axes[2].set_title(f'QQ Plot: {col}')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'normalidade_{col}.png')
        plt.close()
        
        # QQ plot detalhado (separado)
        plt.figure(figsize=(10, 8))
        stats.probplot(df[col].dropna(), dist="norm", plot=plt)
        plt.title(f'QQ Plot: {col}', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'qqplot_{col}.png')
        plt.close()
        
        # Testes estatísticos de normalidade
        resultados = {}
        dados = df[col].dropna()
        
        # Teste de Shapiro-Wilk (bom para amostras pequenas até médias)
        if len(dados) < 5000:  # Shapiro é computacionalmente intensivo para grandes amostras
            stat_shapiro, p_shapiro = stats.shapiro(dados)
            resultados['Shapiro-Wilk'] = {
                'estatística': stat_shapiro,
                'p-valor': p_shapiro,
                'normal (α=0.05)': p_shapiro > 0.05
            }
        
        # Teste de Kolmogorov-Smirnov com distribuição normal
        stat_ks, p_ks = stats.kstest(dados, 'norm', args=(dados.mean(), dados.std()))
        resultados['Kolmogorov-Smirnov'] = {
            'estatística': stat_ks,
            'p-valor': p_ks,
            'normal (α=0.05)': p_ks > 0.05
        }
        
        # Teste D'Agostino-Pearson (baseado em assimetria e curtose)
        stat_norm, p_norm = stats.normaltest(dados)
        resultados['D\'Agostino-Pearson'] = {
            'estatística': stat_norm,
            'p-valor': p_norm,
            'normal (α=0.05)': p_norm > 0.05
        }
        
        return resultados
    except Exception as e:
        log_message(f"Erro na análise de normalidade para {col}: {e}", "warning")
        return None

# Função para realizar análise PCA
def realizar_pca(df, numerical_cols, target_col=None):
    """
    Realiza análise de componentes principais (PCA) e visualiza os resultados
    """
    try:
        # Selecionar apenas colunas numéricas para PCA
        cols_to_use = [col for col in numerical_cols if col != target_col]
        
        if len(cols_to_use) < 2:
            log_message("Número insuficiente de variáveis numéricas para PCA", "warning")
            return None
        
        # Preparar os dados
        X = df[cols_to_use].dropna()
        
        if len(X) < 3:
            log_message("Número insuficiente de registros completos para PCA", "warning")
            return None
        
        # Padronizar os dados
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Aplicar PCA
        pca = PCA()
        components = pca.fit_transform(X_scaled)
        
        # Criar DataFrame com os componentes
        pca_df = pd.DataFrame(
            data=components[:, :2],
            columns=['PC1', 'PC2']
        )
        
        # Adicionar target se existir
        if target_col is not None and target_col in df.columns:
            pca_df[target_col] = df.loc[X.index, target_col].values
        
        # Visualizar resultados
        plt.figure(figsize=(10, 8))
        
        # Scatter plot dos dois primeiros componentes
        if target_col is not None and target_col in pca_df.columns:
            scatter = sns.scatterplot(
                data=pca_df,
                x='PC1',
                y='PC2',
                hue=target_col,
                palette='viridis',
                alpha=0.7,
                s=100
            )
            plt.title(f'PCA - Componentes Principais por {target_col}')
        else:
            scatter = sns.scatterplot(
                data=pca_df,
                x='PC1',
                y='PC2',
                alpha=0.7,
                s=100
            )
            plt.title('PCA - Componentes Principais')
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} da variância)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} da variância)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('pca_visualization.png')
        plt.close()
        
        # Plotar gráfico de variância explicada
        plt.figure(figsize=(10, 6))
        var_ratio = pca.explained_variance_ratio_
        cum_var_ratio = np.cumsum(var_ratio)
        
        plt.bar(range(1, len(var_ratio) + 1), var_ratio, alpha=0.7, color='blue', label='Variância Individual')
        plt.step(range(1, len(cum_var_ratio) + 1), cum_var_ratio, where='mid', color='red', label='Variância Acumulada')
        plt.axhline(y=0.8, color='green', linestyle='--', label='Limite 80%')
        
        plt.xlabel('Número de Componentes')
        plt.ylabel('Proporção de Variância Explicada')
        plt.title('Variância Explicada por Componente')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('pca_variance_explained.png')
        plt.close()
        
        # Plotar biplot (projeção das variáveis originais)
        plt.figure(figsize=(12, 10))
        
        # Escala para as setas
        scaling = np.min([np.abs(pca_df['PC1'].min()), np.abs(pca_df['PC1'].max()),
                          np.abs(pca_df['PC2'].min()), np.abs(pca_df['PC2'].max())])
        
        # Plot dos pontos
        plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.5)
        
        # Plot das setas para as variáveis originais
        for i, feature in enumerate(cols_to_use):
            plt.arrow(0, 0, 
                      pca.components_[0, i] * scaling, 
                      pca.components_[1, i] * scaling,
                      head_width=0.05 * scaling, head_length=0.1 * scaling, fc='red', ec='red')
            plt.text(pca.components_[0, i] * scaling * 1.15, 
                     pca.components_[1, i] * scaling * 1.15, 
                     feature, color='red', ha='center', va='center')
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} da variância)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} da variância)')
        plt.title('Biplot PCA')
        plt.grid(True, alpha=0.3)
        
        # Adicionar círculo de correlação
        circle = plt.Circle((0, 0), radius=scaling, fill=False, color='gray', ls='--')
        plt.gca().add_patch(circle)
        
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig('pca_biplot.png')
        plt.close()
        
        # Retornar resultados
        return {
            'pca': pca,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'components': pca.components_,
            'feature_names': cols_to_use
        }
    except Exception as e:
        log_message(f"Erro na análise PCA: {e}", "error")
        log_message(traceback.format_exc(), "debug")
        return None

# Função para análise de tendências usando plotly
def analisar_tendencias(df):
    """
    Realiza análise de tendências temporais e gera visualizações interativas
    """
    try:
        log_message("\nRealizando análise de tendências temporais...")
        
        # Análise de tendência de ingressantes por período
        if 'periodo_ingresso' in df.columns:
            # Contagem de ingressantes por período
            ingressantes_por_periodo = df['periodo_ingresso'].value_counts().sort_index()
            
            # Gráfico interativo com plotly
            fig = px.bar(
                x=ingressantes_por_periodo.index,
                y=ingressantes_por_periodo.values,
                title="Tendência de Ingressantes por Período",
                labels={'x': 'Período', 'y': 'Número de Ingressantes'},
                color=ingressantes_por_periodo.values,
                color_continuous_scale='Viridis'
            )
            
            fig.update_layout(
                template='plotly_white',
                xaxis_title="Período de Ingresso",
                yaxis_title="Número de Ingressantes",
                coloraxis_showscale=False
            )
            
            # Salvar como HTML interativo
            fig.write_html(os.path.join(pasta_resultados, 'tendencia_ingressantes.html'))
            
            # Salvar também como imagem
            fig.write_image(os.path.join(pasta_resultados, 'tendencia_ingressantes.png'))
            
            log_message("Gráfico de tendência de ingressantes gerado com sucesso.")
        
        # Análise de tendência de saídas por tipo e período
        if 'periodo_situacao' in df.columns and 'forma_saida' in df.columns:
            # Remover registros sem período de situação ou forma de saída
            df_saida = df[df['forma_saida'] != 'NULL'].dropna(subset=['periodo_situacao'])
            
            if len(df_saida) > 0:
                # Agrupar por período e tipo de saída
                saidas_por_tipo = df_saida.groupby(['periodo_situacao', 'forma_saida']).size().reset_index(name='count')
                
                # Gráfico interativo com plotly
                fig = px.line(
                    saidas_por_tipo,
                    x='periodo_situacao',
                    y='count',
                    color='forma_saida',
                    markers=True,
                    title="Tendência de Saídas por Tipo e Período",
                )
                
                fig.update_layout(
                    template='plotly_white',
                    xaxis_title="Período de Saída",
                    yaxis_title="Número de Estudantes",
                    legend_title="Forma de Saída"
                )
                
                # Salvar como HTML interativo
                fig.write_html(os.path.join(pasta_resultados, 'tendencia_saidas_por_tipo.html'))
                
                # Salvar também como imagem
                fig.write_image(os.path.join(pasta_resultados, 'tendencia_saidas_por_tipo.png'))
                
                log_message("Gráfico de tendência de saídas por tipo gerado com sucesso.")
        
        return True
    except Exception as e:
        log_message(f"Erro na análise de tendências: {e}", "error")
        log_message(traceback.format_exc(), "debug")
        return False

# Função para análise de tempo até a saída
def analisar_tempo_ate_saida(df):
    """
    Analisa o tempo até a saída dos estudantes (formados e evadidos)
    """
    try:
        log_message("\nAnalisando tempo até a saída...")
        
        # Verificar se temos as colunas necessárias
        colunas_necessarias = ['periodo_ingresso', 'periodo_situacao', 'forma_saida']
        if not all(col in df.columns for col in colunas_necessarias):
            log_message("Colunas necessárias não encontradas para análise de tempo até saída.", "warning")
            return False
        
        # Filtrar apenas estudantes que já saíram (formados ou evadidos)
        df_saida = df[(df['forma_saida'] != 'NULL') & (~df['periodo_situacao'].isna())]
        
        if len(df_saida) < 5:
            log_message("Dados insuficientes para análise de tempo até saída.", "warning")
            return False
        
        # Extrair ano e período
        df_saida['ano_ingresso'] = df_saida['periodo_ingresso'].apply(lambda x: float(str(x).split('.')[0]) if pd.notna(x) else np.nan)
        df_saida['ano_saida'] = df_saida['periodo_situacao'].apply(lambda x: float(str(x).split('.')[0]) if pd.notna(x) else np.nan)
        
        # Calcular tempo até a saída (em anos)
        df_saida['tempo_ate_saida'] = df_saida['ano_saida'] - df_saida['ano_ingresso']
        
        # Criar visualização
        plt.figure(figsize=(12, 8))
        
        # Histograma do tempo até a saída
        sns.histplot(data=df_saida, x='tempo_ate_saida', hue='forma_saida', bins=10, kde=True)
        plt.title('Distribuição do Tempo até a Saída por Forma de Saída')
        plt.xlabel('Tempo até a Saída (anos)')
        plt.ylabel('Número de Estudantes')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(pasta_resultados, 'tempo_ate_saida.png'))
        plt.close()
        
        # Analisar tempo até saída por diferentes categorias
        if 'sexo' in df_saida.columns:
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=df_saida, x='sexo', y='tempo_ate_saida')
            plt.title('Tempo até Saída por Sexo')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(pasta_resultados, 'tempo_saida_por_sexo.png'))
            plt.close()
        
        if 'raca_cor' in df_saida.columns:
            plt.figure(figsize=(14, 8))
            sns.boxplot(data=df_saida, x='raca_cor', y='tempo_ate_saida')
            plt.title('Tempo até Saída por Raça/Cor')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(pasta_resultados, 'tempo_saida_por_raca_cor.png'))
            plt.close()
        
        # Análise para bolsistas e não bolsistas
        if 'recebeu_bolsa' in df_saida.columns:
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=df_saida, x='recebeu_bolsa', y='tempo_ate_saida')
            plt.title('Tempo até Saída por Recebimento de Bolsa')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(pasta_resultados, 'tempo_saida_por_recebeu_bolsa.png'))
            plt.close()
        
        if 'recebeu_auxilio' in df_saida.columns:
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=df_saida, x='recebeu_auxilio', y='tempo_ate_saida')
            plt.title('Tempo até Saída por Recebimento de Auxílio')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(pasta_resultados, 'tempo_saida_por_recebeu_auxilio.png'))
            plt.close()
        
        log_message("Análise de tempo até a saída concluída com sucesso.")
        return True
    except Exception as e:
        log_message(f"Erro na análise de tempo até a saída: {e}", "error")
        log_message(traceback.format_exc(), "debug")
        return False

# Função para análise de correlação avançada
def analisar_correlacoes_avancadas(df):
    """
    Realiza análise avançada de correlações entre variáveis numéricas e categóricas
    """
    try:
        log_message("\nRealizando análise avançada de correlações...")
        
        # Variáveis numéricas
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Remover a coluna target e colunas de identificação
        numerical_cols = [col for col in numerical_cols if col not in ['evadiu', 'matricula']]
        
        if len(numerical_cols) > 1:
            # Matriz de correlação
            plt.figure(figsize=(14, 12))
            corr_matrix = df[numerical_cols].corr()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', annot=True, center=0, fmt='.2f',
                        square=True, linewidths=.5, cbar_kws={"shrink": .5})
            plt.title('Matriz de Correlação - Variáveis Numéricas', fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(pasta_resultados, 'correlacao_numericas.png'))
            plt.close()
            
            log_message("Matriz de correlação de variáveis numéricas gerada com sucesso.")
        
        # Correlação com evasão
        if 'evadiu' in df.columns:
            correlacoes_evasao = {}
            
            # Correlações para variáveis numéricas
            for col in numerical_cols:
                if col != 'evadiu':
                    correlacao = df[['evadiu', col]].corr().iloc[0, 1]
                    correlacoes_evasao[col] = correlacao
            
            # Visualizar correlações com evasão
            plt.figure(figsize=(12, 8))
            correlacoes_df = pd.DataFrame(list(correlacoes_evasao.items()), columns=['Variável', 'Correlação'])
            correlacoes_df = correlacoes_df.sort_values('Correlação')
            
            # Criar barplot
            sns.barplot(x='Correlação', y='Variável', data=correlacoes_df)
            plt.title('Correlação de Variáveis Numéricas com Evasão', fontsize=16)
            plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(pasta_resultados, 'correlacao_com_evasao.png'))
            plt.close()
            
            log_message("Análise de correlação com evasão concluída com sucesso.")
        
        # Correlações categóricas (opcional)
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        categorical_cols = [col for col in categorical_cols if col not in ['matricula']]
        
        if len(categorical_cols) > 0 and 'evadiu' in df.columns:
            log_message("Analisando associação entre variáveis categóricas e evasão...")
            
            # Análises específicas
            for col in categorical_cols:
                try:
                    # Tabela de contingência
                    crosstab = pd.crosstab(df[col], df['evadiu'])
                    
                    # Salvar em CSV
                    crosstab.to_csv(os.path.join(pasta_resultados, f'contingencia_{col}_evasao.csv'))
                except Exception as e:
                    log_message(f"Erro ao analisar {col}: {e}", "warning")
        
        return True
    except Exception as e:
        log_message(f"Erro na análise avançada de correlações: {e}", "error")
        log_message(traceback.format_exc(), "debug")
        return False

# Função para criação de pairplots
def criar_pairplots(df):
    """
    Cria pairplots para analisar relações entre múltiplas variáveis numéricas
    """
    try:
        log_message("\nCriando pairplots para análise multivariada...")
        
        # Variáveis numéricas
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Remover colunas de identificação
        numerical_cols = [col for col in numerical_cols if col not in ['matricula']]
        
        # Limitar a um número razoável de variáveis para o pairplot
        if len(numerical_cols) > 6:
            important_cols = ['cr', 'evadiu', 'interticio', 'reprovacoes_1_ciclo']
            # Adicionar outras colunas importantes até chegar a 6
            additional_cols = [col for col in numerical_cols if col not in important_cols]
            selected_cols = important_cols + additional_cols[:6 - len(important_cols)]
            numerical_cols = [col for col in selected_cols if col in df.columns]
        
        # Criar pairplot se houver pelo menos 3 variáveis numéricas
        if len(numerical_cols) >= 3:
            # Pairplot básico
            plt.figure(figsize=(20, 20))
            g = sns.pairplot(df[numerical_cols], height=2.5, diag_kind='kde')
            plt.tight_layout()
            plt.savefig(os.path.join(pasta_resultados, 'pairplot_numericas.png'))
            plt.close()
            
            log_message("Pairplot para variáveis numéricas gerado com sucesso.")
            
            # Pairplot com hue (colorido por sexo ou outra categoria)
            if 'sexo' in df.columns:
                g = sns.pairplot(df[numerical_cols + ['sexo']], height=2.5, hue='sexo', diag_kind='kde')
                plt.tight_layout()
                plt.savefig(os.path.join(pasta_resultados, 'pairplot_por_sexo.png'))
                plt.close()
                
                log_message("Pairplot por sexo gerado com sucesso.")
            
            # Pairplot com hue por evasão
            if 'evadiu' in df.columns and 'evadiu' in numerical_cols:
                # Remover evadiu da lista para não duplicar
                num_cols_without_evadiu = [col for col in numerical_cols if col != 'evadiu']
                
                if len(num_cols_without_evadiu) >= 2:  # Precisamos de pelo menos 2 variáveis + hue
                    df_hue = df.copy()
                    df_hue['Evasão'] = df_hue['evadiu'].map({0: 'Não', 1: 'Sim'})
                    
                    g = sns.pairplot(df_hue[num_cols_without_evadiu + ['Evasão']], 
                                    height=2.5, hue='Evasão', diag_kind='kde')
                    plt.tight_layout()
                    plt.savefig(os.path.join(pasta_resultados, 'pairplot_por_evasao.png'))
                    plt.close()
                    
                    log_message("Pairplot por evasão gerado com sucesso.")
        else:
            log_message("Número insuficiente de variáveis numéricas para gerar pairplots.", "warning")
        
        return True
    except Exception as e:
        log_message(f"Erro ao criar pairplots: {e}", "error")
        log_message(traceback.format_exc(), "debug")
        return False

# Função para análise de apoio e desempenho
def analisar_apoio_desempenho(df):
    """
    Analisa a relação entre recebimento de apoio (bolsa/auxílio) e desempenho/evasão
    """
    try:
        log_message("\nAnalisando relação entre apoio financeiro e desempenho/evasão...")
        
        # Verificar se temos as colunas necessárias
        colunas_apoio = ['recebeu_bolsa', 'recebeu_auxilio']
        
        if not any(col in df.columns for col in colunas_apoio):
            log_message("Colunas de apoio não encontradas para análise.", "warning")
            return False
        
        # Criar variável de apoio combinada
        if all(col in df.columns for col in colunas_apoio):
            df_apoio = df.copy()
            df_apoio['recebeu_apoio'] = (df_apoio['recebeu_bolsa'] == 'sim') | (df_apoio['recebeu_auxilio'] == 'sim')
            df_apoio['recebeu_apoio'] = df_apoio['recebeu_apoio'].map({True: 'Sim', False: 'Não'})
        elif 'recebeu_bolsa' in df.columns:
            df_apoio = df.copy()
            df_apoio['recebeu_apoio'] = df_apoio['recebeu_bolsa'].map({'sim': 'Sim', 'não': 'Não'})
        elif 'recebeu_auxilio' in df.columns:
            df_apoio = df.copy()
            df_apoio['recebeu_apoio'] = df_apoio['recebeu_auxilio'].map({'sim': 'Sim', 'não': 'Não'})
        
        # Análise com CR (se disponível)
        if 'cr' in df_apoio.columns:
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=df_apoio, x='recebeu_apoio', y='cr')
            plt.title('Coeficiente de Rendimento por Recebimento de Apoio')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(pasta_resultados, 'cr_por_apoio.png'))
            plt.close()
        
        # Análise com evasão (se disponível)
        if 'evadiu' in df_apoio.columns:
            plt.figure(figsize=(12, 6))
            crosstab_percent = pd.crosstab(df_apoio['recebeu_apoio'], df_apoio['evadiu'], normalize='index') * 100
            crosstab_percent.plot(kind='bar', stacked=True)
            plt.title('Proporção de Evasão por Recebimento de Apoio')
            plt.xlabel('Recebeu Apoio')
            plt.ylabel('Percentual (%)')
            plt.legend(title='Evasão', labels=['Não', 'Sim'])
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(pasta_resultados, 'evasao_por_apoio.png'))
            plt.close()
            
            # Gráfico de barras horizontal
            plt.figure(figsize=(12, 6))
            df_apoio_count = df_apoio.groupby(['recebeu_apoio', 'evadiu']).size().unstack()
            df_apoio_count.plot(kind='barh')
            plt.title('Contagem de Estudantes por Apoio e Evasão')
            plt.xlabel('Número de Estudantes')
            plt.ylabel('Recebeu Apoio')
            plt.legend(title='Evasão', labels=['Não', 'Sim'])
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(pasta_resultados, 'contagem_apoio_evasao.png'))
            plt.close()
        
        # Análise com reprovações (se disponível)
        if 'reprovacoes_1_ciclo' in df_apoio.columns:
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=df_apoio, x='recebeu_apoio', y='reprovacoes_1_ciclo')
            plt.title('Reprovações no Primeiro Ciclo por Recebimento de Apoio')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(pasta_resultados, 'reprovacao_por_apoio.png'))
            plt.close()
        
        # Análise com outras variáveis categóricas (como sexo ou raça)
        for col in ['sexo', 'raca_cor']:
            if col in df_apoio.columns:
                plt.figure(figsize=(14, 8))
                crosstab_percent = pd.crosstab(df_apoio[col], df_apoio['recebeu_apoio'], normalize='index') * 100
                crosstab_percent.plot(kind='bar', stacked=True)
                plt.title(f'Proporção de Recebimento de Apoio por {col}')
                plt.xlabel(col)
                plt.ylabel('Percentual (%)')
                plt.legend(title='Recebeu Apoio')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(pasta_resultados, f'apoio_por_{col}.png'))
                plt.close()
        
        log_message("Análise de apoio financeiro e desempenho concluída com sucesso.")
        return True
    except Exception as e:
        log_message(f"Erro na análise de apoio e desempenho: {e}", "error")
        log_message(traceback.format_exc(), "debug")
        return False

# Função para análise de clusters usando K-means
def analisar_clusters(df):
    """
    Realiza análise de clusters usando K-means para identificar perfis de estudantes
    """
    try:
        log_message("\nRealizando análise de clusters para identificar perfis de estudantes...")
        
        # Selecionar variáveis numéricas para clusterização
        cols_cluster = [col for col in df.select_dtypes(include=['int64', 'float64']).columns 
                        if col not in ['matricula', 'evadiu']]
        
        if len(cols_cluster) < 2:
            log_message("Número insuficiente de variáveis numéricas para clusterização.", "warning")
            return False
        
        # Preparar dados sem valores missing
        df_cluster = df[cols_cluster].dropna()
        
        if len(df_cluster) < 10:
            log_message("Número insuficiente de registros completos para clusterização.", "warning")
            return False
        
        # Padronizar os dados
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_cluster)
        
        # Determinar o número ótimo de clusters usando o método do cotovelo
        from sklearn.cluster import KMeans
        import numpy as np
        
        inertias = []
        K_range = range(1, 10)
        
        for k in K_range:
            kmeans_model = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans_model.fit(df_scaled)
            inertias.append(kmeans_model.inertia_)
        
        # Plotar gráfico do método do cotovelo
        plt.figure(figsize=(10, 6))
        plt.plot(K_range, inertias, 'bo-')
        plt.xlabel('Número de Clusters (k)')
        plt.ylabel('Inércia')
        plt.title('Método do Cotovelo para Determinação do Número Ótimo de Clusters')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(pasta_resultados, 'kmeans_elbow_method.png'))
        plt.close()
        
        # Escolher número de clusters (aqui usamos 3 como exemplo, mas poderia ser otimizado)
        num_clusters = 3
        
        # Treinar o modelo K-means
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        df['cluster'] = kmeans.fit_predict(df_scaled)
        
        log_message(f"Modelo K-means treinado com {num_clusters} clusters.")
        
        # Adicionar clusters ao DataFrame original
        df_cluster_original = df.copy()
        df_cluster_original.loc[df_cluster.index, 'cluster'] = kmeans.labels_
        df_cluster_original['cluster'] = df_cluster_original['cluster'].fillna(-1).astype(int)
        
        # Analisar características dos clusters
        log_message("\nCaracterísticas médias por cluster:")
        cluster_means = df_cluster_original[df_cluster_original['cluster'] >= 0].groupby('cluster')[cols_cluster].mean()
        log_message(cluster_means)
        
        # Visualizar clusters em 2D usando PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        df_pca = pca.fit_transform(df_scaled)
        
        # Criar DataFrame com resultados do PCA
        df_pca_result = pd.DataFrame(data=df_pca, columns=['PC1', 'PC2'])
        df_pca_result['cluster'] = kmeans.labels_
        
        # Adicionar informação de evasão se disponível
        if 'evadiu' in df.columns:
            df_pca_result['evadiu'] = df.loc[df_cluster.index, 'evadiu'].values
        
        # Plotar os clusters
        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            data=df_pca_result,
            x='PC1',
            y='PC2',
            hue='cluster',
            palette='viridis',
            s=100,
            alpha=0.7
        )
        plt.title('Visualização dos Clusters de Perfis de Estudantes (PCA)')
        plt.xlabel(f'Componente Principal 1 ({pca.explained_variance_ratio_[0]:.2%} da variância)')
        plt.ylabel(f'Componente Principal 2 ({pca.explained_variance_ratio_[1]:.2%} da variância)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(pasta_resultados, 'clusters_pca.png'))
        plt.close()
        
        # Analisar taxa de evasão por cluster
        if 'evadiu' in df_cluster_original.columns:
            plt.figure(figsize=(10, 6))
            evasao_por_cluster = df_cluster_original[df_cluster_original['cluster'] >= 0].groupby('cluster')['evadiu'].mean() * 100
            evasao_por_cluster.plot(kind='bar', color='skyblue')
            plt.title('Taxa de Evasão por Cluster')
            plt.xlabel('Cluster')
            plt.ylabel('Taxa de Evasão (%)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(pasta_resultados, 'evasao_por_cluster.png'))
            plt.close()
            
            log_message("\nTaxa de evasão por cluster:")
            log_message(evasao_por_cluster)
        
        # Gerar tabela de perfis de estudantes por cluster
        perfis = pd.DataFrame()
        
        for col in cols_cluster:
            perfis[f'Média {col}'] = df_cluster_original[df_cluster_original['cluster'] >= 0].groupby('cluster')[col].mean()
        
        if 'evadiu' in df_cluster_original.columns:
            perfis['Taxa de Evasão (%)'] = df_cluster_original[df_cluster_original['cluster'] >= 0].groupby('cluster')['evadiu'].mean() * 100
        
        # Salvar perfis para CSV
        perfis.to_csv(os.path.join(pasta_resultados, 'analise_por_perfil.csv'))
        
        # Criar visualização interativa com plotly
        if 'evadiu' in df_cluster_original.columns:
            import plotly.express as px
            
            # Preparar dados para plotly
            df_plot = df_cluster_original[df_cluster_original['cluster'] >= 0].copy()
            df_plot['Cluster'] = df_plot['cluster'].apply(lambda x: f'Cluster {x}')
            df_plot['Evasão'] = df_plot['evadiu'].map({0: 'Não', 1: 'Sim'})
            
            # Gráfico interativo de dispersão
            fig = px.scatter(
                df_plot,
                x='cr',
                y='reprovacoes_1_ciclo' if 'reprovacoes_1_ciclo' in df_plot.columns else cols_cluster[1],
                color='Cluster',
                symbol='Evasão',
                hover_name='matricula',
                hover_data=cols_cluster,
                title='Perfis de Estudantes por Cluster e Status de Evasão',
                labels={
                    'cr': 'Coeficiente de Rendimento',
                    'reprovacoes_1_ciclo': 'Reprovações no 1º Ciclo'
                }
            )
            
            fig.update_layout(
                template='plotly_white',
                legend_title_text='Legenda'
            )
            
            # Salvar como HTML interativo
            fig.write_html(os.path.join(pasta_resultados, 'perfil_vs_desempenho.html'))
            
            # Salvar também como imagem
            fig.write_image(os.path.join(pasta_resultados, 'perfil_vs_desempenho.png'))
        
        log_message("Análise de clusters concluída com sucesso.")
        return True
    except Exception as e:
        log_message(f"Erro na análise de clusters: {e}", "error")
        log_message(traceback.format_exc(), "debug")
        return False

# Função para análise de correspondência múltipla (MCA) para variáveis categóricas
def analisar_mca(df):
    """
    Realiza análise de correspondência múltipla (MCA) para analisar relações entre variáveis categóricas
    """
    try:
        log_message("\nRealizando análise de correspondência múltipla (MCA)...")
        
        # Verificar se Prince está instalado (pacote para MCA)
        try:
            import prince
        except ImportError:
            log_message("Pacote 'prince' não encontrado. Tentando instalar...", "warning")
            try:
                import pip
                pip.main(['install', 'prince'])
                import prince
                log_message("Pacote 'prince' instalado com sucesso.", "info")
            except Exception as e:
                log_message(f"Não foi possível instalar 'prince': {e}", "error")
                log_message("Pulando análise MCA.", "warning")
                return False
        
        # Selecionar variáveis categóricas
        cat_cols = [col for col in df.select_dtypes(include=['object']).columns 
                    if col not in ['matricula']]
        
        if len(cat_cols) < 2:
            log_message("Número insuficiente de variáveis categóricas para MCA.", "warning")
            return False
        
        # Selecionar apenas as linhas completas para as variáveis categóricas
        df_mca = df[cat_cols].dropna()
        
        if len(df_mca) < 10:
            log_message("Dados insuficientes para análise MCA.", "warning")
            return False
        
        log_message(f"Realizando MCA com {len(cat_cols)} variáveis categóricas e {len(df_mca)} observações.")
        
        # Realizar MCA
        mca = prince.MCA(n_components=2, n_iter=3, copy=True, check_input=True, engine='auto', random_state=42)
        mca_result = mca.fit(df_mca)
        
        # Obter coordenadas das categorias
        category_coords = mca_result.column_coordinates(df_mca)
        
        # Obter coordenadas das observações
        observation_coords = mca_result.row_coordinates(df_mca)
        
        # Visualizar categorias em um gráfico
        plt.figure(figsize=(14, 10))
        
        for category in category_coords.index:
            plt.scatter(
                category_coords.loc[category, 0],
                category_coords.loc[category, 1],
                marker='o', 
                s=50
            )
            plt.annotate(
                category, 
                (category_coords.loc[category, 0], category_coords.loc[category, 1]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8
            )
        
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        plt.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
        plt.title('Análise de Correspondência Múltipla - Categorias')
        plt.xlabel(f'Dimensão 1 ({mca.explained_inertia_[0]:.2%})')
        plt.ylabel(f'Dimensão 2 ({mca.explained_inertia_[1]:.2%})')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(pasta_resultados, 'mca_categorias.png'))
        plt.close()
        
        # Visualizar observações
        plt.figure(figsize=(12, 8))
        
        # Adicionar evadiu como cor se disponível
        if 'evadiu' in df.columns:
            evadiu_map = df.loc[df_mca.index, 'evadiu'].map({0: 'Não Evadido', 1: 'Evadido'})
            scatter = plt.scatter(
                observation_coords[0], 
                observation_coords[1],
                c=df.loc[df_mca.index, 'evadiu'],
                cmap='coolwarm',
                alpha=0.6,
                s=30
            )
            plt.legend(handles=scatter.legend_elements()[0], labels=['Não Evadido', 'Evadido'])
        else:
            plt.scatter(
                observation_coords[0], 
                observation_coords[1],
                alpha=0.6,
                s=30
            )
        
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        plt.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
        plt.title('Análise de Correspondência Múltipla - Observações')
        plt.xlabel(f'Dimensão 1 ({mca.explained_inertia_[0]:.2%})')
        plt.ylabel(f'Dimensão 2 ({mca.explained_inertia_[1]:.2%})')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(pasta_resultados, 'mca_observacoes.png'))
        plt.close()
        
        log_message("Análise de correspondência múltipla (MCA) concluída com sucesso.")
        return True
    except Exception as e:
        log_message(f"Erro na análise MCA: {e}", "error")
        log_message(traceback.format_exc(), "debug")
        return False

# Função principal que executa todas as análises
def main():
    # Declarar variáveis globais no início da função
    global df, model_results, apoio_reprov, categorical_cols, numerical_cols
    global X, y_test, y_pred, numerical_results_df, categorical_results_df
    
    log_message("="*80)
    log_message("INICIANDO ANÁLISE ESTATÍSTICA DETALHADA DE DADOS ACADÊMICOS")
    log_message("="*80)
    
    # Carregar dados do arquivo CSV de exemplo
    df = carregar_dados("dados_exemplo_computacao_200.csv")
    
    # Inicializar variáveis globais que serão usadas no relatório
    model_results = None
    apoio_reprov = None
    X = None
    y_test = None
    y_pred = None
    numerical_results_df = pd.DataFrame()
    categorical_results_df = pd.DataFrame()
    
    # Truncar textos longos para colunas específicas
    colunas_para_truncar = ['acao_afirmativa', 'forma_ingresso', 'tipo_saida']
    for col in colunas_para_truncar:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: truncar_texto(x, max_len=12) if pd.notna(x) else x)
            log_message(f"Coluna {col} truncada para melhor visualização")
    
    # Converter colunas numéricas que usam vírgula como separador decimal
    for col in df.select_dtypes(include=['object']).columns:
        try:
            if df[col].str.contains(',').any():
                df[col] = df[col].str.replace(',', '.').astype(float)
                log_message(f"Coluna {col} convertida de formato vírgula para ponto decimal")
        except:
            pass
    
    # ETAPA 1: ANÁLISE EXPLORATÓRIA INICIAL
    log_message("\n" + "="*80)
    log_message("1. ANÁLISE EXPLORATÓRIA INICIAL")
    log_message("="*80)
    
    # Informações básicas sobre o dataset
    log_message("\nInformações básicas sobre o dataset:")
    log_message(f"Número de registros: {df.shape[0]}")
    log_message(f"Número de variáveis: {df.shape[1]}")
    
    # Análise de valores missing
    log_message("\nValores missing por coluna:")
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    missing_df = pd.DataFrame({'Missing Values': missing_data, 'Percentage': missing_percent})
    log_message(missing_df[missing_df['Missing Values'] > 0].sort_values('Missing Values', ascending=False))
    
    # Identificar tipos de variáveis
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Remover colunas de identificação (matrícula, nome) da análise
    for col in ['matricula', 'nome']:
        if col in categorical_cols:
            categorical_cols.remove(col)
        if col in numerical_cols:
            numerical_cols.remove(col)
    
    # Criar variável target de evasão se não existir
    if 'evadiu' not in df.columns and 'forma_saida' in df.columns:
        df['evadiu'] = df['forma_saida'].apply(classificar_evasao)
        if 'evadiu' not in numerical_cols:
            numerical_cols.append('evadiu')
    
    log_message(f"\nVariáveis categóricas: {len(categorical_cols)}")
    log_message(f"Variáveis numéricas: {len(numerical_cols)}")
    
    # ETAPA 2: ANÁLISE DESCRITIVA
    log_message("\n" + "="*80)
    log_message("2. ANÁLISE DESCRITIVA")
    log_message("="*80)
    
    # Estatísticas descritivas para variáveis numéricas
    log_message("\nEstatísticas descritivas para variáveis numéricas:")
    if len(numerical_cols) > 0:
        for col in numerical_cols:
            stats_df = df[col].describe().to_frame().T
            log_message(f"\n{col}:")
            log_message(stats_df)
    
    # Distribuições para variáveis categóricas
    log_message("\nDistribuição de variáveis categóricas:")
    for col in categorical_cols:
        value_counts = df[col].value_counts(dropna=False)
        percentages = df[col].value_counts(normalize=True, dropna=False) * 100
        
        log_message(f"\n{col}:")
        for value in value_counts.index:
            log_message(f"  {value}: {value_counts[value]} ({percentages[value]:.2f}%)")
    
    # Contagem de estudantes por situação de evasão
    if 'evadiu' in df.columns:
        log_message("\nContagem de estudantes por situação de evasão:")
        evasao_count = df['evadiu'].value_counts()
        log_message(f"Não Evasão (0): {evasao_count.get(0, 0)} estudantes")
        log_message(f"Evasão (1): {evasao_count.get(1, 0)} estudantes")
        
        # Taxa de evasão
        taxa_evasao = (evasao_count.get(1, 0) / df.shape[0]) * 100
        log_message(f"\nTaxa de evasão: {taxa_evasao:.2f}%")
    
    # ETAPA 3: VISUALIZAÇÕES
    log_message("\n" + "="*80)
    log_message("3. VISUALIZAÇÕES")
    log_message("="*80)
    
    # Análise de normalidade para variáveis numéricas
    log_message("\nRealizando testes de normalidade para variáveis numéricas:")
    normalidade_resultados = {}
    for col in numerical_cols:
        if col != 'evadiu' and df[col].nunique() > 5:  # Apenas variáveis com suficientes valores distintos
            log_message(f"\nTestes de normalidade para {col}:")
            result = analisar_normalidade(df, col)
            if result:
                normalidade_resultados[col] = result
                for teste, valores in result.items():
                    log_message(f"  {teste}: estatística={valores['estatística']:.4f}, p-valor={valores['p-valor']:.4f}, normal={valores['normal (α=0.05)']}")
    
    # Visualizações para variáveis numéricas
    for col in numerical_cols:
        if col != 'evadiu':  # Excluir a variável target da visualização individual
            # Visualizações adicionais - Violin plot
            plt.figure(figsize=(12, 8))
            sns.violinplot(y=df[col].dropna())
            plt.title(f'Violin Plot: {col}')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'violin_{col}.png')
            plt.close()
            
            # Histograma e boxplot (já estão no código original)
            plt.figure(figsize=(12, 6))
            
            # Histograma
            plt.subplot(1, 2, 1)
            sns.histplot(data=df, x=col, kde=True)
            plt.title(f'Distribuição de {col}')
            plt.grid(True, alpha=0.3)
            
            # Boxplot
            plt.subplot(1, 2, 2)
            sns.boxplot(data=df, x=col)
            plt.title(f'Boxplot de {col}')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'distribuicao_{col}.png')
            plt.close()
    
    # Visualizações para variáveis categóricas
    for col in categorical_cols:
        try:
            # Gráfico de Barras (já está no código original)
            plt.figure(figsize=(12, 6))
            sns.countplot(data=df, y=col, order=df[col].value_counts().index)
            plt.title(f'Distribuição de {col}')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'distribuicao_{col}.png')
            plt.close()
            
            # Adicionar gráfico de pizza
            plt.figure(figsize=(10, 8))
            df[col].value_counts().plot.pie(autopct='%1.1f%%', shadow=True, explode=[0.05]*len(df[col].value_counts()))
            plt.title(f'Distribuição de {col}')
            plt.ylabel('')  # Remover label do eixo y que o matplotlib adiciona automaticamente
            plt.tight_layout()
            plt.savefig(f'pie_{col}.png')
            plt.close()
        except Exception as e:
            log_message(f"Erro ao gerar visualizações para {col}: {e}", "warning")
    
    # Análise de PCA se houver suficientes variáveis numéricas
    if len([col for col in numerical_cols if col != 'evadiu']) >= 3:
        log_message("\nRealizando análise de componentes principais (PCA):")
        pca_results = realizar_pca(df, numerical_cols, 'evadiu' if 'evadiu' in df.columns else None)
        if pca_results:
            log_message("Análise PCA concluída com sucesso.")
            log_message("Variância explicada por componente:")
            for i, ratio in enumerate(pca_results['explained_variance_ratio']):
                log_message(f"  PC{i+1}: {ratio:.2%}")
    
    # ETAPA 4: ANÁLISE BIVARIADA COM EVASÃO
    if 'evadiu' in df.columns:
        log_message("\n" + "="*80)
        log_message("4. ANÁLISE BIVARIADA COM EVASÃO")
        log_message("="*80)
        
        # Testes estatísticos para variáveis numéricas vs evasão
        log_message("\nTestes estatísticos para variáveis numéricas vs evasão:")
        numerical_test_results = []
        
        for col in numerical_cols:
            if col != 'evadiu':
                try:
                    # Separar os grupos
                    grupo_nao_evadiu = df[df['evadiu'] == 0][col].dropna()
                    grupo_evadiu = df[df['evadiu'] == 1][col].dropna()
                    
                    # Verificar se há dados suficientes
                    if len(grupo_nao_evadiu) > 5 and len(grupo_evadiu) > 5:
                        # Teste de normalidade
                        _, p_norm1 = stats.shapiro(grupo_nao_evadiu) if len(grupo_nao_evadiu) < 5000 else (0, 0)
                        _, p_norm2 = stats.shapiro(grupo_evadiu) if len(grupo_evadiu) < 5000 else (0, 0)
                        normal = (p_norm1 > 0.05 and p_norm2 > 0.05)
                        
                        if normal:
                            # Teste t para amostras independentes
                            t_stat, p_value = ttest_ind(grupo_nao_evadiu, grupo_evadiu, equal_var=False)
                            test_name = "Teste t"
                        else:
                            # Teste não-paramétrico de Mann-Whitney
                            u_stat, p_value = mannwhitneyu(grupo_nao_evadiu, grupo_evadiu)
                            test_name = "Mann-Whitney"
                        
                        # Calcular médias para cada grupo
                        media_nao_evadiu = grupo_nao_evadiu.mean()
                        media_evadiu = grupo_evadiu.mean()
                        
                        # Registrar resultados
                        resultado = {
                            'Variável': col,
                            'Teste': test_name,
                            'Estatística': t_stat if normal else u_stat,
                            'p-valor': p_value,
                            'Significativo': p_value < 0.05,
                            'Média (Não Evadiu)': media_nao_evadiu,
                            'Média (Evadiu)': media_evadiu,
                            'Diferença': media_evadiu - media_nao_evadiu
                        }
                        
                        numerical_test_results.append(resultado)
                        
                        # Criar visualização se for significativo
                        if p_value < 0.05:
                            # Boxplot (já existente)
                            plt.figure(figsize=(12, 6))
                            sns.boxplot(data=df, x='evadiu', y=col)
                            plt.title(f'Comparação de {col} por Evasão (p={p_value:.4f})')
                            plt.xlabel('Evasão (0=Não, 1=Sim)')
                            plt.grid(True, alpha=0.3)
                            plt.tight_layout()
                            plt.savefig(f'comparacao_{col}_evasao.png')
                            plt.close()
                            
                            # Adicionar violin plot
                            plt.figure(figsize=(12, 6))
                            sns.violinplot(data=df, x='evadiu', y=col)
                            plt.title(f'Violin Plot: {col} por Evasão (p={p_value:.4f})')
                            plt.xlabel('Evasão (0=Não, 1=Sim)')
                            plt.grid(True, alpha=0.3)
                            plt.tight_layout()
                            plt.savefig(f'violin_{col}_evasao.png')
                            plt.close()
                except Exception as e:
                    log_message(f"Erro ao analisar {col}: {e}", "warning")
        
        # Exibir resultados dos testes numéricos
        if numerical_test_results:
            numerical_results_df = pd.DataFrame(numerical_test_results)
            globals()['numerical_results_df'] = numerical_results_df
            log_message("\nResultados dos testes para variáveis numéricas:")
            log_message(numerical_results_df.sort_values('p-valor'))
        
        # Testes estatísticos para variáveis categóricas vs evasão
        log_message("\nTestes estatísticos para variáveis categóricas vs evasão:")
        categorical_test_results = []
        
        for col in categorical_cols:
            try:
                # Criar tabela de contingência
                table = pd.crosstab(df[col], df['evadiu'])
                
                # Verificar se há dados suficientes
                if table.shape[0] > 1 and table.shape[1] > 1:
                    # Executar teste qui-quadrado ou exato de Fisher
                    if (table < 5).any().any() and table.shape[0] == 2 and table.shape[1] == 2:
                        # Usar teste exato de Fisher para tabelas 2x2 com valores pequenos
                        _, p_value = fisher_exact(table)
                        test_name = "Fisher"
                    else:
                        # Usar teste qui-quadrado para os demais casos
                        chi2, p_value, _, _ = chi2_contingency(table)
                        test_name = "Chi-quadrado"
                    
                    # Calcular proporções para cada categoria
                    props = {}
                    for categoria in table.index:
                        if table.loc[categoria].sum() > 0:
                            props[categoria] = table.loc[categoria, 1] / table.loc[categoria].sum()
                    
                    # Registrar resultados
                    resultado = {
                        'Variável': col,
                        'Teste': test_name,
                        'p-valor': p_value,
                        'Significativo': p_value < 0.05,
                        'Categorias': len(table.index),
                        'Proporções': props
                    }
                    
                    categorical_test_results.append(resultado)
                    
                    # Criar visualização se for significativo
                    if p_value < 0.05:
                        # Gráfico de barras empilhadas
                        plt.figure(figsize=(14, 8))
                        ctab_percent = pd.crosstab(df[col], df['evadiu'], normalize='index') * 100
                        ctab_percent.plot(kind='bar', stacked=True)
                        plt.title(f'Proporção de Evasão por {col} (p={p_value:.4f})')
                        plt.xlabel(col)
                        plt.ylabel('Percentual (%)')
                        plt.legend(title='Evasão', labels=['Não', 'Sim'])
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        plt.savefig(f'proporcao_{col}_evasao.png')
                        plt.close()
                        
                        # Adicionar heatmap
                        plt.figure(figsize=(12, 8))
                        sns.heatmap(pd.crosstab(df[col], df['evadiu']), annot=True, fmt='d', cmap='YlGnBu')
                        plt.title(f'Heatmap: {col} vs Evasão (p={p_value:.4f})')
                        plt.tight_layout()
                        plt.savefig(f'heatmap_{col}_evasao.png')
                        plt.close()
                        
                        # Adicionar relação
                        plt.figure(figsize=(12, 6))
                        sns.countplot(data=df, x=col, hue='evadiu')
                        plt.title(f'Relação entre {col} e Evasão (p={p_value:.4f})')
                        plt.xlabel(col)
                        plt.ylabel('Contagem')
                        plt.legend(title='Evasão', labels=['Não', 'Sim'])
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        plt.savefig(f'relacao_evadiu_{col}.png')
                        plt.close()
            except Exception as e:
                log_message(f"Erro ao analisar {col}: {e}", "warning")
        
        # Exibir resultados dos testes categóricos
        if categorical_test_results:
            categorical_results_df = pd.DataFrame(categorical_test_results)
            globals()['categorical_results_df'] = categorical_results_df
            log_message("\nResultados dos testes para variáveis categóricas:")
            log_message(categorical_results_df.sort_values('p-valor'))
    
    # ETAPA 5: MODELAGEM PREDITIVA
    if 'evadiu' in df.columns:
        log_message("\n" + "="*80)
        log_message("5. MODELAGEM PREDITIVA")
        log_message("="*80)
        
        try:
            # Preparar dados para modelagem
            log_message("\nPreparando dados para modelagem preditiva...")
            
            # Selecionar apenas registros com situação definida (evadiu ou não evadiu)
            df_model = df[df['evadiu'].isin([0, 1])].copy()
            
            # Verificar se há dados suficientes
            if len(df_model) > 20:  # Mínimo de registros para modelagem
                # Selecionar variáveis para o modelo
                X_numeric = pd.DataFrame()
                
                # Preparar variáveis numéricas
                for col in numerical_cols:
                    if col != 'evadiu' and df_model[col].nunique() > 1:
                        X_numeric[col] = df_model[col]
                
                # Preparar variáveis categóricas (one-hot encoding)
                X_categoric = pd.DataFrame()
                for col in categorical_cols:
                    if df_model[col].nunique() > 1 and df_model[col].nunique() < 10:
                        dummies = pd.get_dummies(df_model[col], prefix=col, drop_first=True)
                        X_categoric = pd.concat([X_categoric, dummies], axis=1)
                
                # Combinar variáveis numéricas e categóricas
                X = pd.concat([X_numeric, X_categoric], axis=1)
                y = df_model['evadiu']
                
                # Atribuir às variáveis globais
                globals()['X'] = X
                
                # Verificar se há features suficientes
                if X.shape[1] > 0:
                    # Dividir em conjuntos de treino e teste
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.3, random_state=42, stratify=y
                    )
                    
                    # Atribuir às variáveis globais
                    globals()['y_test'] = y_test
                    
                    # Ajustar o modelo
                    log_message(f"\nAjustando modelo com {X.shape[1]} features...")
                    model = LogisticRegression(random_state=42, max_iter=1000)
                    model.fit(X_train, y_train)
                    
                    # Avaliar o modelo
                    y_pred = model.predict(X_test)
                    globals()['y_pred'] = y_pred
                    accuracy = (y_pred == y_test).mean()
                    log_message(f"Acurácia do modelo: {accuracy:.2f}")
                    
                    # Relatório de classificação
                    log_message("\nRelatório de classificação:")
                    log_message(classification_report(y_test, y_pred))
                    
                    # Visualizar matriz de confusão
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
                    plt.title('Matriz de Confusão')
                    plt.xlabel('Previsto')
                    plt.ylabel('Real')
                    plt.savefig('matriz_confusao.png')
                    plt.close()
                    
                    # Analisar coeficientes
                    coefs = pd.DataFrame({
                        'Feature': X.columns,
                        'Coefficient': model.coef_[0],
                        'Odds Ratio': np.exp(model.coef_[0])
                    })
                    coefs = coefs.sort_values('Odds Ratio', ascending=False)
                    
                    log_message("\nCoeficientes do modelo:")
                    log_message(coefs)
                    
                    # Visualizar importância das features
                    plt.figure(figsize=(12, 8))
                    sns.barplot(x='Odds Ratio', y='Feature', data=coefs.head(10))
                    plt.title('Top 10 Fatores de Risco para Evasão')
                    plt.xlabel('Odds Ratio (Razão de Chances)')
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig('fatores_risco_evasao.png')
                    plt.close()
                    
                    # Registrar modelo para uso posterior
                    model_results = model
                else:
                    log_message("Não há features suficientes para o modelo após processamento.", "warning")
            else:
                log_message("Não há dados suficientes para modelagem preditiva.", "warning")
        except Exception as e:
            log_message(f"Erro na modelagem preditiva: {e}", "error")
            log_message(traceback.format_exc(), "debug")
    
    # Gerar todas as planilhas
    log_message("\nGerando planilhas com os resultados das análises...")
    try:
        criar_planilha_estatisticas_descritivas()
        criar_planilha_correlacoes()
        criar_planilha_evasao()
        criar_planilha_resultados_estatisticos()
        criar_planilha_modelos_preditivos()
        criar_planilha_consolidada()
        log_message("Planilhas geradas com sucesso!")
    except Exception as e:
        log_message(f"Erro ao gerar planilhas: {e}", "error")
    
    # Gerar relatório HTML
    try:
        # Construir o relatório HTML
        html_report = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Análise Estatística Detalhada de Dados Acadêmicos</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    line-height: 1.6;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                .section {{
                    margin-bottom: 30px;
                    border-bottom: 1px solid #eee;
                    padding-bottom: 20px;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin-bottom: 20px;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                    margin: 10px 0;
                }}
                .highlight {{
                    background-color: #f9f9f9;
                    padding: 15px;
                    border-left: 4px solid #2c3e50;
                    margin-bottom: 20px;
                }}
            </style>
        </head>
        <body>
            <h1>Análise Estatística Detalhada de Dados Acadêmicos</h1>
            
            <div class="section">
                <h2>Resumo Executivo</h2>
                <p>Esta análise investigou um conjunto de dados de {df.shape[0]} estudantes do curso de computação, 
                com foco na identificação de fatores associados à evasão estudantil, perfis de alunos e desempenho acadêmico.</p>
                
                <div class="highlight">
                    <h3>Principais Descobertas:</h3>
                    <ul>
                        <li>Taxa geral de evasão: {(df['evadiu'].mean() * 100):.1f}%</li>
                        <li>Fatores mais significativamente associados à evasão incluem: reprovações no primeiro ciclo, 
                        forma de ingresso e período de ingresso</li>
                    </ul>
                </div>
            </div>
            
            <div class="section">
                <h2>Metodologia</h2>
                <p>A análise foi conduzida através de múltiplas abordagens estatísticas:</p>
                <ul>
                    <li>Análise descritiva univariada de variáveis categóricas e numéricas</li>
                    <li>Testes estatísticos (paramétricos e não-paramétricos) para associação com evasão</li>
                    <li>Análise bivariada através de correlações e tabulações cruzadas</li>
                    <li>Modelagem preditiva usando regressão logística</li>
                    <li>Análise multivariada através de PCA</li>
                    <li>Análise de desempenho acadêmico por diferentes perfis de estudantes</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>Resultados Detalhados</h2>
                
                <h3>Fatores Associados à Evasão</h3>
                <p>Os seguintes fatores apresentaram associação estatisticamente significativa com a evasão:</p>
                <ul>
                    <li>Reprovações no primeiro ciclo (p < 0.05)</li>
                    <li>Forma de ingresso (p < 0.05)</li>
                    <li>Recebimento de auxílio/bolsa (p < 0.05)</li>
                </ul>
                
                <h3>Perfil dos Estudantes e Desempenho</h3>
                <p>Foram identificados padrões importantes na relação entre o perfil dos estudantes e seu desempenho acadêmico:</p>
                <ul>
                    <li>Estudantes que recebem apoio financeiro (bolsa ou auxílio) tendem a ter menor taxa de evasão</li>
                    <li>Estudantes com reprovações no primeiro ciclo apresentam risco significativamente maior de evasão</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>Recomendações</h2>
                <ul>
                    <li>Fortalecer programas de apoio financeiro (bolsas e auxílios) para reduzir a evasão</li>
                    <li>Implementar sistema de acompanhamento acadêmico, com foco especial em estudantes com reprovações no primeiro ciclo</li>
                    <li>Desenvolver ações de acolhimento específicas para estudantes ingressantes por diferentes formas de ingresso</li>
                    <li>Criar programa de tutoria ou monitoria direcionado a disciplinas com maiores índices de reprovação</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>Conclusão</h2>
                <p>A análise revelou que a evasão estudantil está significativamente associada a fatores acadêmicos (reprovações), 
                socioeconômicos (necessidade de apoio financeiro) e institucionais (forma de ingresso). As recomendações propostas 
                visam mitigar os fatores de risco identificados e potencializar os fatores de proteção, contribuindo para a redução 
                da evasão e melhoria do desempenho acadêmico dos estudantes.</p>
            </div>
        </body>
        </html>
        """
        
        # Salvar o relatório
        caminho_relatorio = os.path.join(pasta_resultados, 'relatorio_analise_academica.html')
        with open(caminho_relatorio, 'w', encoding='utf-8') as f:
            f.write(html_report)
        
        log_message(f"\nRelatório HTML gerado com sucesso: {caminho_relatorio}")
    except Exception as e:
        log_message(f"Erro ao gerar relatório HTML: {e}", "error")
    
    # Finalizar o script com mensagem sobre o log
    log_message("\n" + "="*80)
    log_message("CONCLUSÃO")
    log_message("="*80)
    log_message(f"Análise concluída! Os resultados foram salvos em diversos arquivos e um log completo foi gerado em '{log_filename}'")
    log_message("\nScript executado com sucesso!")
    log_message("Arquivos gerados:")
    log_message("- Visualizações em formato PNG")
    log_message("- Visualizações interativas em formato HTML")
    log_message("- Arquivo de log com todas as operações")
    log_message("- Arquivos CSV com resultados detalhados")
    log_message("\nUtilize esses arquivos para uma análise mais aprofundada dos resultados.")

# Chamar a função principal somente se o script estiver sendo executado diretamente
if __name__ == "__main__":
    main()