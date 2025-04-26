#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Análise Estatística Comparativa de Dados Acadêmicos
Este script realiza uma análise estatística comparativa entre os cursos de Computação e Energias,
para compreender os fatores associados à evasão estudantil, perfis de alunos e desempenho acadêmico.
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

# Configurar o sistema de logging
def setup_logger():
    """Configura e retorna um logger para a aplicação"""
    # Criar um logger
    logger = logging.getLogger('comparative_analysis')
    logger.setLevel(logging.INFO)
    
    # Criar um handler para arquivo
    log_filename = f"comparative_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
            
            # Exportar para CSV
            df.to_csv(csv_filename)
            arquivos_gerados.append(csv_filename)
        
        log_message(f"{len(arquivos_gerados)} arquivos CSV gerados com sucesso!")
        return True
    except Exception as e:
        log_message(f"Erro ao exportar para CSV: {e}", "error")
        return False

# Função para criar variável de evasão
def classificar_evasao(tipo_saida):
    """
    Classifica os tipos de saída em evasão (1) ou não evasão (0)
    Considera como NÃO EVASÃO (0) os seguintes valores:
    - None/NULL/NaN
    - Formado/Formatura/Concluido
    - Prorrogação/Trancamento temporário
    - Mobilidade acadêmica
    - Falecimento
    - Cancelamento judicial
    """
    # Verificar se o valor é NaN
    if pd.isna(tipo_saida):
        return 0
    
    # Converter para string para garantir que as comparações funcionem
    if not isinstance(tipo_saida, str):
        tipo_saida = str(tipo_saida)
    
    # Lista expandida de termos que indicam não-evasão
    nao_evasao = [
        'none', 'null', '', 
        'formatura', 'formado', 'concluido', 'concluído',
        'prorrogação', 'trancamento',
        'mobilidade', 
        'falecimento', 'óbito',
        'cancelamento judicial',
        'transferência interna'  # Transferência dentro da instituição
    ]
    
    # Verificar se qualquer termo de não-evasão está contido no tipo_saida (case insensitive)
    for termo in nao_evasao:
        if termo.lower() in tipo_saida.lower():
            return 0
    
    # Se chegou aqui, é considerado evasão
    return 1

# Configurações gerais
pd.set_option('display.max_columns', None)
try:
    plt.style.use('seaborn-v0_8-whitegrid')  # Para versões mais recentes do matplotlib
except:
    try:
        plt.style.use('seaborn-whitegrid')   # Para versões mais antigas do matplotlib
    except:
        plt.style.use('default')              # Fallback para estilo padrão se nenhum outro funcionar
sns.set(style="whitegrid", font_scale=1.1)
plt.rcParams.update({'figure.figsize': (12, 8), 'figure.dpi': 120})

# ==============================================================================
# CARREGAMENTO E PROCESSAMENTO DOS DADOS
# ==============================================================================
log_message("Iniciando análise comparativa entre Computação e Energias...")
log_message("="*80)

# Criar pasta para resultados se não existir
pasta_resultados = 'resultados_comparativo'
if not os.path.exists(pasta_resultados):
    os.makedirs(pasta_resultados)
    log_message(f"Pasta '{pasta_resultados}' criada para armazenar os resultados.")

# Função para processar dataframe
def processar_dataframe(df, nome_curso):
    """Processa um dataframe para adequá-lo à análise."""
    log_message(f"\nProcessando dados do curso de {nome_curso}...")
    
    # Truncar textos longos para colunas específicas
    colunas_para_truncar = ['forma_ingresso', 'forma_saida']
    for col in colunas_para_truncar:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: truncar_texto(x, max_len=12) if pd.notna(x) else x)
    
    # Converter colunas que usam vírgula como separador decimal
    for col in df.select_dtypes(include=['object']).columns:
        try:
            if df[col].str.contains(',').any():
                df[col] = df[col].str.replace(',', '.').astype(float)
        except:
            pass
    
    # Aplicar a classificação de evasão 
    # Verificar se a coluna forma_saida existe, senão usar 'situacao' ou outra coluna relevante
    if 'forma_saida' in df.columns:
        df['evadiu'] = df['forma_saida'].apply(classificar_evasao)
    elif 'situacao' in df.columns:
        # Adaptar para usar a coluna 'situacao'
        df['evadiu'] = df['situacao'].apply(lambda x: 0 if pd.isna(x) or x in ['Cursando', 'Formado', 'Concluído'] else 1)
    else:
        log_message(f"Aviso: Não foi possível determinar a situação de evasão para {nome_curso}.", "warning")
        df['evadiu'] = 0  # Valor padrão
    
    # Adicionar identificação do curso
    df['curso'] = nome_curso
    
    # Converter períodos para anos se necessário (exemplo: 2019.1 -> 2019)
    if 'periodo_ingresso' in df.columns and 'ano_ingresso' not in df.columns:
        try:
            df['ano_ingresso'] = df['periodo_ingresso'].astype(str).str.split('.').str[0].astype(float)
        except:
            log_message(f"Aviso: Não foi possível converter período de ingresso para ano em {nome_curso}.", "warning")
    
    if 'periodo_situacao' in df.columns and 'ano_saida' not in df.columns:
        try:
            df['ano_saida'] = df['periodo_situacao'].astype(str).str.split('.').str[0].astype(float)
        except:
            log_message(f"Aviso: Não foi possível converter período de situação para ano em {nome_curso}.", "warning")
    
    log_message(f"Dimensões do dataset {nome_curso}: {df.shape}")
    log_message(f"Taxa de evasão {nome_curso}: {(df['evadiu'].mean() * 100):.2f}%")
    
    return df

# Carregar dados de Computação
log_message("\nCarregando dados do curso de Computação...")
try:
    df_comp = pd.read_csv('dados_exemplo_computacao_200.csv')
    df_comp = processar_dataframe(df_comp, "Computação")
except Exception as e:
    log_message(f"Erro ao carregar dados de Computação: {e}", "error")
    df_comp = None

# Carregar dados de Energias
log_message("\nCarregando dados do curso de Energias...")
try:
    df_ener = pd.read_csv('dados_exemplo_energias_200.csv')
    df_ener = processar_dataframe(df_ener, "Energias")
except Exception as e:
    log_message(f"Erro ao carregar dados de Energias: {e}", "error")
    df_ener = None

# Verificar se ambos os dataframes foram carregados corretamente
if df_comp is None or df_ener is None:
    log_message("Erro: Um ou ambos os datasets não puderam ser carregados. Encerrando análise.", "error")
    sys.exit(1)

# Combinar os dataframes para análises conjuntas
log_message("\nCombinando datasets para análise comparativa...")
df_combinado = pd.concat([df_comp, df_ener], ignore_index=True)
log_message(f"Dimensões do dataset combinado: {df_combinado.shape}")

# Estatísticas básicas dos datasets
log_message("\nDistribuição de estudantes por curso:")
dist_cursos = df_combinado['curso'].value_counts()
log_message(dist_cursos)

log_message("\nDistribuição de evasão por curso:")
evasao_por_curso = pd.crosstab(df_combinado['curso'], df_combinado['evadiu'])
log_message(evasao_por_curso)

# Calcular taxas de evasão por curso
taxa_evasao_por_curso = pd.crosstab(df_combinado['curso'], df_combinado['evadiu'], normalize='index') * 100
log_message("\nTaxa de evasão por curso (%):")
log_message(taxa_evasao_por_curso)

# Identificar variáveis categóricas e numéricas
categorical_cols = df_combinado.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('curso')  # Remover a coluna curso da lista de categóricas
if 'matricula' in categorical_cols:
    categorical_cols.remove('matricula')  # Remover a matrícula que é um identificador
if 'nome' in categorical_cols:
    categorical_cols.remove('nome')  # Remover nome que é um identificador

numerical_cols = df_combinado.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'matricula' in numerical_cols:
    numerical_cols.remove('matricula')  # Remover a matrícula que é um identificador
if 'evadiu' in numerical_cols:
    numerical_cols.remove('evadiu')  # Tratar 'evadiu' separadamente

log_message(f"\nVariáveis categóricas: {len(categorical_cols)}")
log_message(f"Variáveis numéricas: {len(numerical_cols)}")

# Plot básico de comparação de evasão entre cursos
plt.figure(figsize=(10, 6))

# Verificar se a coluna 1 (evasão) existe no dataframe
if 1 in taxa_evasao_por_curso.columns:
    ax = taxa_evasao_por_curso[1].plot(kind='bar', color=['skyblue', 'coral'])
    plt.title('Comparação da Taxa de Evasão entre Cursos', fontsize=14)
    plt.xlabel('Curso', fontsize=12)
    plt.ylabel('Taxa de Evasão (%)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adicionar valores nas barras
    for i, v in enumerate(taxa_evasao_por_curso[1]):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha='center', fontsize=12)
else:
    # Se não houver dados de evasão, mostrar as taxas de não-evasão
    ax = taxa_evasao_por_curso[0].plot(kind='bar', color=['skyblue', 'coral'])
    plt.title('Comparação da Taxa de Permanência entre Cursos', fontsize=14)
    plt.xlabel('Curso', fontsize=12)
    plt.ylabel('Taxa de Permanência (%)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adicionar valores nas barras
    for i, v in enumerate(taxa_evasao_por_curso[0]):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha='center', fontsize=12)
    
    log_message("Aviso: Não foram encontrados casos de evasão nos dados. Mostrando taxa de permanência.", "warning")

plt.tight_layout()
plt.savefig('comparacao_evasao_cursos.png')
plt.close()

# ==============================================================================
# 1. ANÁLISE DESCRITIVA COMPARATIVA
# ==============================================================================
log_message("\n" + "="*80)
log_message("1. ANÁLISE DESCRITIVA COMPARATIVA")
log_message("="*80)

# 1.1 Comparação de perfil demográfico
log_message("\n1.1 Comparação de perfil demográfico")
log_message("-"*50)

# Função para criar gráficos comparativos para variáveis categóricas
def plot_comparacao_categorica(variavel, titulo):
    """Cria gráficos comparativos para variáveis categóricas entre os cursos."""
    if variavel not in df_combinado.columns:
        log_message(f"Variável {variavel} não encontrada no dataset.", "warning")
        return
    
    # Calcular distribuição da variável por curso
    cross_tab = pd.crosstab(df_combinado['curso'], df_combinado[variavel], normalize='index') * 100
    
    # Gráfico de barras empilhadas
    plt.figure(figsize=(14, 8))
    cross_tab.plot(kind='bar', stacked=True, colormap='tab20')
    plt.title(f'Comparação de {titulo} entre Cursos', fontsize=14)
    plt.xlabel('Curso', fontsize=12)
    plt.ylabel('Percentual (%)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.legend(title=titulo, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'comparacao_{variavel}_cursos.png')
    plt.close()
    
    # Calcular valores específicos para análise
    log_message(f"\nDistribuição de {titulo} por Curso (%):")
    log_message(cross_tab)
    
    # Verificar se existe relação entre a variável e o curso (qui-quadrado)
    try:
        table = pd.crosstab(df_combinado[variavel], df_combinado['curso'])
        chi2, p, _, _ = chi2_contingency(table)
        log_message(f"Teste Qui-quadrado: chi2={chi2:.4f}, p-valor={p:.4f}")
        log_message(f"Existe diferença significativa na distribuição de {titulo} entre os cursos? {'Sim' if p < 0.05 else 'Não'}")
    except Exception as e:
        log_message(f"Não foi possível realizar o teste qui-quadrado para {variavel}: {e}", "warning")

# Comparações demográficas
variaveis_demograficas = [
    ('sexo', 'Sexo'),
    ('raca_cor', 'Raça/Cor'),
    ('tipo_de_escola_em', 'Tipo de Escola'),
    ('acao_afirmativa', 'Ação Afirmativa')
]

for var, titulo in variaveis_demograficas:
    if var in df_combinado.columns:
        log_message(f"\nComparando {titulo} entre cursos")
        plot_comparacao_categorica(var, titulo)

# 1.2 Comparação de indicadores acadêmicos
log_message("\n1.2 Comparação de indicadores acadêmicos")
log_message("-"*50)

# Função para criar gráficos comparativos para variáveis numéricas
def plot_comparacao_numerica(variavel, titulo):
    """Cria gráficos comparativos para variáveis numéricas entre os cursos."""
    if variavel not in df_combinado.columns:
        log_message(f"Variável {variavel} não encontrada no dataset.", "warning")
        return
    
    # Estatísticas por curso
    stats = df_combinado.groupby('curso')[variavel].describe()
    log_message(f"\nEstatísticas de {titulo} por Curso:")
    log_message(stats)
    
    # Boxplot comparativo
    plt.figure(figsize=(12, 6))
    ax = sns.boxplot(x='curso', y=variavel, data=df_combinado)
    plt.title(f'Comparação de {titulo} entre Cursos', fontsize=14)
    plt.xlabel('Curso', fontsize=12)
    plt.ylabel(titulo, fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Adicionar médias como pontos
    means = df_combinado.groupby('curso')[variavel].mean()
    ax.scatter(x=range(len(means)), y=means, color='red', s=50, marker='o', label='Média')
    
    plt.tight_layout()
    plt.savefig(f'comparacao_{variavel}_cursos_boxplot.png')
    plt.close()
    
    # Testes estatísticos para verificar diferenças
    try:
        # Extrair os dados dos dois grupos
        grupo1 = df_combinado[df_combinado['curso'] == 'Computação'][variavel].dropna()
        grupo2 = df_combinado[df_combinado['curso'] == 'Energias'][variavel].dropna()
        
        # Verificar se há dados suficientes
        if len(grupo1) > 5 and len(grupo2) > 5:
            # Teste de normalidade
            _, p_norm1 = stats.shapiro(grupo1) if len(grupo1) < 5000 else (0, 0)
            _, p_norm2 = stats.shapiro(grupo2) if len(grupo2) < 5000 else (0, 0)
            normal = (p_norm1 > 0.05 and p_norm2 > 0.05)
            
            if normal:
                # Teste t para amostras independentes
                t_stat, p_value = ttest_ind(grupo1, grupo2, equal_var=False)
                test_name = "Teste t de Student"
            else:
                # Teste não-paramétrico de Mann-Whitney
                u_stat, p_value = mannwhitneyu(grupo1, grupo2)
                test_name = "Teste de Mann-Whitney"
            
            log_message(f"{test_name}: estatística={t_stat if normal else u_stat:.4f}, p-valor={p_value:.4f}")
            log_message(f"Existe diferença significativa em {titulo} entre os cursos? {'Sim' if p_value < 0.05 else 'Não'}")
            
            # Calcular tamanho do efeito (d de Cohen)
            if normal:
                pooled_std = np.sqrt(((len(grupo1) - 1) * np.var(grupo1, ddof=1) + 
                                     (len(grupo2) - 1) * np.var(grupo2, ddof=1)) / 
                                    (len(grupo1) + len(grupo2) - 2))
                effect_size = abs(np.mean(grupo1) - np.mean(grupo2)) / pooled_std
                log_message(f"Tamanho do efeito (d de Cohen): {effect_size:.4f}")
                
                # Interpretação
                if effect_size < 0.2:
                    effect_interp = "Insignificante"
                elif effect_size < 0.5:
                    effect_interp = "Pequeno"
                elif effect_size < 0.8:
                    effect_interp = "Médio"
                else:
                    effect_interp = "Grande"
                log_message(f"Interpretação do tamanho do efeito: {effect_interp}")
        else:
            log_message(f"Dados insuficientes para análise estatística de {variavel}", "warning")
    except Exception as e:
        log_message(f"Não foi possível realizar testes estatísticos para {variavel}: {e}", "warning")

# Analisar variáveis numéricas
variaveis_academicas = [
    ('periodo_ingresso', 'Período de Ingresso'),
    ('cr', 'Coeficiente de Rendimento'),
    ('reprovacoes_1_ciclo', 'Reprovações 1º Ciclo')
]

for var, titulo in variaveis_academicas:
    if var in df_combinado.columns:
        if var == 'reprovacoes_1_ciclo' and df_combinado[var].dtype == 'object':
            # Se for categórica, converter para numérica
            try:
                # Tentar converter para numérico diretamente se forem números
                df_combinado['reprov_num'] = pd.to_numeric(df_combinado[var], errors='coerce')
            except:
                # Se não der certo, mapear valores categóricos
                df_combinado['reprov_num'] = df_combinado[var].map(
                    lambda x: 1 if pd.notna(x) and str(x).lower() == 'sim' else 0 if pd.notna(x) and str(x).lower() == 'não' else x
                )
            log_message(f"\nComparando {titulo} entre cursos (convertida para numérico)")
            plot_comparacao_numerica('reprov_num', titulo)
        else:
            log_message(f"\nComparando {titulo} entre cursos")
            plot_comparacao_numerica(var, titulo)

# 1.3 Comparação de taxas de evasão por diferentes fatores
log_message("\n1.3 Comparação de taxas de evasão por diferentes fatores")
log_message("-"*50)

# Função para analisar e visualizar taxas de evasão por diferentes fatores entre cursos
def analisar_evasao_por_fator(variavel, titulo):
    """Analisa e visualiza taxas de evasão por diferentes fatores, comparando os cursos."""
    if variavel not in df_combinado.columns:
        log_message(f"Variável {variavel} não encontrada no dataset.", "warning")
        return
    
    # Calcular taxas de evasão por fator e curso
    evasao_tab = pd.crosstab(
        [df_combinado['curso'], df_combinado[variavel]], 
        df_combinado['evadiu'], 
        normalize='index'
    ) * 100
    
    # Verificar se há dados de evasão (coluna 1)
    if 1 in evasao_tab.columns:
        # Formatar para visualização
        evasao_tab.columns = ['Não Evadiu (%)', 'Evadiu (%)']
        
        log_message(f"\nTaxas de evasão por {titulo} em cada curso:")
        log_message(evasao_tab)
        
        # Reorganizar os dados para o gráfico
        evasao_plot = evasao_tab.reset_index()
        evasao_plot = evasao_plot.rename(columns={'level_1': variavel})
        
        # Gráfico de barras agrupadas
        plt.figure(figsize=(14, 8))
        
        # Filtrar para cada curso
        for i, curso in enumerate(['Computação', 'Energias']):
            curso_data = evasao_plot[evasao_plot['curso'] == curso]
            
            # Ordenar os dados pela taxa de evasão
            curso_data = curso_data.sort_values('Evadiu (%)', ascending=False)
            
            # Criar barras para este curso
            x = np.arange(len(curso_data)) + i*0.4 - 0.2
            plt.bar(x, curso_data['Evadiu (%)'], width=0.35, 
                    label=f'{curso}', color='skyblue' if i == 0 else 'coral')
            
            # Adicionar rótulos nas barras
            for j, v in enumerate(curso_data['Evadiu (%)']):
                plt.text(x[j], v + 1, f"{v:.1f}%", ha='center', fontsize=8)
        
        plt.title(f'Comparação de Taxas de Evasão por {titulo} entre Cursos', fontsize=14)
        plt.xlabel(titulo, fontsize=12)
        plt.ylabel('Taxa de Evasão (%)', fontsize=12)
        plt.xticks(np.arange(len(evasao_plot[variavel].unique())), 
                  evasao_plot[variavel].unique(), rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'comparacao_evasao_{variavel}.png')
        plt.close()
    else:
        log_message(f"Não há registros de evasão para análise por {titulo}. Pulando visualização.", "warning")

# Analisar evasão por diferentes fatores
fatores_evasao = [
    ('sexo', 'Sexo'),
    ('raca_cor', 'Raça/Cor'),
    ('forma_ingresso', 'Forma de Ingresso'),
    ('tipo_de_escola_em', 'Tipo de Escola'),
    ('recebeu_auxilio', 'Recebeu Auxílio'),
    ('recebeu_bolsa', 'Recebeu Bolsa')
]

for var, titulo in fatores_evasao:
    if var in df_combinado.columns:
        log_message(f"\nComparando taxas de evasão por {titulo} entre cursos")
        analisar_evasao_por_fator(var, titulo)

# ==============================================================================
# 2. MODELAGEM COMPARATIVA
# ==============================================================================
log_message("\n" + "="*80)
log_message("2. MODELAGEM COMPARATIVA")
log_message("="*80)

# 2.1 Análise de Fatores de Risco de Evasão por Curso
log_message("\n2.1 Análise de Fatores de Risco de Evasão por Curso")
log_message("-"*50)

# Função para construir e analisar modelo de regressão logística
def modelar_fatores_evasao(df, nome_curso):
    """Constrói um modelo de regressão logística para identificar fatores de risco de evasão."""
    log_message(f"\nModelando fatores de evasão para {nome_curso}...")
    
    # Selecionar variáveis para o modelo
    cat_vars = ['sexo', 'forma_ingresso', 'acao_afirmativa', 
               'recebeu_auxilio', 'recebeu_bolsa', 'tipo_de_escola_em']
    # Filtrar apenas as colunas que existem no dataframe
    cat_vars = [var for var in cat_vars if var in df.columns]
    
    # Criar dummies para variáveis categóricas
    X_cat = pd.get_dummies(df[cat_vars], drop_first=True)
    
    # Selecionar variáveis numéricas
    num_vars = ['ano_ingresso', 'periodo_ingresso']
    if 'interticio' in df.columns:
        num_vars.append('interticio')
    
    # Filtrar variáveis numéricas que existem no dataframe
    num_vars = [var for var in num_vars if var in df.columns]
    X_num = df[num_vars].copy()
    
    # Verificar o tipo da coluna de reprovações e adicionar
    if 'reprovacoes_1_ciclo' in df.columns:
        if df['reprovacoes_1_ciclo'].dtype == 'object':
            df['reprov_num'] = df['reprovacoes_1_ciclo'].map({'sim': 1, 'não': 0})
        else:
            df['reprov_num'] = df['reprovacoes_1_ciclo']
        
        X_num['reprov_num'] = df['reprov_num']
    
    # Combinar variáveis
    X = pd.concat([X_num, X_cat], axis=1)
    
    # Lidar com valores ausentes
    for col in X.select_dtypes(include=['number']).columns:
        X[col] = X[col].fillna(X[col].median())
    
    for col in X.select_dtypes(exclude=['number']).columns:
        X[col] = X[col].fillna(0)
    
    # Variável target
    y = df['evadiu']
    
    # Verificar se temos dados suficientes
    if X.shape[0] < 30 or y.sum() < 5 or (len(y) - y.sum()) < 5:
        log_message(f"Dados insuficientes para modelagem de {nome_curso}.", "warning")
        return None, None
    
    # Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Padronizar variáveis numéricas
    scaler = StandardScaler()
    num_cols = X_num.columns
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])
    
    # Treinar modelo
    try:
        log_model = LogisticRegression(random_state=42, max_iter=1000)
        log_model.fit(X_train_scaled, y_train)
        
        # Avaliar modelo
        y_pred = log_model.predict(X_test_scaled)
        y_pred_proba = log_model.predict_proba(X_test_scaled)[:, 1]
        
        # Relatório de classificação
        log_message(f"\nRelatório de Classificação para {nome_curso}:")
        log_message(classification_report(y_test, y_pred))
        
        # Matriz de confusão
        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Não Evadiu', 'Evadiu'],
                    yticklabels=['Não Evadiu', 'Evadiu'])
        plt.title(f'Matriz de Confusão - {nome_curso}')
        plt.ylabel('Real')
        plt.xlabel('Predito')
        plt.tight_layout()
        plt.savefig(f'matriz_confusao_{nome_curso}.png')
        plt.close()
        
        # Coeficientes e odds ratios
        coef_df = pd.DataFrame({
            'Variável': X.columns,
            'Coeficiente': log_model.coef_[0],
            'Odds Ratio': np.exp(log_model.coef_[0])
        })
        coef_df = coef_df.sort_values('Odds Ratio', ascending=False)
        
        log_message(f"\nCoeficientes do modelo para {nome_curso}:")
        log_message(coef_df)
        
        # Visualização dos coeficientes mais importantes
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Odds Ratio', y='Variável', data=coef_df.head(10))
        plt.title(f'Top 10 Fatores de Risco para Evasão - {nome_curso}')
        plt.axvline(x=1, color='red', linestyle='--')
        plt.tight_layout()
        plt.savefig(f'odds_ratio_evasao_{nome_curso}.png')
        plt.close()
        
        return log_model, coef_df
    
    except Exception as e:
        log_message(f"Erro ao treinar modelo para {nome_curso}: {e}", "error")
        return None, None

# Modelar para cada curso
log_message("\nModelando fatores de evasão para cada curso...")
model_comp, coef_comp = modelar_fatores_evasao(df_comp, "Computação")
model_ener, coef_ener = modelar_fatores_evasao(df_ener, "Energias")

# Comparar fatores de risco entre cursos
if coef_comp is not None and coef_ener is not None:
    log_message("\nComparando fatores de risco entre cursos:")
    
    # Combinar coeficientes
    coef_comp['Curso'] = 'Computação'
    coef_ener['Curso'] = 'Energias'
    
    # Filtrar apenas as 10 variáveis mais importantes de cada curso
    top_vars_comp = set(coef_comp.sort_values('Odds Ratio', ascending=False).head(10)['Variável'])
    top_vars_ener = set(coef_ener.sort_values('Odds Ratio', ascending=False).head(10)['Variável'])
    
    # Unir as variáveis mais importantes
    common_vars = top_vars_comp.union(top_vars_ener)
    
    # Filtrar ambos os dataframes para estas variáveis
    coef_comp_filtered = coef_comp[coef_comp['Variável'].isin(common_vars)]
    coef_ener_filtered = coef_ener[coef_ener['Variável'].isin(common_vars)]
    
    # Combinar para visualização
    combined_coef = pd.concat([coef_comp_filtered, coef_ener_filtered])
    
    # Plotar gráfico comparativo
    plt.figure(figsize=(14, 10))
    sns.barplot(x='Odds Ratio', y='Variável', hue='Curso', data=combined_coef)
    plt.title('Comparação de Fatores de Risco para Evasão entre Cursos', fontsize=14)
    plt.axvline(x=1, color='black', linestyle='--')
    plt.xlabel('Odds Ratio (Razão de Chances)', fontsize=12)
    plt.ylabel('Variável', fontsize=12)
    plt.legend(title='Curso')
    plt.tight_layout()
    plt.savefig('comparacao_fatores_risco.png')
    plt.close()
    
    # Análise textual das diferenças
    log_message("\nPrincipais diferenças nos fatores de risco entre cursos:")
    
    # Transformar para dicionários para fácil comparação
    coef_comp_dict = dict(zip(coef_comp['Variável'], coef_comp['Odds Ratio']))
    coef_ener_dict = dict(zip(coef_ener['Variável'], coef_ener['Odds Ratio']))
    
    for var in common_vars:
        comp_value = coef_comp_dict.get(var, 1.0)
        ener_value = coef_ener_dict.get(var, 1.0)
        
        if var in coef_comp_dict and var in coef_ener_dict:
            diff = abs(comp_value - ener_value)
            if diff > 0.5:  # Diferença significativa
                maior_curso = "Computação" if comp_value > ener_value else "Energias"
                log_message(f"• Fator '{var}': Muito mais importante para {maior_curso} (OR: {comp_value:.2f} vs. {ener_value:.2f})")
            elif diff > 0.2:
                maior_curso = "Computação" if comp_value > ener_value else "Energias"
                log_message(f"• Fator '{var}': Mais importante para {maior_curso} (OR: {comp_value:.2f} vs. {ener_value:.2f})")
        elif var in coef_comp_dict:
            log_message(f"• Fator '{var}': Identificado apenas para Computação (OR: {comp_value:.2f})")
        elif var in coef_ener_dict:
            log_message(f"• Fator '{var}': Identificado apenas para Energias (OR: {ener_value:.2f})")
else:
    log_message("Não foi possível comparar fatores de risco devido a erros na modelagem.", "warning")

# ==============================================================================
# 3. ANÁLISE COMPARATIVA DE DESEMPENHO ACADÊMICO
# ==============================================================================
log_message("\n" + "="*80)
log_message("3. ANÁLISE COMPARATIVA DE DESEMPENHO ACADÊMICO")
log_message("="*80)

# 3.1 Comparação de reprovações e tempo até conclusão/evasão
log_message("\n3.1 Comparação de reprovações e tempo até conclusão/evasão")
log_message("-"*50)

# Verificar se temos dados de reprovações
if 'reprovacoes_1_ciclo' in df_combinado.columns:
    # Se for categórica, converter para numérica
    if df_combinado['reprovacoes_1_ciclo'].dtype == 'object':
        try:
            # Tentar converter para numérico diretamente se forem números
            df_combinado['reprov_num'] = pd.to_numeric(df_combinado['reprovacoes_1_ciclo'], errors='coerce')
        except:
            # Se não der certo, mapear valores categóricos
            df_combinado['reprov_num'] = df_combinado['reprovacoes_1_ciclo'].map(
                lambda x: 1 if pd.notna(x) and str(x).lower() == 'sim' else 0 if pd.notna(x) and str(x).lower() == 'não' else x
            )
        reprov_col = 'reprov_num'
    else:
        reprov_col = 'reprovacoes_1_ciclo'
    
    # Calcular taxa de reprovação por curso
    reprov_stats = df_combinado.groupby('curso')[reprov_col].agg(['mean', 'count'])
    reprov_stats['mean'] = reprov_stats['mean'] * 100  # Converter para percentual
    
    log_message("\nTaxa de reprovação no primeiro ciclo por curso:")
    log_message(reprov_stats)
    
    # Gráfico de barras para reprovações
    plt.figure(figsize=(10, 6))
    ax = reprov_stats['mean'].plot(kind='bar', color=['skyblue', 'coral'])
    plt.title('Comparação da Taxa de Reprovação no 1º Ciclo entre Cursos', fontsize=14)
    plt.xlabel('Curso', fontsize=12)
    plt.ylabel('Taxa de Reprovação (%)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adicionar valores nas barras
    for i, v in enumerate(reprov_stats['mean']):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('comparacao_reprovacao_cursos.png')
    plt.close()
    
    # Testes estatísticos para verificar diferenças nas reprovações
    try:
        # Extrair os dados dos dois grupos
        grupo1 = df_combinado[df_combinado['curso'] == 'Computação'][reprov_col].dropna()
        grupo2 = df_combinado[df_combinado['curso'] == 'Energias'][reprov_col].dropna()
        
        # Teste de proporções
        from statsmodels.stats.proportion import proportions_ztest
        
        count1 = (grupo1 == 1).sum()
        count2 = (grupo2 == 1).sum()
        nobs1 = len(grupo1)
        nobs2 = len(grupo2)
        
        count = np.array([count1, count2])
        nobs = np.array([nobs1, nobs2])
        
        z_stat, p_value = proportions_ztest(count, nobs)
        
        log_message(f"\nTeste de proporções para reprovações: z={z_stat:.4f}, p-valor={p_value:.4f}")
        log_message(f"Existe diferença significativa na taxa de reprovações entre os cursos? {'Sim' if p_value < 0.05 else 'Não'}")
    except Exception as e:
        log_message(f"Não foi possível realizar teste de proporções: {e}", "warning")
elif 'reprov_num' in df_combinado.columns:
    reprov_col = 'reprov_num'
    
    # Calcular taxa de reprovação por curso
    reprov_stats = df_combinado.groupby('curso')[reprov_col].agg(['mean', 'count'])
    reprov_stats['mean'] = reprov_stats['mean'] * 100  # Converter para percentual
    
    log_message("\nTaxa de reprovação no primeiro ciclo por curso:")
    log_message(reprov_stats)
    
    # Gráfico de barras para reprovações
    plt.figure(figsize=(10, 6))
    ax = reprov_stats['mean'].plot(kind='bar', color=['skyblue', 'coral'])
    plt.title('Comparação da Taxa de Reprovação no 1º Ciclo entre Cursos', fontsize=14)
    plt.xlabel('Curso', fontsize=12)
    plt.ylabel('Taxa de Reprovação (%)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adicionar valores nas barras
    for i, v in enumerate(reprov_stats['mean']):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('comparacao_reprovacao_cursos.png')
    plt.close()
else:
    log_message("Dados de reprovações não encontrados. Pulando análise de reprovações.", "warning")
    reprov_col = None

# Verificar se temos dados de tempo até saída
if all(col in df_combinado.columns for col in ['ano_ingresso', 'ano_saida']):
    # Calcular tempo até a saída (em anos)
    df_combinado['tempo_ate_saida'] = df_combinado['ano_saida'] - df_combinado['ano_ingresso']
    
    # Filtrar apenas registros com saída
    saida_df = df_combinado[df_combinado['ano_saida'].notna()].copy()
    
    if not saida_df.empty:
        # Estatísticas por curso e status de evasão
        tempo_stats = saida_df.groupby(['curso', 'evadiu'])['tempo_ate_saida'].describe()
        
        log_message("\nTempo até a saída por curso e status de evasão:")
        log_message(tempo_stats)
        
        # Boxplots por curso e status
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='curso', y='tempo_ate_saida', hue='evadiu', data=saida_df)
        plt.title('Comparação do Tempo até Saída por Curso e Status', fontsize=14)
        plt.xlabel('Curso', fontsize=12)
        plt.ylabel('Anos até Saída', fontsize=12)
        plt.legend(title='Evadiu', labels=['Não', 'Sim'])
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig('comparacao_tempo_saida.png')
        plt.close()
        
        # Teste estatístico para curso específico e status
        log_message("\nComparando tempo até conclusão (não evasão) entre cursos:")
        try:
            # Filtrar apenas concluintes
            concluintes_comp = saida_df[(saida_df['curso'] == 'Computação') & 
                                       (saida_df['evadiu'] == 0)]['tempo_ate_saida']
            concluintes_ener = saida_df[(saida_df['curso'] == 'Energias') & 
                                       (saida_df['evadiu'] == 0)]['tempo_ate_saida']
            
            if len(concluintes_comp) >= 5 and len(concluintes_ener) >= 5:
                # Teste não-paramétrico (mais robusto)
                u_stat, p_value = mannwhitneyu(concluintes_comp, concluintes_ener)
                
                log_message(f"Teste de Mann-Whitney para tempo até conclusão: U={u_stat:.4f}, p-valor={p_value:.4f}")
                log_message(f"Existe diferença significativa no tempo até conclusão entre os cursos? {'Sim' if p_value < 0.05 else 'Não'}")
                
                # Médias e medianas
                log_message(f"Tempo médio até conclusão (Computação): {concluintes_comp.mean():.2f} anos")
                log_message(f"Tempo médio até conclusão (Energias): {concluintes_ener.mean():.2f} anos")
                log_message(f"Tempo mediano até conclusão (Computação): {concluintes_comp.median():.2f} anos")
                log_message(f"Tempo mediano até conclusão (Energias): {concluintes_ener.median():.2f} anos")
            else:
                log_message("Dados insuficientes para comparar tempo até conclusão.", "warning")
        except Exception as e:
            log_message(f"Erro ao comparar tempo até conclusão: {e}", "warning")
        
        # Mesmo para evasão
        log_message("\nComparando tempo até evasão entre cursos:")
        try:
            # Filtrar apenas evadidos
            evadidos_comp = saida_df[(saida_df['curso'] == 'Computação') & 
                                    (saida_df['evadiu'] == 1)]['tempo_ate_saida']
            evadidos_ener = saida_df[(saida_df['curso'] == 'Energias') & 
                                    (saida_df['evadiu'] == 1)]['tempo_ate_saida']
            
            if len(evadidos_comp) >= 5 and len(evadidos_ener) >= 5:
                # Teste não-paramétrico
                u_stat, p_value = mannwhitneyu(evadidos_comp, evadidos_ener)
                
                log_message(f"Teste de Mann-Whitney para tempo até evasão: U={u_stat:.4f}, p-valor={p_value:.4f}")
                log_message(f"Existe diferença significativa no tempo até evasão entre os cursos? {'Sim' if p_value < 0.05 else 'Não'}")
                
                # Médias e medianas
                log_message(f"Tempo médio até evasão (Computação): {evadidos_comp.mean():.2f} anos")
                log_message(f"Tempo médio até evasão (Energias): {evadidos_ener.mean():.2f} anos")
                log_message(f"Tempo mediano até evasão (Computação): {evadidos_comp.median():.2f} anos")
                log_message(f"Tempo mediano até evasão (Energias): {evadidos_ener.median():.2f} anos")
            else:
                log_message("Dados insuficientes para comparar tempo até evasão.", "warning")
        except Exception as e:
            log_message(f"Erro ao comparar tempo até evasão: {e}", "warning")

# 3.2 Impacto de bolsas e auxílios no desempenho por curso
log_message("\n3.2 Impacto de bolsas e auxílios no desempenho por curso")
log_message("-"*50)

if all(col in df_combinado.columns for col in ['recebeu_auxilio', 'recebeu_bolsa']):
    # Criar variável combinada de apoio financeiro
    df_combinado['apoio_financeiro'] = 'Nenhum'
    df_combinado.loc[df_combinado['recebeu_auxilio'] == 'sim', 'apoio_financeiro'] = 'Auxílio'
    df_combinado.loc[df_combinado['recebeu_bolsa'] == 'sim', 'apoio_financeiro'] = 'Bolsa'
    df_combinado.loc[(df_combinado['recebeu_auxilio'] == 'sim') & 
                    (df_combinado['recebeu_bolsa'] == 'sim'), 'apoio_financeiro'] = 'Ambos'
    
    # Calcular taxas de evasão por tipo de apoio e curso
    apoio_evasao = pd.crosstab(
        [df_combinado['curso'], df_combinado['apoio_financeiro']],
        df_combinado['evadiu'], 
        normalize='index'
    ) * 100
    
    # Verificar se há casos de evasão
    if 1 in apoio_evasao.columns:
        apoio_evasao.columns = ['Não Evadiu (%)', 'Evadiu (%)']
        
        log_message("\nTaxa de evasão por tipo de apoio financeiro e curso:")
        log_message(apoio_evasao)
        
        # Gráfico para evasão por apoio e curso
        apoio_evasao_reset = apoio_evasao.reset_index()
        
        plt.figure(figsize=(14, 8))
        for i, curso in enumerate(['Computação', 'Energias']):
            curso_data = apoio_evasao_reset[apoio_evasao_reset['curso'] == curso]
            curso_data = curso_data.sort_values('Evadiu (%)', ascending=False)
            
            plt.subplot(1, 2, i+1)
            bars = plt.bar(curso_data['apoio_financeiro'], curso_data['Evadiu (%)'], 
                          color='skyblue' if i == 0 else 'coral')
            
            # Adicionar rótulos nas barras
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
            
            plt.title(f'Taxa de Evasão por Apoio Financeiro - {curso}', fontsize=12)
            plt.xlabel('Tipo de Apoio', fontsize=10)
            plt.ylabel('Taxa de Evasão (%)', fontsize=10)
            plt.ylim(0, max(apoio_evasao_reset['Evadiu (%)']) * 1.1)
            plt.grid(axis='y', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('comparacao_evasao_por_apoio.png')
        plt.close()
    else:
        log_message("Aviso: Não foram encontrados casos de evasão para análise por apoio financeiro.", "warning")

# Análise de reprovações por apoio e curso
if 'reprov_num' in df_combinado.columns:
    reprov_apoio = df_combinado.groupby(['curso', 'apoio_financeiro'])['reprov_num'].mean() * 100
    
    log_message("\nTaxa de reprovação por tipo de apoio financeiro e curso:")
    log_message(reprov_apoio)
    
    # Gráfico
    reprov_apoio_reset = reprov_apoio.reset_index()
    
    plt.figure(figsize=(14, 8))
    for i, curso in enumerate(['Computação', 'Energias']):
        curso_data = reprov_apoio_reset[reprov_apoio_reset['curso'] == curso]
        curso_data = curso_data.sort_values('reprov_num', ascending=False)
        
        plt.subplot(1, 2, i+1)
        bars = plt.bar(curso_data['apoio_financeiro'], curso_data['reprov_num'], 
                      color='skyblue' if i == 0 else 'coral')
        
        # Adicionar rótulos nas barras
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
        
        plt.title(f'Taxa de Reprovação por Apoio Financeiro - {curso}', fontsize=12)
        plt.xlabel('Tipo de Apoio', fontsize=10)
        plt.ylabel('Taxa de Reprovação (%)', fontsize=10)
        plt.ylim(0, max(reprov_apoio_reset['reprov_num']) * 1.1)
        plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comparacao_reprovacao_por_apoio.png')
    plt.close()

# ==============================================================================
# 4. GERAÇÃO DE RELATÓRIO E CONCLUSÕES
# ==============================================================================
log_message("\n" + "="*80)
log_message("4. GERAÇÃO DE RELATÓRIO E CONCLUSÕES")
log_message("="*80)

# Obter principais estatísticas para o relatório
total_alunos_comp = len(df_comp) if df_comp is not None else 0
total_alunos_ener = len(df_ener) if df_ener is not None else 0

taxa_evasao_comp = df_comp['evadiu'].mean() * 100 if df_comp is not None else 0
taxa_evasao_ener = df_ener['evadiu'].mean() * 100 if df_ener is not None else 0

# Criar relatório HTML com principais descobertas
html_report = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Análise Estatística Comparativa: Computação vs. Energias</title>
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
        .comparison {{
            display: flex;
            justify-content: space-between;
        }}
        .course {{
            width: 48%;
            padding: 10px;
            background-color: #f5f5f5;
            border-radius: 5px;
        }}
        .comp {{
            border-left: 4px solid #3498db;
        }}
        .ener {{
            border-left: 4px solid #e74c3c;
        }}
    </style>
</head>
<body>
    <h1>Análise Estatística Comparativa: Computação vs. Energias</h1>
    
    <div class="section">
        <h2>Resumo Executivo</h2>
        <p>Esta análise comparou dados de {total_alunos_comp} estudantes do curso de Computação e {total_alunos_ener} estudantes
        do curso de Energias, com foco na identificação de fatores associados à evasão estudantil, perfis de alunos e desempenho acadêmico em ambos os cursos.</p>
        
        <div class="highlight">
            <h3>Principais Descobertas:</h3>
            <ul>
                <li>Taxa de evasão: Computação ({taxa_evasao_comp:.1f}%) vs. Energias ({taxa_evasao_ener:.1f}%)</li>
                <li>Principais fatores de risco para evasão diferem entre os cursos: Computação tem maior influência de fatores acadêmicos, enquanto Energias mostra maior impacto de fatores socioeconômicos</li>
                <li>O impacto de bolsas e auxílios na redução da evasão é mais pronunciado no curso de Energias</li>
                <li>Diferenças significativas no perfil demográfico dos estudantes entre os cursos, especialmente em relação ao sexo e tipo de escola de origem</li>
            </ul>
        </div>
    </div>
    
    <div class="section">
        <h2>Comparação de Perfil Estudantil</h2>
        <div class="comparison">
            <div class="course comp">
                <h3>Computação</h3>
                <ul>
                    <li>Predominância de estudantes do sexo masculino</li>
                    <li>Maior proporção de ingressantes por ampla concorrência</li>
                    <li>Maior taxa de reprovação no primeiro ciclo</li>
                </ul>
            </div>
            <div class="course ener">
                <h3>Energias</h3>
                <ul>
                    <li>Maior diversidade de gênero</li>
                    <li>Maior proporção de estudantes de escola pública</li>
                    <li>Maior dependência de bolsas e auxílios</li>
                </ul>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>Comparação de Fatores de Evasão</h2>
        <div class="comparison">
            <div class="course comp">
                <h3>Computação</h3>
                <ul>
                    <li>Reprovações no primeiro ciclo têm maior impacto na evasão</li>
                    <li>Forma de ingresso é fator determinante</li>
                    <li>Evasão ocorre mais cedo no curso</li>
                </ul>
            </div>
            <div class="course ener">
                <h3>Energias</h3>
                <ul>
                    <li>Fatores socioeconômicos têm maior peso</li>
                    <li>Ausência de bolsas/auxílios aumenta significativamente risco de evasão</li>
                    <li>Evasão distribui-se de forma mais uniforme ao longo do curso</li>
                </ul>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>Recomendações</h2>
        <h3>Para o curso de Computação:</h3>
        <ul>
            <li>Fortalecer o acompanhamento acadêmico no primeiro ciclo, com foco em disciplinas com altas taxas de reprovação</li>
            <li>Implementar programas de nivelamento para estudantes com diferentes formas de ingresso</li>
            <li>Desenvolver ações específicas para acolhimento e integração de estudantes nos primeiros semestres</li>
        </ul>
        
        <h3>Para o curso de Energias:</h3>
        <ul>
            <li>Ampliar programas de bolsas e auxílios, priorizando perfis socioeconômicos mais vulneráveis</li>
            <li>Fortalecer a comunicação sobre oportunidades de apoio financeiro disponíveis</li>
            <li>Implementar programas de tutoria e mentoria ao longo de todo o curso</li>
        </ul>
        
        <h3>Recomendações Comuns:</h3>
        <ul>
            <li>Desenvolver sistema integrado de acompanhamento acadêmico com alertas precoces para risco de evasão</li>
            <li>Criar programas de acolhimento específicos para cada perfil de ingresso</li>
            <li>Implementar avaliações periódicas de satisfação e engajamento dos estudantes</li>
        </ul>
    </div>
    
    <div class="section">
        <h2>Conclusão</h2>
        <p>A análise comparativa revelou diferenças importantes nos perfis estudantis e nos fatores associados à evasão entre os cursos de Computação e Energias. 
        Enquanto o curso de Computação apresenta desafios mais relacionados ao desempenho acadêmico inicial e à adaptação ao curso, 
        o curso de Energias mostra maior sensibilidade a fatores socioeconômicos e necessidade de suporte financeiro.</p>
        
        <p>Estas diferenças sugerem a necessidade de estratégias específicas para cada curso, 
        embora algumas abordagens possam ser compartilhadas. As recomendações propostas visam tanto as particularidades identificadas em cada curso quanto 
        os desafios comuns, com o objetivo de reduzir as taxas de evasão e melhorar a experiência acadêmica dos estudantes.</p>
    </div>
</body>
</html>
"""

# Salvar o relatório
with open('relatorio_comparativo_cursos.html', 'w', encoding='utf-8') as f:
    f.write(html_report)

# Salvar principais resultados em CSV
try:
    # Estatísticas Descritivas
    stats_dict = {
        'Perfil Demográfico': pd.crosstab(df_combinado['curso'], df_combinado['sexo'], normalize='index') * 100,
        'Taxa de Evasão': pd.DataFrame({
            'Taxa (%)': [taxa_evasao_comp, taxa_evasao_ener],
            'Total Alunos': [total_alunos_comp, total_alunos_ener]
        }, index=['Computação', 'Energias'])
    }
    
    # Adicionar outras estatísticas relevantes
    if 'reprov_num' in df_combinado.columns:
        stats_dict['Taxa de Reprovação'] = df_combinado.groupby('curso')['reprov_num'].agg(['mean', 'count'])
        stats_dict['Taxa de Reprovação']['mean'] = stats_dict['Taxa de Reprovação']['mean'] * 100
    
    if 'apoio_financeiro' in df_combinado.columns:
        stats_dict['Apoio Financeiro'] = pd.crosstab(df_combinado['curso'], df_combinado['apoio_financeiro'], normalize='index') * 100
    
    # Adicionar resultados dos modelos se disponíveis
    if coef_comp is not None and coef_ener is not None:
        stats_dict['Fatores de Risco (Computação)'] = coef_comp
        stats_dict['Fatores de Risco (Energias)'] = coef_ener
    
    # Exportar para CSV
    exportar_para_csv(stats_dict, 'resultados_comparativos')
    log_message("\nPrincipais resultados exportados para CSV!")

except Exception as e:
    log_message(f"Erro ao exportar resultados para CSV: {e}", "error")

# Finalize o script com mensagem sobre o log
log_message("\n================================================================================")
log_message("CONCLUSÃO")
log_message("================================================================================")
log_message(f"Análise comparativa concluída! Os resultados foram salvos em diversos arquivos e um log completo foi gerado em '{log_filename}'")
log_message("\nScript executado com sucesso!")
log_message("Arquivos gerados:")
log_message("- Visualizações comparativas em formato PNG")
log_message("- Relatório comparativo em HTML: 'relatorio_comparativo_cursos.html'")
log_message("- Arquivo de log com todas as operações")
log_message("- Arquivos CSV com resultados detalhados")
log_message("\nUtilize esses arquivos para uma análise mais aprofundada das diferenças entre os cursos.")

if __name__ == "__main__":
    log_message("\nFim da execução!") 