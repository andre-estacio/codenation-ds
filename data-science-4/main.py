#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk

from IPython import get_ipython
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# In[2]:


# Algumas configurações para o matplotlib.
#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[3]:


countries = pd.read_csv("countries.csv")


# In[4]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.head(5)


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# In[5]:


# Sua análise começa aqui.

countries['Country'] = countries['Country'].str.strip()
countries['Region'] = countries['Region'].str.strip()
countries = countries.apply(lambda x: x.replace(',', '.', regex = True))
countries.iloc[:,2:] = countries.iloc[:,2:].apply(pd.to_numeric)

countries.head(5)


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[6]:


def q1():
    # Retorne aqui o resultado da questão 1.
    lista = list(countries['Region'].unique())
    resultado = sorted(lista)
    return resultado
    pass


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[7]:


def q2():
    # Retorne aqui o resultado da questão 2.
    discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
    countries['Pop_density'] = discretizer.fit_transform(countries[['Pop_density']])
    resultado = int(countries['Pop_density'][countries['Pop_density'] == 9.0].count())
    return resultado
    pass


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[8]:


def q3():
    # Retorne aqui o resultado da questão 3.
    codificado = OneHotEncoder(sparse = False).fit_transform(countries[['Region', 'Climate']].fillna(0))
    resultado = codificado.shape[1]
    return resultado
    pass


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[9]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[10]:


def q4():
    # Retorne aqui o resultado da questão 4.
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    pipeline = make_pipeline(imputer, scaler)

    countries_filtrado = countries.drop(columns = ['Country', 'Region'])
    columns = countries_filtrado.columns
    pipeline.fit(countries_filtrado)
    arable = pd.DataFrame(pipeline.transform(np.array(test_country[2:]).reshape(1, -1)), columns=columns).loc[:,"Arable"][0]
    resultado = round(float(arable), 3)
    return resultado
    pass


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[11]:


def q5():
    # Retorne aqui o resultado da questão 4.
    quantil_1 = countries['Net_migration'].quantile(0.25)
    quantil_3 = countries['Net_migration'].quantile(0.75)
    iqr = quantil_3 - quantil_1
    outliers_abaixo = countries['Net_migration'][countries['Net_migration'] < (quantil_1 - 1.5 * iqr)].count()
    outliers_acima = countries['Net_migration'][countries['Net_migration'] > (quantil_3 + 1.5 * iqr)].count()
    resultado = (int(outliers_abaixo), int(outliers_acima), False)
    return resultado
    pass


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[12]:


categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)


# In[13]:


def q6():
    # Retorne aqui o resultado da questão 4.
    count_vec = CountVectorizer()
    newsgroup_counts = count_vec.fit_transform(newsgroup.data)
    word_index = count_vec.vocabulary_.get(u'phone')
    count_list = newsgroup_counts.sum(axis=0)
    resultado = int(count_list[0,word_index])
    return resultado
    pass


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[14]:


def q7():
    # Retorne aqui o resultado da questão 4.
    tf_idf = TfidfVectorizer (use_idf=True)
    newsgroup_counts = tf_idf.fit_transform(newsgroup.data)
    word_index = tf_idf.vocabulary_.get(u'phone')
    tfidf_list = newsgroup_counts.sum(axis=0)
    word_tf_idf = tfidf_list[0,word_index]
    resultado = round(float(word_tf_idf), 3)
    return resultado
    pass

