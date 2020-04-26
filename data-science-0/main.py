#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[25]:


def q1():
    # Retorne aqui o resultado da questão 1.
    return black_friday.shape
    pass


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[168]:


def q2():
    # Retorne aqui o resultado da questão 2.
    mulheres1 = black_friday[black_friday.Gender == 'F']
    mulheres2 = mulheres1[mulheres1.Age == '26-35']
    return mulheres2.User_ID.count()
    pass


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[30]:


def q3():
    # Retorne aqui o resultado da questão 3.
    return black_friday.User_ID.nunique()
    pass


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[170]:


def q4():
    # Retorne aqui o resultado da questão 4.
    return len(black_friday.dtypes.value_counts())
    pass


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[182]:


def q5():
    # Retorne aqui o resultado da questão 5.
    nulos = (black_friday.shape[0] - black_friday.dropna().shape[0])/black_friday.shape[0]
    return nulos
    pass


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[184]:


def q6():
    # Retorne aqui o resultado da questão 6.
    maior_nulo = black_friday.isnull().sum().max()
    return maior_nulo
    pass


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[196]:


def q7():
    # Retorne aqui o resultado da questão 7.
    sem_nulos = black_friday[black_friday.notnull()]['Product_Category_3'].mode().item()
    return sem_nulos
    pass


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[163]:


def q8():
    # Retorne aqui o resultado da questão 8.
    mini = black_friday.Purchase.min()
    maxi = black_friday.Purchase.max()

    normalizado = (black_friday.Purchase - mini) / (maxi-mini)
    return normalizado.mean()
    pass


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[162]:


def q9():
    # Retorne aqui o resultado da questão 9.
    media = black_friday.Purchase.mean()
    desvio = black_friday.Purchase.std()

    black_friday.Purchase_padronizada = (black_friday.Purchase - media) / desvio
    entre = black_friday[(black_friday.Purchase_padronizada >= -1) & (black_friday.Purchase_padronizada <= 1)]
    return entre.shape[0]
    pass


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[206]:


def q10():
    # Retorne aqui o resultado da questão 10.
    nulos_filtro = black_friday[black_friday.Product_Category_2.isnull()]
    return nulos_filtro.shape[0] == nulos_filtro.Product_Category_3.isnull().shape[0]
    pass


# In[ ]:




