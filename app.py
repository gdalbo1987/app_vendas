import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date
import joblib as jb

def load_data(file):
    df = pd.read_excel(file)
    return df

st.set_page_config(page_title = 'Previsão de Vendas', layout = 'wide')

st.write('Avaliação Técnica - Giuliano Dal Bó')
st.write('Data da avaliação: 23/09/2024')
st.title('Previsão de Vendas com Machine Learning 📈')
st.warning('O modelo gera a previsão para os 12 meses subsequentes ao da entrada de dados.')
st.success('Hipoteticamente a próxima previsão será gerada a partir dos resultados de vendas de Janeiro de 2024.')

st.sidebar.title('Dados de entrada:')

uploaded_file = st.sidebar.file_uploader('Carregue a Planilha de Vendas:', type = ['xlsx'])

param1 = st.sidebar.number_input('Parâmetro 5', value = None, format = '%0f')
param2 = st.sidebar.number_input('Parâmetro 8', value = None, format = '%0f')
param3 = st.sidebar.number_input('Valor atual das vendas', value = None, format = '%0f')

model = jb.load('model.pkl.z')

if st.sidebar.button('Rodar Modelo'):

    
    if uploaded_file is not None and param1 is not None and param2 is not None and param3 is not None:

        

            df = load_data(uploaded_file)

            df['Data'] = pd.to_datetime(df['Ano'].astype(str) + '-' + df['Mês'].astype(str) + '-01') + pd.offsets.MonthEnd(0)
            
            df.rename(columns = {'TARGET' : 'VENDAS'}, inplace = True)
            df.drop(['PARAMETRO 1',
                    'PARAMETRO 2',
                    'PARAMETRO 3',
                    'PARAMETRO 4',
                    'PARAMETRO 6',
                    'PARAMETRO 7',
                    'Ano',
                    'Mês'], axis = 1, inplace = True)
            
            last_date = df.iloc[-1, -1]
            
            data_run = last_date + pd.DateOffset(months=1)
            
            new_row = pd.DataFrame({'Data': [data_run], 'VENDAS': [float(param3)], 'PARAMETRO 5': [float(param1)], 'PARAMETRO 8': [float(param2)]})

            df = pd.concat([df, new_row])

            df = df.set_index('Data')

            df['LAG_1'] = df['VENDAS'].shift(1).fillna(method = 'bfill')
            df['LAG_2'] = df['VENDAS'].shift(2).fillna(method = 'bfill')
            df['LAG_3'] = df['VENDAS'].shift(3).fillna(method = 'bfill')

            df['D1'] = df['VENDAS'].diff(1).fillna(method = 'bfill')
            df['D2'] = df['VENDAS'].diff(2).fillna(method = 'bfill')
            df['D3'] = df['VENDAS'].diff(3).fillna(method = 'bfill')
            df['D4'] = df['VENDAS'].diff(4).fillna(method = 'bfill')

            ultimo_indice = df.index[-1]

            prox_mes = ultimo_indice + pd.DateOffset(months=1)

            datas_futuras = pd.date_range(start = prox_mes, periods=12, freq='M')

            df_ = pd.DataFrame((df.iloc[-1, :])).T

            order = ['LAG_2', 'D1', 'D3', 'D4', 'LAG_1', 'VENDAS', 'D2', 'LAG_3', 'PARAMETRO 8', 'PARAMETRO 5']

            df_pred = df_[order]

            p = model.predict(df_pred)

            preds = pd.DataFrame({'Data': datas_futuras, 'Previsões': p.T[:, 0]})

            preds = preds.set_index('Data')

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x = df.index,
                y = df['VENDAS'],
                mode = 'lines',
                name = 'Histórico de vendas',
                line = dict(color = 'blue')
            ))

            fig.add_trace(go.Scatter(
                x = preds.index,
                y = preds['Previsões'],
                mode = 'lines',
                name = 'Previsão de vendas',
                line = dict(color = 'green')
            ))
            
            v_line = ultimo_indice + pd.DateOffset(days = 15)

            fig.add_shape(
                type = 'line',
                x0 = v_line, x1 = v_line,
                y0 = 0, y1 = 5000,
                line = dict(color = 'black', width = 10)
            )



            fig.update_layout(
                title = 'Gráfico de Histórico de Vendas com Previsão dos Próximos 12 Meses',
                xaxis_title = 'Data',
                yaxis_title = 'Vendas'
            )

            st.subheader('Principais Indicadores:')

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric('Média de vendas prevista:', int(preds['Previsões'].mean()))

            with col2:
                st.metric('Valor máximo de vendas previsto:', int(preds['Previsões'].max()))
            
            with col3:
                st.metric('Valor mínimo de vendas previsto:', int(preds['Previsões'].min()))

            st.metric('Mês de Pico:', preds['Previsões'].idxmax().strftime('%b'))

            st.plotly_chart(fig, on_select = 'ignore', use_container_width = True)

            
    else:
        st.error('Favor carregar planilha de vendas e inserir dos dados para gerar a previsão.')
        







