from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)

# Carregar o modelo de rede neural pré-treinado
modelo = joblib.load('decision_tree.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prever', methods=['post'])
def prever():
    # Obter os dados do formulário
    idade = int(request.form['age'])
    sexo = int(request.form['sex'])
    tipo_dor_peito = int(request.form['chest pain'])
    pressao_sanguinea_repouso = int(request.form['resting bp s'])
    colesterol_soro = int(request.form['cholesterol'])
    acucar_sangue_jejum = int(request.form['fasting blood sugar'])
    eletrocardiograma_repouso = int(request.form['resting ecg'])
    frequencia_cardiaca_maxima = int(request.form['max heart rate'])
    angina_induzida_exercicio = int(request.form['exercise angina'])
    depressao_st = int(request.form['oldpeak'])
    inclinacao_st_pico = int(request.form['ST slope'])

   
    # Organizar os dados em um array numpy
    dados = np.array([[idade, sexo, tipo_dor_peito, pressao_sanguinea_repouso, colesterol_soro,
                      acucar_sangue_jejum, eletrocardiograma_repouso, frequencia_cardiaca_maxima,
                      angina_induzida_exercicio, depressao_st, inclinacao_st_pico]])
    
    # Verificar se os dados têm o formato correto (11 features)
    if dados.shape[1] != 11:
        return "Erro: Os dados fornecidos não têm o formato correto."

    # Fazer a previsão usando o modelo
    previsao = modelo.predict(dados)

    # Verificar se a previsão é unidimensional e expandir se necessário
    if previsao.ndim == 1:
        previsao = np.expand_dims(previsao, axis=0)

    # Obter o resultado da previsão
    resultado = np.argmax(previsao, axis=1)

    # Interpretar o resultado
    saida = "Doença Cardiovascular" if resultado[0] == 1 else "Normal"

    return render_template('resultado.html', previsao=saida)

if __name__ == '__main__':
    app.run(debug=True)
