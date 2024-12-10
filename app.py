import pandas as pd
from flask import Flask, jsonify, render_template, request

from dotenv import load_dotenv
import os
import numpy as np
load_dotenv()

from ai.query_vectors import search_similar_atividades
from ai.call_llm import call_llm

app = Flask(__name__)

atividades = pd.read_csv('data/atividades.csv')

def busca_atividades(atividade: str) -> str:
    """
    Busca por similaridade em um banco de dados vetorial para encontrar tipos de negócios similares.
    
    Args:
        atividade (str): Nome da atividade a ser buscada.
    
    Returns:
        str: Lista de tipos de negócios similares em formato de string, separados por vírgulas.
    """
    # Realizar a busca no Qdrant
    similares = search_similar_atividades(atividade, 50)

    # Processar os resultados e retornar os tipos de negócios mais similares como uma string
    atividades_similares = atividades[atividades.codigo_atividade.isin(similares)].atividade.to_list()
    atividades_lista = ", ".join(atividades_similares)
    resposta_modelo = call_llm(f"Diga quais tipos de negócio são equivalentes a um(a) '{atividade}' dentre a seguinte lista: '{atividades_lista}'.\
                               Você só deve retornar os negócios que são exatamente o mesmo negócio da atividade solicitada, que contenham a atividade solicitada entre outras coisas, ou sinônimos da atividade solicitada, não apenas negócios em ramos similares.\
                               Retorne uma lista no mesmo formato da lista fornecida, com as atividades separadas por uma vírgula, sem novas linhas.")
    resposta_tratada = resposta_modelo.replace('\n', '').split(', ')
    return resposta_tratada

# Carregar os dados do CSV no início da aplicação
def carregar_dados_populacao():
    df = pd.read_csv("data/populacao_porto_alegre.csv")
    geojson = {
        "type": "FeatureCollection",
        "features": []
    }
    for index, row in df.iterrows():
        # Converter a geometria para formato GeoJSON
        geometry = row["geometry"].replace("POLYGON ", "").replace("(", "").replace(")", "")
        bairro = row["NM_BAIRRO"]
        coordinates = [
            [
                [float(lon), float(lat)]
                for lon, lat in [pair.split() for pair in geometry.split(", ")]
            ]
        ]
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": coordinates,
            },
            "properties": {
                "populacao": int(row["POPULTOT"]),
                "area": "" if np.isnan(row["AreaKm2_p8"]) else row["AreaKm2_p8"],
                "densidade": "" if np.isnan(row["DnsPKm2_p8"]) else row["DnsPKm2_p8"],
                "bairro": bairro if type(bairro) is str else ""
            },
        }
        geojson["features"].append(feature)

    return geojson

def carregar_dados_salas_comerciais():
    df = pd.read_csv("data/imoveis_comerciais.csv")
    geojson = {
        "type": "FeatureCollection",
        "features": []
    }

    for index, row in df.iterrows():
        endereco_raw = f"{row['logradouro']} {row['predio']}"
        bairro = row["bairro"]
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [row["longitude"], row["latitude"]]
            },
            "properties": {
                "endereco": " ".join(endereco_raw.split()).strip(),
                "finalidade": row["des_finalidade"],
                "bairro": bairro.strip() if type(bairro) is str else '',
                "area": row['mtr_area_real'],
                "valor": row['vlr_venal_imovel']
            }
        }
        geojson["features"].append(feature)
    return geojson

def carregar_dados_estabelecimentos():
    df = pd.read_csv("data/alvaras_com_coordenadas.csv")
    geojson = {
        "type": "FeatureCollection",
        "features": []
    }

    for index, row in df.iterrows():
        endereco_raw = f"{row['logradouro']} {row['predio']}"
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [row["longitude"], row["latitude"]]
            },
            "properties": {
                "atividade": row["atividade"],
                "endereco": " ".join(endereco_raw.split()).strip(),
                "bairro": row["bairro"].strip(),
            }
        }
        geojson["features"].append(feature)
    return geojson

estabelecimentos_data = carregar_dados_estabelecimentos()

salas_comerciais_data = carregar_dados_salas_comerciais()

# Variável para armazenar os dados de população carregados
populacao_data = carregar_dados_populacao()

@app.route("/")
def index():
    # Substitua 'minha_chave' pela chave que você usa no MapTiler
    maptiler_key = os.environ.get("MAPTILER_KEY")

    return render_template("index.html", maptiler_key=maptiler_key)

@app.route("/populacao")
def populacao():
    return jsonify(populacao_data)

@app.route("/salas_comerciais")
def salas_comerciais():
    return jsonify(salas_comerciais_data)

@app.route("/estabelecimentos")
def estabelecimentos():
    return jsonify(estabelecimentos_data)

@app.route('/buscar_atividades', methods=['POST'])
def buscar_atividades():
    atividade = request.json['atividade']
    atividades_similares = busca_atividades(atividade)
    return jsonify({'atividades': atividades_similares})

if __name__ == "__main__":
    # app.run(debug=True)
    app.run()
