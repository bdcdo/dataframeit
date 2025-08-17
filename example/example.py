from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd
from dataframeit import dataframeit

TEMPLATE = """
Você é um advogado especialista em direito sanitário.
Irei te apresentar o texto de uma decisão judicial.

Classifique a decisão judicial de acordo com o seguinte formato:

INICIO DAS INSTRUÇÕES DE FORMATAÇÃO
{format}
FIM DAS INSTRUÇÕES DE FORMATAÇÃO

Utilize as informações do texto para preencher os campos.

INICIO DA SENTENÇA
{sentenca}
FIM DA SENTENÇA
"""

descricao_escopo = """Responda 'dentro_do_escopo' se o caso analisado estiver relacionado a casos que cumpram os seguintes requisitos:
1. Trata-se de um pedido de tratamento de saúde.
2. O autor da ação pleiteia tratamento para uma doença rara.

Se o caso não trata de pedido de tratamento de saúde, responda "fora_do_escopo".
Se você estiver em dúvida sobre se o autor da ação tem uma doença rara ou não, responda "escopo_incerto".

ATENÇÃO: É possível que a decisão mencione doenças raras apenas em uma citação de jurisprudência, mas o caso em si do autor não tenha relação com isso.
Considere essa possibilidade ao avaliar se o caso está ou não dentro do escopo.

Se o caso for classificado como "fora_do_escopo", responda todas as perguntas seguintes como "None".
"""

class Tratamento(BaseModel):
    nome: str | None = Field(..., description="Nome do tratamento pleiteado. 'None' apenas se não foi pleiteado tratamento.")
    resultado: Literal['deferido', 'indeferido'] | None = Field(None, description="Se o tratamento indicado foi deferido ou indeferido. 'None' apenas se não foi pleiteado tratamento.")

class RespostaFinalWrapper(BaseModel):
    definicao_escopo: Literal["dentro_do_escopo", "escopo_incerto", "fora_do_escopo"] = Field(...,
                                description=descricao_escopo)
    justificativa: str = Field(...,
                               description="Justificativa da avaliação anterior sobre o escopo do caso.")
    condicoes_de_saude: list[str] | None = Field(None, 
                                                 description="Lista de nomes das condição de saúde do paciente (sejam elas raras ou não). Mantenha a mesma redação utilizada pelo magistrado ao se referir a cada doença.")
    tratamentos: list[Tratamento] | None = None
    genero_paciente: Literal['masculino', 'feminino', 'outro'] | None = Field(None,
                                                                       description='Gênero do autor da ação. Responda outro se houver mais de um autor ou não for possível identificar. Responda None se o caso é fora do escopo.')
    justica_gratuita: Literal['nao', 'sim mas foi negado', 'sim e foi aceito'] | None = Field(None,
                                                        description='Há menção na decisão ao fato de que o autor da ação requereu justiça gratuita?')
    representacao: Literal['sim', 'nao'] | None = Field(None,
                                                        description='O autor da ação é representado por seu responsável/genitor ou seu tutor?')
    danos_morais_pedido: tuple[Literal['sim', 'nao'], float] | None = Field(...,
                                       description='Se o paciente pediu danos morais e, se sim, quanto. Responda "nao" se não há menção a danos morais na decisão. Responda None se o caso é fora do escopo.')
    danos_morais_concedido: tuple[Literal['sim', 'nao'], float] | None = Field(None,
                                       description='Se o magistrado concedeu o pedido danos morais e, se sim, de que valor. Responda None se não há menção a danos morais na decisão.')
    
if __name__=='__main__':
    df = pd.read_excel('clusters_saude_cluster_amostras_rodada5_5porCluster.xlsx')
    
    df_final=dataframeit(df, RespostaFinalWrapper, TEMPLATE)

    df_final.to_excel('clusters_analisados.xlsx', index=False)