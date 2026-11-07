DATA_GROUPS = ('condicoes', 'ocorrencias', 'veiculo', 'via', 'vitimas')

YEARS = (2021, 2022, 2023, 2024, 2025, 2026)
ALL_YEARS = (2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026)

COLUMNS_PER_DATA_GROUP = {
    'condicoes': [
        'numero_ocorrencia_associado',
        'con_ano_fato',
        'con_mes_numerico_fato',
        'con_luminosidade_descricao',
        'con_condicao_sinalizacao_descricao',
        'con_tipo_via_descricao',
        'con_tipo_pavimento_descricao',
        'con_verificacao'
    ],
    'ocorrencias': [
        'numero_ocorrencia_associado',
        'oco_qtd_ocorrencia',
        'oco_natureza_descricao',
        'oco_ano_fato',
        'oco_mes_resumido',
        'oco_mes_numerico_fato',
        'oco_data_fato',
        'oco_dia_da_semana',
        'oco_horario_fato',
        'oco_faixa_1_hora_fato',
        'oco_faixa_6_horas_fato',
        'oco_motivo_presumido_descricao',
        'oco_grupo_local_imediato',
        'oco_local_imediato_longa',
        'oco_tipo_logradouro_descricao',
        'oco_logradouro_nome',
        'oco_nome_bairro',
        'oco_nome_municipio',
        'oco_codigo_municipio',
        'oco_ocorrencia_uf',
        'oco_risp_completa',
        'oco_rmbh',
        'oco_unidade_responsavel_registro_nome',
        'oco_latitude_sirgas2000',
        'oco_longitude_sirgas2000'
    ],
    'veiculo': [
        'numero_ocorrencia_associado',
        'vei_ano_fato',
        'vei_mes_numerico_fato',
        'vei_qtd_veiculo',
        'vei_tipo_veiculo_descricao_longa'
    ],
    'via': [
        'numero_ocorrencia_associado',
        'via_ano_fato',
        'via_mes_numerico_fato',
        'via_boa',
        'via_buracos',
        'via_material_na_pista',
        'via_ondulada',
        'via_outras_no_historico',
        'via_rachadura',
        'via_sulcos_horizontais',
        'via_verificacao'
    ],
    'vitimas': [
        'numero_ocorrencia_associado',
        'vit_ano_fato',
        'vit_mes_numerico_fato',
        'vit_nome_municipio_fato',
        'vit_codigo_municipio_fato',
        'vit_risp_completa_fato',
        'vit_rmbh_fato',
        'vit_ocorrencia_uf_fato',
        'vit_latitude_sirgas2000',
        'vit_longitude_sirgas2000',
        'vit_qtd_envolvido',
        'vit_envolvimento_descricao',
        'vit_envolvimento_codigo',
        'vit_condicao_fisica_descricao',
        'vit_valor_idade_aparente',
        'vit_codigo_sexo',
        'vit_cor_pele_descricao',
        'vit_escolaridade_descricao',
        'vit_tipo_envolvimento',
        'vit_grupo_tipo_envolvimento_codigo',
        'vit_grupo_tipo_envolvimento',
        'vit_faixa_etaria',
        'vit_numero_ocorrencia_env'
    ]
}

ORIGINAL_COLUMNS_NAMES_PER_DATA_GROUP = {
    'vitimas': 'numero_ocorrencia_associado;ano_fato;mes_numerico_fato;nome_municipio_fato;codigo_municipio_fato;risp_completa_fato;rmbh_fato;ocorrencia_uf_fato;latitude_sirgas2000;longitude_sirgas2000;qtd_envolvido;envolvimento_descricao;envolvimento_codigo;condicao_fisica_descricao;valor_idade_aparente;codigo_sexo;cor_pele_descricao;escolaridade_descricao;Tipo Envolvimento;Grupo Tipo Envolvimento - Código;Grupo Tipo Envolvimento;Faixa Etária;numero_ocorrencia_env',
    'via': 'numero_ocorrencia_associado;ano_fato;mes_numerico_fato;BOA;BURACOS;MATERIAL NA PISTA;ONDULADA;OUTRAS (NO HISTORICO);RACHADURA;SULCOS HORIZONTAIS;verificacao',
    'veiculo': 'numero_ocorrencia_associado;ano_fato;mes_numerico_fato;qtd_veiculo;tipo_veiculo_descricao_longa',
    'ocorrencias': 'numero_ocorrencia_associado;qtd_ocorrencia;natureza_descricao;ano_fato;mes_resumido;mes_numerico_fato;data_fato;Dia da Semana;horario_fato;Faixa_1_Hora_Fato;Faixa_6_Horas_Fato;motivo_presumido_descricao;Grupo Local Imediato;local_imediato_longa;tipo_logradouro_descricao;logradouro_nome;nome_bairro;nome_municipio;codigo_municipio;ocorrencia_uf;risp_completa;rmbh;unidade_responsavel_registro_nome;latitude_sirgas2000;longitude_sirgas2000',
    'condicoes': 'numero_ocorrencia_associado;ano_fato;mes_numerico_fato;luminosidade_descricao;condicao_sinalizacao_descricao;tipo_via_descricao;tipo_pavimento_descricao;verificacao'
}

VEHICLES_CATEGORIES = {
    'motocicleta': ('MOTOCICLETA', 'MOTONETA'),
    'bicicleta': ('BICICLETA', 'TRICICLO'),
    'pesado': ('ONIBUS', 'MICROONIBUS', 'CAMINHAO/TRATOR', 'TRATOR MISTO', 'TRATOR DE ESTEIRAS', 'CAMINHAO', 'TRATOR DE RODAS', 'SEMI-REBOQUE', 'REBOQUE')
}

STREET_ADDRESS_ROAD_TYPE_VALUES =  {"RODOVIA"}

ROAD_ACCIDENT_WITH_VICTIM_OCURRENCY_DESCRIPTION = "ACIDENTE DE TRANSITO COM VITIMA"
