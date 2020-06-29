# Bibliotecas necesarias para el aplicativo
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Importamos la base de datos
df = pd.read_csv("BD LUBRICACION.csv")
equipos = df.TAG.unique()
clase =df.CLASE.unique()

# Parametros para los diferentes tipos de reportes
param_desgaste =['CLASIFICACION_DE_EQUIPO','AG_(PLATA)','AL_(ALUMINIO)','CR_(CROMO)', 'CU_(COBRE)', 'FE_(HIERRO)', 'MO_(MOLIBDENO)','NI_(NIQUEL)','PB_(PLOMO)', 'SN_(ESTANO)', 'TI_(TITANIO)']

param_contaminante = ['CLASIFICACION_DE_CONTAMINACION','K_(POTASIO)', 'NA_(SODIO)', 'SI_(SILICIO)', 'V_(VANADIO)']

param_aditivo = ['TAG','MUESTREADA','EDAD_DEL_EQUIPO','EDAD_DEL_ACEITE','CLASIFICACION_DEL_ACEITE','B_(BORO)','BA_(BARIO)', 'CA_(CALCIO)', 'MG_(MAGNESIO)', 'P_(FOSFORO)','ZN_(ZINC)']

param_lubricante = ['CLASIFICACION_DEL_ACEITE', 'INDICE_PQ','VISC@100C_(CST)','OXIDACION_(AB/CM)', '@TAN_(MG_KOH/G)', 'NITRATION_(AB/CM)','TBN_(MG_KOH/G)','HOLLIN_(WT_%)', 'AGUA_(VOL.%)','INDICADOR_DE_REFRIGERANTE']

param_reporte =['TAG','MUESTREADA','EDAD_DEL_EQUIPO','EDAD_DEL_ACEITE','ESTADO_DEL_REPORTE','CLASIFICACION_DEL_ACEITE','CLASIFICACION_DE_EQUIPO','CLASIFICACION_DE_CONTAMINACION']

# Esquema de colores para los reportes principales
col_pat_report = (lambda x: 'background-color : tomato' if x=="ALERTA" 
         else 'background-color : orange' if x=="PRECAUCION" 
         else 'background-color : springgreen' if x=="NORMAL"
         else "")

# Funcion Principal
def run():
    
    # Construcción del sidebar
    st.sidebar.info('LubApp V 0.1')
    selected_clase = st.sidebar.selectbox("Seleccione por clase de equipo",clase) # Seleccion de las clases
    selected_equipo = st.sidebar.selectbox("Seleccione por codigo de equipo",df[df.CLASE==selected_clase].TAG.unique()) 
    var_graf = ["Serie de tiempo", "Correlacion"]
    tipo_graf = st.sidebar.selectbox("Tipo de Grafico", var_graf)
        
    # Funciones Transversales
    def graf_sert(df,equipo,param):
        var_graf_lub= st.selectbox("",param)
        filt_frame = df[df["TAG"]== equipo]
        return px.line(filt_frame, x='MUESTREADA', y=var_graf_lub, title='Comportamiento Variable Vs Tiempo' )
      
    def graf_corr(df,equipo,param_x,param_y):
        var_graf_corr_x = st.selectbox("Variable en X",param_x)
        var_graf_corr_y = st.selectbox("Variable en Y",param_y)
        filt_frame = df[df["TAG"]== equipo]
        return px.scatter(filt_frame, x=var_graf_corr_x, y=var_graf_corr_y, title='Correlacion entre variables' )
    
    def reporte_estado(df): # Funcion de reporte general
        filt_frame = df[df["CLASE"] == selected_clase]
        ult_muestra = dict(filt_frame.groupby("TAG")["MUESTREADA"].max())
        list_index = []
        for key,value in ult_muestra.items():
            index_value=(filt_frame[(filt_frame["TAG"]==key)&(df["MUESTREADA"]==value)].index)
            list_index.append(index_value.item())
        return df.iloc[list_index][param_reporte].style\
                .applymap(col_pat_report)\
                .set_precision(2)  
    
    def reporte_lubricante(df,equipo):
        filt_frame = df[df["TAG"]== equipo]
        ult_muestra = filt_frame.MUESTREADA.max()
        return filt_frame[(filt_frame["TAG"]==equipo)&(filt_frame["MUESTREADA"]== ult_muestra)][param_lubricante].style\
                .applymap(col_pat_report)\
                .set_precision(2)
    
    def reporte_desgaste(df,equipo):
        filt_frame = df[df["TAG"]== equipo]
        ult_muestra = filt_frame.MUESTREADA.max()
        return filt_frame[(filt_frame["TAG"]==equipo)&(filt_frame["MUESTREADA"]== ult_muestra)][param_desgaste].style\
                .applymap(col_pat_report)\
                .set_precision(2)
    
    def reporte_contaminante(df,equipo):
        filt_frame = df[df["TAG"]== equipo]
        ult_muestra = filt_frame.MUESTREADA.max()
        return filt_frame[(filt_frame["TAG"]==equipo)&(filt_frame["MUESTREADA"]== ult_muestra)][param_contaminante].style\
                .applymap(col_pat_report)\
                .set_precision(2)
    
    # Titulo de la aplicación 
    st.header("Aplicativo para la gestión de resultados de lubricación")

    # Reporte general
    st.subheader("Últimos resultados encontrados en la base de datos") 
    st.dataframe(reporte_estado(df)) # impresión reporte general
    
    #Reportes por equipo
    st.subheader("Estado del Lubricante")
    
    # Reporte de estado del lubricante
    st.dataframe(reporte_lubricante(df,selected_equipo))
    param_graf_lub = ['VISC@100C_(CST)','OXIDACION_(AB/CM)', '@TAN_(MG_KOH/G)', 'NITRATION_(AB/CM)','TBN_(MG_KOH/G)','HOLLIN_(WT_%)', 'AGUA_(VOL.%)']
    var_corr_lub_x= ['EDAD_DEL_ACEITE','EDAD_DEL_EQUIPO','VISC@100C_(CST)','OXIDACION_(AB/CM)', '@TAN_(MG_KOH/G)', 'NITRATION_(AB/CM)','TBN_(MG_KOH/G)','HOLLIN_(WT_%)', 'AGUA_(VOL.%)']
    var_corr_lub_y= ['VISC@100C_(CST)','EDAD_DEL_EQUIPO','EDAD_DEL_ACEITE','OXIDACION_(AB/CM)', '@TAN_(MG_KOH/G)', 'NITRATION_(AB/CM)','TBN_(MG_KOH/G)','HOLLIN_(WT_%)', 'AGUA_(VOL.%)']  
   
    if tipo_graf == "Serie de tiempo":   
        st.write(graf_sert(df,selected_equipo,param_graf_lub))
    else:
        st.write(graf_corr(df,selected_equipo,var_corr_lub_x,var_corr_lub_y))
    
    # Reporete de Desgaste   
    st.subheader("Contenido de metales de desgaste")
    st.dataframe(reporte_desgaste(df,selected_equipo))
    
    param_graf_desg = ['FE_(HIERRO)','AG_(PLATA)','AL_(ALUMINIO)','CR_(CROMO)', 'CU_(COBRE)',  'MO_(MOLIBDENO)','NI_(NIQUEL)','PB_(PLOMO)', 'SN_(ESTANO)', 'TI_(TITANIO)']
    var_corr_desg_x= ['EDAD_DEL_ACEITE','EDAD_DEL_EQUIPO','AG_(PLATA)','AL_(ALUMINIO)','CR_(CROMO)', 'CU_(COBRE)', 'FE_(HIERRO)', 'MO_(MOLIBDENO)','NI_(NIQUEL)','PB_(PLOMO)', 'SN_(ESTANO)', 'TI_(TITANIO)']
    var_corr_desg_y= [ 'FE_(HIERRO)','EDAD_DEL_ACEITE','EDAD_DEL_EQUIPO','AG_(PLATA)','AL_(ALUMINIO)','CR_(CROMO)', 'CU_(COBRE)', 'MO_(MOLIBDENO)','NI_(NIQUEL)','PB_(PLOMO)', 'SN_(ESTANO)', 'TI_(TITANIO)'] 
    
    if tipo_graf == "Serie de tiempo":   
        st.write(graf_sert(df,selected_equipo,param_graf_desg))
    else:
        st.write(graf_corr(df,selected_equipo,var_corr_desg_x,var_corr_desg_y))
    
    # Resultado del contaminante
    st.subheader("Contaminación del lubricante")
    st.dataframe(reporte_contaminante(df,selected_equipo))
    
    param_graf_cont = ['K_(POTASIO)', 'NA_(SODIO)', 'SI_(SILICIO)', 'V_(VANADIO)']
    var_corr_cont_x= ['EDAD_DEL_ACEITE','EDAD_DEL_EQUIPO','K_(POTASIO)', 'NA_(SODIO)', 'SI_(SILICIO)', 'V_(VANADIO)']
    var_corr_cont_y= ['K_(POTASIO)','EDAD_DEL_ACEITE','EDAD_DEL_EQUIPO', 'NA_(SODIO)', 'SI_(SILICIO)', 'V_(VANADIO)'] 
    
    if tipo_graf == "Serie de tiempo":   
        st.write(graf_sert(df,selected_equipo,param_graf_cont))
    else:
        st.write(graf_corr(df,selected_equipo,var_corr_cont_x,var_corr_cont_y))

if __name__ == '__main__':
    run()