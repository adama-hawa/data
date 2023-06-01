import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as html
import plotly.express as px
import io 
import chart_studio.plotly as py
import seaborn as sns 
import plotly.express as px
import cufflinks as cf
import altair as alt
import seaborn as sns
import hydralit as hy
import datetime as dt
#from hydralit import HydraHeadApp

from streamlit_option_menu import option_menu
from  PIL import Image
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
cf.go_offline()

#Nous allons lire notre datasets
file_path = "./data_vide_jan0_climat.csv"
data = pd.read_csv(file_path, sep=';', encoding='latin-1')


# convertir le type de la collone heure d'object à datetime
#data["dateheure"] = pd.to_datetime(data["dateheure"], format="%Y%m%d%H%M%S")

# Extraction des composantes de date, heure et minutes-secondes
#data['date'] = data['dateheure'].dt.strftime("%d-%m-%Y")
#data['heure'] = data['dateheure'].dt.strftime('%H')
#data['minutes_secondes'] = data['dateheure'].dt.strftime('%M:%S')
#data['minutes'] = data['dateheure'].dt.strftime('%M')
#data['secondes'] = data['dateheure'].dt.strftime('%S')

# Restreindre la dateheure sur les pics observés afin de voir comment se produisent les phénomènes dans un nouvel dataframe : filtre
#def res(dataFinal, dateDebut, dateFin):
    #dataFinal = dataFinal[(dataFinal['dateheure'] >= dateDebut) & (dataFinal['dateheure'] <= dateFin)]
    #return dataFinal

#data = res(data, "2020-01-03 11:15:00", "2020-01-03 11:30:00")

print_dataset = st.checkbox('Afficher les données')

if print_dataset:
    st.write(data.head(10))
    @st.cache_data
    def convertir_df(df):
        return df.to_csv().encode('utf-8')

    csv = convertir_df(data)

    st.download_button(
        label="Télécharger le dataset (.csv)",
        data=csv,
        file_name='dataset.csv',
        mime='text/csv',
    )

# data.iplot()


#side bar
st.sidebar.markdown('<p class="font">ARPEGE MASTERK</p>', unsafe_allow_html=True)

data_taille = len(data)

# st.write("L'analyse de données se fera sur " +data_taille+ " individus")

with st.sidebar.expander("A propos de cette application"):
     st.write("""
        Décrire l'application  \n  \nSignature
     """)

# menu vertical
with st.sidebar:
    st.sidebar.title("Filtres")
    
    
    # Liste des colonnes autorisées pour les filtres
    allowed_cols = ['pointsCapteur1','pointsCapteur2','pointsCapteur3','pointsCapteur4','pointsCapteur5','pointsCapteur6','totalPoints',]
    
    # Modifier le types des colonnes d'objet à int (pour les données quantitatives)
    #data['heure'] = pd.to_numeric(data['heure'])
    
    #heure_max = data['dateheure'].max()
    #heure_min = data['dateheure'].min()
    #ages = st.slider('Séléctionnez l\'heure', heure_min, heure_max, 11)

    
    all_selected = st.checkbox("Sélectionner toutes les variables")
    selected_filters = []

    if all_selected:
        selected_filters = allowed_cols
    else:
       selected_filters = st.multiselect("Sélectionner toutes les variables", allowed_cols, default=allowed_cols)
    
    data = data[selected_filters]

    all = st.button("Réinitialiser")

    if all :
        selected_filters = allowed_cols
        all_selected = True
        data = data[selected_filters]
    #mask = pd.Series(True, index=data.index)
    # Filtres par colonne 
    #for col in allowed_cols:
        #if col in selected_filters : 
            #mask &= True
        #else :
            #mask &= False
    #data = data[mask]

#st.write(selected_filters)

    
# with st.write("Jours d'hospitalisation"):
#     jours = st.slider('How old are you?', 0, 130, 25)







onglet1, onglet2, onglet3 = st.tabs(["Lecture de données","Statistiques Desciprtives","Variables"])

with onglet1:
    st.header("Lecture des données")

    st.write("Voici les données :")

    # Afficher les données sous forme de tableau interactif
    st.dataframe(data, width=800, height=400)


with onglet2:
   st.header("Statistiques Desciprtives")

   #récupération des nom des colonnes dynamiquement
   nom_col = list(data.columns)

   option_variable1 = st.selectbox(
   'Variable 1',
   (nom_col[1:]),
   key='variable1'
   )
   #st.write('You selected:', option_variable)

   #Calcul des stats déscriptives
   stat_desc = data[option_variable1]
   st.write(stat_desc.describe())

   # Vérification si la variable est numérique
   if stat_desc.dtype == 'float64' or stat_desc.dtype == 'int64':
       # Création d'un diagramme en boîte
       fig, ax = plt.subplots()
       sns.boxplot(x=option_variable1, data=data, ax=ax)
       ax.set_title(option_variable1)

       # Affichage du diagramme en boîte
       st.pyplot(fig)
   else:
       st.write("Cette variable n'est pas numérique, donc elle ne peut pas être représentée graphiquement avec un diagramme en boîte.")
    # Sélectionner uniquement les colonnes numériques et obtenir leurs noms
   numeric_cols = data.select_dtypes(include=np.number)#.columns.drop([""]).tolist()

    # Ajouter les noms des colonnes numériques à la liste nom_col
   nom_col1 = [''] + numeric_cols.columns.tolist() # ajouter une chaîne vide pour la première option de la boîte de sélection

   cor_matrix1 = data[numeric_cols.columns].corr()
   fig, ax = plt.subplots()
   sns.heatmap(cor_matrix1, annot=True, cmap="YlGnBu", ax=ax)
   ax.set_title("Matrice de corrélation")
   st.pyplot(fig)
    # Analyse de Survie
   option_variable2 = st.selectbox(
        'Variable 2',
        (nom_col1[1:]),
        key='variable2'
    )
   option_variable3 = st.selectbox(
        'Variable 3',
        (nom_col1[1:]),
        key='variable3'
    )
    # Sélection des données pour les deux variables
   subset_data = data[[option_variable2, option_variable3]].select_dtypes(include=np.number)
    # Calcul de la matrice de corrélation
   corr_matrix = subset_data.corr()

    # Affichage de la matrice de corrélation avec Seaborn
   fig, ax = plt.subplots()
   sns.heatmap(corr_matrix, annot=True, cmap="YlGnBu", ax=ax)
   ax.set_title("Matrice de corrélation")
   st.pyplot(fig)

with onglet3:
    st.header("Variables")
    # Ajouter du texte
    st.markdown(" Graphique : Veuillez sélectionner deux variables")
    # Récupération des noms des colonnes dynamiquement
    nom_col = list(data.columns)
    # Sélection de la variable à afficher sur l'axe des x
    x_variable = st.selectbox("Sélectionner la variable pour l'axe des x :", nom_col[1:])
    # Sélection de la variable à afficher sur l'axe des y
    y_variable = st.selectbox("Sélectionner la variable pour l'axe des y :", nom_col[1:])
    if data[x_variable].dtype == 'int64' or data[x_variable].dtype == 'float64':

        # Convertir la variable en float64 si elle ne l'est pas déjà
        if data[x_variable].dtype != 'float64':
            data[x_variable] = data[x_variable].astype('float64')

        # Créer le diagramme en boîte
        box_chart = alt.Chart(data).mark_boxplot().encode(
            x=x_variable,
            y=y_variable
        ).interactive()

        # Afficher le diagramme en boîte
        st.subheader("Diagramme en boîte")
        st.altair_chart(box_chart, use_container_width=True)

    # Si l'une des deux variables est qualitative
    else:
        # Création d'un graphique à barres
        fig = px.bar(data, x=x_variable, color=y_variable)
        # Affichage du graphique
        st.subheader("Graphique à barres")
        st.plotly_chart(fig)

    # Création du nuage de points
    scatter_chart = alt.Chart(data).mark_point().encode(
            x=x_variable,
            y=y_variable,
            color=alt.Color(nom_col[1], scale=alt.Scale(range=['red', 'blue']))
        ).interactive()
            
    # Affichage du nuage de points
    st.subheader("Nuage de points")
    st.altair_chart(scatter_chart, use_container_width=True)

        # Ajouter une image
        #image = Image.open('image.png')
        #st.image(image, caption='Voici une image', use_column_width=True)
