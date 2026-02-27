import os
import joblib
import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


app = Dash(__name__, serve_locally=True, suppress_callback_exceptions=True)
app.title = "Dashboard Nutritionnel France"


server = app.server



def safe_load(path, fallback_cols):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=fallback_cols)



df1 = safe_load("Fichier1_Nutriscore_Categories.csv", ["main_category_en", "nutriscore_grade"])

if df1.empty:
    df1 = pd.DataFrame({
        'main_category_en': ['Beverages'] * 50 + ['Sugary snacks'] * 50,
        'nutriscore_grade': ['a', 'b', 'c', 'd', 'e'] * 20
    })

category_keywords = {
    "Animaux": ["cat", "dog", "pet", "animal"],
    "Boissons": ["drink", "juice", "soda", "water", "tea", "coffee", "beverage", "kombucha", "sirop", "nectar", "gazpacho", "thé", "café"],
    "Viande & poisson": ["ham", "chicken", "turkey", "pork", "beef", "sausage", "saucisse", "salmon", "fish", "meat", "poisson", "thon", "tuna"],
    "Produits laitiers": ["milk", "cheese", "yogurt", "cream", "butter", "fromage", "yaourt", "lait", "egg", "oeuf"],
    "Plats préparés": ["pizza", "meal", "ready", "lasagna", "frozen", "soupe", "soup", "sandwich", "burger"],
    "Snacks sucrés": ["biscuit", "cookie", "cake", "chocolate", "candy", "bonbon", "snack", "gaufre"],
    "Boulangerie": ["bread", "bakery", "toast", "pain", "brioche", "crepe"],
    "Fruits & légumes": ["fruit", "vegetable", "salad", "legume", "tomato", "carrot"],
    "Féculents": ["pasta", "rice", "noodle", "pâte", "riz", "lentil"],
    "Sauces & tartinables": ["sauce", "ketchup", "mayo", "mustard", "spread", "hummus"],
    "Confitures & miel": ["jam", "jelly", "honey", "confiture", "miel"],
    "Fruits secs & graines": ["nut", "almond", "cashew", "hazelnut", "graines", "noix"],
    "Huiles & matières grasses": ["oil", "olive oil", "fat", "margarine", "huile"],
    "Céréales & petit-déjeuner": ["cereal", "muesli", "granola", "breakfast"],
    "Produits végétaux": ["plant-based", "vegan", "vegetarian", "tofu", "soy"],
    "Desserts & glaces": ["ice cream", "dessert", "pudding", "sorbet", "glace"],
    "Épicerie": ["grocery", "spice", "sel", "salt", "sugar", "sucre", "flour", "farine"],
}

def clean_category(cat):
    if pd.isna(cat):
        return "Autres"
    cat_lower = str(cat).lower()
    for main_cat, keywords in category_keywords.items():
        for word in keywords:
            if word in cat_lower:
                return main_cat
    return "Autres"

df1["category_clean"] = df1["main_category_en"].apply(clean_category)
df1["nutriscore_grade"] = df1["nutriscore_grade"].astype(str).str.strip().str.lower()
df1 = df1[df1["nutriscore_grade"].isin(["a", "b", "c", "d", "e"])]

all_categories_p1 = sorted(df1["category_clean"].unique())



PATH_P2 = "Fichier4_.csv"
NUTRIENT_COLS = ['sugars_100g', 'fat_100g', 'saturated-fat_100g', 'salt_100g', 'fiber_100g', 'proteins_100g']

if os.path.exists(PATH_P2):
    df2 = pd.read_csv(PATH_P2)


    if 'nutriscore_grade' not in df2.columns:
        df2['nutriscore_grade'] = np.nan
    if 'main_category_en' not in df2.columns:
        df2['main_category_en'] = np.nan

    df2 = df2.dropna(subset=['nutriscore_grade', 'main_category_en'])

    df2['nutriscore_grade'] = df2['nutriscore_grade'].astype(str).str.upper().str.strip()
    df2 = df2[df2['nutriscore_grade'].isin(['A', 'B', 'C', 'D', 'E'])]


    for col in NUTRIENT_COLS:
        if col not in df2.columns:
            df2[col] = np.nan
        else:
            df2[col] = pd.to_numeric(df2[col], errors='coerce')
            df2.loc[(df2[col] < 0) | (df2[col] > 100), col] = np.nan

    category_keywords_p2 = {
        "Animaux": ["cat", "dog", "pet", "animal"],
        "Boissons": ["drink", "juice", "soda", "water", "tea", "coffee", "beverage", "kombucha", "sirop", "nectar", "gazpacho", "thé", "café"],
        "Viande & poisson": ["ham", "chicken", "turkey", "pork", "beef", "sausage", "salmon", "fish", "meat", "poisson", "thon", "tuna"],
        "Produits laitiers": ["milk", "cheese", "yogurt", "cream", "butter", "fromage", "yaourt", "lait", "egg", "oeuf"],
        "Plats préparés": ["pizza", "meal", "ready", "lasagna", "frozen", "soupe", "soup", "sandwich", "burger"],
        "Snacks sucrés": ["biscuit", "cookie", "cake", "chocolate", "candy", "bonbon", "snack", "gaufre"],
        "Boulangerie": ["bread", "bakery", "toast", "pain", "brioche", "crepe"],
        "Fruits & légumes": ["fruit", "vegetable", "salad", "legume", "tomato", "carrot"],
        "Féculents": ["pasta", "rice", "noodle", "pâte", "riz", "lentil"],
        "Sauces & tartinables": ["sauce", "ketchup", "mayo", "mustard", "spread", "hummus"],
        "Confitures & miel": ["jam", "jelly", "honey", "confiture", "miel"],
        "Fruits secs & graines": ["nut", "almond", "cashew", "hazelnut", "graines", "noix"],
        "Huiles & matières grasses": ["oil", "olive oil", "fat", "margarine", "huile"],
        "Céréales & petit-déjeuner": ["cereal", "muesli", "granola", "breakfast"],
        "Produits végétaux": ["plant-based", "vegan", "vegetarian", "tofu", "soy"],
        "Desserts & glaces": ["ice cream", "dessert", "pudding", "sorbet", "glace"],
        "Épicerie": ["grocery", "spice", "sel", "salt", "sugar", "sucre", "flour", "farine"]
    }

    def clean_category_p2(cat):
        cat_lower = str(cat).lower()
        for main_cat, keywords in category_keywords_p2.items():
            for word in keywords:
                if word in cat_lower:
                    return main_cat
        return "Autres"

    df2["category_clean"] = df2["main_category_en"].apply(clean_category_p2)
    all_categories_p2 = sorted(df2["category_clean"].unique())
else:
    df2 = pd.DataFrame(columns=['nutriscore_grade', 'main_category_en', 'category_clean'] + NUTRIENT_COLS)
    all_categories_p2 = []



DATA_PATH = "Fichier4.csv"
MODEL_PATH = "nutriscore_lr.joblib"
TARGET = "nutriscore_grade"
FEATURES = ["energy_100g", "fat_100g", "saturated-fat_100g", "carbohydrates_100g",
            "sugars_100g", "fiber_100g", "proteins_100g", "salt_100g"]
VALID_GRADES = ["a", "b", "c", "d", "e"]

def get_or_train():
    if os.path.exists(MODEL_PATH):
        bundle = joblib.load(MODEL_PATH)
        return bundle["model"], bundle.get("acc")

    if not os.path.exists(DATA_PATH):
        return None, None

    df = pd.read_csv(DATA_PATH)
    if TARGET not in df.columns:
        return None, None

    df[TARGET] = df[TARGET].astype(str).str.strip().str.lower()
    df = df[df[TARGET].isin(VALID_GRADES)]


    for f in FEATURES:
        if f not in df.columns:
            df[f] = np.nan

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    pipe.fit(X_train, y_train)
    acc = accuracy_score(y_test, pipe.predict(X_test))
    joblib.dump({"model": pipe, "acc": acc}, MODEL_PATH)
    return pipe, acc

model4, acc4 = get_or_train()



if os.path.exists("Fichier_3.csv"):
    df3 = pd.read_csv("Fichier_3.csv")
    if "nutriscore_grade" not in df3.columns:
        df3["nutriscore_grade"] = "unknown"
    df3["type"] = df3["nutriscore_grade"].apply(lambda x: "Inconnus" if str(x).lower() == "unknown" else "Connus")
    counts3 = df3["type"].value_counts().reset_index()
    counts3.columns = ["type", "count"]
else:
    counts3 = pd.DataFrame({"type": ["Connus", "Inconnus"], "count": [70, 30]})

fig_pie = px.pie(
    counts3, names="type", values="count",
    color="type", color_discrete_map={"Connus": "#2ecc71", "Inconnus": "#e74c3c"},
    hole=0
)
fig_pie.update_layout(showlegend=True, margin=dict(t=0, b=0, l=0, r=0), height=300)



def input_row(label, id_, placeholder):
    return html.Div([
        html.Label(f"{label} :", style={'flex': '1', 'fontWeight': 'bold', 'fontSize': '14px'}),
        dcc.Input(
            id=id_,
            type="number",
            placeholder=placeholder,
            style={'flex': '1', 'padding': '8px', 'borderRadius': '5px', 'border': '1px solid #ccc'}
        ),
        html.Span("kJ" if "energy" in id_ else "g", style={'width': '30px', 'marginLeft': '10px', 'color': '#777'})
    ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '15px'})


def make_header(active_page):
    btn_styles = {
        1: {'borderRadius': '25px', 'border': '2px solid #0056b3', 'padding': '10px 25px', 'backgroundColor': 'white', 'color': '#0056b3', 'fontWeight': 'bold', 'cursor': 'pointer'},
        2: {'borderRadius': '25px', 'border': 'none', 'padding': '10px 25px', 'backgroundColor': 'white', 'color': '#0056b3', 'fontWeight': 'bold', 'cursor': 'pointer'},
        3: {'borderRadius': '25px', 'border': 'none', 'padding': '10px 25px', 'backgroundColor': 'white', 'color': '#0056b3', 'fontWeight': 'bold', 'cursor': 'pointer'},
    }
    btn_styles[active_page]['border'] = '2px solid #0056b3'
    btn_styles[active_page]['backgroundColor'] = '#e8f0fe'

    return html.Div(
        style={'backgroundColor': '#a1c4fd', 'padding': '25px 0', 'textAlign': 'center', 'boxShadow': '0 2px 10px rgba(0,0,0,0.1)'},
        children=[
            html.H1(
                "Analyse du Nutri-Score et des nutriments des produits alimentaires en France",
                style={'margin': '0 0 5px 0', 'fontWeight': '900', 'fontSize': '28px', 'color': '#000'}
            ),
            html.P(
                "Ce tableau de bord vise à analyser la qualité nutritionnelle des produits alimentaires commercialisés en France à partir des données Open Food Facts, et à évaluer la capacité de prédiction du Nutri-Score à partir des nutriments.",
                style={'margin': '0 0 18px 0', 'fontSize': '13px', 'color': '#333', 'maxWidth': '900px',
                       'marginLeft': 'auto', 'marginRight': 'auto', 'lineHeight': '1.5'}
            ),
            html.Div(style={'display': 'flex', 'justifyContent': 'center', 'gap': '30px'}, children=[
                html.Button("Analyse du Nutriscore", id='btn-page-1', n_clicks=0, style=btn_styles[1]),
                html.Button("Analyse des nutriments (sucre, graisses, sel ...)", id='btn-page-2', n_clicks=0, style=btn_styles[2]),
                html.Button("Modèle de Prédiction", id='btn-page-3', n_clicks=0, style=btn_styles[3]),
            ])
        ]
    )



def layout_page1():
    return html.Div(style={'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#f4f7f9', 'paddingBottom': '50px'}, children=[
        make_header(1),

        html.Div(style={'margin': '25px auto', 'width': '95%', 'maxWidth': '1200px', 'backgroundColor': 'white', 'padding': '15px',
                        'borderRadius': '8px', 'border': '1px solid #dee2e6'}, children=[
            html.Label("Filtre de sélection des catégories :", style={'fontWeight': 'bold', 'color': '#333'}),
            dcc.Dropdown(
                id='category-picker-1',
                options=[{'label': c, 'value': c} for c in all_categories_p1],
                value=all_categories_p1,
                multi=True,
                style={'marginTop': '10px'}
            ),
        ]),

        html.Div(style={'textAlign': 'center', 'margin': '40px 0'}, children=[
            html.Button("Nutri-Score par catégories (KPI + histogramme)",
                        style={'borderRadius': '30px', 'border': '2px solid #0056b3', 'padding': '12px 45px',
                               'backgroundColor': '#e8f0fe', 'color': '#0056b3', 'fontWeight': 'bold', 'fontSize': '16px'}),
        ]),

        html.Div(id='kpi-container', style={'display': 'flex', 'justifyContent': 'center', 'margin': '0 auto', 'width': '95%',
                                            'maxWidth': '1200px', 'gap': '20px'}),

        html.Div(style={'width': '95%', 'maxWidth': '1200px', 'margin': '30px auto', 'backgroundColor': 'white', 'padding': '30px',
                        'borderRadius': '12px', 'boxShadow': '0 4px 20px rgba(0,0,0,0.08)'}, children=[
            html.H3(id='graph-title', style={'textAlign': 'center', 'fontStyle': 'italic', 'color': '#444', 'marginBottom': '20px'}),
            dcc.Graph(id='nutriscore-hist-1', config={'displayModeBar': False})
        ])
    ])


def layout_page2():
    return html.Div(style={'fontFamily': 'Arial', 'backgroundColor': '#f4f7f9', 'minHeight': '100vh'}, children=[
        make_header(2),

        html.Div(style={'margin': '20px auto', 'width': '95%', 'maxWidth': '1200px', 'backgroundColor': 'white', 'padding': '15px',
                        'borderRadius': '8px'}, children=[
            html.Div(style={'marginBottom': '15px'}, children=[
                html.Label("Nutriment à analyser :", style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '10px'}),
                dcc.RadioItems(
                    id='nutrient-picker',
                    options=[
                        {'label': 'Sucre', 'value': 'sugars_100g'},
                        {'label': 'Matières grasses', 'value': 'fat_100g'},
                        {'label': 'Graisses saturées', 'value': 'saturated-fat_100g'},
                        {'label': 'Sel', 'value': 'salt_100g'},
                        {'label': 'Fibres', 'value': 'fiber_100g'},
                        {'label': 'Protéines', 'value': 'proteins_100g'},
                    ],
                    value='sugars_100g',
                    inline=True,
                    inputStyle={'marginRight': '6px', 'accentColor': '#0056b3', 'width': '18px', 'height': '18px', 'cursor': 'pointer'},
                    labelStyle={'marginRight': '25px', 'fontSize': '14px', 'fontWeight': '600', 'color': '#333',
                                'cursor': 'pointer', 'display': 'inline-flex', 'alignItems': 'center', 'gap': '6px'},
                ),
            ]),
            html.Div(children=[
                html.Label("Filtre de sélection des catégories :", style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '5px'}),
                dcc.Dropdown(
                    id='category-picker',
                    options=[{'label': c, 'value': c} for c in all_categories_p2],
                    value=all_categories_p2,
                    multi=True
                ),
            ]),
        ]),

        html.Div(style={
            'display': 'flex',
            'margin': '0 auto',
            'width': '95%',
            'maxWidth': '1200px',
            'gap': '30px',
            'height': '650px',
            'alignItems': 'flex-start'
        }, children=[

            html.Div(style={'flex': '1', 'height': '100%', 'display': 'flex', 'flexDirection': 'column'}, children=[
                html.Div(id='products-kpi', style={'backgroundColor': 'white', 'padding': '15px', 'borderRadius': '12px', 'textAlign': 'center',
                                                  'boxShadow': '0 4px 10px rgba(0,0,0,0.05)', 'marginBottom': '10px',
                                                  'borderTop': '6px solid #007bff'}),

                html.Div(id='global-kpi', style={'backgroundColor': 'white', 'padding': '25px', 'borderRadius': '12px', 'textAlign': 'center',
                                                'boxShadow': '0 4px 10px rgba(0,0,0,0.05)', 'marginBottom': '20px'}),

                html.H3("Détails des moyennes par catégorie sélectionnée", style={'textAlign': 'center', 'fontSize': '15px', 'marginBottom': '10px'}),

                html.Div(id='category-kpis', style={'overflowY': 'auto', 'flex': '1', 'paddingRight': '10px'})
            ]),

            html.Div(style={'flex': '1.8', 'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '12px',
                            'boxShadow': '0 4px 10px rgba(0,0,0,0.05)', 'height': '100%', 'display': 'flex', 'flexDirection': 'column'}, children=[
                html.H3(id='heatmap-title', style={'textAlign': 'center', 'margin': '0 0 10px 0'}),
                dcc.Graph(id='sugar-heatmap', config={'displayModeBar': False}, style={'flex': '1'})
            ])
        ])
    ])


def layout_page3():
    acc_display = f"{acc4 * 100:.0f}%" if acc4 is not None else "N/A"
    return html.Div(style={'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#f4f7f9', 'minHeight': '100vh'}, children=[
        make_header(3),

        html.Div(style={'backgroundColor': 'white', 'padding': '20px', 'marginTop': '20px'}, children=[

            html.Div([
                html.Button("Impact des données manquantes et prédiction du Nutri-Score",
                            style={'backgroundColor': '#e3f2fd', 'border': '1px solid #2196f3', 'borderRadius': '20px',
                                   'padding': '10px 30px', 'color': '#1976d2', 'fontWeight': 'bold', 'fontSize': '16px'})
            ], style={'textAlign': 'center', 'marginBottom': '40px'}),

            html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'maxWidth': '1200px', 'margin': '0 auto',
                            'alignItems': 'flex-start'}, children=[

                html.Div(style={'width': '45%'}, children=[
                    html.P("En raison du nombre important de données manquantes au sein de la colonne Nutri-Score de la base Open Food Facts, nous avons mis en place un modèle de régression logistique afin de prédire les scores d'indexation nutritionnelle manquants.",
                           style={'fontStyle': 'italic', 'fontSize': '16px', 'textAlign': 'center', 'lineHeight': '1.5', 'color': '#333'}),

                    html.H4("Pourcentage des produits avec Nutri-Score connu vs inconnu",
                            style={'textAlign': 'center', 'marginTop': '40px', 'color': '#455a64'}),

                    dcc.Graph(figure=fig_pie, config={'displayModeBar': False}, style={'height': '300px'}),

                    html.P("Une part importante de produits présente des informations incomplètes, ce qui limite la précision des analyses.",
                           style={'fontStyle': 'italic', 'fontSize': '13px', 'textAlign': 'center', 'color': '#666', 'marginTop': '10px'})
                ]),

                html.Div(style={'width': '45%', 'backgroundColor': '#f8f9fa', 'padding': '20px', 'borderRadius': '10px', 'border': '1px solid #ddd'}, children=[
                    html.Div([
                        html.H4("Prédiction du Nutri-Score (Régression Logistique)", style={'margin': '0', 'fontSize': '16px'}),
                    ], style={'borderBottom': '1px solid #ddd', 'paddingBottom': '10px', 'marginBottom': '20px'}),

                    html.P(f"Accuracy du modèle (test) : {acc_display}",
                           style={'textAlign': 'center', 'fontWeight': 'bold', 'marginBottom': '20px'}),

                    input_row("Énergie", "in_energy", "Ex : 1800"),
                    input_row("Graisses", "in_fat", "Ex : 8.0"),
                    input_row("Graisses saturées", "in_satfat", "Ex : 3.2"),
                    input_row("Glucides", "in_carb", "Ex : 55"),
                    input_row("Sucres", "in_sugars", "Ex : 12.5"),
                    input_row("Fibres", "in_fiber", "Ex : 4.5"),
                    input_row("Protéines", "in_proteins", "Ex : 6"),
                    input_row("Sel", "in_salt", "Ex : 0.8"),

                    html.Div([
                        html.Button("Prédire", id="btn_predict", n_clicks=0,
                                    style={'backgroundColor': '#4caf50', 'color': 'white', 'border': 'none', 'padding': '12px 60px',
                                           'borderRadius': '5px', 'fontWeight': 'bold', 'cursor': 'pointer', 'fontSize': '16px'})
                    ], style={'textAlign': 'center', 'marginTop': '20px'}),

                    html.Div(id="pred_bar", children="Prédiction: Nutri-Score: —",
                             style={'marginTop': '25px', 'backgroundColor': '#e8f5e9', 'padding': '15px', 'borderRadius': '5px',
                                    'border': '1px solid #c8e6c9', 'color': '#2e7d32', 'textAlign': 'center',
                                    'fontWeight': 'bold', 'fontSize': '18px'})
                ])
            ])
        ])
    ])



app.layout = html.Div([
    dcc.Store(id='current-page', data=1),
    html.Div(id='page-content')
])



@app.callback(
    Output('current-page', 'data'),
    [Input('btn-page-1', 'n_clicks'),
     Input('btn-page-2', 'n_clicks'),
     Input('btn-page-3', 'n_clicks')],
    prevent_initial_call=True
)
def navigate(n1, n2, n3):
    from dash import ctx
    triggered = ctx.triggered_id
    if triggered == 'btn-page-1':
        return 1
    elif triggered == 'btn-page-2':
        return 2
    elif triggered == 'btn-page-3':
        return 3
    return 1


@app.callback(
    Output('page-content', 'children'),
    Input('current-page', 'data')
)
def render_page(page):
    if page == 1:
        return layout_page1()
    elif page == 2:
        return layout_page2()
    elif page == 3:
        return layout_page3()
    return layout_page1()



@app.callback(
    [Output('kpi-container', 'children'),
     Output('nutriscore-hist-1', 'figure'),
     Output('graph-title', 'children')],
    [Input('category-picker-1', 'value')]
)
def update_dashboard_p1(selected_categories):
    if not selected_categories:
        dff = df1
        title_suffix = "Toutes les catégories"
    else:
        dff = df1[df1["category_clean"].isin(selected_categories)]
        title_suffix = ", ".join(selected_categories[:3]) + ("..." if len(selected_categories) > 3 else "")

    total = len(dff)
    counts = dff["nutriscore_grade"].value_counts()

    ab_total = counts.get('a', 0) + counts.get('b', 0)
    de_total = counts.get('d', 0) + counts.get('e', 0)
    per_ab = (ab_total / total * 100) if total > 0 else 0
    per_de = (de_total / total * 100) if total > 0 else 0

    kpis = [
        html.Div(style={'borderTop': '6px solid #007bff', 'backgroundColor': 'white', 'padding': '25px', 'borderRadius': '8px',
                        'boxShadow': '0 4px 12px rgba(0,0,0,0.1)', 'textAlign': 'center', 'flex': '1'}, children=[
            html.P("Produits analysés", style={'color': '#666', 'fontWeight': 'bold'}),
            html.H2(f"{total/1000:.1f}k" if total >= 1000 else f"{total}", style={'color': '#0056b3', 'fontSize': '32px'})
        ]),
        html.Div(style={'borderTop': '6px solid #28a745', 'backgroundColor': 'white', 'padding': '25px', 'borderRadius': '8px',
                        'boxShadow': '0 4px 12px rgba(0,0,0,0.1)', 'textAlign': 'center', 'flex': '1'}, children=[
            html.P("Score A & B (Sain)", style={'color': '#666', 'fontWeight': 'bold'}),
            html.H2([f"{per_ab:.2f}", html.Span("%", style={'fontSize': '18px'})], style={'color': '#28a745', 'fontSize': '32px'})
        ]),
        html.Div(style={'borderTop': '6px solid #dc3545', 'backgroundColor': 'white', 'padding': '25px', 'borderRadius': '8px',
                        'boxShadow': '0 4px 12px rgba(0,0,0,0.1)', 'textAlign': 'center', 'flex': '1'}, children=[
            html.P("Score D & E (Gras/Sucré)", style={'color': '#666', 'fontWeight': 'bold'}),
            html.H2([f"{per_de:.1f}", html.Span("%", style={'fontSize': '18px'})], style={'color': '#dc3545', 'fontSize': '32px'})
        ]),
    ]

    fig = go.Figure(data=[go.Bar(
        x=['A', 'B', 'C', 'D', 'E'],
        y=[counts.get(g, 0) for g in ['a', 'b', 'c', 'd', 'e']],
        marker_color=['#008b4c', '#80bc29', '#fca300', '#eb6101', '#d7191c'],
        width=0.7
    )])
    fig.update_layout(plot_bgcolor='#f8f9fa', margin=dict(t=10, b=20, l=50, r=20), height=400)

    return kpis, fig, f"Répartition des produits selon leur Nutri-Score : {title_suffix}"



NUTRIENT_LABELS = {
    'sugars_100g': 'Sucre',
    'fat_100g': 'Matières grasses',
    'saturated-fat_100g': 'Graisses saturées',
    'salt_100g': 'Sel',
    'fiber_100g': 'Fibres',
    'proteins_100g': 'Protéines',
}

@app.callback(
    [Output('products-kpi', 'children'),
     Output('global-kpi', 'children'),
     Output('category-kpis', 'children'),
     Output('sugar-heatmap', 'figure'),
     Output('heatmap-title', 'children')],
    [Input('category-picker', 'value'),
     Input('nutrient-picker', 'value')]
)
def update_dashboard_p2(selected_categories, selected_nutrient):
    if not selected_categories:
        selected_categories = all_categories_p2
    if not selected_nutrient:
        selected_nutrient = 'sugars_100g'

    nutrient_label = NUTRIENT_LABELS.get(selected_nutrient, selected_nutrient)


    if df2 is None or df2.empty:
        fig_vide = go.Figure()
        fig_vide.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            annotations=[dict(text="Aucune donnée (Fichier4_.csv manquant ou vide)",
                              xref="paper", yref="paper", x=0.5, y=0.5,
                              showarrow=False, font=dict(size=16, color="#aaa"))]
        )
        return (
            [html.P("Aucun produit", style={'color': '#e74c3c', 'fontWeight': 'bold'})],
            [html.P("Aucune donnée disponible", style={'color': '#e74c3c'})],
            [],
            fig_vide,
            f"Moyennes de {nutrient_label.lower()} (g/100g) par catégorie et Nutri-Score"
        )


    if selected_nutrient not in df2.columns:
        fig_vide = go.Figure()
        fig_vide.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            annotations=[dict(text=f"Colonne « {selected_nutrient} » non disponible",
                              xref="paper", yref="paper", x=0.5, y=0.5,
                              showarrow=False, font=dict(size=16, color="#aaa"))]
        )
        return (
            [html.P("Aucun produit", style={'color': '#e74c3c', 'fontWeight': 'bold'})],
            [html.P(f"Nutriment non disponible : {nutrient_label}", style={'color': '#e74c3c'})],
            [],
            fig_vide,
            f"Moyennes de {nutrient_label.lower()} (g/100g) par catégorie et Nutri-Score"
        )

    dff = df2[df2["category_clean"].isin(selected_categories)].copy()
    dff = dff.dropna(subset=[selected_nutrient])

    if dff.empty or dff[selected_nutrient].isna().all():
        fig_vide = go.Figure()
        fig_vide.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            annotations=[dict(text=f"Données non disponibles pour « {nutrient_label} »",
                              xref="paper", yref="paper", x=0.5, y=0.5,
                              showarrow=False, font=dict(size=16, color="#aaa"))]
        )
        return (
            [html.P("Aucun produit", style={'color': '#e74c3c', 'fontWeight': 'bold'})],
            [html.P(f"Aucune donnée disponible pour : {nutrient_label}", style={'color': '#e74c3c', 'fontWeight': 'bold'})],
            [],
            fig_vide,
            f"Moyennes de {nutrient_label.lower()} (g/100g) par catégorie et Nutri-Score"
        )

    avg_total = float(dff[selected_nutrient].mean())
    nb_produits = int(len(dff))

    products_card = [
        html.P("Produits analysés (nutriments)", style={'color': '#666', 'fontWeight': 'bold', 'margin': '0 0 5px 0', 'fontSize': '13px'}),
        html.H2(f"{nb_produits/1000:.1f}k" if nb_produits >= 1000 else f"{nb_produits}",
                style={'color': '#0056b3', 'fontSize': '28px', 'margin': '0'}),
        html.P("Nombre de produits utilisés pour l'analyse nutritionnelle.",
               style={'color': '#7f8c8d', 'fontSize': '11px', 'margin': '5px 0 0 0'})
    ]

    global_card = [
        html.H2(f"{avg_total:.2f} g", style={'color': '#0056b3', 'fontSize': '36px', 'margin': '0'}),
        html.P(f"Taux de {nutrient_label.lower()} moyen cumulé / 100g", style={'color': '#7f8c8d', 'fontSize': '13px'})
    ]


    cat_stats = dff.groupby('category_clean')[selected_nutrient].mean().sort_values(ascending=False)
    cat_cards = [
        html.Div([
            html.P(f"Moyenne de {nutrient_label.lower()} de : {cat}",
                   style={'margin': '0', 'fontSize': '12px', 'fontWeight': 'bold', 'color': '#555'}),
            html.H3(f"{val:.2f} g/100g", style={'margin': '5px 0 0 0', 'color': '#0056b3', 'fontSize': '18px'})
        ], style={'backgroundColor': 'white', 'padding': '15px', 'borderRadius': '8px',
                  'marginBottom': '10px', 'borderLeft': "6px solid #ddd", 'boxShadow': '0 2px 5px rgba(0,0,0,0.05)'})
        for cat, val in cat_stats.items()
    ]


    pdf = dff.groupby(['category_clean', 'nutriscore_grade'])[selected_nutrient].mean().reset_index()
    pivot_table = pdf.pivot(index='category_clean', columns='nutriscore_grade', values=selected_nutrient)

    available_grades = [g for g in ["A", "B", "C", "D", "E"] if g in pivot_table.columns]
    pivot_table = pivot_table[available_grades]


    nutriscore_colorscale = [
        [0.0,  '#008b4c'],
        [0.25, '#80bc29'],
        [0.5,  '#fca300'],
        [0.75, '#eb6101'],
        [1.0,  '#d7191c'],
    ]

    fig = px.imshow(
        pivot_table,
        labels=dict(x="Nutri-Score", y="", color=f"{nutrient_label} (g)"),
        x=pivot_table.columns,
        y=pivot_table.index,
        color_continuous_scale=nutriscore_colorscale,
        text_auto=".1f",
        aspect="auto"
    )
    fig.update_layout(
        autosize=True,
        margin=dict(t=30, b=30, l=150, r=10),
        xaxis=dict(side="top"),
        yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    fig.update_traces(xgap=5, ygap=5)

    graph_title = f"Moyennes de {nutrient_label.lower()} (g/100g) par catégorie et Nutri-Score"

    return products_card, global_card, cat_cards, fig, graph_title



@app.callback(
    Output("pred_bar", "children"),
    Input("btn_predict", "n_clicks"),
    [State("in_energy", "value"), State("in_fat", "value"), State("in_satfat", "value"),
     State("in_carb", "value"), State("in_sugars", "value"), State("in_fiber", "value"),
     State("in_proteins", "value"), State("in_salt", "value")]
)
def do_predict(n, energy, fat, satfat, carb, sugars, fiber, proteins, salt):
    if n == 0:
        return "Prédiction: Nutri-Score: —"

    if None in [energy, fat, satfat, carb, sugars, fiber, proteins, salt]:
        return "Veuillez remplir tous les champs."

    if model4 is None:
        return "Modèle non disponible (fichier de données manquant)."

    row = pd.DataFrame([[energy, fat, satfat, carb, sugars, fiber, proteins, salt]], columns=FEATURES)
    pred = model4.predict(row)[0].upper()
    return f"Prédiction: Nutri-Score: {pred}"



if __name__ == "__main__":

    app.run(host="0.0.0.0", port=8050, debug=False)