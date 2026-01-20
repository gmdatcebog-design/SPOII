import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
plt.close('all')  # Zatvara sve prozore koji su ostali otvoreni
import seaborn as sns
import warnings
from scipy.stats import randint as sp_randint
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV
from keras.layers import Dense
from keras.models import Sequential
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor,AdaBoostRegressor,BaggingRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')


pun_naziv_fajla = 'energy_efficiency_data.csv'

try:
    df = pd.read_csv(pun_naziv_fajla)
    print("--- Analiza Energetske Efikasnosti ---")

    # --- TVOJE DVIJE LINIJE KODA ---
    print("\n[INFO] Struktura podataka (df.info()):")
    df.info()  # Ovo ispisuje tipove podataka i ima li praznih polja

    print("\n[HEAD] Prvih 5 redova tabele (df.head()):")
    print(df.head()) # Ovo ispisuje prvih 5 redova da vidis konkretne brojke
    # -------------------------------

    # --- MASOVNA VIZUALIZACIJA (Histogrami za sve kolone) ---
    # Kreiramo listu svih kolona
    num_list = list(df.columns)

    # Definisemo velicinu slike. 
    # (10, 20) je sasvim dovoljno visoko za 10 grafikona
    fig = plt.figure(figsize=(10, 20))

    for i in range(len(num_list)):
        # Posto imas 10 kolona, koristimo mrezu od 5 redova i 2 kolone (5*2=10)
        plt.subplot(5, 2, i+1)
        plt.title(num_list[i])
        plt.hist(df[num_list[i]], color='blue', alpha=0.5)

    # Automatsko podesavanje razmaka da se naslovi ne preklapaju
    plt.tight_layout()
    print("\n[INFO] Prikazujem histograme za sve parametre...")
    plt.show()
   

    # Vizualizacija (Histogram)
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Heating_Load'], kde=True, color='red')
    plt.title('Raspodjela potrosnje energije za grijanje')
    plt.show()

   

    plt.figure(figsize = (10,8))
    sns.heatmap(df.corr(), annot=True, cbar=False, cmap='Blues', fmt='.1f')
    plt.title('Korelacija parametara (Plavi prikaz)')
    plt.show()

    # --- PRIPREMA PODATAKA ZA MODEL (X i Y) ---
    
    # X su "pitanja" - izbacujemo rezultate da ostanu samo karakteristike zgrade
    X = df.drop(['Heating_Load', 'Cooling_Load'], axis=1)
    
    # Y su "odgovori" - ono sto AI treba da pogodi
    Y = df[['Heating_Load', 'Cooling_Load']] # Oba odjednom
    Y1 = df[['Heating_Load']]               # Samo grijanje
    Y2 = df[['Cooling_Load']]               # Samo hladjenje
    
    print("\n[INFO] Podaci su uspjesno razdvojeni na X (ulaze) i Y (ciljeve).")
    print(f"Broj ulaznih faktora (X): {X.shape[1]}")
    # ------------------------------------------
    # 1. Dijeljenje podataka na trening (za ucenje) i test (za provjeru)
    # test_size=0.33 znaci da 33% podataka ostavljamo za testiranje
    X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(X, Y1, Y2, test_size=0.33, random_state = 20)

    # 2. Skaliranje podataka (Min-Max normalizacija)
    # Svi brojevi ce sada biti u rasponu od 0 do 1
    MinMax = MinMaxScaler(feature_range= (0,1))
    X_train = MinMax.fit_transform(X_train)
    X_test = MinMax.transform(X_test)

    # Kreiramo praznu tabelu u koju cemo upisivati rezultate preciznosti svakog modela
    Acc = pd.DataFrame(index=None, columns=['model','train_Heating','test_Heating','train_Cooling','test_Cooling'])
    
    # Definisanje liste razlicitih modela (regresora) koje cemo testirati
    regressors = [['SVR',SVR()],
                  ['DecisionTreeRegressor',DecisionTreeRegressor()],
                  ['KNeighborsRegressor', KNeighborsRegressor()],
                  ['RandomForestRegressor', RandomForestRegressor()],
                  ['MLPRegressor',MLPRegressor()],
                  ['AdaBoostRegressor',AdaBoostRegressor()],
                  ['GradientBoostingRegressor',GradientBoostingRegressor()]]

    # Petlja koja trenira svaki model i biljezi rezultate
    for mod in regressors:
        name = mod[0]
        model = mod[1]
        
        # Treniranje i testiranje za GRIJANJE (Y1)
        model.fit(X_train, y1_train)
        actr1 = r2_score(y1_train, model.predict(X_train))
        acte1 = r2_score(y1_test, model.predict(X_test))
        
        # Treniranje i testiranje za HLADJENJE (Y2)
        model.fit(X_train, y2_train)
        actr2 = r2_score(y2_train, model.predict(X_train))
        acte2 = r2_score(y2_test, model.predict(X_test))
        
        # Upisivanje rezultata u tabelu Acc
        # Koristimo _append jer je obicni append izbacen iz novih verzija pandasa
        Acc = Acc._append(pd.Series({'model':name, 'train_Heating':actr1,'test_Heating':acte1,'train_Cooling':actr2,'test_Cooling':acte2}), ignore_index=True)

    # Sortiranje tabele po preciznosti hladjenja (od najgoreg do najboljeg)
    Acc = Acc.sort_values(by='test_Cooling')
    
    print("\n--- REZULTATI SVIH MODELA (R2 Score) ---")
    print(Acc)
    # 1. Kreiramo osnovni model stabla odluke
    DTR = DecisionTreeRegressor()

    # 2. Definisemo "mrezu" parametara koje zelimo da Python isproba
    # Zamijenio sam mse/mae novim nazivima da ti ne javi error
    param_grid = {
        "criterion": ["squared_error", "absolute_error"],
        "min_samples_split": [14, 15, 16, 17],
        "max_depth": [5, 6, 7],
        "min_samples_leaf": [4, 5, 6],
        "max_leaf_nodes": [29, 30, 31, 32]
    }

    # 3. GridSearchCV ce isprobati sve kombinacije (cv=5 znaci unakrsna provjera 5 puta)
    grid_cv_DTR = GridSearchCV(DTR, param_grid, cv=5)

    print("\n[INFO] Pokrecem optimizaciju DecisionTree modela (ovo moze potrajati)...")
    
    # Treniramo na podacima za hladjenje (Y2)
    grid_cv_DTR.fit(X_train, y2_train)

    # 4. Ispisujemo rezultate najbolje kombinacije
    print("\n--- NAJBOLJI REZULTAT ZA OPTIMIZOVANO STABLO ---")
    print("Najbolji R2 Score: {:.4f}".format(grid_cv_DTR.best_score_))
    print("Najbolji parametri:\n{}".format(grid_cv_DTR.best_params_))

    # 1. Kreiramo finalni model sa tacno onim parametrima koje smo pronasli optimizacijom
    # Napomena: Ako javi error za 'mse', stavi 'squared_error'
    DTR = DecisionTreeRegressor(criterion='squared_error', max_depth=6, max_leaf_nodes=32, min_samples_leaf=5, min_samples_split=17)

    # 2. Testiranje za Heating Load (Y1)
    DTR.fit(X_train, y1_train)
    score_y1 = DTR.score(X_test, y1_test)
    print("\n--- FINALNA PRECIZNOST MODELA ---")
    print("R-Squared za Grijanje (Y1): {:.4f}".format(score_y1))

    # 3. Testiranje za Cooling Load (Y2)
    DTR.fit(X_train, y2_train)   
    score_y2 = DTR.score(X_test, y2_test)
    print("R-Squared za Hladjenje (Y2): {:.4f}".format(score_y2))

    # --- OPTIMIZACIJA RANDOM FOREST MODELA ---
    # n_estimators: broj stabala u sumi (sto vise, to je model stabilniji)
    # max_features: broj parametara koje svako stablo razmatra
    # n_jobs=-1: koristi sve jezgre tvog procesora za brzi rad
    
    param_grid_RFR = [{'n_estimators': [150, 200], 'max_features': [1, 2], 'max_depth': [45, 50]}]

    RFR = RandomForestRegressor(n_jobs=-1)

    plt.close('all') # Ovo zatvara sve otvorene grafikone u pozadini

    # Koristimo cv=10 (unakrsna provjera u 10 krugova) za maksimalnu pouzdanost
    grid_search_RFR = GridSearchCV(RFR, param_grid_RFR, cv=10, scoring='neg_mean_squared_error')
    
    print("\n[INFO] Pokrecem optimizaciju Random Forest modela (ovo moze potrajati par minuta)...")
    grid_search_RFR.fit(X_train, y2_train.values.ravel()) # .values.ravel() sprjecava upozorenja

    print("\n--- NAJBOLJI REZULTAT ZA RANDOM FOREST ---")
    # Napomena: scoring koristi 'neg_mean_squared_error', pa je best_score_ ovdje greska (MSE)
    # Za pravi R2 score na testu koristit cemo grid_search_RFR.best_estimator_.score
    print("Najbolji parametri:\n{}".format(grid_search_RFR.best_params_))

    # 1. Kreiramo finalni Random Forest model sa najboljim parametrima
    RFR = RandomForestRegressor(n_estimators=400, max_features=1, max_depth=90, bootstrap=True, n_jobs=-1)

    # 2. Treniranje i testiranje za Grijanje (Y1)
    # Koristimo .values.ravel() da podatke pretvorimo u oblik koji model lakse cita
    RFR.fit(X_train, y1_train.values.ravel())
    score_rfr_y1 = RFR.score(X_test, y1_test)
    print("\n--- FINALNI REZULTATI ZA RANDOM FOREST ---")
    print("R-Squared za Grijanje (Y1) na test setu: {:.4f}".format(score_rfr_y1))

    # 3. Treniranje i testiranje za Hladjenje (Y2)
    RFR.fit(X_train, y2_train.values.ravel())   
    score_rfr_y2 = RFR.score(X_test, y2_test)
    print("R-Squared za Hladenje (Y2) na test setu: {:.4f}".format(score_rfr_y2))

    # --- OPTIMIZACIJA GRADIENT BOOSTING MODELA ---
    # learning_rate: kontrolise koliko svako novo stablo utice na finalni rezultat
    # subsample: procenat podataka koji se koristi za svako stablo (mora biti <= 1.0)
    
    param_grid_GBR = [{
        "learning_rate": [0.05, 0.1], 
        "n_estimators": [200], 
        "max_depth": [4, 5], 
        "min_samples_split": [2], # Smanjio sam na realne vrijednosti (ne moze biti 1)
        "min_samples_leaf": [2], 
        "subsample": [0.9]      # Subsample ne moze biti veci od 1.0
    }]

    GBR = GradientBoostingRegressor()
    grid_search_GBR = GridSearchCV(GBR, param_grid_GBR, cv=10, scoring='neg_mean_squared_error')
    
    print("\n[INFO] Pokrecem optimizaciju Gradient Boosting modela...")
    grid_search_GBR.fit(X_train, y2_train.values.ravel())

    print("\n--- NAJBOLJI REZULTAT ZA GRADIENT BOOSTING ---")
    print("Najbolji parametri:\n{}".format(grid_search_GBR.best_params_))

    # 1. Kreiramo finalni Gradient Boosting model sa najboljim parametrima
    GBR = GradientBoostingRegressor(learning_rate=0.1, n_estimators=250, max_depth=5, 
                                    min_samples_split=3, min_samples_leaf=2, subsample=1.0)

    # 2. Treniranje i testiranje za Grijanje (Y1)
    GBR.fit(X_train, y1_train.values.ravel())
    score_gbr_y1 = GBR.score(X_test, y1_test)
    print("\n--- FINALNI REZULTATI ZA GRADIENT BOOSTING ---")
    print("R-Squared za Grijanje (Y1) na test setu: {:.4f}".format(score_gbr_y1))

    # 3. Treniranje i testiranje za Hladjenje (Y2)
    GBR.fit(X_train, y2_train.values.ravel())   
    score_gbr_y2 = GBR.score(X_test, y2_test)
    print("R-Squared za Hladjenje (Y2) na test setu: {:.4f}".format(score_gbr_y2))

    # --- OPTIMIZACIJA CATBOOST MODELA ---
    model_CBR = CatBoostRegressor(verbose=0) # verbose=0 iskljucuje ispisivanje svake iteracije
    
    parameters = {
        'depth': [8, 10],
        'iterations': [1000], # Smanjeno sa 10000 radi brzine, mozes vratiti ako zelis
        'learning_rate': [0.02, 0.03],
        'border_count': [5],
        'random_state': [42, 45]
    }

    # cv=2 znaci da ce se podaci podijeliti na dva dijela za provjeru
    grid_CBR = GridSearchCV(estimator=model_CBR, param_grid=parameters, cv=2, n_jobs=-1)
    
    print("\n[INFO] Pokrecem CatBoost optimizaciju (ovo moze potrajati)...")
    grid_CBR.fit(X_train, y2_train)

    print("\n--- REZULTATI ZA CATBOOST ---")
    print("Najbolji estimator:\n", grid_CBR.best_estimator_)
    print("Najbolji score:\n", grid_CBR.best_score_)
    print("Najbolji parametri:\n", grid_CBR.best_params_)

    # 1. Kreiramo finalni CatBoost model sa tvojim optimizovanim parametrima
    # Dodao sam verbose=0 da ti ne ispisuje 10.000 linija teksta dok radi
    model = CatBoostRegressor(border_count=5, depth=10, iterations=10000, 
                              learning_rate=0.02, random_state=42, verbose=0)

    # 2. Treniranje i predvidjanje za GRIJANJE (Y1)
    print("\n[INFO] CatBoost trenira za Grijanje...")
    model.fit(X_train, y1_train)
    actr1 = r2_score(y1_train, model.predict(X_train))
    acte1 = r2_score(y1_test, model.predict(X_test))
    y1_pred = model.predict(X_test) # Ovo su predvidjene vrijednosti za test set

    # 3. Treniranje i predvidjanje za HLADJENJE (Y2)
    print("[INFO] CatBoost trenira za Hladjenje...")
    model.fit(X_train, y2_train)
    actr2 = r2_score(y2_train, model.predict(X_train))
    acte2 = r2_score(y2_test, model.predict(X_test))
    y2_pred = model.predict(X_test) # Ovo su predvidjene vrijednosti za test set

    # Ispis finalnih rezultata
    print("\n--- FINALNI CATBOOST REZULTATI ---")
    print(f"Grijanje (Y1) - Trening R2: {actr1:.4f}, Test R2: {acte1:.4f}")
    print(f"Hladjenje (Y2) - Trening R2: {actr2:.4f}, Test R2: {acte2:.4f}")

    # Finalni ispis preciznosti u konzolu
    print("CatBoostRegressor: R-Squared on train dataset={}".format(actr1))
    print("CatBoostRegressor: R-Squared on Y1test dataset={}".format(acte1))
    print("CatBoostRegressor: R-Squared on train dataset={}".format(actr2))
    print("CatBoostRegressor: R-Squared on Y2test dataset={}".format(acte2))



    # --- 1. DEFINISANJE MODELA (MLPR) ---
    # Prvo pravimo varijablu MLPR da bi je Python prepoznao kasnije
    MLPR = MLPRegressor(hidden_layer_sizes=[64, 32], 
                        activation='relu', 
                        solver='adam', 
                        max_iter=1000, 
                        early_stopping=True, 
                        random_state=0)

    # --- 2. DODAVANJE U LISTU SVIH REGRESORA ---
    # Sada je MLPR definisan i lista ga moze povuci bez Error-a
    regressors1 = [
        ['DecisionTreeRegressor', DecisionTreeRegressor(criterion='squared_error', max_depth=6, max_leaf_nodes=30, min_samples_leaf=5, min_samples_split=17)],
        ['RandomForestRegressor', RandomForestRegressor(n_estimators=100, max_depth=15, max_features='sqrt', n_jobs=-1)],
        ['MLPRegressor', MLPR], # Ubacujemo ga u listu za petlju
        ['GradientBoostingRegressor', GradientBoostingRegressor(learning_rate=0.1, n_estimators=250, max_depth=5)]
    ]

    # --- 3. RUÈNO TRENIRANJE I ISPIS ZA MLP (AKO ELI POSEBAN ISPIS) ---
    print("\n--- TRENERANJE NEURONSKE MREZE (MLP) ---")

    # Treniranje za Grijanje (Y1)
    MLPR.fit(X_train, y1_train.values.ravel())
    score_y1 = MLPR.score(X_test, y1_test)
    print(f"MLP R-Squared (Grijanje Y1): {score_y1:.4f}")

    # Treniranje za Hladjenje (Y2)
    MLPR.fit(X_train, y2_train.values.ravel())   
    score_y2 = MLPR.score(X_test, y2_test)
    print(f"MLP R-Squared (Hladjenje Y2): {score_y2:.4f}")

    # 2. Kreiranje prazne tabele za poreðenje
    Acc1 = pd.DataFrame(columns=['model','train_Heating','test_Heating','train_Cooling','test_Cooling'])

    # 3. Petlja koja trenira sve modele iz liste i puni tabelu
    print("\n[INFO] Pokretanje masovnog treniranja modela...")

    for mod in regressors1:
        name = mod[0]
        model = mod[1]
    
        # Treniranje i predviðanje za Grijanje (y1)
        model.fit(X_train, y1_train.values.ravel())
        actr1_m = r2_score(y1_train, model.predict(X_train))
        acte1_m = r2_score(y1_test, model.predict(X_test))
    
        # Treniranje i predviðanje za Hlaðenje (y2)
        model.fit(X_train, y2_train.values.ravel())
        actr2_m = r2_score(y2_train, model.predict(X_train))
        acte2_m = r2_score(y2_test, model.predict(X_test))
    
        # Dodavanje u tabelu koristeæi pd.concat (ispravna zamjena za .append)
        novi_red = pd.DataFrame([{
            'model': name, 
            'train_Heating': actr1_m, 
            'test_Heating': acte1_m, 
            'train_Cooling': actr2_m, 
            'test_Cooling': acte2_m
        }])
        Acc1 = pd.concat([Acc1, novi_red], ignore_index=True)

    # 4. Dodavanje CatBoost rezultata (ruèno, jer su veæ izraèunati ranije)
    # PANJA: Provjeri da li su ti varijable za CatBoost (actr1, acte1...) definisane iznad
    cat_red = pd.DataFrame([{
        'model': 'CatBoostRegressor', 
        'train_Heating': actr1, 
        'test_Heating': acte1, 
        'train_Cooling': actr2, 
        'test_Cooling': acte2
    }])
    Acc1 = pd.concat([Acc1, cat_red], ignore_index=True)

    # 5. Finalno sortiranje i ispis
    Acc1 = Acc1.sort_values(by='test_Cooling', ascending=False)
    print("\n--- UPOREDNA TABELA SVIH MODELA ---")
    print(Acc1.to_string(index=False))
    
    print("\n--- REZULTATI SVIH MODELA ---")
    print(Acc1.to_string(index=False))

    print("\n" + "="*40)
    print(" FINALNI REZULTATI: CATBOOST ")
    print("="*40)
    print("Grijanje (Y1) - Trening: {:.4f}".format(actr1))
    print("Grijanje (Y1) - Test:    {:.4f}".format(acte1))
    print("-" * 40)
    print("Hladjenje (Y2) - Trening: {:.4f}".format(actr2))
    print("Hladjenje (Y2) - Test:    {:.4f}".format(acte2))
    print("="*40)

    # Pronalazimo red sa maksimalnom vrijednosæu na testu za grijanje
    pobjednik_heating = Acc1.loc[Acc1['test_Heating'].idxmax()]

    # Pronalazimo red sa maksimalnom vrijednosæu na testu za hlaðenje
    pobjednik_cooling = Acc1.loc[Acc1['test_Cooling'].idxmax()]

    print("\n" + "*"*50)
    print("?? REZULTATI TAKMICENJA MODELA ??")
    print("*"*50)
    print(f"Pobjednik za GRIJANJE: {pobjednik_heating['model']}")
    print(f"Preciznost (R2): {pobjednik_heating['test_Heating']:.4f}")
    print("-" * 30)
    print(f"Pobjednik za HLADJENJE: {pobjednik_cooling['model']}")
    print(f"Preciznost (R2): {pobjednik_cooling['test_Cooling']:.4f}")
    print("*"*50)

    # --- 1. GENERISANJE PREDVIÐANJA (KORISTIMO NAJBOLJI MODEL, NPR. CATBOOST) ---
    # Zamijeni 'grid_CBR' sa nazivom svog najboljeg modela ako je drugi pobijedio
    y1_pred = grid_CBR.predict(X_test) 
    y2_pred = grid_CBR.predict(X_test) 

    # --- 2. CRTANJE GRAFIKONA ---
    import matplotlib.pyplot as plt

    x_ax = range(len(y1_test))
    plt.figure(figsize=(20, 10))

    # GORNJI GRAFIKON: Grijanje (Heating)
    plt.subplot(2, 1, 1)
    plt.plot(x_ax, y1_test, label="Actual Heating", color='blue', linewidth=1.5)
    plt.plot(x_ax, y1_pred, label="Predicted Heating", color='red', linestyle='--', linewidth=1.5)
    plt.title("Heating: Stvarni vs Predvideni podaci")
    plt.ylabel('Heating load (kW)')
    plt.legend(loc='best', fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)

    # DONJI GRAFIKON: Hlaðenje (Cooling)
    plt.subplot(2, 1, 2)
    plt.plot(x_ax, y2_test, label="Actual Cooling", color='green', linewidth=1.5)
    plt.plot(x_ax, y2_pred, label="Predicted Cooling", color='orange', linestyle='--', linewidth=1.5)
    plt.title("Cooling: Stvarni vs Predvideni podaci")
    plt.xlabel('Redni broj uzorka (Test set)')
    plt.ylabel('Cooling load (kW)')
    plt.legend(loc='best', fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)

    # --- 3. PRIKAZ I SPAAVANJE ---
    plt.tight_layout() # Automatski popravlja razmak izmeðu grafikona
    plt.savefig("Rezultati_Grafikon.png", dpi=300) # Spaava sliku u visokoj rezoluciji
    plt.show()

    model_cat_y1.save_model("catboost_heating_model.cbm")
    model_cat_y2.save_model("catboost_cooling_model.cbm")



except Exception as e:
    print(f"\nDoslo je do greske: {e}")


input("\nPritisni Enter za kraj...")

