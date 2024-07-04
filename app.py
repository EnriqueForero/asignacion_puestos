
import matplotlib
matplotlib.use('Agg')

import streamlit as st
import pulp
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict  # Añadimos esta importación


# Definición de bloques y sub-bloques
BLOCKS = [
    {"stations": ["H11", "H12", "J11", "J12", "K11", "K12", "M11", "M12", "N11", "N12"], "is_window": True},
    {"stations": ["T12", "V11", "V12"], "is_window": True},
    {"stations": ["A8", "A9", "B8", "B9", "D8", "D9", "E8", "E9", "G8", "G9"], "is_window": False},
    {"stations": ["J8", "J9", "K8", "K9"], "is_window": False},
    {"stations": ["S8", "S9", "T8", "T9", "V8", "V9", "W8", "W9"], "is_window": False}
]

# Puntajes de estaciones
STATION_SCORES = {
    "H11": 3, "H12": 3, "J11": 3, "J12": 3, "K11": 3, "K12": 3,
    "M11": 3, "M12": 3, "N11": 3, "N12": 3,
    "T12": 3, "V11": 3, "V12": 3,
    "A8": 1, "A9": 1, "B8": 1, "B9": 1,
    "D8": 1, "D9": 1, "E8": 1, "E9": 1,
    "G8": 1, "G9": 1,
    "J8": 1, "J9": 1, "K8": 1, "K9": 1,
    "S8": 1, "S9": 1, "T8": 1, "T9": 1,
    "V8": 1, "V9": 1, "W8": 1, "W9": 1,
}

PREFERENCE_FACTORS = {
    "fulfilled": 1.5,
    "random": 1.0,
    "unfulfilled": 0.5
}

def load_teams_from_json(file_path: str) -> dict:
    with open(file_path, 'r') as f:
        return json.load(f)

def calculate_total_utility(utility_per_station: Dict[str, float]) -> float:
    """
    Calcula la utilidad total sumando las utilidades individuales de todas las estaciones.
    """
    return sum(utility_per_station.values())

def optimize_with_multiple_starts(teams: dict, num_starts: int = 10):
    best_assignments = None
    best_utility_per_station = None
    best_total_utility = float('-inf')

    for _ in range(num_starts):
        assignments, utility_per_station = optimize_assignments(teams)
        total_utility = calculate_total_utility(utility_per_station)

        if total_utility > best_total_utility:
            best_assignments = assignments
            best_utility_per_station = utility_per_station
            best_total_utility = total_utility

    return best_assignments, best_utility_per_station, best_total_utility

def optimize_assignments(teams: dict):
    # Crear el problema
    prob = pulp.LpProblem("Workstation Assignment", pulp.LpMaximize)

    # Crear variables de decisión
    x = pulp.LpVariable.dicts("assign",
                              ((t, b) for t in teams for b in range(len(BLOCKS))),
                              cat='Binary')

    # Función objetivo
    prob += pulp.lpSum(x[t, b] * sum(STATION_SCORES[s] for s in BLOCKS[b]['stations']) * teams[t]['utility'] *
                       (PREFERENCE_FACTORS["fulfilled"] if (teams[t]['preference'] == 'window' and BLOCKS[b]['is_window']) or
                                                          (teams[t]['preference'] == 'wall' and not BLOCKS[b]['is_window']) else
                        PREFERENCE_FACTORS["random"] if teams[t]['preference'] == 'random' else
                        PREFERENCE_FACTORS["unfulfilled"])
                       for t in teams for b in range(len(BLOCKS)))

    # Restricciones
    # Cada equipo se asigna a exactamente un bloque
    for t in teams:
        prob += pulp.lpSum(x[t, b] for b in range(len(BLOCKS))) == 1

    # La capacidad de cada bloque no se excede
    for b in range(len(BLOCKS)):
        prob += pulp.lpSum(x[t, b] * teams[t]['size'] for t in teams) <= len(BLOCKS[b]['stations'])

    # Resolver el problema
    # Ajustar parámetros del solver
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=60, gapRel=0.01)
    prob.solve(solver)

    # Extraer la solución y calcular la utilidad individual
    assignments = {}
    utility_per_station = {}
    for b in range(len(BLOCKS)):
        block_stations = BLOCKS[b]['stations'].copy()
        for t in teams:
            if x[t, b].value() == 1:
                team_size = teams[t]['size']
                for _ in range(team_size):
                    if block_stations:
                        station = block_stations.pop(0)
                        assignments[station] = t
                        # Calcular utilidad individual
                        is_window = BLOCKS[b]['is_window']
                        base_score = STATION_SCORES.get(station, 1 if not is_window else 2)
                        team_utility = teams[t]['utility']
                        preference = teams[t]['preference']
                        if (preference == 'window' and is_window) or (preference == 'wall' and not is_window):
                            preference_factor = PREFERENCE_FACTORS["fulfilled"]
                        elif preference == 'random':
                            preference_factor = PREFERENCE_FACTORS["random"]
                        else:
                            preference_factor = PREFERENCE_FACTORS["unfulfilled"]
                        utility_per_station[station] = (base_score * team_utility * preference_factor) / team_size

    # Asignar estaciones vacías
    for block in BLOCKS:
        for station in block['stations']:
            if station not in assignments:
                assignments[station] = "Vacío"
                utility_per_station[station] = 0

    return assignments, utility_per_station

def visualize_grid(assignments: Dict[str, str], utility_per_station: Dict[str, float], teams: Dict, fig_width: int = 15, fig_height: int = 6):
    """
    Visualiza la asignación de puestos de trabajo en una cuadrícula, incluyendo la utilidad individual.

    Args:
        assignments (Dict[str, str]): Asignaciones de estaciones a equipos.
        utility_per_station (Dict[str, float]): Utilidad por estación.
        teams (Dict): Configuración de equipos.
        fig_width (int): Ancho de la figura. Por defecto es 15.
        fig_height (int): Alto de la figura. Por defecto es 6.
    """
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim(0, 23)
    ax.set_ylim(0, 5)
    ax.axis('off')

    colors = {team: info["color"] for team, info in teams.items()}
    colors["Vacío"] = "white"  # Color para puestos vacíos

    x_mapping = {chr(65+i): i for i in range(23)}

    for station, team in assignments.items():
        col = x_mapping[station[0]]
        row = 12 - int(station[1:])
        rect = patches.Rectangle((col, row), 1, 1, facecolor=colors[team], edgecolor='black')
        ax.add_patch(rect)
        if team != "Vacío":
            ax.text(col+0.5, row+0.7, team[0], ha='center', va='center', color='white' if team in ["Industrias", "Proyectos"] else 'black', fontweight='bold')
            individual_utility = utility_per_station.get(station, 0)
            ax.text(col+0.5, row+0.3, f"{individual_utility:.2f}", ha='center', va='center', fontsize=8)
        else:
            ax.text(col+0.5, row+0.5, station, ha='center', va='center', fontsize=8)

    ax.text(-1, 2, 'VENTANA', rotation=90, va='center', fontweight='bold')
    ax.text(-1, 4, 'PARED', rotation=90, va='center', fontweight='bold')

    legend_elements = [patches.Patch(facecolor=color, edgecolor='black', label=team)
                       for team, color in colors.items()]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05),
              ncol=5, fancybox=True, shadow=True)

    plt.title('Asignación de puestos de trabajo y utilidad individual', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig  # Devuelve la figura en lugar de mostrarla

def main(teams_file: str):
    teams = load_teams_from_json(teams_file)
    assignments, utility_per_station, total_utility = optimize_with_multiple_starts(teams)

    df = pd.DataFrame([(station, team) for station, team in assignments.items()],
                      columns=['Estación', 'Equipo'])
    print("Resumen de asignaciones:")
    print(df)

    print("\nDistribución de equipos por tipo de estación:")

    # Crear un diccionario que mapee cada estación a su tipo (ventana o pared)
    station_types = {}
    for block in BLOCKS:
        for station in block['stations']:
            station_types[station] = 'Ventana' if block['is_window'] else 'Pared'

    # Calcular la distribución
    distribution = df.groupby('Equipo').apply(
        lambda x: pd.Series({
            'Ventana': sum(station_types[station] == 'Ventana' for station in x['Estación']),
            'Pared': sum(station_types[station] == 'Pared' for station in x['Estación'])
        })
    )

    print(distribution)

    # Calcular utility_per_station
    utility_per_station = {}
    for station, team in assignments.items():
        if team != "Vacío":
            is_window = any(station in block['stations'] for block in BLOCKS if block['is_window'])
            base_score = STATION_SCORES.get(station, 1 if not is_window else 2)
            team_utility = teams[team]['utility']
            preference = teams[team]['preference']

            if (preference == 'window' and is_window) or (preference == 'wall' and not is_window):
                preference_factor = PREFERENCE_FACTORS["fulfilled"]
            elif preference == 'random':
                preference_factor = PREFERENCE_FACTORS["random"]
            else:
                preference_factor = PREFERENCE_FACTORS["unfulfilled"]

            utility_per_station[station] = base_score * team_utility * preference_factor
        else:
            utility_per_station[station] = 0

    # Calcular y imprimir la utilidad total
    total_utility = calculate_total_utility(utility_per_station)
    print(f"\nMejor utilidad total alcanzada: {total_utility:.2f}")

    visualize_grid(assignments, utility_per_station, teams)

    # Verificar si los equipos están juntos en sus bloques asignados
    print("\nVerificación de equipos juntos por bloque:")
    for team in teams:
        team_stations = [station for station, assigned_team in assignments.items() if assigned_team == team]
        if team_stations:
            team_block = next(block for block in BLOCKS if team_stations[0] in block['stations'])
            is_window = team_block['is_window']
            if all(station in team_block['stations'] for station in team_stations):
                print(f"El equipo {team} está junto en el bloque {'de ventana' if is_window else 'de pared'}")
            else:
                print(f"ALERTA: El equipo {team} está dividido entre bloques")

    # Verificar si los equipos están en su ubicación preferida
    print("\nVerificación de ubicación preferida de equipos:")
    for team, info in teams.items():
        team_stations = [station for station, assigned_team in assignments.items() if assigned_team == team]
        if team_stations:
            is_window = any(station in block['stations'] for block in BLOCKS if block['is_window'] for station in team_stations)
            if info['preference'] == 'window' and is_window:
                print(f"El equipo {team} está en la ventana (preferencia cumplida)")
            elif info['preference'] == 'wall' and not is_window:
                print(f"El equipo {team} está en la pared (preferencia cumplida)")
            elif info['preference'] == 'random':
                print(f"El equipo {team} está en {'la ventana' if is_window else 'la pared'} (preferencia aleatoria)")
            else:
                print(f"ALERTA: El equipo {team} no está en su ubicación preferida")

#

# Ejemplo de uso
teams_json = {
    "Analítica": {"size": 4, "color": "gold", "preference": "wall", "utility": 2},
    "Manufacturas": {"size": 4, "color": "silver", "preference": "random", "utility": 2},
    "Proyectos": {"size": 4, "color": "violet", "preference": "window", "utility": 3},
    "Logística": {"size": 3, "color": "chocolate", "preference": "random", "utility": 2},
    "Turismo": {"size": 3, "color": "yellow", "preference": "random", "utility": 2},
    "Agroalimentos": {"size": 3, "color": "seagreen", "preference": "random", "utility": 2},
    "Industrias": {"size": 2, "color": "skyblue", "preference": "random", "utility": 2},
    "Practicantes": {"size": 5, "color": "seashell", "preference": "wall", "utility": -1}
}

import tempfile

with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_file:
    json.dump(teams_json, temp_file)
    temp_file_path = temp_file.name

main(temp_file_path)

import os
os.unlink(temp_file_path)


# Configuración de la página de Streamlit
st.set_page_config(page_title="Optimización de Asignación de Puestos", layout="wide")

st.title("Optimización de Asignación de Puestos de Trabajo")

# Sidebar para parámetros ajustables
st.sidebar.header("Parámetros de Equipos")

# Ajuste de tamaño de equipo y utilidad
updated_teams = {}
for team, info in teams_json.items():
    st.sidebar.subheader(f"Equipo: {team}")
    
    # Ajuste de tamaño del equipo
    team_size = st.sidebar.number_input(
        f"Número de personas en {team}",
        min_value=1,
        max_value=10,
        value=info['size'],
        step=1
    )
    
    # Ajuste de utilidad
    initial_value = float(info['utility'])
    initial_value = max(min(initial_value, 5.0), -1.0)  # Limitar el valor entre -1.0 y 5.0
    team_utility = st.sidebar.slider(
        f"Utilidad de {team}",
        min_value=-1.0,
        max_value=5.0,
        value=initial_value,
        step=0.1
    )
    
    # Ajuste de preferencias
    team_preference = st.sidebar.selectbox(
        f"Preferencia de {team}",
        options=['window', 'wall', 'random'],
        index=['window', 'wall', 'random'].index(info['preference'])
    )
    
    # Actualizar la información del equipo
    updated_teams[team] = {
        'size': team_size,
        'utility': team_utility,
        'preference': team_preference,
        'color': info['color']  # Mantener el color original
    }

# Botón para ejecutar la optimización
if st.sidebar.button("Optimizar asignaciones"):
    # Guardar los datos actualizados en un archivo temporal
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_file:
        json.dump(updated_teams, temp_file)
        temp_file_path = temp_file.name

    # Ejecutar la optimización
    assignments, utility_per_station, total_utility = optimize_with_multiple_starts(updated_teams)

    # Mostrar resultados
    st.subheader("Resumen de asignaciones")
    df = pd.DataFrame([(station, team) for station, team in assignments.items()],
                      columns=['Estación', 'Equipo'])
    st.dataframe(df)

    st.subheader("Distribución de equipos por tipo de estación")
    station_types = {}
    for block in BLOCKS:
        for station in block['stations']:
            station_types[station] = 'Ventana' if block['is_window'] else 'Pared'

    distribution = df.groupby('Equipo').apply(
        lambda x: pd.Series({
            'Ventana': sum(station_types[station] == 'Ventana' for station in x['Estación']),
            'Pared': sum(station_types[station] == 'Pared' for station in x['Estación'])
        })
    )
    st.dataframe(distribution)

    st.subheader("Utilidad total")
    st.write(f"Mejor utilidad total alcanzada: {total_utility:.2f}")

    # Visualización de la asignación
    st.subheader("Visualización de la asignación")
    fig = visualize_grid(assignments, utility_per_station, updated_teams)
    st.pyplot(fig)

    # Verificar si los equipos están juntos en sus bloques asignados
    st.subheader("Verificación de equipos juntos por bloque:")
    team_block_info = []
    for team in updated_teams:
        team_stations = [station for station, assigned_team in assignments.items() if assigned_team == team]
        if team_stations:
            team_block = next(block for block in BLOCKS if team_stations[0] in block['stations'])
            is_window = team_block['is_window']
            if all(station in team_block['stations'] for station in team_stations):
                team_block_info.append(f"El equipo {team} está junto en el bloque {'de ventana' if is_window else 'de pared'}")
            else:
                team_block_info.append(f"ALERTA: El equipo {team} está dividido entre bloques")
    
    for info in team_block_info:
        if "ALERTA" in info:
            st.warning(info)
        else:
            st.info(info)

    # Verificar si los equipos están en su ubicación preferida
    st.subheader("Verificación de ubicación preferida de equipos:")
    team_preference_info = []
    for team, info in updated_teams.items():
        team_stations = [station for station, assigned_team in assignments.items() if assigned_team == team]
        if team_stations:
            is_window = any(station in block['stations'] for block in BLOCKS if block['is_window'] for station in team_stations)
            if info['preference'] == 'window' and is_window:
                team_preference_info.append(f"El equipo {team} está en la ventana (preferencia cumplida)")
            elif info['preference'] == 'wall' and not is_window:
                team_preference_info.append(f"El equipo {team} está en la pared (preferencia cumplida)")
            elif info['preference'] == 'random':
                team_preference_info.append(f"El equipo {team} está en {'la ventana' if is_window else 'la pared'} (preferencia aleatoria)")
            else:
                team_preference_info.append(f"ALERTA: El equipo {team} no está en su ubicación preferida")
    
    for info in team_preference_info:
        if "ALERTA" in info:
            st.warning(info)
        else:
            st.success(info)

    # Limpiar el archivo temporal
    os.unlink(temp_file_path)

else:
    st.write("Ajusta los parámetros en la barra lateral y haz clic en 'Optimizar asignaciones' para ver los resultados.")
