# app.py
import os
import tempfile
import numpy as np
import pandas as pd
from scipy.stats import expon

from flask import Flask, request, jsonify
from flask_cors import CORS

import wntr


# =====================================================
# CONFIG
# =====================================================
DEFAULT_EPICENTER = (787671.8935, 9992673.3202)
EJE_X=787671.8935
EJE_Y=9992673.3202
TOTAL_DURATION_S = 24 * 3600
MINIMUM_PRESSURE = 3.52
REQUIRED_PRESSURE = 14.06

LEAK_START_TIME_S = 5 * 3600  # 5h

# Semilla para reproducibilidad (si quieres aleatorio, quita esto)
np.random.seed(13315)

def infer_material_hw(C):
    if C >= 140:
        return "PVC / PEAD"
    elif C >= 120:
        return "Acero / Hierro dúctil"
    elif C >= 100:
        return "Hierro fundido"
    else:
        return "Tubería antigua"

# =====================================================
# CORE: RUN SIMULATION
# =====================================================
def run_simulation(inp_storage_file, magnitude: float, depth: float,x:float,y:float):
    """
    inp_storage_file: archivo recibido (werkzeug FileStorage)
    magnitude: magnitud del sismo (ej: 6.5)
    depth: profundidad en metros (ej: 10000)
    x: coordenada x del epicentro
    y: coordenada y del epicentro
    """

    tmp_path = None
    try:
        # -----------------------------
        # Guardar archivo temporal .inp
        # -----------------------------
        with tempfile.NamedTemporaryFile(delete=False, suffix=".inp") as tmp:
            tmp_path = tmp.name
            inp_storage_file.save(tmp_path)

        # -----------------------------
        # Cargar red
        # -----------------------------
        wn = wntr.network.WaterNetworkModel(tmp_path)

        # -----------------------------
        # Terremoto + parámetros
        # -----------------------------
        DEFAULT_EPICENTER = (x, y)
        epicenter = DEFAULT_EPICENTER
        earthquake = wntr.scenario.Earthquake(epicenter, magnitude, depth)

        # -----------------------------
        # Distancias, PGA, PGV, RR (por tubería)
        # -----------------------------
        R = earthquake.distance_to_epicenter(wn, element_type=wntr.network.Pipe)
        pga = earthquake.pga_attenuation_model(R)
        pgv = earthquake.pgv_attenuation_model(R)
        RR = earthquake.repair_rate_model(pgv)

        # Longitud por tubería
        L = pd.Series(wn.query_link_attribute("length", link_type=wntr.network.Pipe))

        # -----------------------------
        # Curva de fragilidad
        # -----------------------------
        pipe_FC = wntr.scenario.FragilityCurve()
        pipe_FC.add_state("Minor Leak", 1, {"Default": expon(scale=0.2)})
        pipe_FC.add_state("Major Leak", 2, {"Default": expon()})

        # -----------------------------
        # Curva de fragilidad (para FRONT)
        # -----------------------------
        x_fc = np.linspace(0, 5, 100)  # mismo rango típico que wntr usa
        minor_fc = expon(scale=0.2).cdf(x_fc)
        major_fc = expon().cdf(x_fc)

        fragility_curve_data = {
            "x": x_fc.tolist(),
            "states": [
                {
                    "name": "Minor Leak",
                    "y": minor_fc.tolist()
                },
                {
                    "name": "Major Leak",
                    "y": major_fc.tolist()
                }
            ]
        }



        # Probabilidades de daño por tubería
        RR_x_L = RR * L
        pipe_Pr = pipe_FC.cdf_probability(RR_x_L)

        # Muestreo de estado de daño
        pipe_damage_state = pipe_FC.sample_damage_state(pipe_Pr)

        # -----------------------------
        # Config hidráulica
        # -----------------------------
        wn.options.hydraulic.demand_model = "PDD"
        wn.options.time.duration = TOTAL_DURATION_S
        wn.options.hydraulic.minimum_pressure = MINIMUM_PRESSURE
        wn.options.hydraulic.required_pressure = REQUIRED_PRESSURE

        # -----------------------------
        # Agregar fugas (crea nodos Leak_*)
        # -----------------------------
        # Guardaremos una tabla de fugas mientras iteramos
        leaks_data = []

        for pipe_name, damage_state in pipe_damage_state.items():
            pipe = wn.get_link(pipe_name)
            pipe_diameter = pipe.diameter
            leak_area = 0.0
            leak_type = "None"

            if damage_state == "Major Leak":
                leak_diameter = 0.25 * pipe_diameter
                leak_area = np.pi / 4 * leak_diameter**2
                leak_type = "Major Leak"

            elif damage_state == "Minor Leak":
                leak_diameter = 0.10 * pipe_diameter
                leak_area = np.pi / 4 * leak_diameter**2
                leak_type = "Minor Leak"

            if leak_area > 0:
                # split_pipe crea un nodo Leak_<pipe_name>
                wn = wntr.morph.split_pipe(
                    wn, pipe_name, pipe_name + "_A", "Leak_" + pipe_name
                )
                leak_node_name = "Leak_" + pipe_name
                leak_node = wn.get_node(leak_node_name)
                leak_node.add_leak(wn, area=leak_area, start_time=LEAK_START_TIME_S)

                # Guardar registro de fuga (tabla)
                # Nota: coordenadas del nodo de fuga pueden ser None según la red;
                # WNTR suele asignar coordenadas por interpolación si están disponibles.
                x, y = leak_node.coordinates if leak_node.coordinates else (None, None)
                leaks_data.append(
                    {
                        "id": leak_node_name,
                        "pipe_origin": pipe_name,
                        "leak_type": leak_type,
                        "leak_area": float(leak_area),
                        "start_time_h": LEAK_START_TIME_S / 3600.0,
                        "x": float(x) if x is not None else None,
                        "y": float(y) if y is not None else None,
                    }
                )

        # -----------------------------
        # Simulación
        # -----------------------------
        sim = wntr.sim.WNTRSimulator(wn)
        results = sim.run_sim()

        
        # -----------------------------
        # Presiones a ~24h
        # -----------------------------
        pressure = results.node["pressure"]
        pressure.index = pressure.index / 3600.0

        time_24h = pressure.index[abs(pressure.index - 24).argmin()]
        pressure_24h = pressure.loc[time_24h, wn.junction_name_list]

        # -----------------------------
        # Summary (KPIs)
        # -----------------------------
        nodes_below_required = int((pressure_24h < REQUIRED_PRESSURE).sum())
        total_nodes = int(len(pressure_24h))
        percent_low = (nodes_below_required / total_nodes * 100) if total_nodes else 0.0

        summary = {
            "time_analyzed_h": round(float(time_24h), 2),
            "pressure_min": round(float(pressure_24h.min()), 2),
            "pressure_max": round(float(pressure_24h.max()), 2),
            "pressure_mean": round(float(pressure_24h.mean()), 2),
            "nodes_below_required_percent": round(float(percent_low), 2),
            "avg_minor_leak_prob": round(float(pipe_Pr["Minor Leak"].mean()), 3),
            "avg_major_leak_prob": round(float(pipe_Pr["Major Leak"].mean()), 3),
            "pipes_major_leak_gt_05": int((pipe_Pr["Major Leak"] > 0.5).sum()),
        }

        # -----------------------------
        # Nodes (para mapa / tabla)
        # -----------------------------
        nodes_data = []
        for node_name in wn.junction_name_list:
            node = wn.get_node(node_name)
            x, y = node.coordinates if node.coordinates else (None, None)

            nodes_data.append(
                {
                    "id": node_name,
                    "x": float(x) if x is not None else None,
                    "y": float(y) if y is not None else None,
                    "pressure": float(pressure_24h.get(node_name, 0.0)),
                    "elevation": float(node.elevation),
                    "demand": float(node.demand_timeseries_list[0].base_value),
                    "below_required": bool(pressure_24h.get(node_name, 0.0) < REQUIRED_PRESSURE),
                }
            )

        # -----------------------------
        # Pipes (para mapas y tablas)
        # -----------------------------
        pipes_data = []
        for pipe_name in wn.pipe_name_list:
            pipe = wn.get_link(pipe_name)
            vel_series = results.link["velocity"]
            speed_24h = vel_series.loc[0, pipe_name]
            pipes_data.append(
                {
                    "id": pipe_name,
                    "from": pipe.start_node_name,
                    "to": pipe.end_node_name,
                    "damage_state": pipe_damage_state.get(pipe_name, "None"),
                    "distance": float(R.get(pipe_name, 0.0)),
                    "pga": float(pga.get(pipe_name, 0.0)),
                    "pgv": float(pgv.get(pipe_name, 0.0)),
                    "repair_rate": float(RR.get(pipe_name, 0.0)),
                    # NUEVO (para tablas)
                    "length": float(L.get(pipe_name, 0.0)),
                    "diameter": pipe.diameter, 
                    "material": infer_material_hw(pipe.roughness),
                    "damage_index": float(RR_x_L.get(pipe_name, 0.0)),  # RR * L
                    "p_minor_leak": float(pipe_Pr["Minor Leak"].get(pipe_name, 0.0)),
                    "p_major_leak": float(pipe_Pr["Major Leak"].get(pipe_name, 0.0)),
                    "speed": float(speed_24h)
                }
            )

        # -----------------------------
        # Ranking de fuga (top N)
        # -----------------------------
        # results.node['leak_demand'] existe cuando hay leaks; si no hay, puede venir vacío
        leak_ranking_data = []
        if "leak_demand" in results.node:
            leaked_demand = results.node["leak_demand"].copy()
            leaked_demand.index = leaked_demand.index / 3600.0

            # Suma total por nodo de fuga
            leak_sum = leaked_demand.sum().sort_values(ascending=False)

            # Filtrar solo los Leak_* si deseas (recomendado)
            leak_sum = leak_sum[leak_sum.index.astype(str).str.startswith("Leak_")]

            top_n = 10
            leak_sum = leak_sum.head(top_n)

            leak_ranking_data = [
                {"leak_node": str(node_id), "total_leak": float(val)}
                for node_id, val in leak_sum.items()
            ]

        # -----------------------------
        # Respuesta final
        # -----------------------------
        return {
            "summary": summary,
            "nodes": nodes_data,
            "pipes": pipes_data,
            "leaks": leaks_data,
            "leak_ranking": leak_ranking_data,
            "fragility_curve": fragility_curve_data
        }

    finally:
        # Siempre limpiar el temporal
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

