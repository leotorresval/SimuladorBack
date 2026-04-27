# app.py
import os
import tempfile
import numpy as np
import pandas as pd
from scipy.stats import expon

from flask import Flask, request, jsonify
from flask_cors import CORS

import wntr
from pyproj import Transformer
import services.storage as storage

# =====================================================
# APP
# =====================================================
app = Flask(__name__)
CORS(app)


# =====================================================
# COORDINATE TRANSFORM (UTM 17S → WGS84)
# =====================================================
UTM_TO_WGS84 = Transformer.from_crs(
    "EPSG:32717",  # UTM Zona 17 Sur (Quito)
    "EPSG:4326",   # Lat/Lng
    always_xy=True
)


def utm_to_latlng(x, y):
    if x is None or y is None:
        return None, None
    lon, lat = UTM_TO_WGS84.transform(x, y)
    return lat, lon


# =====================================================
# CONFIG
# =====================================================
TOTAL_DURATION_S = 24 * 3600
MINIMUM_PRESSURE = 3.52
REQUIRED_PRESSURE = 14.06
LEAK_START_TIME_S = 5 * 3600

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
# CORE SIMULATION
# =====================================================
def run_simulation(inp_storage_file, magnitude, depth, epicenter_x, epicenter_y):

    tmp_path = None
    try:
        # -----------------------------
        # Guardar archivo temporal
        # -----------------------------
        with tempfile.NamedTemporaryFile(delete=False, suffix=".inp") as tmp:
            tmp_path = tmp.name
            inp_storage_file.save(tmp_path)

        # -----------------------------
        # Cargar red
        # -----------------------------
        wn = wntr.network.WaterNetworkModel(tmp_path)

        # -----------------------------
        # Epicentro (UTM)
        # -----------------------------
        epicenter_utm = (epicenter_x, epicenter_y)
        epicenter_lat, epicenter_lng = utm_to_latlng(epicenter_x, epicenter_y)

        earthquake = wntr.scenario.Earthquake(
            epicenter_utm, magnitude, depth
        )

        # -----------------------------
        # Daño sísmico
        # -----------------------------
        R = earthquake.distance_to_epicenter(
            wn, element_type=wntr.network.Pipe
        )
        pga = earthquake.pga_attenuation_model(R)
        pgv = earthquake.pgv_attenuation_model(R)
        RR = earthquake.repair_rate_model(pgv)

        L = pd.Series(
            wn.query_link_attribute("length", link_type=wntr.network.Pipe)
        )

        pipe_FC = wntr.scenario.FragilityCurve()
        pipe_FC.add_state("Minor Leak", 1, {"Default": expon(scale=0.2)})
        pipe_FC.add_state("Major Leak", 2, {"Default": expon()})

        x_fc = np.linspace(0, 5, 100)
        fragility_curve_data = {
            "x": x_fc.tolist(),
            "states": [
                {"name": "Minor Leak", "y": expon(scale=0.2).cdf(x_fc).tolist()},
                {"name": "Major Leak", "y": expon().cdf(x_fc).tolist()},
            ],
        }

        RR_x_L = RR * L
        pipe_Pr = pipe_FC.cdf_probability(RR_x_L)
        pipe_damage_state = pipe_FC.sample_damage_state(pipe_Pr)

        # -----------------------------
        # Config hidráulica
        # -----------------------------
        wn.options.hydraulic.demand_model = "PDD"
        wn.options.time.duration = TOTAL_DURATION_S
        wn.options.hydraulic.minimum_pressure = MINIMUM_PRESSURE
        wn.options.hydraulic.required_pressure = REQUIRED_PRESSURE

        # -----------------------------
        # Fugas
        # -----------------------------
        leaks_data = []

        for pipe_name, damage_state in pipe_damage_state.items():
            pipe = wn.get_link(pipe_name)
            leak_area = 0.0
            leak_type = "None"

            if damage_state == "Major Leak":
                d = 0.25 * pipe.diameter
                leak_area = np.pi / 4 * d**2
                leak_type = "Major Leak"

            elif damage_state == "Minor Leak":
                d = 0.10 * pipe.diameter
                leak_area = np.pi / 4 * d**2
                leak_type = "Minor Leak"

            if leak_area > 0:
                wn = wntr.morph.split_pipe(
                    wn, pipe_name, pipe_name + "_A", "Leak_" + pipe_name
                )
                leak_node = wn.get_node("Leak_" + pipe_name)
                leak_node.add_leak(
                    wn, area=leak_area, start_time=LEAK_START_TIME_S
                )

                x, y = leak_node.coordinates if leak_node.coordinates else (None, None)
                lat, lng = utm_to_latlng(x, y)

                leaks_data.append({
                    "id": leak_node.name,
                    "pipe_origin": pipe_name,
                    "leak_type": leak_type,
                    "leak_area": float(leak_area),
                    "start_time_h": LEAK_START_TIME_S / 3600,
                    "x": x,
                    "y": y,
                    "lat": lat,
                    "lng": lng,
                })

        # -----------------------------
        # Simulación
        # -----------------------------
        sim = wntr.sim.WNTRSimulator(wn)
        results = sim.run_sim()

        pressure = results.node["pressure"]
        pressure.index /= 3600


        # =====================================================
        # CURVA DE PRESIÓN PROMEDIO
        # =====================================================
        avg_pressure = pressure.mean(axis=1)

        pressure_avg_curve = {
            "time": avg_pressure.index.tolist(),
            "values": avg_pressure.tolist()
        }




        t24 = pressure.index[abs(pressure.index - 24).argmin()]
        pressure_24h = pressure.loc[t24, wn.junction_name_list]

        # =====================
        # RESILIENCIA
        # =====================
        print("WSA start")
        expected_demand = wntr.metrics.expected_demand(wn)
        demand = results.node['demand']

        wsa = wntr.metrics.water_service_availability(
            expected_demand,
            demand
        )

        wsa_avg = float(wsa.mean().mean())
        print("WSA done")

        # Todini
        print("Todini start")
        head = results.node['head']
        pressure = results.node['pressure']
        flowrate = results.link['flowrate']

        pump_flowrate = flowrate.loc[:, wn.pump_name_list]

        todini_index = wntr.metrics.todini_index(
            head,
            pressure,
            demand,
            pump_flowrate,
            wn,
            REQUIRED_PRESSURE
        )

        todini_avg = float(todini_index.mean())
        print("Todini done")

        # # Entropía
        # print("Entropía start")
        # flow_snapshot = flowrate.loc[12*3600, :]
        # flow_snapshot = abs(flow_snapshot)
        # flow_snapshot = flow_snapshot.sort_values(ascending=False).head(500)

        # G = wn.to_graph(link_weight=flow_snapshot)

        # _, system_entropy = wntr.metrics.entropy(G)
        # print("Entropía done")



        leak_series = results.node["leak_demand"]

        
        leak_demand_curve = {
            "time": (leak_series.index / 3600).tolist(),
            "series": []
        }

        for node_name in leak_series.columns:
            leak_demand_curve["series"].append({
                "name": node_name,
                "y": (leak_series[node_name] * 1000).tolist()  # L/s
            })

        # -----------------------------
        # tuberias fix
        # -----------------------------
        
        leaked_sum = leak_series.sum().sort_values(ascending=False)
        pipes_to_fix = leaked_sum[leaked_sum > 0]
        num_pipes_to_repair = sum(
            1 for damage_state in pipe_damage_state.values
            if damage_state in ['Minor Leak', 'Major Leak']
        )
        
        df_pipes_to_fix = pipes_to_fix.reset_index()
        df_pipes_to_fix.columns = ['Tuberia', 'Fuga_acumulada']

        # Quitar el prefijo 'Leak_' para recuperar el nombre original de la tubería
        df_pipes_to_fix['Tuberia'] = df_pipes_to_fix['Tuberia'].str.replace('Leak_', '', regex=False)

        # Agregar estado de daño
        df_damage = pd.Series(pipe_damage_state, name='Estado_daño').reset_index()
        df_damage.columns = ['Tuberia', 'Estado_daño']

        # Asegurar mismo tipo de dato en ambas columnas
        df_pipes_to_fix['Tuberia'] = df_pipes_to_fix['Tuberia'].astype(str)
        df_damage['Tuberia'] = df_damage['Tuberia'].astype(str)

        # Unir información
        df_export = df_pipes_to_fix.merge(df_damage, on='Tuberia', how='left')
        df_export.insert(0, 'Prioridad', range(1, len(df_export) + 1))
        
        df_export = df_export[
            [ 'Tuberia','Prioridad', 'Estado_daño', 'Fuga_acumulada']
        ]
        pipes_fix_list = df_export.to_dict(orient="records")


        # -----------------------------
        # Summary
        # -----------------------------
        summary = {
            "time_analyzed_h": round(float(t24), 2),
            "pressure_min": round(float(pressure_24h.min()), 2),
            "pressure_max": round(float(pressure_24h.max()), 2),
            "pressure_mean": round(float(pressure_24h.mean()), 2),
            "nodes_below_required_percent": round(
                float((pressure_24h < REQUIRED_PRESSURE).mean() * 100), 2
            ),
            "avg_minor_leak_prob": round(pipe_Pr["Minor Leak"].mean(), 3),
            "avg_major_leak_prob": round(pipe_Pr["Major Leak"].mean(), 3),
            "wsa_avg": round(wsa_avg, 3),
            "todini_index": round(todini_avg, 3),
             "pipes_to_fix": pipes_fix_list,
            # "system_entropy": round(float(system_entropy), 3),
        }
        # -----------------------------
        # Nodes
        # -----------------------------
        nodes_data = []
        for node_name in wn.junction_name_list:
            node = wn.get_node(node_name)
            x, y = node.coordinates if node.coordinates else (None, None)
            lat, lng = utm_to_latlng(x, y)
            demand_m3s = node.demand_timeseries_list[0].base_value

            nodes_data.append({
                "id": node_name,
                "x": x,
                "y": y,
                "lat": lat,
                "lng": lng,
                "pressure": float(pressure_24h.get(node_name, 0.0)),
                "elevation": float(node.elevation),
                "demand": float(node.demand_timeseries_list[0].base_value),
                "demand_lps": float(demand_m3s * 1000),
                "below_required": bool(
                    pressure_24h.get(node_name, 0.0) < REQUIRED_PRESSURE
                ),
            })

        # -----------------------------
        # Pipes
        # -----------------------------
        pipes_data = []
        vel_series = results.link["velocity"]
        flow_series = results.link["flowrate"]
        for pipe_name in wn.pipe_name_list:
            pipe = wn.get_link(pipe_name)
            pipes_data.append({
                "id": pipe_name,
                "from": pipe.start_node_name,
                "to": pipe.end_node_name,
                "damage_state": pipe_damage_state.get(pipe_name, "None"),
                "distance": float(R.get(pipe_name, 0.0)),
                "pga": float(pga.get(pipe_name, 0.0)),
                "pgv": float(pgv.get(pipe_name, 0.0)),
                "repair_rate": float(RR.get(pipe_name, 0.0)),
                "length": float(L.get(pipe_name, 0.0)),
                "diameter": pipe.diameter,
                "material": infer_material_hw(pipe.roughness),
                "damage_index": float(RR_x_L.get(pipe_name, 0.0)),
                "p_minor_leak": float(pipe_Pr["Minor Leak"].get(pipe_name, 0.0)),
                "p_major_leak": float(pipe_Pr["Major Leak"].get(pipe_name, 0.0)),
                "speed": float(vel_series.loc[0, pipe_name]),
                "flowrate": float(flow_series.loc[0, pipe_name]),   # m³/s
                "flowrate_lps": float(flow_series.loc[0, pipe_name] * 1000),  # L/s
            })


        # -----------------------------
        # RESPONSE
        # -----------------------------
        result = {
            "epicenter": {
                "x": epicenter_x,
                "y": epicenter_y,
                "magnitude": magnitude,
                "depth": depth,
                "lat": epicenter_lat,
                "lng": epicenter_lng,
            },
            "summary": summary,
            "nodes": nodes_data,
            "pipes": pipes_data,
            "leaks": leaks_data,
            "fragility_curve": fragility_curve_data,
            "leak_demand_curve": leak_demand_curve,
            "pressure_avg_curve": pressure_avg_curve,
        }
        storage.LAST_SIMULATION = result
        storage.save_simulation(result)
        return result

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

