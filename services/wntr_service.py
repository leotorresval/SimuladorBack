import wntr
import numpy as np
import pandas as pd
import tempfile
import os
from scipy.stats import expon


def run_simulation(inp_file, magnitude, depth):
    # =====================================================
    # Guardar archivo temporal
    # =====================================================
    with tempfile.NamedTemporaryFile(delete=False, suffix=".inp") as tmp:
        inp_path = tmp.name
        inp_file.save(inp_path)

    # =====================================================
    # Cargar red
    # =====================================================
    wn = wntr.network.WaterNetworkModel(inp_path)

    # =====================================================
    # Parámetros fijos (puedes exponerlos luego si quieres)
    # =====================================================
    epicenter = (787671.8935, 9992673.3202)
    total_duration = 24 * 3600
    minimum_pressure = 3.52
    required_pressure = 14.06

    leak_start_time = 5 * 3600
    leak_repair_time = 15 * 3600

    np.random.seed(13315)

    # =====================================================
    # Terremoto
    # =====================================================
    earthquake = wntr.scenario.Earthquake(epicenter, magnitude, depth)

    # =====================================================
    # PGA, PGV, Repair Rate
    # =====================================================
    R = earthquake.distance_to_epicenter(
        wn, element_type=wntr.network.Pipe
    )
    pga = earthquake.pga_attenuation_model(R)
    pgv = earthquake.pgv_attenuation_model(R)
    RR = earthquake.repair_rate_model(pgv)

    L = pd.Series(
        wn.query_link_attribute('length', link_type=wntr.network.Pipe)
    )

    # =====================================================
    # Fragilidad
    # =====================================================
    pipe_FC = wntr.scenario.FragilityCurve()
    pipe_FC.add_state('Minor Leak', 1, {'Default': expon(scale=0.2)})
    pipe_FC.add_state('Major Leak', 2, {'Default': expon()})

    pipe_Pr = pipe_FC.cdf_probability(RR * L)
    pipe_damage_state = pipe_FC.sample_damage_state(pipe_Pr)

    # =====================================================
    # Configuración hidráulica
    # =====================================================
    wn.options.hydraulic.demand_model = 'PDD'
    wn.options.time.duration = total_duration
    wn.options.hydraulic.minimum_pressure = minimum_pressure
    wn.options.hydraulic.required_pressure = required_pressure

    # =====================================================
    # Agregar fugas
    # =====================================================
    for pipe_name, damage_state in pipe_damage_state.items():
        pipe = wn.get_link(pipe_name)
        pipe_diameter = pipe.diameter
        leak_area = 0.0

        if damage_state == 'Major Leak':
            leak_diameter = 0.25 * pipe_diameter
            leak_area = np.pi / 4 * leak_diameter**2

        elif damage_state == 'Minor Leak':
            leak_diameter = 0.10 * pipe_diameter
            leak_area = np.pi / 4 * leak_diameter**2

        if leak_area > 0:
            wn = wntr.morph.split_pipe(
                wn, pipe_name, pipe_name + '_A', 'Leak_' + pipe_name
            )
            leak_node = wn.get_node('Leak_' + pipe_name)
            leak_node.add_leak(
                wn, area=leak_area, start_time=leak_start_time
            )

    # =====================================================
    # Simulación
    # =====================================================
    sim = wntr.sim.WNTRSimulator(wn)
    results = sim.run_sim()

    # =====================================================
    # Presiones a 24h
    # =====================================================
    pressure = results.node['pressure']
    pressure.index = pressure.index / 3600

    time_24h = pressure.index[
        abs(pressure.index - 24).argmin()
    ]

    pressure_24h = pressure.loc[time_24h, wn.junction_name_list]

    pressure_df = pressure_24h.reset_index()
    pressure_df.columns = ['node', 'pressure']

    # =====================================================
    # Métricas
    # =====================================================
    summary = {
        "time_analyzed_h": round(time_24h, 2),
        "pressure_min": round(pressure_24h.min(), 2),
        "pressure_max": round(pressure_24h.max(), 2),
        "pressure_mean": round(pressure_24h.mean(), 2),
        "nodes_below_required_percent": round(
            (pressure_24h < required_pressure).sum()
            / len(pressure_24h) * 100, 2
        ),
        "avg_minor_leak_prob": round(pipe_Pr['Minor Leak'].mean(), 3),
        "avg_major_leak_prob": round(pipe_Pr['Major Leak'].mean(), 3),
        "pipes_major_leak_gt_05": int(
            (pipe_Pr['Major Leak'] > 0.5).sum()
        )
    }

    # =====================================================
    # Datos de red para mapas (Leaflet / ECharts)
    # =====================================================
    pipes_data = []
    for pipe_name in wn.pipe_name_list:
        pipe = wn.get_link(pipe_name)
        pipes_data.append({
            "id": pipe_name,
            "from": pipe.start_node_name,
            "to": pipe.end_node_name,
            "damage_state": pipe_damage_state.get(pipe_name, "None"),
            "repair_rate": float(RR.get(pipe_name, 0))
        })

    nodes_data = []
    for node_name in wn.junction_name_list:
        node = wn.get_node(node_name)
        nodes_data.append({
            "id": node_name,
            "x": node.coordinates[0],
            "y": node.coordinates[1],
            "pressure": float(pressure_24h.get(node_name, 0))
        })

    # =====================================================
    # Limpiar archivo temporal
    # =====================================================
    os.remove(inp_path)

    return {
        "summary": summary,
        "nodes": nodes_data,
        "pipes": pipes_data
    }
