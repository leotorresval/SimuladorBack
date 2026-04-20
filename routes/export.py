from flask import Blueprint, request, send_file, abort
import pandas as pd
import tempfile
import os

import services.storage as storage

export_bp = Blueprint("export", __name__)

@export_bp.route("/export/<table>")
def export_table(table):

    simulation = storage.load_simulation()
    if simulation is None:
        abort(400, "No hay simulacion previa")

    if table not in simulation:
        abort(404, "Tabla no encontrada")

    data = simulation[table]

    if table == "summary":
        merged = {**simulation["summary"], **simulation["epicenter"]}
        data = [merged]

    df = pd.DataFrame(data)
    fmt = request.args.get("format", "csv")

    # =====================
    # CSV
    # =====================
    if fmt == "csv":
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        df.to_csv(tmp.name, index=False)
        tmp.close()

        return send_file(
            tmp.name,
            mimetype="text/csv",
            as_attachment=True,
            download_name=f"{table}.csv"
        )

    # =====================
    # XLSX
    # =====================
    if fmt == "xlsx":
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
        df.to_excel(tmp.name, index=False, engine="openpyxl")
        tmp.close()

        return send_file(
            tmp.name,
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            as_attachment=True,
            download_name=f"{table}.xlsx"
        )

    abort(400, "Formato no soportado")
