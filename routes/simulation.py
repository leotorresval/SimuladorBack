from flask import Blueprint, request, jsonify
from services.wntr_service import run_simulation

simulation_bp = Blueprint("simulation", __name__)

@simulation_bp.route("/simulate", methods=["POST"])
def simulate():
    try:
        inp_file = request.files.get("inp_file")
        magnitude = request.form.get("magnitude", type=float)
        depth = request.form.get("depth", type=float)
        x = request.form.get("x", type=float)
        y = request.form.get("y", type=float)
        if not inp_file or magnitude is None or depth is None or x is None or y is None:
            return jsonify({
                "error": "Debe enviar inp_file, magnitude, depth, x y y en el formulario."
            }), 400

        result = run_simulation(inp_file, magnitude, depth,x,y)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
