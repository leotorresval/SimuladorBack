from flask import Blueprint, request, jsonify
from services.wntr_service import run_simulation

simulation_bp = Blueprint("simulation", __name__)

@simulation_bp.route("/simulate", methods=["POST"])
def simulate():
    try:
        inp_file = request.files.get("inp_file")
        magnitude = request.form.get("magnitude", type=float)
        depth = request.form.get("depth", type=float)

        if not inp_file or magnitude is None or depth is None:
            return jsonify({
                "error": "Debe enviar inp_file, magnitude y depth"
            }), 400

        result = run_simulation(inp_file, magnitude, depth)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
