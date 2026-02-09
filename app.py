from flask import Flask
from flask_cors import CORS
from routes.simulation import simulation_bp
from routes.export import export_bp

def create_app():
    app = Flask(__name__)

    CORS(
        app,
        resources={r"/api/*": {
            "origins": [
                "http://localhost:5173",
                "https://simulador.torvalsoft.com"
            ]
        }},
        supports_credentials=False
    )

    app.register_blueprint(simulation_bp, url_prefix="/api")
    app.register_blueprint(export_bp, url_prefix="/api")

    return app
