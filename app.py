import os
import json
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from groq import Groq
from dotenv import load_dotenv
from ml_model import (
    train_models, load_models, predict,
    get_pd_curves, inverse_predict,
    TARGETS, UNITS, BN_MIN, BN_MAX, AO_MIN, AO_MAX
)

load_dotenv()

app = Flask(__name__)
CORS(app)
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Load pre-trained models on startup
_payload = None


def get_payload():
    return _payload


SYSTEM_PROMPT = """You are an expert materials science AI assistant for a composite material property prediction tool.

The composite material is: Carbon Fiber (20% fixed) + Boron Nitride (BN) + Aluminium Oxide (Al2O3).

Properties predicted:
- Tensile Strength (GPa): measures how much pulling force the material can withstand
- Young's Modulus (GPa): measures material stiffness / resistance to elastic deformation  
- Hardness (HV Vickers): measures resistance to surface deformation
- Buckling Strength (N/m): measures the material's ability to resist sudden structural instability or collapse when subjected to compressive loads.

ML Models used:
- GPR (Gaussian Process Regression): probabilistic model, trained on 9 experimental points, gives real uncertainty estimates
- GBM (Gradient Boosting Machine): ensemble of 60 decision trees, trained on 1000+ weighted points (90% experimental weight)
- Auto-selection: model with higher R² is chosen per property

Dataset: 9 experimental points (BN/AO at 2.5, 5.0, 7.5 wt%), 500 theory points, 500 FEA simulation points.
Experimental data carries 90% weight, theory+FEA carry 10%.

Valid input range: BN and AO both between 2.5% and 7.5%.

Be concise, technical but accessible. When users ask about predictions, explain the material science behind why those values make sense. Always mention uncertainty when GPR is used."""


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def status():
    return jsonify({"models_loaded": _payload is not None})


@app.route("/api/train", methods=["POST"])
def train():
    global _payload
    files = request.files
    if "theory" not in files or "fea" not in files or "exp" not in files:
        return jsonify({"error": "Upload theory, fea, and exp Excel files"}), 400

    os.makedirs("uploads", exist_ok=True)
    theory_path = "uploads/theory.xlsx"
    fea_path    = "uploads/fea.xlsx"
    exp_path    = "uploads/exp.xlsx"

    files["theory"].save(theory_path)
    files["fea"].save(fea_path)
    files["exp"].save(exp_path)

    try:
        _payload = train_models(theory_path, fea_path, exp_path)
        return jsonify({
            "success": True,
            "r2_scores": _payload["all_r2"],
            "exp_count": len(_payload["exp_data"]),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
def predict_route():
    if _payload is None:
        return jsonify({"error": "Models not trained yet. Upload datasets first."}), 400

    data = request.json
    bn   = float(data.get("bn", 5.0))
    ao   = float(data.get("ao", 5.0))

    if not (BN_MIN <= bn <= BN_MAX and AO_MIN <= ao <= AO_MAX):
        return jsonify({"error": f"Values must be between {BN_MIN} and {BN_MAX}"}), 400

    results = predict(bn, ao, _payload)
    curves  = get_pd_curves(bn, ao, _payload)

    return jsonify({
        "bn":      bn,
        "ao":      ao,
        "results": results,
        "curves":  curves,
        "exp_data": _payload["exp_data"],
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    data     = request.json
    messages = data.get("messages", [])
    context  = data.get("prediction_context", None)

    system = SYSTEM_PROMPT
    if context:
        system += f"\n\nCurrent prediction context:\nBN={context.get('bn')}%  AO={context.get('ao')}%\n"
        for prop, val in context.get("results", {}).items():
            system += f"  {prop}: {val['value']} {val['unit']} (model={val['model']}, R²={val['r2']})\n"

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": system}] + messages,
            max_tokens=600,
            temperature=0.7,
        )
        return jsonify({"reply": response.choices[0].message.content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/inverse_predict", methods=["POST"])
def inverse_predict_route():
    if _payload is None:
        return jsonify({"error": "Models not trained yet. Upload datasets first."}), 400

    data = request.json
    prop_keys = {
        "tensile":  "Tensile",
        "youngs":   "Youngs",
        "hardness": "Hardness",
        "buckling": "Buckling",
    }
    targets = {}
    weights = {}
    for key, prop in prop_keys.items():
        val = data.get(key)
        targets[prop] = float(val) if val not in (None, "", "null") else None
        w = data.get("weight_" + key)
        weights[prop] = float(w) if w not in (None, "", "null") else 1.0

    try:
        result = inverse_predict(targets, _payload, weights=weights, grid_size=150)
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
