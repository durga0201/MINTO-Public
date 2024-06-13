from flask import Flask, request, jsonify
from flask_cors import CORS
from minto import Experiment  # Assuming your Experiment class is in a file named experiment.py


app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Initialize a global experiment dictionary to store experiment instances
experiments = {}


@app.route('/create_experiment', methods=['POST'])
def create_experiment():
    data = request.json
    name = data.get('name')
    version = data.get('version')
    savedir = data.get('savedir')
    
    experiment = Experiment(name=name, version=version, savedir=savedir)
    experiment_key = f"{name}_v{version}"
    experiments[experiment_key] = experiment
    
    return jsonify({"message": f"Experiment {experiment_key} created successfully!"}), 201

@app.route('/run_experiment', methods=['POST'])
def run_experiment():
    data = request.json
    experiment_key = data.get('experiment_key')
    
    if experiment_key not in experiments:
        return jsonify({"error": "Experiment not found"}), 404
    
    experiment = experiments[experiment_key]
    experiment.run()
    
    return jsonify({"message": f"Experiment {experiment_key} run successfully!"}), 200

@app.route('/log_parameter', methods=['POST'])
def log_parameter():
    data = request.json
    experiment_key = data.get('experiment_key')
    parameter_name = data.get('parameter_name')
    parameter_value = data.get('parameter_value')
    
    if experiment_key not in experiments:
        return jsonify({"error": "Experiment not found"}), 404
    
    experiment = experiments[experiment_key]
    experiment.log_parameter(parameter_name, parameter_value)
    
    return jsonify({"message": "Parameter logged successfully!"}), 200

@app.route('/log_result', methods=['POST'])
def log_result():
    data = request.json
    experiment_key = data.get('experiment_key')
    result_name = data.get('result_name')
    result_value = data.get('result_value')
    
    if experiment_key not in experiments:
        return jsonify({"error": "Experiment not found"}), 404
    
    experiment = experiments[experiment_key]
    experiment.log_result(result_name, result_value)
    
    return jsonify({"message": "Result logged successfully!"}), 200

@app.route('/save_experiment', methods=['POST'])
def save_experiment():
    data = request.json
    experiment_key = data.get('experiment_key')
    save_path = data.get('save_path')
    
    if experiment_key not in experiments:
        return jsonify({"error": "Experiment not found"}), 404
    
    experiment = experiments[experiment_key]
    experiment.save(path=save_path)
    
    return jsonify({"message": "Experiment saved successfully!"}), 200

@app.route('/load_experiment', methods=['POST'])
def load_experiment():
    data = request.json
    load_path = data.get('load_path')
    version = data.get('version')
    
    experiment = Experiment().load(path=load_path, version=version)
    experiment_key = f"{experiment.name}_v{experiment.version}"
    experiments[experiment_key] = experiment
    
    return jsonify({"message": f"Experiment {experiment_key} loaded successfully!"}), 200

@app.route('/list_versions', methods=['POST'])
def list_versions():
    data = request.json
    experiment_name = data.get('experiment_name')
    savedir = data.get('savedir')
    
    experiment = Experiment(name=experiment_name, savedir=savedir)
    versions = experiment.list_versions()
    
    return jsonify({"versions": versions}), 200

@app.route('/delete_version', methods=['POST'])
def delete_version():
    data = request.json
    experiment_key = data.get('experiment_key')
    version = data.get('version')
    savedir = data.get('savedir')
    
    if experiment_key not in experiments:
        return jsonify({"error": "Experiment not found"}), 404
    
    experiment = experiments[experiment_key]
    experiment.delete_version(version=version, path=savedir)
    
    return jsonify({"message": "Version deleted successfully!"}), 200

if __name__ == '__main__':
    app.run(debug=True)