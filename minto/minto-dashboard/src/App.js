import React, { useState, useEffect } from 'react';
import axios from 'axios';

function App() {
  const [experiments, setExperiments] = useState([]);
  const [experimentName, setExperimentName] = useState('');
  const [experimentVersion, setExperimentVersion] = useState('');
  const [saveDir, setSaveDir] = useState('');

  useEffect(() => {
    // Load initial data if necessary
  }, []);

  const createExperiment = () => {
    axios.post('http://localhost:5000/create_experiment', {
      name: experimentName,
      version: experimentVersion,
      savedir: saveDir
    }).then(response => {
      alert(response.data.message);
      // Optionally reload experiments list
    }).catch(error => {
      alert(error.response.data.error);
    });
  };

  const runExperiment = (experimentKey) => {
    axios.post('http://localhost:5000/run_experiment', {
      experiment_key: experimentKey
    }).then(response => {
      alert(response.data.message);
      // Optionally reload experiments list
    }).catch(error => {
      alert(error.response.data.error);
    });
  };

  const logParameter = (experimentKey, parameterName, parameterValue) => {
    axios.post('http://localhost:5000/log_parameter', {
      experiment_key: experimentKey,
      parameter_name: parameterName,
      parameter_value: parameterValue
    }).then(response => {
      alert(response.data.message);
      // Optionally reload experiments list
    }).catch(error => {
      alert(error.response.data.error);
    });
  };

  const logResult = (experimentKey, resultName, resultValue) => {
    axios.post('http://localhost:5000/log_result', {
      experiment_key: experimentKey,
      result_name: resultName,
      result_value: resultValue
    }).then(response => {
      alert(response.data.message);
      // Optionally reload experiments list
    }).catch(error => {
      alert(error.response.data.error);
    });
  };

  const saveExperiment = (experimentKey, savePath) => {
    axios.post('http://localhost:5000/save_experiment', {
      experiment_key: experimentKey,
      save_path: savePath
    }).then(response => {
      alert(response.data.message);
      // Optionally reload experiments list
    }).catch(error => {
      alert(error.response.data.error);
    });
  };

  return (
    <div className="App">
      <h1>Minto Dashboard</h1>
      <div>
        <h2>Create Experiment</h2>
        <input type="text" placeholder="Experiment Name" value={experimentName} onChange={e => setExperimentName(e.target.value)} />
        <input type="text" placeholder="Experiment Version" value={experimentVersion} onChange={e => setExperimentVersion(e.target.value)} />
        <input type="text" placeholder="Save Directory" value={saveDir} onChange={e => setSaveDir(e.target.value)} />
        <button onClick={createExperiment}>Create</button>
      </div>
      {/* Add more UI components for other functionalities like running experiment, logging data, etc. */}
    </div>
  );
}

export default App;