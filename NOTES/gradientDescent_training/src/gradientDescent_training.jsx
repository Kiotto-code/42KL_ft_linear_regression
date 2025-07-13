import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ScatterChart, Scatter } from 'recharts';

const GradientDescentTraining = () => {
  const [step, setStep] = useState(0);
  const [isTraining, setIsTraining] = useState(false);
  const [learningRate, setLearningRate] = useState(0.01);
  const [batchSize, setBatchSize] = useState(32);
  const [algorithm, setAlgorithm] = useState('sgd');
  const [lossHistory, setLossHistory] = useState([]);
  const [currentParams, setCurrentParams] = useState({ w: 0.5, b: 0.3 });
  const [gradients, setGradients] = useState({ w: 0, b: 0 });

  // Generate synthetic data for demonstration
  const generateData = (n = 100) => {
    const data = [];
    for (let i = 0; i < n; i++) {
      const x = Math.random() * 10;
      const y = 2 * x + 1 + (Math.random() - 0.5) * 2; // y = 2x + 1 + noise
      data.push({ x, y });
    }
    return data;
  };

  const [trainingData] = useState(generateData(100));

  // Calculate loss (Mean Squared Error)
  const calculateLoss = (params, data) => {
    const { w, b } = params;
    let totalLoss = 0;
    data.forEach(point => {
      const prediction = w * point.x + b;
      const error = prediction - point.y;
      totalLoss += error * error;
    });
    return totalLoss / data.length;
  };

  // Calculate gradients
  const calculateGradients = (params, data) => {
    const { w, b } = params;
    let dwSum = 0, dbSum = 0;
    
    data.forEach(point => {
      const prediction = w * point.x + b;
      const error = prediction - point.y;
      dwSum += 2 * error * point.x;
      dbSum += 2 * error;
    });

    return {
      w: dwSum / data.length,
      b: dbSum / data.length
    };
  };

  // Update parameters based on algorithm
  const updateParameters = (params, grads, lr, algorithm) => {
    switch (algorithm) {
      case 'sgd':
        return {
          w: params.w - lr * grads.w,
          b: params.b - lr * grads.b
        };
      case 'momentum':
        // Simplified momentum (would need velocity state in real implementation)
        return {
          w: params.w - lr * grads.w * 0.9,
          b: params.b - lr * grads.b * 0.9
        };
      default:
        return params;
    }
  };

  // Training step
  const trainStep = () => {
    // Get batch of data
    const batchData = trainingData.slice(0, Math.min(batchSize, trainingData.length));
    
    // Calculate gradients
    const grads = calculateGradients(currentParams, batchData);
    setGradients(grads);
    
    // Update parameters
    const newParams = updateParameters(currentParams, grads, learningRate, algorithm);
    setCurrentParams(newParams);
    
    // Calculate and record loss
    const loss = calculateLoss(newParams, trainingData);
    setLossHistory(prev => [...prev, { step: step + 1, loss }]);
    
    setStep(prev => prev + 1);
  };

  // Auto-training
  useEffect(() => {
    let interval;
    if (isTraining) {
      interval = setInterval(trainStep, 100);
    }
    return () => clearInterval(interval);
  }, [isTraining, currentParams, step, learningRate, batchSize, algorithm]);

  const resetTraining = () => {
    setStep(0);
    setIsTraining(false);
    setLossHistory([]);
    setCurrentParams({ w: 0.5, b: 0.3 });
    setGradients({ w: 0, b: 0 });
  };

  const currentLoss = lossHistory.length > 0 ? lossHistory[lossHistory.length - 1].loss : 0;

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gradient-to-br from-blue-50 to-indigo-100 min-h-screen">
      <div className="bg-white rounded-lg shadow-xl p-8">
        <h1 className="text-3xl font-bold text-center mb-8 text-gray-800">
          🎯 Gradient Descent Training Simulator
        </h1>
        
        {/* Controls */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          <div className="bg-gray-50 p-4 rounded-lg">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Learning Rate
            </label>
            <input
              type="range"
              min="0.001"
              max="0.1"
              step="0.001"
              value={learningRate}
              onChange={(e) => setLearningRate(parseFloat(e.target.value))}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
            />
            <span className="text-sm text-gray-600">{learningRate.toFixed(3)}</span>
          </div>
          
          <div className="bg-gray-50 p-4 rounded-lg">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Batch Size
            </label>
            <input
              type="range"
              min="1"
              max="100"
              value={batchSize}
              onChange={(e) => setBatchSize(parseInt(e.target.value))}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
            />
            <span className="text-sm text-gray-600">{batchSize}</span>
          </div>
          
          <div className="bg-gray-50 p-4 rounded-lg">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Algorithm
            </label>
            <select
              value={algorithm}
              onChange={(e) => setAlgorithm(e.target.value)}
              className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              <option value="sgd">SGD</option>
              <option value="momentum">Momentum</option>
            </select>
          </div>
          
          <div className="bg-gray-50 p-4 rounded-lg flex flex-col justify-center">
            <button
              onClick={() => setIsTraining(!isTraining)}
              className={`px-4 py-2 rounded-md font-medium transition-colors mb-2 ${
                isTraining 
                  ? 'bg-red-500 hover:bg-red-600 text-white' 
                  : 'bg-green-500 hover:bg-green-600 text-white'
              }`}
            >
              {isTraining ? 'Stop Training' : 'Start Training'}
            </button>
            <button
              onClick={resetTraining}
              className="px-4 py-2 bg-gray-500 hover:bg-gray-600 text-white rounded-md font-medium transition-colors"
            >
              Reset
            </button>
          </div>
        </div>

        {/* Current Status */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
          <div className="bg-blue-50 p-4 rounded-lg text-center">
            <div className="text-2xl font-bold text-blue-600">{step}</div>
            <div className="text-sm text-gray-600">Training Steps</div>
          </div>
          <div className="bg-green-50 p-4 rounded-lg text-center">
            <div className="text-2xl font-bold text-green-600">{currentLoss.toFixed(4)}</div>
            <div className="text-sm text-gray-600">Current Loss</div>
          </div>
          <div className="bg-purple-50 p-4 rounded-lg text-center">
            <div className="text-lg font-bold text-purple-600">
              w: {currentParams.w.toFixed(3)}, b: {currentParams.b.toFixed(3)}
            </div>
            <div className="text-sm text-gray-600">Parameters</div>
          </div>
          <div className="bg-orange-50 p-4 rounded-lg text-center">
            <div className="text-sm font-bold text-orange-600">
              ∇w: {gradients.w.toFixed(3)}, ∇b: {gradients.b.toFixed(3)}
            </div>
            <div className="text-sm text-gray-600">Gradients</div>
          </div>
        </div>

        {/* Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Loss Chart */}
          <div className="bg-white border rounded-lg p-4">
            <h3 className="text-lg font-semibold mb-4">Loss Over Time</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={lossHistory}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="step" />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="loss" stroke="#3b82f6" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Data Visualization */}
          <div className="bg-white border rounded-lg p-4">
            <h3 className="text-lg font-semibold mb-4">Data & Current Model</h3>
            <ResponsiveContainer width="100%" height={300}>
              <ScatterChart data={trainingData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="x" />
                <YAxis dataKey="y" />
                <Tooltip />
                <Scatter dataKey="y" fill="#94a3b8" />
                <Line 
                  type="monotone" 
                  dataKey={x => currentParams.w * x + currentParams.b}
                  stroke="#ef4444" 
                  strokeWidth={3}
                />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Training Process Explanation */}
        <div className="mt-8 bg-gray-50 p-6 rounded-lg">
          <h2 className="text-2xl font-bold mb-4 text-gray-800">🧠 How Gradient Descent Training Works</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h3 className="text-lg font-semibold mb-3 text-blue-600">1. The Training Loop</h3>
              <div className="space-y-2 text-sm">
                <div className="flex items-center space-x-2">
                  <span className="w-6 h-6 bg-blue-100 rounded-full flex items-center justify-center text-blue-600 font-bold">1</span>
                  <span>Forward Pass: Make predictions using current parameters</span>
                </div>
                <div className="flex items-center space-x-2">
                  <span className="w-6 h-6 bg-blue-100 rounded-full flex items-center justify-center text-blue-600 font-bold">2</span>
                  <span>Calculate Loss: Measure how wrong our predictions are</span>
                </div>
                <div className="flex items-center space-x-2">
                  <span className="w-6 h-6 bg-blue-100 rounded-full flex items-center justify-center text-blue-600 font-bold">3</span>
                  <span>Backward Pass: Calculate gradients of loss w.r.t. parameters</span>
                </div>
                <div className="flex items-center space-x-2">
                  <span className="w-6 h-6 bg-blue-100 rounded-full flex items-center justify-center text-blue-600 font-bold">4</span>
                  <span>Update Parameters: Move in direction opposite to gradient</span>
                </div>
              </div>
            </div>

            <div>
              <h3 className="text-lg font-semibold mb-3 text-green-600">2. Key Concepts</h3>
              <div className="space-y-2 text-sm">
                <div><strong>Learning Rate:</strong> Controls how big steps we take</div>
                <div><strong>Batch Size:</strong> How many samples we use per update</div>
                <div><strong>Gradient:</strong> Direction of steepest increase in loss</div>
                <div><strong>Convergence:</strong> When loss stops decreasing significantly</div>
              </div>
            </div>
          </div>

          <div className="mt-6 p-4 bg-white rounded-lg border-l-4 border-blue-500">
            <h4 className="font-semibold text-gray-800 mb-2">💡 Pro Tips:</h4>
            <ul className="text-sm space-y-1 text-gray-700">
              <li>• Start with a moderate learning rate (0.01-0.1)</li>
              <li>• Use larger batch sizes for more stable gradients</li>
              <li>• Monitor loss - it should generally decrease over time</li>
              <li>• If loss explodes, reduce learning rate</li>
              <li>• If loss plateaus, try increasing learning rate or changing algorithm</li>
            </ul>
          </div>
        </div>

        {/* Mathematical Formulation */}
        <div className="mt-8 bg-indigo-50 p-6 rounded-lg">
          <h2 className="text-2xl font-bold mb-4 text-indigo-800">📐 Mathematical Formulation</h2>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div>
              <h3 className="text-lg font-semibold mb-3 text-indigo-600">Update Rule</h3>
              <div className="bg-white p-4 rounded border font-mono text-sm">
                θ = θ - α * ∇J(θ)
              </div>
              <div className="text-sm mt-2 text-gray-600">
                Where θ are parameters, α is learning rate, ∇J(θ) is the gradient
              </div>
            </div>
            <div>
              <h3 className="text-lg font-semibold mb-3 text-indigo-600">Loss Function (MSE)</h3>
              <div className="bg-white p-4 rounded border font-mono text-sm">
                J(θ) = (1/2m) * Σ(h(x) - y)²
              </div>
              <div className="text-sm mt-2 text-gray-600">
                Where m is batch size, h(x) is prediction, y is true value
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default GradientDescentTraining;