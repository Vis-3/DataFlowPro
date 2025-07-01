import React, { useState, useEffect, useRef } from 'react';

// Main App component
const App = () => {
    // State variables for data, UI control, and user messages
    const [selectedFile, setSelectedFile] = useState(null);
    const [dataSummary, setDataSummary] = useState(null);
    const [headData, setHeadData] = useState(null);
    const [tailData, setTailData] = useState(null);
    const [descriptiveStats, setDescriptiveStats] = useState(null);
    const [outlierInfo, setOutlierInfo] = useState(null);
    const [skewnessInfo, setSkewnessInfo] = useState(null);
    const [correlationMatrixData, setCorrelationMatrixData] = useState(null);
    const [plots, setPlots] = useState({});
    const [columnDetails, setColumnDetails] = useState({});
    const [message, setMessage] = useState('');
    const [errorMessage, setErrorMessage] = useState('');
    const [loading, setLoading] = useState(false);
    const [aiInsights, setAiInsights] = useState('');
    const [aiLoading, setAiLoading] = useState(false);
    const [transformationHistory, setTransformationHistory] = useState([]);
    const [futureTransformations, setFutureTransformations] = useState([]);
    const [showExportCodeModal, setShowExportCodeModal] = useState(false);
    const [exportCode, setExportCode] = useState('');

    // State for Model Building and Evaluation
    const [targetVariable, setTargetVariable] = useState('');
    const [testSize, setTestSize] = useState(0.2);
    const [randomState, setRandomState] = useState(42);
    const [selectedModel, setSelectedModel] = useState('');
    const [modelMetrics, setModelMetrics] = useState(null);
    const [modelMessage, setModelMessage] = useState('');
    const [modelPlots, setModelPlots] = useState({}); // New state for model evaluation plots

    // State to manage current active tab in the main content area
    const [activeTab, setActiveTab] = useState('summary');

    // State for the notepad
    const [notepadContent, setNotepadContent] = useState(() => {
        // Initialize notepad content from localStorage if available
        return localStorage.getItem('notepadContent') || '';
    });

    // Ref for messages to auto-scroll
    const messagesEndRef = useRef(null);
    // Ref for file input to clear its value programmatically
    const fileInputRef = useRef(null);

    // Backend URL (replace with your actual backend URL if different)
    const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://127.0.0.1:5000';

    // Scroll to bottom of messages whenever they update
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [message, errorMessage, aiInsights, modelMessage]);

    // Save notepad content to localStorage whenever it changes
    useEffect(() => {
        localStorage.setItem('notepadContent', notepadContent);
    }, [notepadContent]);

    /**
     * Helper function to display a message (success or error)
     * @param {string} msg - The message text.
     * @param {boolean} isError - True if it's an error message, false for success.
     */
    const displayMessage = (msg, isError = false) => {
        if (isError) {
            setErrorMessage(msg);
            setMessage('');
        } else {
            setMessage(msg);
            setErrorMessage('');
        }
        setTimeout(() => {
            setMessage('');
            setErrorMessage('');
        }, 5000); // Clear messages after 5 seconds
    };

    /**
     * Handles file selection by the user.
     * @param {Event} event - The file input change event.
     */
    const handleFileChange = (event) => {
        setSelectedFile(event.target.files[0]);
    };

    /**
     * Resets all UI elements and data to their initial state.
     */
    const resetAll = () => {
        setSelectedFile(null);
        // Explicitly clear the file input's value
        if (fileInputRef.current) {
            fileInputRef.current.value = '';
        }
        setDataSummary(null);
        setHeadData(null);
        setTailData(null);
        setDescriptiveStats(null);
        setOutlierInfo(null);
        setSkewnessInfo(null);
        setCorrelationMatrixData(null);
        setPlots({});
        setColumnDetails({});
        setMessage('');
        setErrorMessage('');
        setAiInsights('');
        setTransformationHistory([]);
        setFutureTransformations([]);
        setTargetVariable('');
        setSelectedModel('');
        setModelMetrics(null);
        setModelMessage('');
        setModelPlots({}); // Reset model plots
        displayMessage('All data and transformations reset!', false);
    };

    /**
     * Sends the selected CSV file to the backend for upload and initial profiling.
     */
    const handleUpload = async () => {
        if (!selectedFile) {
            displayMessage('Please select a CSV file first.', true);
            return;
        }

        setLoading(true);
        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await fetch(`${BACKEND_URL}/upload_csv`, {
                method: 'POST',
                body: formData,
            });
            const result = await response.json();

            if (result.status === 'success') {
                setDataSummary(result.data_summary);
                setHeadData(JSON.parse(result.data_preview_head));
                setTailData(JSON.parse(result.data_preview_tail));
                setDescriptiveStats(result.descriptive_statistics);
                setOutlierInfo(result.outlier_info);
                setSkewnessInfo(result.skewness_info);
                setCorrelationMatrixData(result.correlation_matrix_data);
                setPlots(result.plots);
                setColumnDetails(result.column_details);
                setTransformationHistory([]); // Reset history on new upload
                setFutureTransformations([]);
                setAiInsights(''); // Clear previous AI insights
                setTargetVariable(''); // Reset model selection
                setSelectedModel('');
                setModelMetrics(null);
                setModelMessage('');
                setModelPlots({}); // Reset model plots
                displayMessage(result.message);
            } else {
                displayMessage(result.message, true);
            }
        } catch (error) {
            console.error('Error uploading file:', error);
            displayMessage(`Error uploading file: ${error.message}`, true);
        } finally {
            setLoading(false);
        }
    };

    /**
     * Generic function to send a transformation request to the backend.
     * @param {string} endpoint - The API endpoint for the transformation.
     * @param {object} payload - The data payload for the request.
     * @param {string} successMessage - Message to display on successful transformation.
     */
    const applyTransformation = async (endpoint, payload, successMessage) => {
        if (!dataSummary) {
            displayMessage('No data loaded. Please upload a CSV first.', true);
            return;
        }

        setLoading(true);
        try {
            const response = await fetch(`${BACKEND_URL}${endpoint}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            const result = await response.json();

            if (result.status === 'success') {
                // Save current state to history before updating
                setTransformationHistory(prev => [
                    ...prev,
                    {
                        dataSummary, headData, tailData, descriptiveStats,
                        outlierInfo, skewnessInfo, correlationMatrixData,
                        plots, columnDetails,
                        targetVariable, testSize, randomState, selectedModel, modelMetrics, modelMessage, modelPlots
                    }
                ]);
                setFutureTransformations([]); // Clear future history on new transformation

                setDataSummary(result.data_summary);
                setHeadData(JSON.parse(result.data_preview_head));
                setTailData(JSON.parse(result.data_preview_tail));
                setDescriptiveStats(result.descriptive_statistics);
                setOutlierInfo(result.outlier_info);
                setSkewnessInfo(result.skewness_info);
                setCorrelationMatrixData(result.correlation_matrix_data);
                setPlots(result.plots);
                setColumnDetails(result.column_details);
                setAiInsights(''); // Clear AI insights as data has changed
                setTargetVariable(''); // Reset model selection as columns might change
                setSelectedModel('');
                setModelMetrics(null);
                setModelMessage('');
                setModelPlots({}); // Reset model plots
                displayMessage(result.message || successMessage);
            } else {
                displayMessage(result.message, true);
            }
        } catch (error) {
            console.error(`Error applying transformation (${endpoint}):`, error);
            displayMessage(`Error applying transformation: ${error.message}`, true);
        } finally {
            setLoading(false);
        }
    };

    /**
     * Undoes the last transformation.
     */
    const undoTransformation = () => {
        if (transformationHistory.length > 0) {
            setFutureTransformations(prev => [
                {
                    dataSummary, headData, tailData, descriptiveStats,
                    outlierInfo, skewnessInfo, correlationMatrixData,
                    plots, columnDetails,
                    targetVariable, testSize, randomState, selectedModel, modelMetrics, modelMessage, modelPlots
                },
                ...prev
            ]);
            const previousState = transformationHistory.pop();
            setTransformationHistory([...transformationHistory]); // Update state to trigger re-render

            setDataSummary(previousState.dataSummary);
            setHeadData(previousState.headData);
            setTailData(previousState.tailData);
            setDescriptiveStats(previousState.descriptiveStats);
            setOutlierInfo(previousState.outlierInfo);
            setSkewnessInfo(previousState.skewnessInfo);
            setCorrelationMatrixData(previousState.correlationMatrixData);
            setPlots(previousState.plots);
            setColumnDetails(previousState.columnDetails);
            setTargetVariable(previousState.targetVariable);
            setTestSize(previousState.testSize);
            setRandomState(previousState.randomState);
            setSelectedModel(previousState.selectedModel);
            setModelMetrics(previousState.modelMetrics);
            setModelMessage(previousState.modelMessage);
            setModelPlots(previousState.modelPlots); // Restore model plots
            setAiInsights(''); // Clear AI insights as data has reverted

            displayMessage("Last transformation undone.");
        } else {
            displayMessage("No transformations to undo.", true);
        }
    };

    /**
     * Redoes the last undone transformation.
     */
    const redoTransformation = () => {
        if (futureTransformations.length > 0) {
            setTransformationHistory(prev => [
                ...prev,
                {
                    dataSummary, headData, tailData, descriptiveStats,
                    outlierInfo, skewnessInfo, correlationMatrixData,
                    plots, columnDetails,
                    targetVariable, testSize, randomState, selectedModel, modelMetrics, modelMessage, modelPlots
                }
            ]);
            const nextState = futureTransformations.shift();
            setFutureTransformations([...futureTransformations]); // Update state to trigger re-render

            setDataSummary(nextState.dataSummary);
            setHeadData(nextState.headData);
            setTailData(nextState.tailData);
            setDescriptiveStats(nextState.descriptiveStats);
            setOutlierInfo(nextState.outlierInfo);
            setSkewnessInfo(nextState.skewnessInfo);
            setCorrelationMatrixData(nextState.correlationMatrixData);
            setPlots(nextState.plots);
            setColumnDetails(nextState.columnDetails);
            setTargetVariable(nextState.targetVariable);
            setTestSize(nextState.testSize);
            setRandomState(nextState.randomState);
            setSelectedModel(nextState.selectedModel);
            setModelMetrics(nextState.modelMetrics);
            setModelMessage(nextState.modelMessage);
            setModelPlots(nextState.modelPlots); // Restore model plots
            setAiInsights(''); // Clear AI insights as data has changed

            displayMessage("Last transformation redone.");
        } else {
            displayMessage("No transformations to redo.", true);
        }
    };

    /**
     * Fetches AI-powered insights from the backend.
     */
    const getAiInsights = async () => {
        if (!dataSummary) {
            displayMessage('No data loaded. Please upload a CSV first.', true);
            return;
        }
        setAiLoading(true);
        setAiInsights(''); // Clear previous insights
        try {
            const response = await fetch(`${BACKEND_URL}/get_ai_insights`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({}), // No specific payload needed yet
            });
            const result = await response.json();
            if (result.status === 'success') {
                setAiInsights(result.ai_insights);
                displayMessage(result.message);
            } else {
                displayMessage(result.message, true);
            }
        } catch (error) {
            console.error('Error fetching AI insights:', error);
            displayMessage(`Error fetching AI insights: ${error.message}`, true);
        } finally {
            setAiLoading(false);
        }
    };

    /**
     * Handles downloading the processed CSV.
     */
    const handleDownloadCSV = async () => {
        if (!dataSummary) {
            displayMessage('No data available to download. Please upload a CSV first.', true);
            return;
        }
        setLoading(true);
        try {
            const response = await fetch(`${BACKEND_URL}/download_csv`, {
                method: 'GET',
            });
            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'processed_data.csv';
                document.body.appendChild(a);
                a.click();
                a.remove();
                window.URL.revokeObjectURL(url);
                displayMessage('Processed CSV downloaded successfully!');
            } else {
                const errorText = await response.text();
                displayMessage(`Error downloading CSV: ${errorText}`, true);
            }
        } catch (error) {
            console.error('Error downloading CSV:', error);
            displayMessage(`Error downloading CSV: ${error.message}`, true);
        } finally {
            setLoading(false);
        }
    };

    /**
     * Requests Python export code from the backend.
     */
    const handleExportCode = async () => {
        setLoading(true);
        try {
            // The backend needs to know the history of transformations applied
            const response = await fetch(`${BACKEND_URL}/export_code`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ transformations: transformationHistory }),
            });
            const result = await response.json();
            if (result.status === 'success') {
                setExportCode(result.code);
                setShowExportCodeModal(true);
                displayMessage(result.message);
            } else {
                displayMessage(result.message, true);
            }
        } catch (error) {
            console.error('Error exporting code:', error);
            displayMessage(`Error exporting code: ${error.message}`, true);
        } finally {
            setLoading(false);
        }
    };

    /**
     * Copies the exported code to the clipboard.
     */
    const copyExportedCode = () => {
        const textArea = document.createElement('textarea');
        textArea.value = exportCode;
        document.body.appendChild(textArea);
        textArea.select();
        try {
            document.execCommand('copy');
            displayMessage('Code copied to clipboard!');
        } catch (err) {
            console.error('Failed to copy text: ', err);
            displayMessage('Failed to copy code to clipboard.', true);
        }
        document.body.removeChild(textArea);
    };

    // --- Model Training Functions ---

    /**
     * Handles the training of the selected machine learning model.
     */
    const handleTrainModel = async () => {
        if (!dataSummary) {
            displayMessage('No data loaded. Please upload a CSV first.', true);
            return;
        }
        if (!targetVariable) {
            displayMessage('Please select a target variable.', true);
            return;
        }
        if (!selectedModel) {
            displayMessage('Please select a model.', true);
            return;
        }
        
        setLoading(true);
        setModelMetrics(null); // Clear previous metrics
        setModelMessage(''); // Clear previous message
        setModelPlots({}); // Clear previous plots

        const payload = {
            target_variable: targetVariable,
            test_size: parseFloat(testSize), // Ensure float
            random_state: parseInt(randomState), // Ensure int
            selected_model: selectedModel,
        };

        try {
            const response = await fetch(`${BACKEND_URL}/train_model`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            const result = await response.json();

            if (result.status === 'success') {
                setModelMetrics(result.model_metrics);
                setModelPlots(result.model_metrics.plots || {}); // Store plots
                setModelMessage(result.message);
                displayMessage(result.message);
            } else {
                setModelMetrics(null); // Clear metrics on error
                setModelPlots({}); // Clear plots on error
                setModelMessage(result.message);
                displayMessage(result.message, true);
            }
        } catch (error) {
            console.error('Error training model:', error);
            setModelMetrics(null);
            setModelPlots({});
            setModelMessage(`Error training model: ${error.message}`);
            displayMessage(`Error training model: ${error.message}`, true);
        } finally {
            setLoading(false);
        }
    };

    // Filter numerical columns for target variable selection in regression/classification
    const numericalColumns = dataSummary ? Object.keys(dataSummary.data_types).filter(
        col => dataSummary.data_types[col].includes('int') || dataSummary.data_types[col].includes('float')
    ) : [];

    // Filter all columns for target variable selection in clustering (as it uses all features)
    const allColumns = dataSummary ? Object.keys(dataSummary.data_types) : [];

    // Render UI
    return (
        <div className="min-h-screen bg-gray-100 font-sans text-gray-800 flex flex-col md:flex-row">
            {/* Sidebar */}
            <aside className="w-full md:w-64 bg-gray-800 text-white p-6 flex flex-col space-y-4">
                <h1 className="text-3xl font-bold mb-6 text-indigo-300">DataFlow Pro</h1>

                {/* File Upload Section */}
                <div className="mb-4">
                    <label className="block text-gray-300 text-sm font-medium mb-2" htmlFor="file-upload">
                        Upload CSV
                    </label>
                    <input
                        type="file"
                        id="file-upload"
                        accept=".csv"
                        onChange={handleFileChange}
                        ref={fileInputRef} // Attach ref here
                        className="block w-full text-sm text-gray-400
                                   file:mr-4 file:py-2 file:px-4
                                   file:rounded-full file:border-0
                                   file:text-sm file:font-semibold
                                   file:bg-indigo-500 file:text-white
                                   hover:file:bg-indigo-600 cursor-pointer"
                    />
                    {selectedFile && (
                        <p className="mt-2 text-sm text-gray-400">Selected: {selectedFile.name}</p>
                    )}
                    <button
                        onClick={handleUpload}
                        disabled={!selectedFile || loading}
                        className="mt-4 w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg shadow-md
                                   transition duration-300 ease-in-out disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        {loading ? 'Processing...' : 'Upload & Profile'}
                    </button>
                </div>

                {/* Notepad */}
                <div className="mb-4">
                    <label htmlFor="notepad" className="block text-gray-300 text-sm font-medium mb-2">
                        My Notes
                    </label>
                    <textarea
                        id="notepad"
                        value={notepadContent}
                        onChange={(e) => setNotepadContent(e.target.value)}
                        placeholder="Jot down your thoughts here..."
                        rows="6"
                        className="w-full p-2 rounded-lg bg-gray-700 text-white border border-gray-600 focus:outline-none focus:ring-1 focus:ring-indigo-500 resize-y"
                    ></textarea>
                    <p className="mt-1 text-xs text-gray-400">Notes are saved in your browser (local session).</p>
                </div>

                {/* Main Actions */}
                <div className="space-y-3">
                    <button
                        onClick={getAiInsights}
                        disabled={!dataSummary || aiLoading || loading}
                        className="w-full bg-purple-600 hover:bg-purple-700 text-white font-bold py-2 px-4 rounded-lg shadow-md
                                   transition duration-300 ease-in-out disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        {aiLoading ? 'Generating Insights...' : 'Get AI Insights'}
                    </button>
                    <button
                        onClick={undoTransformation}
                        disabled={transformationHistory.length === 0 || loading}
                        className="w-full bg-yellow-600 hover:bg-yellow-700 text-white font-bold py-2 px-4 rounded-lg shadow-md
                                   transition duration-300 ease-in-out disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        Undo Last
                    </button>
                    <button
                        onClick={redoTransformation}
                        disabled={futureTransformations.length === 0 || loading}
                        className="w-full bg-orange-600 hover:bg-orange-700 text-white font-bold py-2 px-4 rounded-lg shadow-md
                                   transition duration-300 ease-in-out disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        Redo Last
                    </button>
                    <button
                        onClick={resetAll}
                        disabled={!dataSummary && transformationHistory.length === 0 || loading}
                        className="w-full bg-red-600 hover:bg-red-700 text-white font-bold py-2 px-4 rounded-lg shadow-md
                                   transition duration-300 ease-in-out disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        Reset All
                    </button>
                    <button
                        onClick={handleDownloadCSV}
                        disabled={!dataSummary || loading}
                        className="w-full bg-green-600 hover:bg-green-700 text-white font-bold py-2 px-4 rounded-lg shadow-md
                                   transition duration-300 ease-in-out disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        Download Processed CSV
                    </button>
                    <button
                        onClick={handleExportCode}
                        disabled={transformationHistory.length === 0 || loading}
                        className="w-full bg-teal-600 hover:bg-teal-700 text-white font-bold py-2 px-4 rounded-lg shadow-md
                                   transition duration-300 ease-in-out disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        Export Python Code
                    </button>
                </div>
            </aside>

            {/* Main Content Area */}
            <main className="flex-1 p-8 bg-white overflow-auto">
                <h2 className="text-3xl font-bold text-gray-900 mb-6">Data Analysis & Preprocessing</h2>

                {/* Messages */}
                {message && (
                    <div className="bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded-lg relative mb-4" role="alert">
                        <span className="block sm:inline">{message}</span>
                    </div>
                )}
                {errorMessage && (
                    <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded-lg relative mb-4" role="alert">
                        <span className="block sm:inline">{errorMessage}</span>
                    </div>
                )}

                {/* Tab Navigation */}
                {dataSummary && (
                    <div className="mb-6 border-b border-gray-200">
                        <nav className="-mb-px flex space-x-8" aria-label="Tabs">
                            <button
                                onClick={() => setActiveTab('summary')}
                                className={`${activeTab === 'summary' ? 'border-indigo-500 text-indigo-600' : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'}
                                    whitespace-nowrap py-3 px-1 border-b-2 font-medium text-sm transition-colors duration-200`}
                            >
                                Data Summary
                            </button>
                            <button
                                onClick={() => setActiveTab('preprocessing')}
                                className={`${activeTab === 'preprocessing' ? 'border-indigo-500 text-indigo-600' : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'}
                                    whitespace-nowrap py-3 px-1 border-b-2 font-medium text-sm transition-colors duration-200`}
                            >
                                Preprocessing
                            </button>
                            <button
                                onClick={() => setActiveTab('feature_engineering')}
                                className={`${activeTab === 'feature_engineering' ? 'border-indigo-500 text-indigo-600' : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'}
                                    whitespace-nowrap py-3 px-1 border-b-2 font-medium text-sm transition-colors duration-200`}
                            >
                                Feature Engineering
                            </button>
                            <button
                                onClick={() => setActiveTab('model_building')}
                                className={`${activeTab === 'model_building' ? 'border-indigo-500 text-indigo-600' : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'}
                                    whitespace-nowrap py-3 px-1 border-b-2 font-medium text-sm transition-colors duration-200`}
                            >
                                Model Building & Evaluation
                            </button>
                        </nav>
                    </div>
                )}

                {/* Conditional Content based on activeTab */}
                {!dataSummary && (
                    <div className="text-center py-10 text-gray-500">
                        <p className="text-lg">Upload a CSV file to get started!</p>
                        <p className="text-md mt-2">Your data summary and preprocessing options will appear here.</p>
                    </div>
                )}

                {dataSummary && activeTab === 'summary' && (
                    <div className="space-y-8">
                        {/* Overall Data Summary */}
                        <section className="bg-gray-50 p-6 rounded-lg shadow-sm">
                            <h3 className="text-2xl font-semibold text-gray-800 mb-4">Overall Data Overview</h3>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-lg">
                                <p><strong>Rows:</strong> {dataSummary.num_rows}</p>
                                <p><strong>Columns:</strong> {dataSummary.num_cols}</p>
                            </div>
                        </section>

                        {/* Column Details: Data Types and Missing Values */}
                        <section className="bg-gray-50 p-6 rounded-lg shadow-sm">
                            <h3 className="text-2xl font-semibold text-gray-800 mb-4">Column Details: Data Types & Missing Values</h3>
                            <div className="overflow-x-auto">
                                <table className="min-w-full divide-y divide-gray-200">
                                    <thead className="bg-gray-200">
                                        <tr>
                                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-600 uppercase tracking-wider rounded-tl-lg">Column Name</th>
                                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-600 uppercase tracking-wider">Data Type</th>
                                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-600 uppercase tracking-wider">Missing Count</th>
                                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-600 uppercase tracking-wider rounded-tr-lg">Missing %</th>
                                        </tr>
                                    </thead>
                                    <tbody className="bg-white divide-y divide-gray-200">
                                        {Object.keys(dataSummary.data_types).map(colName => (
                                            <tr key={colName}>
                                                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{colName}</td>
                                                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-700">{dataSummary.data_types[colName]}</td>
                                                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-700">{dataSummary.missing_values[colName]}</td>
                                                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-700">{dataSummary.missing_percentages[colName]}%</td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </section>

                        {/* Descriptive Statistics */}
                        <section className="bg-gray-50 p-6 rounded-lg shadow-sm">
                            <h3 className="text-2xl font-semibold text-gray-800 mb-4">Descriptive Statistics (Numerical)</h3>
                            {descriptiveStats && (
                                <div className="overflow-x-auto">
                                    <table className="min-w-full divide-y divide-gray-200">
                                        <thead className="bg-gray-200">
                                            <tr>
                                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-600 uppercase tracking-wider rounded-tl-lg">Statistic</th>
                                                {descriptiveStats['count'] && Object.keys(descriptiveStats['count']).map(col => (
                                                    <th key={col} className="px-6 py-3 text-left text-xs font-medium text-gray-600 uppercase tracking-wider">{col}</th>
                                                ))}
                                            </tr>
                                        </thead>
                                        <tbody className="bg-white divide-y divide-gray-200">
                                            {Object.keys(descriptiveStats).map(statName => (
                                                <tr key={statName}>
                                                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{statName}</td>
                                                    {descriptiveStats[statName] && Object.values(descriptiveStats[statName]).map((val, idx) => (
                                                        <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-gray-700">
                                                            {val !== null ? (typeof val === 'number' ? val.toFixed(2) : val) : 'N/A'}
                                                        </td>
                                                    ))}
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            )}
                        </section>

                        {/* Outlier Information */}
                        <section className="bg-gray-50 p-6 rounded-lg shadow-sm">
                            <h3 className="text-2xl font-semibold text-gray-800 mb-4">Outlier Information (IQR Method)</h3>
                            {outlierInfo && Object.keys(outlierInfo).length > 0 ? (
                                <div className="overflow-x-auto">
                                    <table className="min-w-full divide-y divide-gray-200">
                                        <thead className="bg-gray-200">
                                            <tr>
                                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-600 uppercase tracking-wider rounded-tl-lg">Column</th>
                                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-600 uppercase tracking-wider">Outlier Count</th>
                                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-600 uppercase tracking-wider">Outlier Percentage</th>
                                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-600 uppercase tracking-wider">Lower Bound</th>
                                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-600 uppercase tracking-wider rounded-tr-lg">Upper Bound</th>
                                            </tr>
                                        </thead>
                                        <tbody className="bg-white divide-y divide-gray-200">
                                            {Object.keys(outlierInfo).map(col => outlierInfo[col] ? (
                                                <tr key={col}>
                                                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{col}</td>
                                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-700">{outlierInfo[col].count}</td>
                                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-700">{outlierInfo[col].percentage}%</td>
                                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-700">{outlierInfo[col].lower_bound}</td>
                                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-700">{outlierInfo[col].upper_bound}</td>
                                                </tr>
                                            ) : null)}
                                        </tbody>
                                    </table>
                                </div>
                            ) : (
                                <p className="text-gray-600">No outlier information available (might be no numerical columns).</p>
                            )}
                        </section>

                        {/* Skewness Information */}
                        <section className="bg-gray-50 p-6 rounded-lg shadow-sm">
                            <h3 className="text-2xl font-semibold text-gray-800 mb-4">Skewness Information (Numerical Features)</h3>
                            {skewnessInfo && Object.keys(skewnessInfo).length > 0 ? (
                                <div className="overflow-x-auto">
                                    <table className="min-w-full divide-y divide-gray-200">
                                        <thead className="bg-gray-200">
                                            <tr>
                                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-600 uppercase tracking-wider rounded-tl-lg">Column</th>
                                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-600 uppercase tracking-wider rounded-tr-lg">Skewness</th>
                                            </tr>
                                        </thead>
                                        <tbody className="bg-white divide-y divide-gray-200">
                                            {Object.keys(skewnessInfo).map(col => (
                                                <tr key={col}>
                                                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{col}</td>
                                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-700">
                                                        {skewnessInfo[col] !== null ? skewnessInfo[col].toFixed(2) : 'N/A'}
                                                    </td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            ) : (
                                <p className="text-gray-600">No skewness information available (might be no numerical columns).</p>
                            )}
                        </section>

                        {/* Data Preview */}
                        <section className="bg-gray-50 p-6 rounded-lg shadow-sm">
                            <h3 className="text-2xl font-semibold text-gray-800 mb-4">Data Preview (Head & Tail)</h3>
                            <div className="mb-6">
                                <h4 className="text-xl font-medium text-gray-700 mb-2">Head (First 5 Rows)</h4>
                                {headData && headData.columns && headData.data ? (
                                    <div className="overflow-x-auto">
                                        <table className="min-w-full divide-y divide-gray-200">
                                            <thead className="bg-gray-200">
                                                <tr>
                                                    {headData.columns.map(col => (
                                                        <th key={col} className="px-6 py-3 text-left text-xs font-medium text-gray-600 uppercase tracking-wider rounded-tl-lg rounded-tr-lg">{col}</th>
                                                    ))}
                                                </tr>
                                            </thead>
                                            <tbody className="bg-white divide-y divide-gray-200">
                                                {headData.data.map((row, rowIndex) => (
                                                    <tr key={rowIndex}>
                                                        {row.map((cell, cellIndex) => (
                                                            <td key={cellIndex} className="px-6 py-4 whitespace-nowrap text-sm text-gray-700">{cell === null ? 'NaN' : cell}</td>
                                                        ))}
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                    </div>
                                ) : <p className="text-gray-600">No head data available.</p>}
                            </div>

                            <div>
                                <h4 className="text-xl font-medium text-gray-700 mb-2">Tail (Last 5 Rows)</h4>
                                {tailData && tailData.columns && tailData.data ? (
                                    <div className="overflow-x-auto">
                                        <table className="min-w-full divide-y divide-gray-200">
                                            <thead className="bg-gray-200">
                                                <tr>
                                                    {tailData.columns.map(col => (
                                                        <th key={col} className="px-6 py-3 text-left text-xs font-medium text-gray-600 uppercase tracking-wider rounded-tl-lg rounded-tr-lg">{col}</th>
                                                    ))}
                                                </tr>
                                            </thead>
                                            <tbody className="bg-white divide-y divide-gray-200">
                                                {tailData.data.map((row, rowIndex) => (
                                                    <tr key={rowIndex}>
                                                        {row.map((cell, cellIndex) => (
                                                            <td key={cellIndex} className="px-6 py-4 whitespace-nowrap text-sm text-gray-700">{cell === null ? 'NaN' : cell}</td>
                                                        ))}
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                    </div>
                                ) : <p className="text-gray-600">No tail data available.</p>}
                            </div>
                        </section>

                        {/* Visualizations */}
                        <section className="bg-gray-50 p-6 rounded-lg shadow-sm">
                            <h3 className="text-2xl font-semibold text-gray-800 mb-4">Visualizations</h3>
                            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                                {Object.keys(plots).length > 0 ? (
                                    Object.keys(plots).map(plotName => (
                                        plots[plotName] ? (
                                            <div key={plotName} className="border border-gray-200 rounded-lg overflow-hidden shadow-sm">
                                                <img
                                                    src={`data:image/png;base64,${plots[plotName]}`}
                                                    alt={plotName.replace(/_/g, ' ').toUpperCase()}
                                                    className="w-full h-auto object-contain"
                                                />
                                            </div>
                                        ) : null
                                    ))
                                ) : (
                                    <p className="text-gray-600 col-span-full">No plots available (e.g., no numerical columns for histograms/boxplots, or not enough for correlation heatmap).</p>
                                )}
                            </div>
                        </section>

                        {/* AI Insights Section */}
                        <section className="bg-indigo-50 p-6 rounded-lg shadow-sm">
                            <h3 className="text-2xl font-semibold text-indigo-800 mb-4">AI Insights & Recommendations</h3>
                            {aiLoading ? (
                                <p className="text-indigo-600">Loading AI insights, please wait...</p>
                            ) : aiInsights ? (
                                <div className="prose max-w-none" dangerouslySetInnerHTML={{ __html: aiInsights.replace(/\n/g, '<br/>') }} />
                            ) : (
                                <p className="text-indigo-600">Click "Get AI Insights" to generate a comprehensive data analysis report.</p>
                            )}
                        </section>
                    </div>
                )}

                {dataSummary && activeTab === 'preprocessing' && (
                    <div className="space-y-8">
                        <h3 className="text-2xl font-semibold text-gray-800 mb-4">Preprocessing Steps</h3>

                        {/* Impute Missing Values */}
                        <section className="bg-white p-6 rounded-lg shadow-sm border border-blue-200">
                            <h4 className="text-xl font-medium text-gray-700 mb-4">Impute Missing Values</h4>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                <div>
                                    <label htmlFor="imputeColumn" className="block text-sm font-medium text-gray-700">Column to Impute:</label>
                                    <select
                                        id="imputeColumn"
                                        className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
                                        defaultValue=""
                                    >
                                        <option value="">Select Column or All</option>
                                        <option value="all_numerical">All Numerical Columns</option>
                                        <option value="all_categorical">All Categorical Columns</option>
                                        {dataSummary && Object.keys(dataSummary.data_types).map(col => (
                                            <option key={col} value={col}>{col} ({dataSummary.data_types[col]})</option>
                                        ))}
                                    </select>
                                </div>
                                <div>
                                    <label htmlFor="imputeStrategy" className="block text-sm font-medium text-gray-700">Imputation Strategy:</label>
                                    <select
                                        id="imputeStrategy"
                                        className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
                                        defaultValue=""
                                    >
                                        <option value="">Select Strategy</option>
                                        <option value="mean">Mean (Numerical)</option>
                                        <option value="median">Median (Numerical)</option>
                                        <option value="mode">Mode (Numerical & Categorical)</option>
                                        <option value="knn">KNN (Numerical)</option>
                                        <option value="mice">MICE (Numerical)</option>
                                    </select>
                                </div>
                                <div>
                                    <label htmlFor="knnNeighbors" className="block text-sm font-medium text-gray-700">K-Neighbors (for KNN):</label>
                                    <input
                                        type="number"
                                        id="knnNeighbors"
                                        defaultValue="5"
                                        min="1"
                                        className="mt-1 block w-full pl-3 pr-3 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
                                    />
                                </div>
                                <div className="md:col-span-2">
                                    <button
                                        onClick={() => {
                                            const column = document.getElementById('imputeColumn').value;
                                            const strategy = document.getElementById('imputeStrategy').value;
                                            const k_neighbors = document.getElementById('knnNeighbors').value;
                                            applyTransformation('/preprocess/impute_missing', { column, strategy, k_neighbors }, `Imputing missing values with ${strategy}...`);
                                        }}
                                        disabled={loading || !dataSummary}
                                        className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-2 px-4 rounded-lg shadow-md transition duration-300 ease-in-out disabled:opacity-50 disabled:cursor-not-allowed"
                                    >
                                        Apply Imputation
                                    </button>
                                </div>
                            </div>
                        </section>

                        {/* Remove Missing Data */}
                        <section className="bg-white p-6 rounded-lg shadow-sm border border-blue-200">
                            <h4 className="text-xl font-medium text-gray-700 mb-4">Remove Missing Data</h4>
                            <div className="flex flex-col md:flex-row space-y-2 md:space-y-0 md:space-x-4">
                                <button
                                    onClick={() => applyTransformation('/preprocess/remove_missing_rows', {}, 'Removing rows with missing values...')}
                                    disabled={loading || !dataSummary}
                                    className="flex-1 bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-2 px-4 rounded-lg shadow-md transition duration-300 ease-in-out disabled:opacity-50 disabled:cursor-not-allowed"
                                >
                                    Remove Rows with Missing
                                </button>
                                <button
                                    onClick={() => applyTransformation('/preprocess/remove_missing_cols', {}, 'Removing columns with missing values...')}
                                    disabled={loading || !dataSummary}
                                    className="flex-1 bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-2 px-4 rounded-lg shadow-md transition duration-300 ease-in-out disabled:opacity-50 disabled:cursor-not-allowed"
                                >
                                    Remove Columns with Missing
                                </button>
                            </div>
                        </section>

                        {/* Remove Duplicates */}
                        <section className="bg-white p-6 rounded-lg shadow-sm border border-blue-200">
                            <h4 className="text-xl font-medium text-gray-700 mb-4">Remove Duplicate Rows</h4>
                            <button
                                onClick={() => applyTransformation('/preprocess/remove_duplicates', {}, 'Removing duplicate rows...')}
                                disabled={loading || !dataSummary}
                                className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-2 px-4 rounded-lg shadow-md transition duration-300 ease-in-out disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                                Remove Duplicates
                            </button>
                        </section>

                        {/* Drop Column */}
                        <section className="bg-white p-6 rounded-lg shadow-sm border border-blue-200">
                            <h4 className="text-xl font-medium text-gray-700 mb-4">Drop Column</h4>
                            <div className="grid grid-cols-1 gap-4">
                                <div>
                                    <label htmlFor="dropColumnName" className="block text-sm font-medium text-gray-700">Column to Drop:</label>
                                    <select
                                        id="dropColumnName"
                                        className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
                                        defaultValue=""
                                    >
                                        <option value="">Select Column</option>
                                        {dataSummary && Object.keys(dataSummary.data_types).map(col => (
                                            <option key={col} value={col}>{col}</option>
                                        ))}
                                    </select>
                                </div>
                                <div>
                                    <button
                                        onClick={() => {
                                            const column_name = document.getElementById('dropColumnName').value;
                                            applyTransformation('/data/drop_column', { column_name }, `Dropping column ${column_name}...`);
                                        }}
                                        disabled={loading || !dataSummary}
                                        className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-2 px-4 rounded-lg shadow-md transition duration-300 ease-in-out disabled:opacity-50 disabled:cursor-not-allowed"
                                    >
                                        Drop Selected Column
                                    </button>
                                </div>
                            </div>
                        </section>

                        {/* Scale Features */}
                        <section className="bg-white p-6 rounded-lg shadow-sm border border-blue-200">
                            <h4 className="text-xl font-medium text-gray-700 mb-4">Scale Numerical Features</h4>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                <div>
                                    <label htmlFor="scaleColumns" className="block text-sm font-medium text-gray-700">Columns to Scale (Numerical):</label>
                                    <select
                                        id="scaleColumns"
                                        multiple
                                        className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md h-32"
                                    >
                                        {dataSummary && Object.keys(dataSummary.data_types)
                                            .filter(col => dataSummary.data_types[col].includes('int') || dataSummary.data_types[col].includes('float'))
                                            .map(col => (
                                                <option key={col} value={col}>{col}</option>
                                            ))}
                                    </select>
                                    <p className="mt-1 text-xs text-gray-500">Hold Ctrl/Cmd to select multiple.</p>
                                </div>
                                <div>
                                    <label htmlFor="scalerType" className="block text-sm font-medium text-gray-700">Scaler Type:</label>
                                    <select
                                        id="scalerType"
                                        className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
                                        defaultValue=""
                                    >
                                        <option value="">Select Scaler</option>
                                        <option value="standard">StandardScaler</option>
                                        <option value="minmax">MinMaxScaler</option>
                                        <option value="robust">RobustScaler</option>
                                    </select>
                                </div>
                                <div className="md:col-span-2">
                                    <button
                                        onClick={() => {
                                            const selectElement = document.getElementById('scaleColumns');
                                            const columns = Array.from(selectElement.options)
                                                .filter(option => option.selected)
                                                .map(option => option.value);
                                            const scaler_type = document.getElementById('scalerType').value;
                                            applyTransformation('/transform/scale_features', { columns, scaler_type }, `Scaling features with ${scaler_type}...`);
                                        }}
                                        disabled={loading || !dataSummary}
                                        className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-2 px-4 rounded-lg shadow-md transition duration-300 ease-in-out disabled:opacity-50 disabled:cursor-not-allowed"
                                    >
                                        Apply Scaling
                                    </button>
                                </div>
                            </div>
                        </section>

                        {/* Apply Skewness Transformation */}
                        <section className="bg-white p-6 rounded-lg shadow-sm border border-blue-200">
                            <h4 className="text-xl font-medium text-gray-700 mb-4">Apply Skewness Transformation</h4>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                <div>
                                    <label htmlFor="skewTransformColumn" className="block text-sm font-medium text-gray-700">Column to Transform (Numerical):</label>
                                    <select
                                        id="skewTransformColumn"
                                        className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
                                        defaultValue=""
                                    >
                                        <option value="">Select Column</option>
                                        {dataSummary && Object.keys(dataSummary.data_types)
                                            .filter(col => dataSummary.data_types[col].includes('int') || dataSummary.data_types[col].includes('float'))
                                            .map(col => (
                                                <option key={col} value={col}>{col}</option>
                                            ))}
                                    </select>
                                </div>
                                <div>
                                    <label htmlFor="transformType" className="block text-sm font-medium text-gray-700">Transformation Type:</label>
                                    <select
                                        id="transformType"
                                        className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
                                        defaultValue=""
                                    >
                                        <option value="">Select Type</option>
                                        <option value="log">Log (x + 1)</option>
                                        <option value="sqrt">Square Root</option>
                                        <option value="boxcox">Box-Cox (requires x &gt; 0)</option>
                                        <option value="yeo-johnson">Yeo-Johnson</option>
                                    </select>
                                </div>
                                <div className="md:col-span-2">
                                    <button
                                        onClick={() => {
                                            const column_name = document.getElementById('skewTransformColumn').value;
                                            const transform_type = document.getElementById('transformType').value;
                                            applyTransformation('/transform/apply_skew_transform', { column_name, transform_type }, `Applying ${transform_type} transformation to ${column_name}...`);
                                        }}
                                        disabled={loading || !dataSummary}
                                        className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-2 px-4 rounded-lg shadow-md transition duration-300 ease-in-out disabled:opacity-50 disabled:cursor-not-allowed"
                                    >
                                        Apply Transformation
                                    </button>
                                </div>
                            </div>
                        </section>

                        {/* Handle Outliers */}
                        <section className="bg-white p-6 rounded-lg shadow-sm border border-blue-200">
                            <h4 className="text-xl font-medium text-gray-700 mb-4">Handle Outliers</h4>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                <div>
                                    <label htmlFor="outlierColumn" className="block text-sm font-medium text-gray-700">Column to Handle (Numerical):</label>
                                    <select
                                        id="outlierColumn"
                                        className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
                                        defaultValue=""
                                    >
                                        <option value="">Select Column</option>
                                        {dataSummary && Object.keys(dataSummary.data_types)
                                            .filter(col => dataSummary.data_types[col].includes('int') || dataSummary.data_types[col].includes('float'))
                                            .map(col => (
                                                <option key={col} value={col}>{col}</option>
                                            ))}
                                    </select>
                                </div>
                                <div>
                                    <label htmlFor="outlierMethod" className="block text-sm font-medium text-gray-700">Method:</label>
                                    <select
                                        id="outlierMethod"
                                        className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
                                        defaultValue="winsorize"
                                    >
                                        <option value="winsorize">Winsorize (Cap)</option>
                                        <option value="trim">Trim (Remove)</option>
                                    </select>
                                </div>
                                <div>
                                    <label htmlFor="lowerQuantile" className="block text-sm font-medium text-gray-700">Lower Bound Quantile (e.g., 0.01):</label>
                                    <input
                                        type="number"
                                        id="lowerQuantile"
                                        step="0.01"
                                        min="0"
                                        max="1"
                                        defaultValue="0.01"
                                        className="mt-1 block w-full pl-3 pr-3 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
                                    />
                                </div>
                                <div>
                                    <label htmlFor="upperQuantile" className="block text-sm font-medium text-gray-700">Upper Bound Quantile (e.g., 0.99):</label>
                                    <input
                                        type="number"
                                        id="upperQuantile"
                                        step="0.01"
                                        min="0"
                                        max="1"
                                        defaultValue="0.99"
                                        className="mt-1 block w-full pl-3 pr-3 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
                                    />
                                </div>
                                <div className="md:col-span-2">
                                    <button
                                        onClick={() => {
                                            const column_name = document.getElementById('outlierColumn').value;
                                            const method = document.getElementById('outlierMethod').value;
                                            const lower_bound_quantile = parseFloat(document.getElementById('lowerQuantile').value);
                                            const upper_bound_quantile = parseFloat(document.getElementById('upperQuantile').value);
                                            applyTransformation('/transform/handle_outliers', { column_name, method, lower_bound_quantile, upper_bound_quantile }, `Handling outliers in ${column_name} with ${method} method...`);
                                        }}
                                        disabled={loading || !dataSummary}
                                        className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-2 px-4 rounded-lg shadow-md transition duration-300 ease-in-out disabled:opacity-50 disabled:cursor-not-allowed"
                                    >
                                        Apply Outlier Handling
                                    </button>
                                </div>
                            </div>
                        </section>
                    </div>
                )}

                {dataSummary && activeTab === 'feature_engineering' && (
                    <div className="space-y-8">
                        <h3 className="text-2xl font-semibold text-gray-800 mb-4">Feature Engineering Steps</h3>

                        {/* One-Hot Encode */}
                        <section className="bg-white p-6 rounded-lg shadow-sm border border-green-200">
                            <h4 className="text-xl font-medium text-gray-700 mb-4">One-Hot Encode Categorical Column</h4>
                            <div className="grid grid-cols-1 gap-4">
                                <div>
                                    <label htmlFor="oneHotColumn" className="block text-sm font-medium text-gray-700">Column to Encode (Categorical):</label>
                                    <select
                                        id="oneHotColumn"
                                        className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
                                        defaultValue=""
                                    >
                                        <option value="">Select Column</option>
                                        {dataSummary && Object.keys(dataSummary.data_types)
                                            .filter(col => dataSummary.data_types[col].includes('object') || dataSummary.data_types[col].includes('category'))
                                            .map(col => (
                                                <option key={col} value={col}>{col}</option>
                                            ))}
                                    </select>
                                </div>
                                <div>
                                    <button
                                        onClick={() => {
                                            const column_name = document.getElementById('oneHotColumn').value;
                                            applyTransformation('/transform/one_hot_encode', { column_name }, `One-hot encoding ${column_name}...`);
                                        }}
                                        disabled={loading || !dataSummary}
                                        className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-2 px-4 rounded-lg shadow-md transition duration-300 ease-in-out disabled:opacity-50 disabled:cursor-not-allowed"
                                    >
                                        Apply One-Hot Encoding
                                    </button>
                                </div>
                            </div>
                        </section>

                        {/* Frequency Encode */}
                        <section className="bg-white p-6 rounded-lg shadow-sm border border-green-200">
                            <h4 className="text-xl font-medium text-gray-700 mb-4">Frequency Encode Categorical Column</h4>
                            <div className="grid grid-cols-1 gap-4">
                                <div>
                                    <label htmlFor="freqEncodeColumn" className="block text-sm font-medium text-gray-700">Column to Encode (Categorical):</label>
                                    <select
                                        id="freqEncodeColumn"
                                        className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
                                        defaultValue=""
                                    >
                                        <option value="">Select Column</option>
                                        {dataSummary && Object.keys(dataSummary.data_types)
                                            .filter(col => dataSummary.data_types[col].includes('object') || dataSummary.data_types[col].includes('category'))
                                            .map(col => (
                                                <option key={col} value={col}>{col}</option>
                                            ))}
                                    </select>
                                </div>
                                <div>
                                    <button
                                        onClick={() => {
                                            const column_name = document.getElementById('freqEncodeColumn').value;
                                            applyTransformation('/transform/frequency_encode', { column_name }, `Frequency encoding ${column_name}...`);
                                        }}
                                        disabled={loading || !dataSummary}
                                        className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-2 px-4 rounded-lg shadow-lg transition duration-300 ease-in-out disabled:opacity-50 disabled:cursor-not-allowed"
                                    >
                                        Apply Frequency Encoding
                                    </button>
                                </div>
                            </div>
                        </section>

                        {/* Create Interaction Feature */}
                        <section className="bg-white p-6 rounded-lg shadow-sm border border-green-200">
                            <h4 className="text-xl font-medium text-gray-700 mb-4">Create Interaction Feature (Col1 * Col2)</h4>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                <div>
                                    <label htmlFor="interactionCol1" className="block text-sm font-medium text-gray-700">Column 1 (Numerical):</label>
                                    <select
                                        id="interactionCol1"
                                        className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
                                        defaultValue=""
                                    >
                                        <option value="">Select Column</option>
                                        {numericalColumns.map(col => (
                                            <option key={col} value={col}>{col}</option>
                                        ))}
                                    </select>
                                </div>
                                <div>
                                    <label htmlFor="interactionCol2" className="block text-sm font-medium text-gray-700">Column 2 (Numerical):</label>
                                    <select
                                        id="interactionCol2"
                                        className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
                                        defaultValue=""
                                    >
                                        <option value="">Select Column</option>
                                        {numericalColumns.map(col => (
                                            <option key={col} value={col}>{col}</option>
                                        ))}
                                    </select>
                                </div>
                                <div className="md:col-span-2">
                                    <button
                                        onClick={() => {
                                            const column1 = document.getElementById('interactionCol1').value;
                                            const column2 = document.getElementById('interactionCol2').value;
                                            applyTransformation('/feature_engineer/create_interaction', { column1, column2 }, `Creating interaction feature ${column1}_x_${column2}...`);
                                        }}
                                        disabled={loading || !dataSummary}
                                        className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-2 px-4 rounded-lg shadow-md transition duration-300 ease-in-out disabled:opacity-50 disabled:cursor-not-allowed"
                                    >
                                        Create Interaction Feature
                                    </button>
                                </div>
                            </div>
                        </section>

                        {/* Create Ratio Feature */}
                        <section className="bg-white p-6 rounded-lg shadow-sm border border-green-200">
                            <h4 className="text-xl font-medium text-gray-700 mb-4">Create Ratio Feature (Num / Den)</h4>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                <div>
                                    <label htmlFor="numeratorColumn" className="block text-sm font-medium text-gray-700">Numerator Column (Numerical):</label>
                                    <select
                                        id="numeratorColumn"
                                        className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
                                        defaultValue=""
                                    >
                                        <option value="">Select Column</option>
                                        {numericalColumns.map(col => (
                                            <option key={col} value={col}>{col}</option>
                                        ))}
                                    </select>
                                </div>
                                <div>
                                    <label htmlFor="denominatorColumn" className="block text-sm font-medium text-gray-700">Denominator Column (Numerical):</label>
                                    <select
                                        id="denominatorColumn"
                                        className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
                                        defaultValue=""
                                    >
                                        <option value="">Select Column</option>
                                        {numericalColumns.map(col => (
                                            <option key={col} value={col}>{col}</option>
                                        ))}
                                    </select>
                                </div>
                                <div className="md:col-span-2">
                                    <button
                                        onClick={() => {
                                            const numerator_column = document.getElementById('numeratorColumn').value;
                                            const denominator_column = document.getElementById('denominatorColumn').value;
                                            applyTransformation('/feature_engineer/create_ratio', { numerator_column, denominator_column }, `Creating ratio feature ${numerator_column}_div_${denominator_column}...`);
                                        }}
                                        disabled={loading || !dataSummary}
                                        className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-2 px-4 rounded-lg shadow-md transition duration-300 ease-in-out disabled:opacity-50 disabled:cursor-not-allowed"
                                    >
                                        Create Ratio Feature
                                    </button>
                                </div>
                            </div>
                        </section>

                        {/* Create Lagged Feature */}
                        <section className="bg-white p-6 rounded-lg shadow-sm border border-green-200">
                            <h4 className="text-xl font-medium text-gray-700 mb-4">Create Lagged Feature (Time Series)</h4>
                            <p className="text-sm text-gray-600 mb-4">Requires a numerical feature, a grouping column (e.g., 'Country Name'), and a time column (e.g., 'Year').</p>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                <div>
                                    <label htmlFor="lagColumn" className="block text-sm font-medium text-gray-700">Feature Column (Numerical):</label>
                                    <select
                                        id="lagColumn"
                                        className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
                                        defaultValue=""
                                    >
                                        <option value="">Select Column</option>
                                        {numericalColumns.map(col => (
                                            <option key={col} value={col}>{col}</option>
                                        ))}
                                    </select>
                                </div>
                                <div>
                                    <label htmlFor="groupColumn" className="block text-sm font-medium text-gray-700">Group Column (e.g., Country):</label>
                                    <select
                                        id="groupColumn"
                                        className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
                                        defaultValue=""
                                    >
                                        <option value="">Select Column</option>
                                        {dataSummary && Object.keys(dataSummary.data_types).map(col => (
                                            <option key={col} value={col}>{col}</option>
                                        ))}
                                    </select>
                                </div>
                                <div>
                                    <label htmlFor="timeColumn" className="block text-sm font-medium text-gray-700">Time Column (Numerical or Date):</label>
                                    <select
                                        id="timeColumn"
                                        className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
                                        defaultValue=""
                                    >
                                        <option value="">Select Column</option>
                                        {dataSummary && Object.keys(dataSummary.data_types).map(col => (
                                            <option key={col} value={col}>{col}</option>
                                        ))}
                                    </select>
                                </div>
                                <div>
                                    <label htmlFor="lagPeriods" className="block text-sm font-medium text-gray-700">Lag Periods (e.g., 1 for previous year):</label>
                                    <input
                                        type="number"
                                        id="lagPeriods"
                                        min="1"
                                        defaultValue="1"
                                        className="mt-1 block w-full pl-3 pr-3 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
                                    />
                                </div>
                                <div className="md:col-span-2">
                                    <button
                                        onClick={() => {
                                            const column_name = document.getElementById('lagColumn').value;
                                            const group_column = document.getElementById('groupColumn').value;
                                            const time_column = document.getElementById('timeColumn').value;
                                            const periods = parseInt(document.getElementById('lagPeriods').value);
                                            applyTransformation('/feature_engineer/create_lagged_feature', { column_name, group_column, time_column, periods }, `Creating lagged feature for ${column_name} by ${periods} periods...`);
                                        }}
                                        disabled={loading || !dataSummary}
                                        className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-2 px-4 rounded-lg shadow-md transition duration-300 ease-in-out disabled:opacity-50 disabled:cursor-not-allowed"
                                    >
                                        Create Lagged Feature
                                    </button>
                                </div>
                            </div>
                        </section>
                    </div>
                )}

                {dataSummary && activeTab === 'model_building' && (
                    <div className="space-y-8">
                        <h3 className="text-2xl font-semibold text-gray-800 mb-4">Model Building & Evaluation</h3>

                        <section className="bg-white p-6 rounded-lg shadow-sm border border-purple-200">
                            <h4 className="text-xl font-medium text-gray-700 mb-4">Select Target & Model</h4>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                <div>
                                    <label htmlFor="targetVariable" className="block text-sm font-medium text-gray-700">Target Variable (Y):</label>
                                    <select
                                        id="targetVariable"
                                        value={targetVariable}
                                        onChange={(e) => setTargetVariable(e.target.value)}
                                        className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
                                        disabled={loading}
                                    >
                                        <option value="">Select Target Column</option>
                                        {/* For target variable, generally all columns can be a target depending on the problem */}
                                        {allColumns.map(col => (
                                            <option key={col} value={col}>{col} ({dataSummary.data_types[col]})</option>
                                        ))}
                                    </select>
                                </div>
                                <div>
                                    <label htmlFor="selectedModel" className="block text-sm font-medium text-gray-700">Select Model:</label>
                                    <select
                                        id="selectedModel"
                                        value={selectedModel}
                                        onChange={(e) => setSelectedModel(e.target.value)}
                                        className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
                                        disabled={loading}
                                    >
                                        <option value="">Select ML Model</option>
                                        <optgroup label="Regression Models">
                                            <option value="linear_regression">Linear Regression</option>
                                            <option value="random_forest_regressor">Random Forest Regressor</option>
                                        </optgroup>
                                        <optgroup label="Classification Models">
                                            <option value="logistic_regression">Logistic Regression</option>
                                            <option value="decision_tree_classifier">Decision Tree Classifier</option>
                                        </optgroup>
                                        <optgroup label="Clustering Models">
                                            <option value="kmeans">K-Means Clustering</option>
                                        </optgroup>
                                    </select>
                                </div>
                                <div>
                                    <label htmlFor="testSize" className="block text-sm font-medium text-gray-700">Test Size (e.g., 0.2 for 20%):</label>
                                    <input
                                        type="number"
                                        id="testSize"
                                        step="0.01"
                                        min="0.05"
                                        max="0.95"
                                        value={testSize}
                                        onChange={(e) => setTestSize(e.target.value)}
                                        className="mt-1 block w-full pl-3 pr-3 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
                                        disabled={loading}
                                    />
                                </div>
                                <div>
                                    <label htmlFor="randomState" className="block text-sm font-medium text-gray-700">Random State:</label>
                                    <input
                                        type="number"
                                        id="randomState"
                                        value={randomState}
                                        onChange={(e) => setRandomState(e.target.value)}
                                        className="mt-1 block w-full pl-3 pr-3 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
                                        disabled={loading}
                                    />
                                </div>
                                <div className="md:col-span-2">
                                    <button
                                        onClick={handleTrainModel}
                                        disabled={loading || !dataSummary || !targetVariable || !selectedModel}
                                        className="w-full bg-purple-600 hover:bg-purple-700 text-white font-bold py-2 px-4 rounded-lg shadow-md transition duration-300 ease-in-out disabled:opacity-50 disabled:cursor-not-allowed"
                                    >
                                        {loading ? 'Training Model...' : 'Train & Evaluate Model'}
                                    </button>
                                </div>
                            </div>
                        </section>

                        {modelMessage && (
                            <div className={`px-4 py-3 rounded-lg relative ${modelMetrics ? 'bg-green-100 border border-green-400 text-green-700' : 'bg-red-100 border border-red-400 text-red-700'} mb-4`} role="alert">
                                <span className="block sm:inline">{modelMessage}</span>
                            </div>
                        )}

                        {modelMetrics && (
                            <section className="bg-gray-50 p-6 rounded-lg shadow-sm">
                                <h4 className="text-xl font-medium text-gray-700 mb-4">Model Evaluation Results ({modelMetrics.model_type})</h4>
                                <div className="overflow-x-auto mb-6">
                                    <table className="min-w-full divide-y divide-gray-200">
                                        <thead className="bg-gray-200">
                                            <tr>
                                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-600 uppercase tracking-wider rounded-tl-lg">Metric</th>
                                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-600 uppercase tracking-wider rounded-tr-lg">Value</th>
                                            </tr>
                                        </thead>
                                        <tbody className="bg-white divide-y divide-gray-200">
                                            {Object.keys(modelMetrics).filter(key => key !== 'model_type' && key !== 'message' && key !== 'plots' && key !== 'class_labels' && key !== 'is_multiclass').map(metricName => (
                                                <tr key={metricName}>
                                                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{metricName.replace(/_/g, ' ').toUpperCase()}</td>
                                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-700">
                                                        {typeof modelMetrics[metricName] === 'number' ? modelMetrics[metricName].toFixed(4) : modelMetrics[metricName]}
                                                    </td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>

                                {/* Model Evaluation Plots */}
                                {Object.keys(modelPlots).length > 0 && (
                                    <>
                                        <h4 className="text-xl font-medium text-gray-700 mb-4">Model Evaluation Plots</h4>
                                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                            {Object.keys(modelPlots).map(plotName => (
                                                modelPlots[plotName] ? (
                                                    <div key={plotName} className="border border-gray-200 rounded-lg overflow-hidden shadow-sm">
                                                        <img
                                                            src={`data:image/png;base64,${modelPlots[plotName]}`}
                                                            alt={plotName.replace(/_/g, ' ').toUpperCase()}
                                                            className="w-full h-auto object-contain"
                                                        />
                                                    </div>
                                                ) : null
                                            ))}
                                        </div>
                                    </>
                                )}
                            </section>
                        )}
                    </div>
                )}
                <div ref={messagesEndRef} /> {/* For auto-scrolling messages */}
            </main>

            {/* Export Code Modal */}
            {showExportCodeModal && (
                <div className="fixed inset-0 bg-gray-600 bg-opacity-75 flex items-center justify-center z-50 p-4">
                    <div className="bg-white rounded-lg shadow-xl p-6 w-full max-w-3xl max-h-[90vh] overflow-y-auto">
                        <h3 className="text-2xl font-bold mb-4 text-gray-800">Exported Python Code</h3>
                        <div className="bg-gray-800 text-white rounded-lg p-4 mb-4 overflow-x-auto">
                            <pre><code>{exportCode}</code></pre>
                        </div>
                        <div className="flex justify-end space-x-4">
                            <button
                                onClick={copyExportedCode}
                                className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg shadow-md transition duration-300 ease-in-out"
                            >
                                Copy Code
                            </button>
                            <button
                                onClick={() => setShowExportCodeModal(false)}
                                className="bg-gray-300 hover:bg-gray-400 text-gray-800 font-bold py-2 px-4 rounded-lg shadow-md transition duration-300 ease-in-out"
                            >
                                Close
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default App;
