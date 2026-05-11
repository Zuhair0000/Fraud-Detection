import { useState } from "react";
import axios from "axios";

function App() {
  const [formData, setFormData] = useState({
    income: 50000,
    name_email_similarity: 0.8,
    prev_address_months_count: 24,
    device_os: "windows",
    payment_type: "AC",
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: e.target.type === "number" ? Number(value) : value,
    });
  };

  const checkTransaction = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await axios.post(
        "http://localhost:8000/predict_fraud",
        formData,
      );
      setResult(response.data);
    } catch (err) {
      setError(
        err.message ||
          "Failed to connect to the backend API. Is the server running?",
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 py-12 px-4 sm:px-6 lg:px-8 font-sans text-slate-900 flex items-center justify-center">
      <div className="max-w-2xl w-full bg-white shadow-xl rounded-2xl overflow-hidden border border-slate-100">
        {/* HEADER SECTION */}
        <div className="bg-indigo-600 px-8 py-6 text-white">
          <h1 className="text-2xl font-bold tracking-tight">
            Deep Learning Transaction Monitor
          </h1>
          <p className="text-indigo-100 mt-1 text-sm">
            Powered by PyTorch & Explainable AI (XAI)
          </p>
        </div>

        <div className="p-8">
          {/* THE INPUT FORM */}
          <form onSubmit={checkTransaction} className="space-y-6">
            <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
              <label className="block text-sm font-semibold text-slate-700">
                Reported Income
                <input
                  type="number"
                  name="income"
                  value={formData.income}
                  onChange={handleInputChange}
                  required
                  className="mt-2 block w-full rounded-lg border-slate-300 bg-slate-50 px-4 py-3 text-slate-900 border focus:border-indigo-500 focus:bg-white focus:ring-2 focus:ring-indigo-200 outline-none transition-all sm:text-sm"
                />
              </label>

              <label className="block text-sm font-semibold text-slate-700">
                Name/Email Similarity (0.0 - 1.0)
                <input
                  type="number"
                  step="0.01"
                  name="name_email_similarity"
                  value={formData.name_email_similarity}
                  onChange={handleInputChange}
                  className="mt-2 block w-full rounded-lg border-slate-300 bg-slate-50 px-4 py-3 text-slate-900 border focus:border-indigo-500 focus:bg-white focus:ring-2 focus:ring-indigo-200 outline-none transition-all sm:text-sm"
                />
              </label>

              <label className="block text-sm font-semibold text-slate-700 sm:col-span-2">
                Months at Previous Address
                <input
                  type="number"
                  name="prev_address_months_count"
                  value={formData.prev_address_months_count}
                  onChange={handleInputChange}
                  className="mt-2 block w-full rounded-lg border-slate-300 bg-slate-50 px-4 py-3 text-slate-900 border focus:border-indigo-500 focus:bg-white focus:ring-2 focus:ring-indigo-200 outline-none transition-all sm:text-sm"
                />
              </label>

              <label className="block text-sm font-semibold text-slate-700">
                Device OS
                <select
                  name="device_os"
                  value={formData.device_os}
                  onChange={handleInputChange}
                  className="mt-2 block w-full rounded-lg border-slate-300 bg-slate-50 px-4 py-3 text-slate-900 border focus:border-indigo-500 focus:bg-white focus:ring-2 focus:ring-indigo-200 outline-none transition-all sm:text-sm cursor-pointer"
                >
                  <option value="windows">Windows</option>
                  <option value="macintosh">Macintosh</option>
                  <option value="linux">Linux</option>
                  <option value="x11">X11</option>
                </select>
              </label>

              <label className="block text-sm font-semibold text-slate-700">
                Payment Type
                <select
                  name="payment_type"
                  value={formData.payment_type}
                  onChange={handleInputChange}
                  className="mt-2 block w-full rounded-lg border-slate-300 bg-slate-50 px-4 py-3 text-slate-900 border focus:border-indigo-500 focus:bg-white focus:ring-2 focus:ring-indigo-200 outline-none transition-all sm:text-sm cursor-pointer"
                >
                  <option value="AC">AC (ACH/Bank Transfer)</option>
                  <option value="CC">CC (Credit Card)</option>
                </select>
              </label>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full mt-6 flex justify-center py-3.5 px-4 border border-transparent rounded-lg shadow-sm text-sm font-bold text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition-colors disabled:bg-indigo-300 disabled:cursor-wait"
            >
              {loading ? (
                <span className="flex items-center gap-2">
                  <svg
                    className="animate-spin h-5 w-5 text-white"
                    xmlns="http://www.w3.org/2000/svg"
                    fill="none"
                    viewBox="0 0 24 24"
                  >
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    ></circle>
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                    ></path>
                  </svg>
                  Analyzing Transaction...
                </span>
              ) : (
                "Run Security Audit"
              )}
            </button>
          </form>

          {/* ERROR HANDLING */}
          {error && (
            <div className="mt-6 p-4 bg-rose-50 border-l-4 border-rose-500 text-rose-800 rounded-r-md shadow-sm">
              <strong className="font-semibold">Connection Error:</strong>{" "}
              {error}
            </div>
          )}

          {/* THE RESULTS PANEL */}
          {result && (
            <div className="mt-8 animate-fade-in-up">
              <div
                className={`p-6 border-l-4 rounded-r-xl shadow-md ${
                  result.fraud_detected
                    ? "border-rose-500 bg-rose-50"
                    : "border-emerald-500 bg-emerald-50"
                }`}
              >
                <div className="flex items-center gap-3 mb-2">
                  <span className="text-2xl">
                    {result.fraud_detected ? "🚨" : "✅"}
                  </span>
                  <h2
                    className={`text-xl font-extrabold ${result.fraud_detected ? "text-rose-900" : "text-emerald-900"}`}
                  >
                    {result.fraud_detected
                      ? "HIGH RISK DETECTED"
                      : "TRANSACTION CLEARED"}
                  </h2>
                </div>

                <p
                  className={`text-sm mb-4 ${result.fraud_detected ? "text-rose-700" : "text-emerald-700"}`}
                >
                  <strong className="font-semibold">Network Confidence:</strong>{" "}
                  <span className="text-lg">
                    {(result.confidence * 100).toFixed(2)}%
                  </span>
                </p>

                {/* Render the XAI Audit Trail if it exists */}
                {result.fraud_detected && Array.isArray(result.audit_trail) && (
                  <div className="mt-5 pt-5 border-t border-rose-200">
                    <h3 className="text-sm font-bold text-rose-900 uppercase tracking-wider mb-3 flex items-center gap-2">
                      <svg
                        className="w-5 h-5"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                        xmlns="http://www.w3.org/2000/svg"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth="2"
                          d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"
                        ></path>
                      </svg>
                      XAI Feature Attribution
                    </h3>
                    <p className="text-sm text-rose-700 mb-3">
                      The neural network identified the following anomalous
                      feature weights:
                    </p>
                    <ul className="space-y-2">
                      {result.audit_trail.map((reason, index) => (
                        <li
                          key={index}
                          className="flex items-start text-sm text-rose-800 font-medium bg-rose-100/50 px-3 py-2 rounded-md"
                        >
                          <span className="mr-2 text-rose-500">•</span>
                          {reason}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
