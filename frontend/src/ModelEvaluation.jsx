import { useEffect, useState } from 'react'

const MODEL_PATHS = {
  'CNN V2': 'cnn_v2',
  'Spatial ViT': 'spatial_vit',
  'Hybrid CNN-Transformer': 'hybrid_spatial',
}

const METRICS = [
  { key: 'accuracy', label: 'Accuracy', color: 'green' },
  { key: 'precision', label: 'Precision', color: 'purple' },
  { key: 'recall', label: 'Recall', color: 'orange' },
  { key: 'f1_score', label: 'F1 Score', color: 'green' },
]

function percentage(value) {
  return Number.isFinite(value) ? `${(value * 100).toFixed(2)}%` : '--'
}

function ModelEvaluation() {
  const [report, setReport] = useState(null)
  const [selectedModel, setSelectedModel] = useState('CNN V2')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false

    async function loadReport() {
      try {
        const response = await fetch(
          '/evaluation/prediction_analysis_summary.json'
        )

        if (!response.ok) {
          throw new Error('Unable to load model evaluation summary.')
        }

        const data = await response.json()

        if (!cancelled) {
          setReport(data)
          setError('')
        }
      } catch (err) {
        if (!cancelled) {
          setError(err.message)
        }
      } finally {
        if (!cancelled) {
          setLoading(false)
        }
      }
    }

    loadReport()

    return () => {
      cancelled = true
    }
  }, [])

  const model = report?.[selectedModel]
  const metadata = model?.evaluation_metadata
  const matrix = model?.confusion_matrix
  const modelFolder = MODEL_PATHS[selectedModel]

  return (
    <section className="evaluation-section" id="evaluation">
      <div className="panel evaluation-intro">
        <div className="panel-header">
          <div>
            <div className="eyebrow">MODEL PERFORMANCE LAB</div>
            <h3>Model Evaluation Center</h3>
            <p>
              Inspect classification metrics, confusion matrices and
              evaluation protocols for the three trained architectures.
            </p>
          </div>
          <span className="panel-tag">EVALUATION REPORTS</span>
        </div>

        <div className="evaluation-notice">
          <strong>Research context</strong>
          <p>
            These results were generated using synthetic mock data.
            Evaluation protocols differ between models, so the metrics
            should not be interpreted as a controlled head-to-head benchmark
            or evidence of real-world crop disease performance.
          </p>
        </div>
      </div>

      {loading && (
        <div className="loading-message">Loading evaluation reports...</div>
      )}

      {error && (
        <div className="error-message">
          <strong>Evaluation data unavailable</strong>
          <p>{error}</p>
        </div>
      )}

      {report && (
        <>
          <section className="panel evaluation-panel">
            <div className="panel-header">
              <div>
                <div className="eyebrow">ARCHITECTURE SELECTION</div>
                <h3>Choose a model</h3>
                <p>Select a model to inspect its evaluation results.</p>
              </div>
              <span className="panel-tag">3 MODELS</span>
            </div>

            <div className="evaluation-model-selector">
              {Object.keys(MODEL_PATHS).map((name) => (
                <button
                  key={name}
                  type="button"
                  className={`evaluation-model-button ${
                    selectedModel === name ? 'selected' : ''
                  }`}
                  onClick={() => setSelectedModel(name)}
                  aria-pressed={selectedModel === name}
                >
                  {name}
                </button>
              ))}
            </div>
          </section>

          {model && (
            <>
              <section className="evaluation-metrics">
                {METRICS.map((metric) => (
                  <div className="metric-card" key={metric.key}>
                    <div className="metric-top">
                      <span className="metric-label">{metric.label}</span>
                      <span className={`metric-icon ${metric.color}`}>
                        ●
                      </span>
                    </div>
                    <div className="metric-value">
                      {percentage(model[metric.key])}
                    </div>
                    <div className="metric-description">
                      {selectedModel} evaluation
                    </div>
                  </div>
                ))}
              </section>

              <div className="evaluation-layout">
                <section className="panel">
                  <div className="panel-header">
                    <div>
                      <div className="eyebrow">CLASSIFICATION RESULTS</div>
                      <h3>Confusion Matrix</h3>
                      <p>Actual classes versus predicted classes.</p>
                    </div>
                    <span className="panel-tag">
                      {model.total_samples?.toLocaleString()} SAMPLES
                    </span>
                  </div>

                  {matrix && (
                    <div className="confusion-matrix-wrapper">
                      <div className="matrix-axis-label">
                        Predicted class
                      </div>
                      <table className="confusion-matrix">
                        <thead>
                          <tr>
                            <th>Actual</th>
                            <th>Healthy</th>
                            <th>Stressed</th>
                          </tr>
                        </thead>
                        <tbody>
                          <tr>
                            <th>Healthy</th>
                            <td className="matrix-correct">
                              {matrix[0]?.[0] ?? '--'}
                            </td>
                            <td className="matrix-error">
                              {matrix[0]?.[1] ?? '--'}
                            </td>
                          </tr>
                          <tr>
                            <th>Stressed</th>
                            <td className="matrix-error">
                              {matrix[1]?.[0] ?? '--'}
                            </td>
                            <td className="matrix-correct">
                              {matrix[1]?.[1] ?? '--'}
                            </td>
                          </tr>
                        </tbody>
                      </table>
                      <div className="matrix-legend">
                        <span><i className="legend-correct" /> Correct</span>
                        <span><i className="legend-error" /> Misclassified</span>
                      </div>
                    </div>
                  )}

                  <div className="evaluation-image">
                    <h4>Generated Confusion Matrix</h4>
                    <img
                      src={`/evaluation/${modelFolder}/confusion_matrix.png`}
                      alt={`${selectedModel} confusion matrix visualization`}
                      loading="lazy"
                    />
                  </div>
                </section>

                <section className="panel evaluation-details">
                  <div className="eyebrow">EXPERIMENT CONFIGURATION</div>
                  <h3>Evaluation Details</h3>

                  <div className="evaluation-detail-row">
                    <span>Architecture</span>
                    <strong>{selectedModel}</strong>
                  </div>
                  <div className="evaluation-detail-row">
                    <span>Evaluation protocol</span>
                    <strong>{metadata?.evaluation_protocol ?? '--'}</strong>
                  </div>
                  <div className="evaluation-detail-row">
                    <span>Training samples</span>
                    <strong>
                      {metadata?.training_samples?.toLocaleString() ?? '--'}
                    </strong>
                  </div>
                  <div className="evaluation-detail-row">
                    <span>Test samples</span>
                    <strong>
                      {metadata?.test_samples?.toLocaleString() ??
                        model.total_samples?.toLocaleString() ?? '--'}
                    </strong>
                  </div>
                  <div className="evaluation-detail-row">
                    <span>Patch size</span>
                    <strong>{metadata?.patch_size ?? '--'}</strong>
                  </div>
                  <div className="evaluation-detail-row">
                    <span>Classification rule</span>
                    <strong>{metadata?.classification_rule ?? 'Argmax'}</strong>
                  </div>

                  {metadata?.classification_threshold !== undefined && (
                    <div className="evaluation-detail-row">
                      <span>Classification threshold</span>
                      <strong>{metadata.classification_threshold}</strong>
                    </div>
                  )}

                  <div className="evaluation-limitations">
                    <h4>Evaluation limitations</h4>
                    <ul>
                      {(metadata?.limitations ?? []).map((item, index) => (
                        <li key={index}>{item}</li>
                      ))}
                    </ul>
                  </div>
                </section>
              </div>

              <section className="panel evaluation-panel">
                <div className="panel-header">
                  <div>
                    <div className="eyebrow">MODEL COMPARISON</div>
                    <h3>Performance Overview</h3>
                    <p>
                      Reported scores for each model under its own
                      evaluation protocol.
                    </p>
                  </div>
                </div>

                <div className="comparison-list">
                  {Object.entries(report).map(([name, values]) => (
                    <div className="comparison-model" key={name}>
                      <div className="comparison-model-heading">
                        <strong>{name}</strong>
                        <span>{percentage(values.accuracy)} accuracy</span>
                      </div>
                      {METRICS.map((metric) => (
                        <div className="comparison-metric" key={metric.key}>
                          <span>{metric.label}</span>
                          <div className="distribution-track">
                            <div
                              className="distribution-fill healthy-fill"
                              style={{
                                width: `${Math.max(
                                  0,
                                  Math.min(100, (values[metric.key] ?? 0) * 100)
                                )}%`,
                              }}
                            />
                          </div>
                          <strong>{percentage(values[metric.key])}</strong>
                        </div>
                      ))}
                    </div>
                  ))}
                </div>
              </section>

              <section className="panel evaluation-panel">
                <div className="panel-header">
                  <div>
                    <div className="eyebrow">PREDICTION ANALYSIS</div>
                    <h3>Actual vs Predicted</h3>
                    <p>
                      Generated evaluation visualizations from the Python
                      analysis pipeline.
                    </p>
                  </div>
                  <span className="panel-tag">SYNTHETIC TEST DATA</span>
                </div>

                <div className="evaluation-visual-grid">
                  <figure>
                    <img
                      src="/evaluation/actual_vs_predicted.png"
                      alt="Actual versus predicted classification analysis"
                      loading="lazy"
                    />
                    <figcaption>Actual versus predicted classes</figcaption>
                  </figure>
                  <figure>
                    <img
                      src="/evaluation/class_distribution.png"
                      alt="Class distribution in evaluation data"
                      loading="lazy"
                    />
                    <figcaption>Evaluation class distribution</figcaption>
                  </figure>
                </div>
              </section>
            </>
          )}
        </>
      )}
    </section>
  )
}

export default ModelEvaluation
