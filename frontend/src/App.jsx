
import { useEffect, useMemo, useState } from 'react'
import { DeckGL } from '@deck.gl/react'
import { COORDINATE_SYSTEM, OrthographicView } from '@deck.gl/core'
import { ScatterplotLayer } from '@deck.gl/layers'
import './App.css'

function App() {
  const [dataset, setDataset] = useState(null)
  const [predictions, setPredictions] = useState(null)

  const [loadError, setLoadError] = useState('')
  const [predictionError, setPredictionError] = useState('')

  const [dataMode, setDataMode] = useState('reference')
  const [showHealthy, setShowHealthy] = useState(true)
  const [showStressed, setShowStressed] = useState(true)
  const [viewMode, setViewMode] = useState('all')

  // Load reference labels and CNN predictions independently.
  useEffect(() => {
    let cancelled = false

    async function loadData() {
      try {
        const response = await fetch('/crop_sample.json')

        if (!response.ok) {
          throw new Error('Unable to load crop_sample.json')
        }

        const data = await response.json()

        if (!cancelled) {
          setDataset(data)
          setLoadError('')
        }
      } catch (error) {
        if (!cancelled) {
          setLoadError(error.message)
        }
      }

      try {
        const response = await fetch('/cnn_predictions.json')

        if (!response.ok) {
          throw new Error('Unable to load cnn_predictions.json')
        }

        const data = await response.json()

        if (!cancelled) {
          setPredictions(data)
          setPredictionError('')
        }
      } catch (error) {
        if (!cancelled) {
          setPredictionError(error.message)
        }
      }
    }

    loadData()

    return () => {
      cancelled = true
    }
  }, [])

  // Select the active data source.
  const activeData =
    dataMode === 'predictions' ? predictions : dataset

  const pixelData = useMemo(
    () => activeData?.pixels ?? [],
    [activeData]
  )

  // Filter the selected dataset.
  const filteredPixels = useMemo(() => {
    return pixelData.filter((pixel) => {
      if (pixel.stressed && !showStressed) return false
      if (!pixel.stressed && !showHealthy) return false
      return true
    })
  }, [pixelData, showHealthy, showStressed])

  // Calculate metrics according to the active data source.
  const healthyCount =
    dataMode === 'predictions'
      ? predictions?.predicted_healthy_pixels ?? 0
      : dataset?.healthy_pixels ?? 0

  const stressedCount =
    dataMode === 'predictions'
      ? predictions?.predicted_stressed_pixels ?? 0
      : dataset?.stressed_pixels ?? 0

  const totalPixels = activeData?.total_pixels ?? 0

  const stressPercentage =
    totalPixels > 0
      ? ((stressedCount / totalPixels) * 100).toFixed(1)
      : '0.0'

  const metadata = dataset?.metadata
    // Dynamic analytics based on the active dataset.
  const healthyPercentage =
    totalPixels > 0
      ? ((healthyCount / totalPixels) * 100).toFixed(1)
      : '0.0'

  const stressedPercentage =
    totalPixels > 0
      ? ((stressedCount / totalPixels) * 100).toFixed(1)
      : '0.0'

  const visibleHealthyCount = filteredPixels.filter(
    (pixel) => !pixel.stressed
  ).length

  const visibleStressedCount = filteredPixels.filter(
    (pixel) => pixel.stressed
  ).length

  const averageStressedProbability = useMemo(() => {
    if (dataMode !== 'predictions' || !pixelData.length) {
      return null
    }

    const totalProbability = pixelData.reduce(
      (sum, pixel) => sum + pixel.stressed_probability,
      0
    )

    return (totalProbability / pixelData.length) * 100
  }, [dataMode, pixelData])

  // Deck.gl pixel visualization.
  const layers = [
    new ScatterplotLayer({
      id: 'hyperspectral-pixels',
      data: filteredPixels,
      coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
      getPosition: (pixel) => pixel.position,
      getRadius: 0.42,
      radiusUnits: 'common',
      radiusMinPixels: 2,
      radiusMaxPixels: 8,
      getFillColor: (pixel) =>
        pixel.stressed
          ? [239, 91, 91, 230]
          : [64, 190, 137, 220],
      stroked: false,
      pickable: true,
      autoHighlight: true,
      highlightColor: [255, 255, 255, 180],
      updateTriggers: {
        getFillColor: [showHealthy, showStressed, dataMode],
      },
    }),
  ]

  const initialViewState = {
    longitude: 0,
    latitude: 0,
    target: [32, 32, 0],
    zoom: 3.2,
    minZoom: 1,
    maxZoom: 8,
    pitch: 0,
    bearing: 0,
  }

  const selectDataMode = (mode) => {
    setDataMode(mode)
    setShowHealthy(true)
    setShowStressed(true)
    setViewMode('all')
  }

  return (
    <div className="app-shell">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">T</div>
          <div>
            <h2>TerraSpectra</h2>
            <span>INTELLIGENCE PLATFORM</span>
          </div>
        </div>

        <div className="sidebar-label">WORKSPACE</div>

        <nav className="navigation">
          <a className="nav-item active" href="#overview">
            <span>◈</span>
            Dashboard
          </a>

          <a className="nav-item" href="#analysis">
            <span>▦</span>
            Crop Analysis
          </a>

          <a className="nav-item" href="#visualization">
            <span>▧</span>
            Spectral Map
          </a>

          <a className="nav-item" href="#dataset">
            <span>◫</span>
            Dataset Details
          </a>
        </nav>

        <div className="sidebar-bottom">
          <div className="status-indicator">
            <span className="status-dot" />
            <div>
              <strong>System Online</strong>
              <small>Local analysis mode</small>
            </div>
          </div>

          <div className="sidebar-footer">
            TerraSpectra v1.0.0
            <br />
            Hyperspectral Analytics
          </div>
        </div>
      </aside>

      {/* Main content */}
      <main className="main-content">
        {/* Top bar */}
        <header className="topbar">
          <div>
            <div className="breadcrumb">
              WORKSPACE <span>/</span> OVERVIEW
            </div>
            <h1>Crop Intelligence Dashboard</h1>
          </div>

          <div className="topbar-right">
            <div className="live-status">
              <span className="status-dot" />
              {dataset && predictions
                ? 'DATA & MODEL LOADED'
                : 'LOADING DATA'}
            </div>

            <div className="profile-avatar">TS</div>
          </div>
        </header>

        {/* Welcome section */}
        <section className="welcome-section" id="overview">
          <div>
            <div className="eyebrow">HYPERSPECTRAL ANALYTICS</div>
            <h2>Understand crop conditions through spectral data.</h2>
            <p>
              Explore synthetic hyperspectral reference labels and
              trained CNN pixel predictions through an interactive
              analytics workspace.
            </p>
          </div>

          <div className="welcome-badge">
            <span className="badge-icon">✦</span>
            <div>
              <strong>Sample Dataset</strong>
              <small>200 spectral bands</small>
            </div>
          </div>
        </section>

        {/* Loading and error messages */}
        {loadError && (
          <div className="error-message">
            <strong>Reference dataset loading failed</strong>
            <p>{loadError}</p>
            <small>
              Confirm crop_sample.json exists inside frontend/public.
            </small>
          </div>
        )}

        {predictionError && (
          <div className="error-message">
            <strong>CNN prediction loading failed</strong>
            <p>{predictionError}</p>
            <small>
              Confirm cnn_predictions.json exists inside frontend/public.
            </small>
          </div>
        )}

        {!activeData && (
          <div className="loading-message">
            Loading hyperspectral data...
          </div>
        )}

        {/* Data source selector */}
        <section className="panel controls-panel" style={{ marginBottom: '22px' }}>
          <div className="panel-header">
            <div>
              <div className="eyebrow">ANALYSIS SOURCE</div>
              <h3>Choose Data View</h3>
              <p>Compare reference labels with trained CNN predictions</p>
            </div>
          </div>

          <div className="view-options">
            <button
              className={
                dataMode === 'reference'
                  ? 'view-option selected'
                  : 'view-option'
              }
              onClick={() => selectDataMode('reference')}
              disabled={!dataset}
            >
              Reference Labels
            </button>

            <button
              className={
                dataMode === 'predictions'
                  ? 'view-option selected'
                  : 'view-option'
              }
              onClick={() => selectDataMode('predictions')}
              disabled={!predictions}
            >
              CNN Predictions
            </button>
          </div>

          <div className="control-divider" />

          <div className="dataset-disclaimer">
            <span>ⓘ</span>
            <p>
              {dataMode === 'predictions'
                ? `Displaying predictions from ${predictions?.model ?? 'the trained model'} using checkpoint ${predictions?.checkpoint ?? '--'}. Classification threshold: ${predictions?.classification_threshold ?? '--'}.`
                : 'Displaying the original synthetic reference labels generated for pipeline testing.'}
            </p>
          </div>
        </section>

        {/* Metrics */}
        <section className="metrics" id="analysis">
          <div className="metric-card">
            <div className="metric-top">
              <span className="metric-label">TOTAL PIXELS</span>
              <span className="metric-icon purple">▦</span>
            </div>
            <div className="metric-value">
              {totalPixels.toLocaleString()}
            </div>
            <div className="metric-description">
              Pixels in active dataset
            </div>
          </div>

          <div className="metric-card">
            <div className="metric-top">
              <span className="metric-label">HEALTHY PIXELS</span>
              <span className="metric-icon green">✦</span>
            </div>
            <div className="metric-value">
              {healthyCount.toLocaleString()}
            </div>
            <div className="metric-description">
              <span className="green-text">
                {dataMode === 'predictions'
                  ? 'CNN predicted healthy'
                  : 'Reference healthy class'}
              </span>
            </div>
          </div>

          <div className="metric-card">
            <div className="metric-top">
              <span className="metric-label">STRESSED PIXELS</span>
              <span className="metric-icon red">△</span>
            </div>
            <div className="metric-value">
              {stressedCount.toLocaleString()}
            </div>
            <div className="metric-description">
              <span className="red-text">
                {dataMode === 'predictions'
                  ? 'CNN predicted stressed'
                  : 'Reference stressed class'}
              </span>
            </div>
          </div>

          <div className="metric-card">
            <div className="metric-top">
              <span className="metric-label">STRESSED SHARE</span>
              <span className="metric-icon orange">◉</span>
            </div>
            <div className="metric-value">{stressPercentage}%</div>
            <div className="metric-description">
              Share of active dataset pixels
            </div>
          </div>
        </section>
        {/* Prediction Analytics */}
<section className="panel analytics-panel">
  <div className="panel-header">
    <div>
      <div className="eyebrow">MODEL INSIGHTS</div>
      <h3>Pixel Classification Analytics</h3>
      <p>
        {dataMode === 'predictions'
          ? 'Summary of CNN-generated pixel classifications'
          : 'Summary of synthetic reference classifications'}
      </p>
    </div>

    <span className="panel-tag">LIVE ANALYTICS</span>
  </div>

  <div className="metrics analytics-metrics">
    <div className="metric-card">
      <div className="metric-top">
        <span className="metric-label">HEALTHY SHARE</span>
      </div>
      <div className="metric-value green-text">
        {healthyPercentage}%
      </div>
      <div className="metric-description">
        {healthyCount.toLocaleString()} pixels classified healthy
      </div>
    </div>

    <div className="metric-card">
      <div className="metric-top">
        <span className="metric-label">STRESSED SHARE</span>
      </div>
      <div className="metric-value red-text">
        {stressedPercentage}%
      </div>
      <div className="metric-description">
        {stressedCount.toLocaleString()} pixels classified stressed
      </div>
    </div>

    <div className="metric-card">
      <div className="metric-top">
        <span className="metric-label">VISIBLE PIXELS</span>
      </div>
      <div className="metric-value">
        {filteredPixels.length.toLocaleString()}
      </div>
      <div className="metric-description">
        {visibleHealthyCount.toLocaleString()} healthy ·{' '}
        {visibleStressedCount.toLocaleString()} stressed
      </div>
    </div>

    <div className="metric-card">
      <div className="metric-top">
        <span className="metric-label">AVERAGE STRESS PROBABILITY</span>
      </div>
      <div className="metric-value">
        {averageStressedProbability === null
          ? '--'
          : `${averageStressedProbability.toFixed(1)}%`}
      </div>
      <div className="metric-description">
        {dataMode === 'predictions'
          ? 'Mean CNN stressed-class probability'
          : 'Available in CNN prediction mode'}
      </div>
    </div>
  </div>

  <div className="distribution-row">
    <span>Healthy distribution</span>
    <strong>{healthyPercentage}%</strong>
  </div>

  <div className="distribution-track">
    <div
      className="distribution-fill healthy-fill"
      style={{ width: `${healthyPercentage}%` }}
    />
  </div>

  <div className="distribution-row">
    <span>Stressed distribution</span>
    <strong>{stressedPercentage}%</strong>
  </div>

  <div className="distribution-track">
    <div
      className="distribution-fill stressed-fill"
      style={{ width: `${stressedPercentage}%` }}
    />
  </div>
</section>

        {/* Visualization and controls */}
        <section className="analysis-layout" id="visualization">
          <div className="panel visualization-panel">
            <div className="panel-header">
              <div>
                <div className="eyebrow">SPATIAL EXPLORER</div>
                <h3>
                  {dataMode === 'predictions'
                    ? 'CNN Prediction Map'
                    : 'Hyperspectral Reference Map'}
                </h3>
                <p>
                  {dataMode === 'predictions'
                    ? 'Pixel-level classifications generated by the trained CNN'
                    : 'Interactive visualization of synthetic reference labels'}
                </p>
              </div>

              <span className="panel-tag">
                {metadata?.width ?? 64} × {metadata?.height ?? 64} GRID
              </span>
            </div>

            <div className="map-legend">
              <span>
                <i className="legend-dot healthy-dot" />
                Healthy
              </span>
              <span>
                <i className="legend-dot stressed-dot" />
                Stressed
              </span>
            </div>

            <div className="map-container">
              {activeData && (
                <DeckGL
                  views={new OrthographicView()}
                  initialViewState={initialViewState}
                  controller={true}
                  layers={layers}
                  getTooltip={({ object }) => {
                    if (!object) return null

                    const details = [
                      `${object.stressed ? 'Stressed' : 'Healthy'} pixel`,
                      `Position: ${object.position.join(', ')}`,
                    ]

                    if (dataMode === 'predictions') {
                      details.push(
                        `Stressed probability: ${(object.stressed_probability * 100).toFixed(2)}%`
                      )
                    } else {
                      details.push(
                        `Mean spectral value: ${object.value}`
                      )
                    }

                    return {
                      text: details.join('\n'),
                    }
                  }}
                />
              )}
            </div>

            <div className="map-footer">
              <span>
                Displaying {filteredPixels.length.toLocaleString()} of{' '}
                {totalPixels.toLocaleString()} pixels
              </span>
              <span>Drag to pan · Scroll to zoom</span>
            </div>
          </div>

          {/* Visualization controls */}
          <div className="panel controls-panel">
            <div className="panel-header">
              <div>
                <div className="eyebrow">CONFIGURATION</div>
                <h3>Map Controls</h3>
                <p>Customize visible pixel classes</p>
              </div>
            </div>

            <div className="control-section">
              <label className="control-label">DISPLAY MODE</label>

              <div className="view-options">
                <button
                  className={
                    viewMode === 'all'
                      ? 'view-option selected'
                      : 'view-option'
                  }
                  onClick={() => {
                    setViewMode('all')
                    setShowHealthy(true)
                    setShowStressed(true)
                  }}
                >
                  All Pixels
                </button>

                <button
                  className={
                    viewMode === 'healthy'
                      ? 'view-option selected'
                      : 'view-option'
                  }
                  onClick={() => {
                    setViewMode('healthy')
                    setShowHealthy(true)
                    setShowStressed(false)
                  }}
                >
                  Healthy
                </button>

                <button
                  className={
                    viewMode === 'stressed'
                      ? 'view-option selected'
                      : 'view-option'
                  }
                  onClick={() => {
                    setViewMode('stressed')
                    setShowHealthy(false)
                    setShowStressed(true)
                  }}
                >
                  Stressed
                </button>
              </div>
            </div>

            <div className="control-divider" />

            <div className="control-section">
              <label className="control-label">PIXEL VISIBILITY</label>

              <label className="filter-option">
                <span>
                  <i className="legend-dot healthy-dot" />
                  Healthy pixels
                </span>
                <input
                  type="checkbox"
                  checked={showHealthy}
                  onChange={(event) => {
                    setShowHealthy(event.target.checked)
                    setViewMode('custom')
                  }}
                />
              </label>

              <label className="filter-option">
                <span>
                  <i className="legend-dot stressed-dot" />
                  Stressed pixels
                </span>
                <input
                  type="checkbox"
                  checked={showStressed}
                  onChange={(event) => {
                    setShowStressed(event.target.checked)
                    setViewMode('custom')
                  }}
                />
              </label>
            </div>

            <div className="control-divider" />

            <div className="control-section">
              <label className="control-label">CLASS DISTRIBUTION</label>

              <div className="distribution-row">
                <span>Healthy</span>
                <strong>{healthyCount.toLocaleString()}</strong>
              </div>

              <div className="distribution-track">
                <div
                  className="distribution-fill healthy-fill"
                  style={{
                    width: `${
                      totalPixels > 0
                        ? (healthyCount / totalPixels) * 100
                        : 0
                    }%`,
                  }}
                />
              </div>

              <div className="distribution-row">
                <span>Stressed</span>
                <strong>{stressedCount.toLocaleString()}</strong>
              </div>

              <div className="distribution-track">
                <div
                  className="distribution-fill stressed-fill"
                  style={{
                    width: `${
                      totalPixels > 0
                        ? (stressedCount / totalPixels) * 100
                        : 0
                    }%`,
                  }}
                />
              </div>
            </div>

            <button
              className="reset-button"
              onClick={() => {
                setShowHealthy(true)
                setShowStressed(true)
                setViewMode('all')
              }}
            >
              ↻ Reset Visualization
            </button>
          </div>
        </section>

        {/* Dataset information */}
        <section className="panel dataset-panel" id="dataset">
          <div className="panel-header">
            <div>
              <div className="eyebrow">DATA INFORMATION</div>
              <h3>Dataset Overview</h3>
              <p>
                Metadata loaded from the Python preprocessing pipeline
              </p>
            </div>

            <span className="panel-tag">
              {dataMode === 'predictions' ? 'CNN OUTPUT' : 'JSON SOURCE'}
            </span>
          </div>

          <div className="dataset-grid">
            <div className="dataset-item">
              <span>Dataset Type</span>
              <strong>
                {activeData?.dataset_type ?? 'Not loaded'}
              </strong>
            </div>

            <div className="dataset-item">
              <span>Image Width</span>
              <strong>{metadata?.width ?? '--'} px</strong>
            </div>

            <div className="dataset-item">
              <span>Image Height</span>
              <strong>{metadata?.height ?? '--'} px</strong>
            </div>

            <div className="dataset-item">
              <span>Spectral Bands</span>
              <strong>{metadata?.bands ?? '--'}</strong>
            </div>

            <div className="dataset-item">
              <span>Data Type</span>
              <strong>{metadata?.dtype ?? '--'}</strong>
            </div>

            <div className="dataset-item">
              <span>Coordinate Reference</span>
              <strong>{metadata?.crs ?? 'Not georeferenced'}</strong>
            </div>

            {dataMode === 'predictions' && (
              <>
                <div className="dataset-item">
                  <span>Model Architecture</span>
                  <strong>{predictions?.model ?? '--'}</strong>
                </div>

                <div className="dataset-item">
                  <span>Checkpoint</span>
                  <strong>{predictions?.checkpoint ?? '--'}</strong>
                </div>

                <div className="dataset-item">
                  <span>Classification Threshold</span>
                  <strong>
                    {predictions?.classification_threshold ?? '--'}
                  </strong>
                </div>

                <div className="dataset-item">
                  <span>Inference Device</span>
                  <strong>{predictions?.metadata?.device ?? '--'}</strong>
                </div>
              </>
            )}
          </div>

          <div className="dataset-disclaimer">
            <span>ⓘ</span>
            <p>
              {dataMode === 'predictions'
                ? predictions?.description
                : dataset?.description}
              {' '}
              This project currently uses synthetic mock data for
              pipeline testing. Model predictions do not establish
              real-world crop disease performance.
            </p>
          </div>
        </section>

        {/* Footer */}
        <footer className="app-footer">
          <span>TerraSpectra · Hyperspectral Crop Analysis</span>
          <span>Academic prototype · Synthetic sample data</span>
        </footer>
      </main>
    </div>
  )
}

export default App
