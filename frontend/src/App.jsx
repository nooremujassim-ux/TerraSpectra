
import { useEffect, useMemo, useState } from 'react'
import { DeckGL } from '@deck.gl/react'
import { COORDINATE_SYSTEM, OrthographicView } from '@deck.gl/core'
import { ScatterplotLayer } from '@deck.gl/layers'
import './App.css'

const DATA_URL = '/crop_sample.json'

function App() {
  const [dataset, setDataset] = useState(null)
  const [loadError, setLoadError] = useState('')
  const [showHealthy, setShowHealthy] = useState(true)
  const [showStressed, setShowStressed] = useState(true)
  const [viewMode, setViewMode] = useState('all')

  // Load the exported Python dataset.
  useEffect(() => {
    let cancelled = false

    fetch(DATA_URL)
      .then((response) => {
        if (!response.ok) {
          throw new Error('Unable to load crop_sample.json')
        }

        return response.json()
      })
      .then((data) => {
        if (!cancelled) {
          setDataset(data)
          setLoadError('')
        }
      })
      .catch((error) => {
        if (!cancelled) {
          setLoadError(error.message)
        }
      })

    return () => {
      cancelled = true
    }
  }, [])

  const pixelData = useMemo(() => dataset?.pixels ?? [], [dataset])

  // Filter pixels based on the selected classes.
  const filteredPixels = useMemo(() => {
    return pixelData.filter((pixel) => {
      if (pixel.stressed && !showStressed) return false
      if (!pixel.stressed && !showHealthy) return false
      return true
    })
  }, [pixelData, showHealthy, showStressed])

  const healthyCount = dataset?.healthy_pixels ?? 0
  const stressedCount = dataset?.stressed_pixels ?? 0
  const totalPixels = dataset?.total_pixels ?? 0

  const stressPercentage =
    totalPixels > 0
      ? ((stressedCount / totalPixels) * 100).toFixed(1)
      : '0.0'

  const metadata = dataset?.metadata

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
        getFillColor: [showHealthy, showStressed],
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
              {dataset ? 'DATA LOADED' : 'LOADING DATA'}
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
              Explore pixel-level sample observations and spectral information
              through an interactive analytics workspace.
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
            <strong>Dataset loading failed</strong>
            <p>{loadError}</p>
            <small>
              Confirm that crop_sample.json exists inside frontend/public.
            </small>
          </div>
        )}

        {!dataset && !loadError && (
          <div className="loading-message">
            Loading hyperspectral sample data...
          </div>
        )}

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
              Pixels in sample image
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
              <span className="green-text">Healthy mock class</span>
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
              <span className="red-text">Synthetic stressed class</span>
            </div>
          </div>

          <div className="metric-card">
            <div className="metric-top">
              <span className="metric-label">STRESSED SHARE</span>
              <span className="metric-icon orange">◉</span>
            </div>
            <div className="metric-value">{stressPercentage}%</div>
            <div className="metric-description">
              Share of synthetic sample pixels
            </div>
          </div>
        </section>

        {/* Visualization and controls */}
        <section className="analysis-layout" id="visualization">
          <div className="panel visualization-panel">
            <div className="panel-header">
              <div>
                <div className="eyebrow">SPATIAL EXPLORER</div>
                <h3>Hyperspectral Pixel Map</h3>
                <p>Interactive visualization of sample pixel classes</p>
              </div>

              <span className="panel-tag">64 × 64 GRID</span>
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
              {dataset && (
                <DeckGL
                  initialViewState={initialViewState}
                  controller={true}
                  layers={layers}
                  getTooltip={({ object }) =>
                    object
                      ? {
                          text:
                            `${object.stressed ? 'Stressed' : 'Healthy'} pixel\n` +
                            `Position: ${object.position.join(', ')}\n` +
                            `Mean spectral value: ${object.value}`,
                        }
                      : null
                  }
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
                  className={viewMode === 'all' ? 'view-option selected' : 'view-option'}
                  onClick={() => {
                    setViewMode('all')
                    setShowHealthy(true)
                    setShowStressed(true)
                  }}
                >
                  All Pixels
                </button>

                <button
                  className={viewMode === 'healthy' ? 'view-option selected' : 'view-option'}
                  onClick={() => {
                    setViewMode('healthy')
                    setShowHealthy(true)
                    setShowStressed(false)
                  }}
                >
                  Healthy
                </button>

                <button
                  className={viewMode === 'stressed' ? 'view-option selected' : 'view-option'}
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
                    width: `${totalPixels > 0 ? (healthyCount / totalPixels) * 100 : 0}%`,
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
                    width: `${totalPixels > 0 ? (stressedCount / totalPixels) * 100 : 0}%`,
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
              ↺ Reset Visualization
            </button>
          </div>
        </section>

        {/* Dataset information */}
        <section className="panel dataset-panel" id="dataset">
          <div className="panel-header">
            <div>
              <div className="eyebrow">DATA INFORMATION</div>
              <h3>Dataset Overview</h3>
              <p>Metadata loaded from the Python preprocessing pipeline</p>
            </div>

            <span className="panel-tag">JSON SOURCE</span>
          </div>

          <div className="dataset-grid">
            <div className="dataset-item">
              <span>Dataset Type</span>
              <strong>
                {dataset?.dataset_type ?? 'Not loaded'}
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
          </div>

          <div className="dataset-disclaimer">
            <span>ⓘ</span>
            <p>
              {dataset?.description ??
                'Dataset information will appear after loading.'}
              {' '}This is synthetic mock data created for pipeline testing.
              It does not represent real crop observations or verified disease
              conditions.
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
