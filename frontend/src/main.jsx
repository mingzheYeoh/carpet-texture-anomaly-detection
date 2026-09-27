import React, { useEffect, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import './style.css'

const models = [
  { id: 'patchcore', name: 'PatchCore', detail: 'Deep features + spatial localization', tag: 'Heatmap' },
  { id: 'lbp', name: 'LBP', detail: 'Local texture + One-Class SVM', tag: '160 features' },
  { id: 'lbp_glcm', name: 'LBP + GLCM', detail: 'Texture & co-occurrence + One-Class SVM', tag: '192 features' },
]
const names = { patchcore: 'PatchCore', lbp_ocsvm: 'LBP', lbp_glcm_ocsvm: 'LBP + GLCM' }
const number = (value, digits = 4) => Number(value).toFixed(digits)
const percent = value => `${(Number(value) * 100).toFixed(1)}%`

async function request(url, options) {
  let response
  try { response = await fetch(url, options) }
  catch { throw new Error('Cannot reach the local service. Start the API, then retry.') }
  const body = await response.json().catch(() => null)
  if (!response.ok || !body) throw new Error(typeof body?.detail === 'string' ? body.detail : 'The local service is unavailable. Check the server and retry.')
  return body
}

function Icon({ name, ...props }) {
  const paths = {
    upload: <><path d="M12 16V4m-4 4 4-4 4 4"/><path d="M4 15v5h16v-5"/></>,
    scan: <><path d="M8 3H3v5m13-5h5v5M3 16v5h5m13-5v5h-5M3 12h18"/></>,
    chart: <><path d="M4 3v17h17M8 16v-5m5 5V6m5 10v-8"/></>,
    arrow: <path d="M5 12h14m-6-6 6 6-6 6"/>,
    check: <path d="m5 12 4 4L19 6"/>,
    alert: <><path d="m12 3 10 18H2L12 3Z"/><path d="M12 9v5m0 3v.5"/></>,
    image: <><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8" cy="8" r="1"/><path d="m3 17 6-6 4 4 3-3 5 5"/></>,
  }
  return <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" {...props}>{paths[name]}</svg>
}

function Results({ data, error, retry }) {
  const [condition, setCondition] = useState('clean')
  if (error) return <div className="error-box" role="alert">{error}<button className="quiet-button ml-4" onClick={retry}>Retry</button></div>
  if (!data) return <p role="status" className="p-8 text-stone-400">Loading measured results…</p>
  const rows = data.brightness.filter(row => row.condition === condition)
  return <section className="results-page">
    <div className="results-intro"><div><h2>Evidence, not estimates.</h2><p>Measured on the same 117 MVTec AD carpet test images.</p></div><span className="pill">Frozen models & thresholds</span></div>
    <div className="flex flex-wrap items-center justify-between gap-4 mb-8">
      <div className="segmented" aria-label="Evaluation condition">{[['clean', 'Original'], ['brightness_0.8', 'Brightness ×0.8'], ['brightness_1.2', 'Brightness ×1.2']].map(([key, label]) => <button key={key} aria-pressed={condition === key} className={condition === key ? 'active' : ''} onClick={() => setCondition(key)}>{label}</button>)}</div>
      <span className="text-sm text-stone-400">28 normal / 89 anomalous</span>
    </div>
    <div className="results-grid">{rows.map(row => <article className="metric-panel" key={row.model}>
      <div className="flex items-center justify-between"><h3>{names[row.model]}</h3><Icon name="chart" /></div>
      <div className="auc-value">{number(row.auroc, 3)}<span>Image AUROC</span></div>
      <div className="metric-track" aria-hidden="true"><div style={{ width: percent(row.auroc) }} /></div>
      <dl className="stats"><div><dt>F1 score</dt><dd>{number(row.f1)}</dd></div><div><dt>Normal false positives</dt><dd>{percent(row.false_positive_rate)}</dd></div><div><dt>AUROC change from original</dt><dd>{Number(row.delta_auroc) >= 0 ? '+' : ''}{number(row.delta_auroc)}</dd></div></dl>
    </article>)}</div>
    <div className="table-wrap"><table><caption>Primary evaluation · original images</caption><thead><tr><th>Model</th><th>Precision</th><th>Recall</th><th>Pixel AUROC</th><th>Median / p95</th><th>Device</th></tr></thead><tbody>{data.clean.map(row => <tr key={row.model}><th>{names[row.model]}</th><td>{number(row.precision)}</td><td>{number(row.recall)}</td><td>{row.pixel_auroc ? number(row.pixel_auroc) : 'Not available'}</td><td>{number(row.latency_median_ms, 1)} / {number(row.latency_p95_ms, 1)} ms</td><td>{row.device.startsWith('cuda') ? 'GPU' : 'CPU'}</td></tr>)}</tbody></table></div>
    <div className="research-note"><Icon name="alert" /><p>Higher F1 alone can hide false alarms. Brightness variants reuse the same images; they are not independent samples. CPU and GPU timings describe practical deployment, not equal-compute comparisons. This study covers one carpet category, not arbitrary textiles.</p></div>
    <p className="source-note">Source: local measured comparison tables · MVTec AD / MVTec Software GmbH · Dataset CC BY-NC-SA 4.0</p>
  </section>
}

function App() {
  const [tab, setTab] = useState('inspect')
  const [model, setModel] = useState('patchcore')
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState('')
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [dragging, setDragging] = useState(false)
  const [split, setSplit] = useState(50)
  const [opacity, setOpacity] = useState(65)
  const [status, setStatus] = useState(null)
  const [statusError, setStatusError] = useState('')
  const [reports, setReports] = useState(null)
  const [reportError, setReportError] = useState('')
  const input = useRef(null)
  const selection = useRef(0)
  const refresh = () => { setStatusError(''); request('/api/status').then(setStatus).catch(e => { setStatus(null); setStatusError(e.message) }) }
  const fetchReports = () => { setReportError(''); request('/api/results').then(setReports).catch(e => setReportError(e.message)) }
  useEffect(() => { refresh(); fetchReports() }, [])
  useEffect(() => {
    if (!file) { setPreview(''); return }
    const url = URL.createObjectURL(file)
    setPreview(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  async function choose(candidate) {
    if (busy || !candidate) return
    const id = ++selection.current
    setResult(null); setError(''); setFile(null)
    if (candidate.size > 10 * 1024 * 1024) { setError('File exceeds 10 MB. Compress or resize it and try again.'); return }
    if (!['image/png', 'image/jpeg'].includes(candidate.type)) { setError('Choose a PNG or JPEG image.'); return }
    try {
      const bitmap = await createImageBitmap(candidate)
      const pixels = bitmap.width * bitmap.height
      bitmap.close()
      if (id !== selection.current) return
      if (pixels > 20_000_000) { setError('Image exceeds 20 megapixels. Resize it and try again.'); return }
      setFile(candidate)
    } catch { if (id === selection.current) setError('Cannot read this image. Choose an undamaged PNG or JPEG.') }
  }
  async function sample(key) {
    if (busy) return
    const id = ++selection.current
    setError(''); setFile(null); setResult(null)
    try {
      const response = await fetch(`/api/samples/${key}`)
      if (!response.ok) throw new Error('Local sample is missing. Upload your own carpet image.')
      const blob = await response.blob()
      if (id === selection.current) await choose(new File([blob], `carpet-sample-${key}.png`, { type: 'image/png' }))
    } catch (e) { if (id === selection.current) setError(e.message) }
  }
  async function inspect() {
    if (!file || busy) return
    setBusy(true); setResult(null); setError('')
    try {
      const response = await request(`/api/predict?model=${model}`, { method: 'POST', headers: { 'Content-Type': file.type }, body: file })
      setResult(response); setSplit(50)
    } catch (e) { setError(e.message) }
    finally { setBusy(false) }
  }
  const selected = status?.models.find(item => item.id === model)
  const anomaly = result?.decision === 'anomaly'
  return <div className="app-shell">
    <header className="topbar">
      <a className="brand" href="/" aria-label="Carpet Lab home"><span className="brand-mark" aria-hidden="true"><i/><i/><i/></span><span>Carpet<span className="font-light text-stone-400">Lab</span></span></a>
      <nav aria-label="Main navigation"><button className={tab === 'inspect' ? 'selected' : ''} aria-current={tab === 'inspect' ? 'page' : undefined} onClick={() => setTab('inspect')}><Icon name="scan"/>Inspection</button><button className={tab === 'results' ? 'selected' : ''} aria-current={tab === 'results' ? 'page' : undefined} onClick={() => setTab('results')}><Icon name="chart"/>Experiment results</button></nav>
      <div className="local-status"><span className={`status-dot ${status ? 'online' : ''}`}/>{status ? 'Local engine connected' : statusError ? 'Engine offline' : 'Connecting…'}</div>
    </header>
    <main>
      <div className="page-heading"><div><h1>{tab === 'inspect' ? 'A closer look at every thread.' : 'The experiment, in numbers.'}</h1><p>{tab === 'inspect' ? 'Inspect carpet texture. Find what breaks the pattern.' : 'Handcrafted texture features meet pretrained deep features.'}</p></div><span className="version-label">Texture inspection / Research edition</span></div>
      {statusError && <div className="error-box mb-5" role="alert">{statusError}<button className="quiet-button ml-4" onClick={refresh}>Reconnect</button></div>}
      {tab === 'results' ? <Results data={reports} error={reportError} retry={fetchReports}/> : <div className="workbench">
        <aside className="input-panel">
          <div className="panel-title"><span>Image source</span><Icon name="image"/></div>
          <input ref={input} id="image-upload" type="file" accept="image/png,image/jpeg" className="sr-only" aria-label="Upload carpet image" disabled={busy} onChange={e => { choose(e.target.files?.[0]); e.target.value = '' }}/>
          <button className={`dropzone ${dragging ? 'dragging' : ''}`} disabled={busy} onClick={() => input.current.click()} onDragOver={e => { e.preventDefault(); setDragging(true) }} onDragLeave={() => setDragging(false)} onDrop={e => { e.preventDefault(); setDragging(false); choose(e.dataTransfer.files?.[0]) }}><Icon name="upload" width="28" height="28"/><strong>{file ? 'Replace image' : 'Drop your image here'}</strong><span>or click to browse</span><small>PNG / JPEG · Up to 10 MB, 20 MP</small></button>
          <div className="samples"><span>Try a local sample</span><div><button disabled={busy} onClick={() => sample('a')}>Sample A</button><button disabled={busy} onClick={() => sample('b')}>Sample B</button></div></div>
          <fieldset disabled={busy} className="model-picker"><legend>Inspection model</legend>{models.map(item => <label key={item.id} className={`model-option ${model === item.id ? 'chosen' : ''}`}><input type="radio" name="model" value={item.id} checked={model === item.id} onChange={() => { setModel(item.id); setResult(null); setError('') }}/><div><strong>{item.name}</strong><span>{item.detail}</span><small>{item.tag}</small></div><span className="radio-indicator" aria-hidden="true"/></label>)}</fieldset>
          <button className="inspect-button" onClick={inspect} disabled={!file || busy || !status || !selected?.available}>{busy ? <><span className="spinner"/>Inspecting…</> : <>Run inspection<Icon name="arrow"/></>}</button>
          {selected && !selected.available && <p role="alert" className="text-amber-200 text-sm mt-3">Model artifacts are missing. Restore this model's local run to inspect.</p>}
          <p className="privacy-note">Images stay on this machine.<br/>Uploads are not saved.</p>
        </aside>
        <section className="viewer-panel" aria-label="Image inspection workspace">
          <div className="viewer-heading"><span className="truncate">{file ? file.name : 'Inspection workspace'}</span><span>{result ? `${result.width} × ${result.height} px` : 'Full image'}</span></div>
          <div className={`image-stage ${busy ? 'is-scanning' : ''}`} aria-busy={busy}>
            {preview ? <div className="image-frame"><img className="original-image" src={result?.original || preview} alt="Uploaded carpet texture"/>{result?.heatmap && <><div className="heatmap-layer" style={{ clipPath: `inset(0 0 0 ${split}%)` }}><img src={result.heatmap} style={{ opacity: opacity / 100 }} alt="Relative anomaly heatmap overlay"/></div><div className="split-line" style={{ left: `${split}%` }}><span>‹ ›</span></div><input className="comparison-slider" type="range" min="0" max="100" value={split} onChange={e => setSplit(Number(e.target.value))} aria-label="Original and heatmap comparison"/><span className="image-label left-label">Original</span><span className="image-label right-label">Heatmap</span></>}{busy && <div className="scan-line"/>}</div> : <div className="empty-stage"><div className="weave-art" aria-hidden="true"><div className="focus-bracket"/></div><h2>Every texture has a pattern.</h2><p>Upload a carpet image to inspect it.</p><span>Full-frame analysis · 256 × 256 model input</span></div>}
          </div>
          <div className="viewer-controls">{result?.heatmap ? <><label className="opacity-control">Overlay opacity<input type="range" min="0" max="100" value={opacity} onChange={e => setOpacity(Number(e.target.value))}/><output>{opacity}%</output></label><span className="color-key">Low <i/> High</span></> : <p>{result ? 'Localization unavailable for this model.' : 'PatchCore reveals relative anomaly intensity across the image.'}</p>}</div>
          <p className="viewer-footnote">{result?.heatmap ? 'Relative colors, scaled within this image. Display controls never change the score or decision.' : 'Use carpet surface images. This detector is not validated for other materials or objects.'}</p>
        </section>
        <aside className="result-panel" aria-label="Inspection result">
          <div className="panel-title"><span>Inspection result</span><span className="tiny-square"/></div>
          {error && <div className="error-box" role="alert">{error}</div>}
          <div className={`decision ${result ? anomaly ? 'anomaly' : 'normal' : ''}`} role="status" aria-live="polite"><div className="decision-symbol">{result ? <Icon name={anomaly ? 'alert' : 'check'} width="30" height="30"/> : <Icon name="scan" width="30" height="30"/>}</div><span>{busy ? 'Reading the texture' : result ? 'Inspection complete' : 'Ready when you are'}</span><h2>{busy ? 'Analyzing…' : result ? anomaly ? 'Anomaly detected' : 'Normal texture' : 'Awaiting image'}</h2><p>{busy ? 'First use loads the model. This can take a moment.' : result ? anomaly ? 'The score exceeds the frozen threshold.' : 'The score is at or below the frozen threshold.' : 'Choose an image and run an inspection to see the result.'}</p></div>
          <dl className="score-list"><div><dt>Raw anomaly score</dt><dd>{result ? number(result.score) : '—'}</dd></div><div><dt>Decision threshold</dt><dd>{result ? number(result.threshold) : '—'}</dd></div><div><dt>Model</dt><dd>{models.find(item => item.id === model).name}</dd></div><div><dt>Scoring time</dt><dd>{result ? `${number(result.inference_ms, 1)} ms` : '—'}</dd></div><div><dt>Device</dt><dd>{result ? result.device.startsWith('cuda') ? 'GPU / CUDA' : 'CPU' : '—'}</dd></div></dl>
          <div className="score-note"><span className="note-mark">i</span><p>Scores are not probabilities and cannot be compared across models. The threshold comes from held-out normal images.</p></div>
          <div className="localization-note"><h3>Spatial localization</h3><p>{model === 'patchcore' ? 'Drag the image divider to compare the original texture with the heatmap.' : 'Localization unavailable. This baseline produces an image-level score only.'}</p></div>
        </aside>
      </div>}
    </main>
    <footer><span><span className="status-dot online"/>On-device inference</span><span>{status?.device || 'Local research workspace'}</span><span>MVTec AD · Carpet category</span></footer>
  </div>
}

createRoot(document.getElementById('root')).render(<App />)
