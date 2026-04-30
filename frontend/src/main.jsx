import React, { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { Home, Activity, IndianRupee, BarChart3, Database } from 'lucide-react';
import './styles.css';

const API = 'http://127.0.0.1:8000';
const initialForm = {
  area_sqft: 1800,
  bedrooms: 3,
  bathrooms: 2,
  location: 'Urban',
  property_age: 8,
  parking: 1,
  furnishing: 'Semi-Furnished',
  floors: 2,
  has_garden: 'No',
  distance_to_city_km: 7.5,
};

function App() {
  const [form, setForm] = useState(initialForm);
  const [health, setHealth] = useState('Checking');
  const [metrics, setMetrics] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [impact, setImpact] = useState([]);
  const [sample, setSample] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  async function apiGet(path) {
    const res = await fetch(`${API}${path}`);
    if (!res.ok) throw new Error(`${path} failed`);
    return res.json();
  }

  async function refresh() {
    try {
      setError('');
      const h = await apiGet('/health');
      setHealth(h.status);
      const m = await apiGet('/metrics');
      setMetrics(m);
      const i = await apiGet('/feature-impact');
      setImpact(i.top_features || []);
      const s = await apiGet('/sample-data?limit=6');
      setSample(s);
    } catch (e) {
      setHealth('Backend not connected');
      setError('Start backend: python -m uvicorn app.main:app --reload');
    }
  }

  useEffect(() => { refresh(); }, []);

  function update(key, value) {
    const numeric = ['area_sqft', 'bedrooms', 'bathrooms', 'property_age', 'parking', 'floors', 'distance_to_city_km'];
    setForm({ ...form, [key]: numeric.includes(key) ? Number(value) : value });
  }

  async function trainModel() {
    try {
      setLoading(true); setError('');
      const res = await fetch(`${API}/train`, { method: 'POST' });
      if (!res.ok) throw new Error('Training failed');
      const data = await res.json();
      setMetrics(data);
      const i = await apiGet('/feature-impact');
      setImpact(i.top_features || []);
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  }

  async function predict() {
    try {
      setLoading(true); setError(''); setPrediction(null);
      const res = await fetch(`${API}/predict`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(form) });
      const data = await res.json();
      if (!res.ok) throw new Error(JSON.stringify(data));
      setPrediction(data);
    } catch (e) { setError('Prediction failed. Check backend and input values.'); }
    finally { setLoading(false); }
  }

  const best = metrics?.results?.[metrics.best_model];

  return <div className="app">
    <header className="hero">
      <div>
        <p className="eyebrow">Machine Learning Regression Dashboard</p>
        <h1><Home size={36}/> House Price Prediction</h1>
        <p>Predict estimated property prices using area, rooms, location, furnishing and distance features.</p>
      </div>
      <div className="status"><Activity size={18}/> API: <b>{health}</b></div>
    </header>

    {error && <div className="error">{error}</div>}

    <section className="cards">
      <div className="card"><Database/><span>Dataset Rows</span><b>{metrics?.rows || '-'}</b></div>
      <div className="card"><BarChart3/><span>Best Model</span><b>{metrics?.best_model || '-'}</b></div>
      <div className="card"><span>R² Score</span><b>{best ? best.r2.toFixed(3) : '-'}</b></div>
      <div className="card"><span>RMSE</span><b>{best ? `₹${Math.round(best.rmse).toLocaleString('en-IN')}` : '-'}</b></div>
    </section>

    <main className="grid">
      <section className="panel formPanel">
        <h2>Property Details</h2>
        <div className="formGrid">
          <label>Area Sq.ft<input type="number" value={form.area_sqft} onChange={e=>update('area_sqft', e.target.value)} /></label>
          <label>Bedrooms<input type="number" value={form.bedrooms} onChange={e=>update('bedrooms', e.target.value)} /></label>
          <label>Bathrooms<input type="number" value={form.bathrooms} onChange={e=>update('bathrooms', e.target.value)} /></label>
          <label>Property Age<input type="number" value={form.property_age} onChange={e=>update('property_age', e.target.value)} /></label>
          <label>Parking<input type="number" value={form.parking} onChange={e=>update('parking', e.target.value)} /></label>
          <label>Floors<input type="number" value={form.floors} onChange={e=>update('floors', e.target.value)} /></label>
          <label>Distance to City KM<input type="number" value={form.distance_to_city_km} onChange={e=>update('distance_to_city_km', e.target.value)} /></label>
          <label>Location<select value={form.location} onChange={e=>update('location', e.target.value)}><option>Metro</option><option>Urban</option><option>Suburban</option><option>Rural</option></select></label>
          <label>Furnishing<select value={form.furnishing} onChange={e=>update('furnishing', e.target.value)}><option>Unfurnished</option><option>Semi-Furnished</option><option>Fully-Furnished</option></select></label>
          <label>Garden<select value={form.has_garden} onChange={e=>update('has_garden', e.target.value)}><option>Yes</option><option>No</option></select></label>
        </div>
        <div className="actions"><button onClick={predict} disabled={loading}>Predict Price</button><button className="secondary" onClick={trainModel} disabled={loading}>Train Model</button><button className="ghost" onClick={refresh}>Refresh</button></div>
      </section>

      <section className="panel resultPanel">
        <h2>Prediction Result</h2>
        {prediction ? <div className="priceBox"><IndianRupee size={28}/><div><small>Estimated Price</small><strong>₹{prediction.predicted_price_inr.toLocaleString('en-IN')}</strong><p>{prediction.predicted_price_lakh} lakh • {prediction.model_name}</p></div></div> : <p className="muted">Enter property details and click Predict Price.</p>}
        <h3>Top Feature Impacts</h3>
        <div className="bars">{impact.map((x, idx)=><div key={idx} className="barRow"><span>{x.feature}</span><div><i style={{width: `${Math.max(4, x.impact*100)}%`}}></i></div><em>{x.impact}</em></div>)}</div>
      </section>
    </main>

    <section className="panel tablePanel">
      <h2>Sample Housing Data</h2>
      <div className="tableWrap"><table><thead><tr>{sample[0] && Object.keys(sample[0]).map(k=><th key={k}>{k}</th>)}</tr></thead><tbody>{sample.map((r,i)=><tr key={i}>{Object.values(r).map((v,j)=><td key={j}>{typeof v === 'number' ? v.toLocaleString('en-IN') : v}</td>)}</tr>)}</tbody></table></div>
    </section>
  </div>
}

createRoot(document.getElementById('root')).render(<App />);
