import { BrowserRouter, Routes, Route } from 'react-router-dom';
import './index.css';
import './layout.css';
import AppLayout    from './components/AppLayout';
import Dashboard    from './pages/Dashboard';
import QuantAnalysis from './pages/QuantAnalysis';
import MarketView   from './pages/MarketView';
import Portfolio    from './pages/Portfolio';
import ModelStatus  from './pages/ModelStatus';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<AppLayout />}>
          <Route index              element={<Dashboard />} />
          <Route path="quant"       element={<QuantAnalysis />} />
          <Route path="market"      element={<MarketView />} />
          <Route path="portfolio"   element={<Portfolio />} />
          <Route path="model"       element={<ModelStatus />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;

