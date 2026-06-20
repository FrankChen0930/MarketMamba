import { BrowserRouter, Routes, Route } from 'react-router-dom';
import './index.css';
import './layout.css';
import AppLayout    from './components/AppLayout';
import Home         from './pages/Home';
import Dashboard    from './pages/Dashboard';
import QuantAnalysis from './pages/QuantAnalysis';
import MarketView   from './pages/MarketView';
import Portfolio    from './pages/Portfolio';
import ModelStatus  from './pages/ModelStatus';
import InvestmentSim from './pages/InvestmentSim';
import TradingSignals from './pages/TradingSignals';
import DualSignals    from './pages/DualSignals';


function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<AppLayout />}>
          <Route index              element={<Home />} />
          <Route path="dashboard"   element={<Dashboard />} />
          <Route path="quant"       element={<QuantAnalysis />} />
          <Route path="market"      element={<MarketView />} />
          <Route path="portfolio"   element={<Portfolio />} />
          <Route path="model"       element={<ModelStatus />} />
          <Route path="sim"         element={<InvestmentSim />} />
          <Route path="scanner"     element={<TradingSignals />} />
          <Route path="dual"        element={<DualSignals />} />

        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;

