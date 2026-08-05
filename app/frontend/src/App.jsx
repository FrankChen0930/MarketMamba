import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import './index.css';
import './layout.css';
import AppLayout    from './components/AppLayout';
import Home         from './pages/Home';
import QuantAnalysis from './pages/QuantAnalysis';
import MarketView   from './pages/MarketView';
import Pipeline      from './pages/Pipeline';
import Research      from './pages/Research';

// 高信念量化模型 /conviction/*
import ConvictionLayout      from './pages/ConvictionLayout';
import ConvictionHome        from './pages/ConvictionHome';
import TradingSignals        from './pages/TradingSignals';
import ConvictionPredictions from './pages/ConvictionPredictions';
import ConvictionBacktest    from './pages/ConvictionBacktest';
import ConvictionVersions    from './pages/ConvictionVersions';
import Portfolio             from './pages/Portfolio';

// 廣度量化模型 /breadth/*
import BreadthLayout      from './pages/BreadthLayout';
import BreadthHome        from './pages/BreadthHome';
import BreadthPredictions from './pages/BreadthPredictions';
import BreadthPortfolio   from './pages/BreadthPortfolio';
import InvestmentSim      from './pages/InvestmentSim';
import BreadthVersions    from './pages/BreadthVersions';
import DualSignals        from './pages/DualSignals';

// 模型分歧看板
import CompareBoard from './pages/CompareBoard';


function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<AppLayout />}>
          <Route index element={<Home />} />

          {/* 高信念量化模型 */}
          <Route path="conviction" element={<ConvictionLayout />}>
            <Route index            element={<ConvictionHome />} />
            <Route path="signals"     element={<TradingSignals />} />
            <Route path="predictions" element={<ConvictionPredictions />} />
            <Route path="backtest"    element={<ConvictionBacktest />} />
            <Route path="versions"    element={<ConvictionVersions />} />
            <Route path="portfolio"   element={<Portfolio />} />
          </Route>

          {/* 廣度量化模型 */}
          <Route path="breadth" element={<BreadthLayout />}>
            <Route index            element={<BreadthHome />} />
            <Route path="predictions" element={<BreadthPredictions />} />
            <Route path="portfolio"   element={<BreadthPortfolio />} />
            <Route path="backtest"    element={<InvestmentSim />} />
            <Route path="versions"    element={<BreadthVersions />} />
            <Route path="compare"     element={<DualSignals />} />
          </Route>

          <Route path="compare"  element={<CompareBoard />} />
          <Route path="quant"    element={<QuantAnalysis />} />
          <Route path="market"   element={<MarketView />} />
          {/* 研究紀錄（作品集頁）。/pipeline 併入 /research/pipeline */}
          <Route path="research" element={<Research />}>
            <Route path="pipeline" element={<Pipeline />} />
          </Route>

          {/* 舊路徑重導向（2026-07-30 頁面樹重整） */}
          <Route path="dashboard" element={<Navigate to="/breadth/predictions" replace />} />
          <Route path="scanner"   element={<Navigate to="/conviction/signals" replace />} />
          <Route path="dual"      element={<Navigate to="/breadth/compare" replace />} />
          <Route path="sim"       element={<Navigate to="/breadth/backtest" replace />} />
          <Route path="portfolio" element={<Navigate to="/conviction/portfolio" replace />} />
          <Route path="model"     element={<Navigate to="/breadth/versions" replace />} />
          <Route path="pipeline"  element={<Navigate to="/research/pipeline" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
