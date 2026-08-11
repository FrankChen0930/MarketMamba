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
import ConvictionPredictions from './pages/ConvictionPredictions';
import ConvictionBacktest    from './pages/ConvictionBacktest';
import ConvictionVersions    from './pages/ConvictionVersions';
import Portfolio             from './pages/Portfolio';

// 廣度量化模型 /breadth/*（V6.2，目前上線中的版本）
import BreadthLayout      from './pages/BreadthLayout';
import BreadthHome        from './pages/BreadthHome';
import BreadthPredictions from './pages/BreadthPredictions';
import BreadthPortfolio   from './pages/BreadthPortfolio';
import BreadthVersions    from './pages/BreadthVersions';

// 前一版 V6.1 /legacy/*（已凍結，保留對照用）
import LegacyLayout  from './pages/LegacyLayout';
import LegacyHome    from './pages/LegacyHome';
import TradingSignals from './pages/TradingSignals';
import InvestmentSim  from './pages/InvestmentSim';
import DualSignals    from './pages/DualSignals';

// 模型分歧看板（尚未串接，暫時不放進主導覽）
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
            <Route path="predictions" element={<ConvictionPredictions />} />
            <Route path="backtest"    element={<ConvictionBacktest />} />
            <Route path="versions"    element={<ConvictionVersions />} />
            <Route path="portfolio"   element={<Portfolio />} />
          </Route>

          {/* 廣度量化模型（V6.2） */}
          <Route path="breadth" element={<BreadthLayout />}>
            <Route index            element={<BreadthHome />} />
            <Route path="predictions" element={<BreadthPredictions />} />
            <Route path="portfolio"   element={<BreadthPortfolio />} />
            <Route path="versions"    element={<BreadthVersions />} />
          </Route>

          {/* 前一版 V6.1（已凍結）。這三頁的元件本身沒有改動，只是換掛在這裡 */}
          <Route path="legacy" element={<LegacyLayout />}>
            <Route index          element={<LegacyHome />} />
            <Route path="signals" element={<TradingSignals />} />
            <Route path="sim"     element={<InvestmentSim />} />
            <Route path="dual"    element={<DualSignals />} />
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
          <Route path="scanner"   element={<Navigate to="/legacy/signals" replace />} />
          <Route path="dual"      element={<Navigate to="/legacy/dual" replace />} />
          <Route path="sim"       element={<Navigate to="/legacy/sim" replace />} />
          <Route path="portfolio" element={<Navigate to="/conviction/portfolio" replace />} />
          <Route path="model"     element={<Navigate to="/breadth/versions" replace />} />
          <Route path="pipeline"  element={<Navigate to="/research/pipeline" replace />} />

          {/* V6.1 拆頁重導向（2026-08-11）。三個舊網址都還活著，不留死連結 */}
          <Route path="conviction/signals" element={<Navigate to="/legacy/signals" replace />} />
          <Route path="breadth/backtest"   element={<Navigate to="/legacy/sim" replace />} />
          <Route path="breadth/compare"    element={<Navigate to="/legacy/dual" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
