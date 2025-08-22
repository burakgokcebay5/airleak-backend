"""
Advanced Pressure Range Optimization Module for LOB Test
Şanzıman LOB testi için gelişmiş basınç aralığı optimizasyonu
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from scipy import stats
from scipy.stats import norm
from scipy.optimize import minimize, differential_evolution
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import warnings
warnings.filterwarnings('ignore')

@dataclass
class PressureRangeResult:
    """Basınç aralığı optimizasyon sonucu"""
    channel_id: int
    optimal_pressure: float  # Optimal basınç değeri
    min_pressure: float      # Minimum basınç
    max_pressure: float      # Maximum basınç
    pressure_range: float    # Basınç aralığı (max - min)
    stability_score: float   # Stabilite skoru (düşük varyans = yüksek stabilite)
    confidence_interval: Tuple[float, float]
    success_rate: float      # Bu aralıkta başarı oranı
    variability_threshold: float  # İzin verilen maksimum değişkenlik
    
@dataclass
class ChannelOptimizationResult:
    """Kanal optimizasyon sonucu"""
    channel_results: Dict[int, PressureRangeResult]
    overall_success_rate: float
    bimal_correlation: float  # Bimal testi ile korelasyon
    stability_index: float    # Genel stabilite indeksi
    risk_score: float        # Risk skoru (0-1 arası)
    recommendations: List[str]
    convergence_history: Dict[str, List[float]]
    sensitivity_analysis: Dict[int, Dict[str, float]]

class PressureRangeOptimizer:
    """
    Basınç aralığı optimizasyonu için gelişmiş algoritmalar
    Her kanal için min-max değerleri ve değişim aralığını optimize eder
    """
    
    def __init__(self, data: pd.DataFrame, bimal_data: Optional[pd.DataFrame] = None):
        self.data = data
        self.bimal_data = bimal_data
        self.channel_stats = self._calculate_channel_statistics()
        self.correlation_matrix = self._calculate_correlations()
        
    def _calculate_channel_statistics(self) -> Dict[int, Dict[str, float]]:
        """Her kanal için istatistiksel analiz"""
        stats_dict = {}
        
        for i in range(1, 5):
            # Firebase'den gelen veriye göre kolon isimlerini ayarla
            channel_col = f'channel_{i}' if f'channel_{i}' in self.data.columns else f'deger_{i}_parsed'
            if channel_col in self.data.columns:
                channel_data = self.data[channel_col].dropna()
                
                # İstatistiksel değerleri hesapla
                # Test verilerinde tek değer var, min-max ayrı kolonlarda değil
                # Bu yüzden değerlerin dağılımından makul bir aralık tahmin ediyoruz
                
                # Verideki gerçek min-max değerler
                data_min = channel_data.min() if len(channel_data) > 0 else 0
                data_max = channel_data.max() if len(channel_data) > 0 else 3
                data_std = channel_data.std() if len(channel_data) > 1 else 0.1
                
                # Test sırasında olası değişkenlik: standart sapmanın 0.5-1 katı kadar
                # Gerçek testlerde basınç 0.05-0.2 bar arasında değişir
                estimated_range = min(0.2, max(0.05, data_std * 0.75))
                
                stats_dict[i] = {
                    'mean': channel_data.mean() if len(channel_data) > 0 else 1.5,
                    'std': data_std,
                    'min': data_min,
                    'max': data_max,
                    'range_mean': estimated_range,  # Tahmini test aralığı
                    'range_std': estimated_range * 0.3,  # Aralıktaki değişkenlik
                    'cv': data_std / (channel_data.mean() + 1e-6) if len(channel_data) > 0 else 0.1,
                    'skewness': stats.skew(channel_data) if len(channel_data) > 1 else 0,
                    'kurtosis': stats.kurtosis(channel_data) if len(channel_data) > 1 else 0,
                    'q1': channel_data.quantile(0.25) if len(channel_data) > 0 else 1.0,
                    'q3': channel_data.quantile(0.75) if len(channel_data) > 0 else 2.0,
                    'median': channel_data.median() if len(channel_data) > 0 else 1.5
                }
        
        return stats_dict
    
    def _calculate_correlations(self) -> np.ndarray:
        """Kanallar arası korelasyon matrisi"""
        channels = []
        for i in range(1, 5):
            channel_col = f'channel_{i}' if f'channel_{i}' in self.data.columns else f'deger_{i}_parsed'
            if channel_col in self.data.columns:
                channels.append(self.data[channel_col].fillna(0))
        
        if len(channels) > 0:
            return np.corrcoef(channels)
        return np.eye(4)
    
    def optimize_pressure_ranges(self, method='advanced_bayesian') -> ChannelOptimizationResult:
        """
        Basınç aralıklarını optimize et
        
        Methods:
        - advanced_bayesian: Gelişmiş Bayesian optimizasyon
        - multi_objective: Çok amaçlı optimizasyon (başarı oranı + stabilite)
        - robust: Robust optimizasyon (outlier'lara dayanıklı)
        - adaptive: Adaptif optimizasyon (dinamik aralık ayarlama)
        """
        
        if method == 'advanced_bayesian':
            return self._advanced_bayesian_optimization()
        elif method == 'multi_objective':
            return self._multi_objective_optimization()
        elif method == 'robust':
            return self._robust_optimization()
        elif method == 'adaptive':
            return self._adaptive_optimization()
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _advanced_bayesian_optimization(self) -> ChannelOptimizationResult:
        """Gelişmiş Bayesian optimizasyon ile basınç aralıklarını bul"""
        channel_results = {}
        convergence_history = {'success_rate': [], 'stability': [], 'risk': []}
        
        for channel_id in range(1, 5):
            if channel_id not in self.channel_stats:
                continue
            
            stats = self.channel_stats[channel_id]
            
            # Gaussian Process modeli oluştur
            kernel = Matern(length_scale=1.0, nu=2.5)
            gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
            
            # Başlangıç noktaları
            n_initial = 20
            X_sample = []
            y_sample = []
            
            for _ in range(n_initial):
                # Rastgele basınç parametreleri
                pressure = np.random.uniform(stats['min'], stats['max'])
                range_size = np.random.uniform(0, stats['range_mean'] + stats['range_std'])
                
                # Bu parametrelerle başarı oranını hesapla
                success_rate = self._evaluate_pressure_config(
                    channel_id, pressure, range_size
                )
                
                X_sample.append([pressure, range_size])
                y_sample.append(success_rate)
            
            # GP modelini eğit
            X_sample = np.array(X_sample)
            y_sample = np.array(y_sample)
            gp.fit(X_sample, y_sample)
            
            # Optimizasyon döngüsü
            for iteration in range(50):
                # Acquisition function (Expected Improvement)
                def acquisition(x):
                    mu, sigma = gp.predict(x.reshape(1, -1), return_std=True)
                    best_y = np.max(y_sample)
                    
                    with np.errstate(divide='warn'):
                        Z = (mu - best_y) / sigma
                        ei = sigma * (Z * norm.cdf(Z) + norm.pdf(Z))
                    
                    return -ei[0]
                
                # Yeni nokta bul
                bounds = [(stats['min'], stats['max']), 
                         (0, stats['range_mean'] + 2 * stats['range_std'])]
                
                result = minimize(
                    acquisition,
                    x0=[stats['mean'], stats['range_mean']],
                    bounds=bounds,
                    method='L-BFGS-B'
                )
                
                # Yeni noktayı değerlendir
                new_x = result.x
                new_y = self._evaluate_pressure_config(channel_id, new_x[0], new_x[1])
                
                # Modeli güncelle
                X_sample = np.vstack([X_sample, new_x])
                y_sample = np.append(y_sample, new_y)
                gp.fit(X_sample, y_sample)
            
            # En iyi konfigürasyonu bul
            best_idx = np.argmax(y_sample)
            best_pressure = X_sample[best_idx, 0]
            best_range = X_sample[best_idx, 1]
            
            # Güven aralığını hesapla
            mu, sigma = gp.predict(X_sample[best_idx].reshape(1, -1), return_std=True)
            confidence_interval = (
                max(0, best_pressure - 2 * sigma[0]),
                min(3, best_pressure + 2 * sigma[0])
            )
            
            # Stabilite skorunu hesapla
            stability_score = 1.0 / (1.0 + stats['cv'])
            
            channel_results[channel_id] = PressureRangeResult(
                channel_id=channel_id,
                optimal_pressure=best_pressure,
                min_pressure=max(0, best_pressure - best_range/2),
                max_pressure=min(3, best_pressure + best_range/2),
                pressure_range=best_range,
                stability_score=stability_score,
                confidence_interval=confidence_interval,
                success_rate=y_sample[best_idx],
                variability_threshold=stats['range_mean'] + stats['range_std']
            )
            
            convergence_history['success_rate'].append(y_sample[best_idx])
            convergence_history['stability'].append(stability_score)
        
        # Genel metrikleri hesapla
        overall_success = np.mean([r.success_rate for r in channel_results.values()])
        stability_index = np.mean([r.stability_score for r in channel_results.values()])
        
        # Bimal korelasyonunu hesapla
        bimal_correlation = self._calculate_bimal_correlation(channel_results)
        
        # Risk skorunu hesapla
        risk_score = self._calculate_risk_score(channel_results)
        
        # Öneriler oluştur
        recommendations = self._generate_recommendations(channel_results)
        
        # Duyarlılık analizi
        sensitivity = self._sensitivity_analysis(channel_results)
        
        return ChannelOptimizationResult(
            channel_results=channel_results,
            overall_success_rate=overall_success,
            bimal_correlation=bimal_correlation,
            stability_index=stability_index,
            risk_score=risk_score,
            recommendations=recommendations,
            convergence_history=convergence_history,
            sensitivity_analysis=sensitivity
        )
    
    def _multi_objective_optimization(self) -> ChannelOptimizationResult:
        """Çok amaçlı optimizasyon: Başarı oranı + Stabilite + Min varyans"""
        channel_results = {}
        
        for channel_id in range(1, 5):
            if channel_id not in self.channel_stats:
                continue
            
            stats = self.channel_stats[channel_id]
            
            # Çok amaçlı fonksiyon
            def multi_objective(x):
                pressure, range_size = x
                
                # Amaç 1: BİMAL UYUMU (EN ÖNEMLİ - Bimal referans test!)
                bimal_score = self._evaluate_bimal_compatibility(channel_id, pressure)
                
                # Amaç 2: Başarı oranını maksimize et
                success_rate = self._evaluate_pressure_config(channel_id, pressure, range_size)
                
                # Amaç 3: Stabiliteyi maksimize et (düşük varyans, dar aralık)
                variance_penalty = range_size / (stats['range_mean'] + 1e-6)
                
                # Ağırlıklı toplam - BİMAL EN YÜKSEK AĞIRLIK
                # Bimal: %40, Başarı: %35, Stabilite: %25
                return -(0.4 * bimal_score + 0.35 * success_rate + 0.25 * (1 - variance_penalty))
            
            # Differential Evolution ile optimize et
            # Aralık için makul sınırlar: min 0.05, max 0.3 bar
            bounds = [(stats['min'], stats['max']), 
                     (0.05, min(0.3, stats['range_mean'] + stats['range_std']))]
            
            result = differential_evolution(
                multi_objective,
                bounds,
                maxiter=100,
                popsize=15,
                seed=42
            )
            
            best_pressure = result.x[0]
            best_range = min(0.3, max(0.05, result.x[1]))  # Aralığı sınırla
            
            # Min/max değerleri hesapla
            min_pressure = max(0, best_pressure - best_range/2)
            max_pressure = min(3, best_pressure + best_range/2)
            actual_range = max_pressure - min_pressure
            
            channel_results[channel_id] = PressureRangeResult(
                channel_id=channel_id,
                optimal_pressure=best_pressure,
                min_pressure=min_pressure,
                max_pressure=max_pressure,
                pressure_range=actual_range,
                stability_score=1.0 / (1.0 + stats['cv']),
                confidence_interval=(best_pressure - 0.1, best_pressure + 0.1),
                success_rate=self._evaluate_pressure_config(channel_id, best_pressure, actual_range),
                variability_threshold=stats['range_mean'] + stats['range_std']
            )
        
        return self._create_optimization_result(channel_results)
    
    def _robust_optimization(self) -> ChannelOptimizationResult:
        """Robust optimizasyon: Outlier'lara ve gürültüye dayanıklı"""
        channel_results = {}
        
        for channel_id in range(1, 5):
            if channel_id not in self.channel_stats:
                continue
            
            channel_col = f'channel_{channel_id}' if f'channel_{channel_id}' in self.data.columns else f'deger_{channel_id}_parsed'
            if channel_col not in self.data.columns:
                continue
            channel_data = self.data[channel_col].dropna()
            
            # Robust istatistikler (median ve MAD)
            median = channel_data.median()
            mad = np.median(np.abs(channel_data - median))
            
            # Trimmed mean (üst ve alt %10'u at)
            trimmed_mean = stats.trim_mean(channel_data, 0.1)
            
            # IQR bazlı aralık
            q1 = channel_data.quantile(0.25)
            q3 = channel_data.quantile(0.75)
            iqr = q3 - q1
            
            # Robust optimal değerler
            optimal_pressure = trimmed_mean
            
            # Makul bir aralık belirle (çok dar veya çok geniş olmamalı)
            # IQR'nin yarısını kullan, minimum 0.05, maksimum 0.3 bar
            pressure_range = min(0.3, max(0.05, iqr * 0.5))
            
            # Min/max değerleri optimal etrafında simetrik olarak ayarla
            min_pressure = max(0, optimal_pressure - pressure_range/2)
            max_pressure = min(3, optimal_pressure + pressure_range/2)
            
            # Gerçek aralığı güncelle (sınırlamalardan sonra)
            actual_range = max_pressure - min_pressure
            
            # Winsorized varyans (extreme değerleri sınırla)
            if len(channel_data) > 10:
                winsorized_data = stats.mstats.winsorize(channel_data, limits=[0.05, 0.05])
                cv = np.std(winsorized_data) / (np.mean(winsorized_data) + 1e-6)
            else:
                # Az veri varsa winsorize yapma
                cv = channel_data.std() / (channel_data.mean() + 1e-6)
            
            stability_score = 1.0 / (1.0 + cv)
            
            channel_results[channel_id] = PressureRangeResult(
                channel_id=channel_id,
                optimal_pressure=optimal_pressure,
                min_pressure=min_pressure,
                max_pressure=max_pressure,
                pressure_range=actual_range,
                stability_score=stability_score,
                confidence_interval=(q1, q3),
                success_rate=self._evaluate_pressure_config(channel_id, optimal_pressure, actual_range),
                variability_threshold=2 * mad
            )
        
        return self._create_optimization_result(channel_results)
    
    def _adaptive_optimization(self) -> ChannelOptimizationResult:
        """Adaptif optimizasyon: Zaman içinde değişen koşullara uyum sağlar"""
        channel_results = {}
        
        for channel_id in range(1, 5):
            if channel_id not in self.channel_stats:
                continue
            
            channel_col = f'channel_{channel_id}' if f'channel_{channel_id}' in self.data.columns else f'deger_{channel_id}_parsed'
            if channel_col not in self.data.columns:
                continue
            channel_data = self.data[channel_col].dropna()
            
            # Zaman bazlı analiz (son veriler daha önemli)
            if 'timestamp' in self.data.columns or 'date' in self.data.columns:
                # Son verilere daha fazla ağırlık ver
                weights = np.linspace(0.5, 1.0, len(channel_data))
            else:
                weights = np.ones(len(channel_data))
            
            # Ağırlıklı ortalama ve standart sapma
            weighted_mean = np.average(channel_data, weights=weights)
            weighted_var = np.average((channel_data - weighted_mean)**2, weights=weights)
            weighted_std = np.sqrt(weighted_var)
            
            # Adaptif aralık (son dönem performansına göre)
            recent_data = channel_data.tail(int(len(channel_data) * 0.3))
            recent_std = recent_data.std()
            
            # Kalman filtresi benzeri güncelleme
            process_noise = 0.01
            measurement_noise = weighted_std ** 2 + 1e-6
            
            kalman_gain = process_noise / (process_noise + measurement_noise)
            optimal_pressure = weighted_mean + kalman_gain * (recent_data.mean() - weighted_mean)
            
            # Adaptif aralık: standart sapmanın 1.5 katı, min 0.05, max 0.25 bar
            pressure_range = min(0.25, max(0.05, recent_std * 1.5))
            
            # Min/max değerleri optimal etrafında simetrik olarak ayarla
            min_pressure = max(0, optimal_pressure - pressure_range/2)
            max_pressure = min(3, optimal_pressure + pressure_range/2)
            
            # Gerçek aralığı güncelle
            actual_range = max_pressure - min_pressure
            
            channel_results[channel_id] = PressureRangeResult(
                channel_id=channel_id,
                optimal_pressure=optimal_pressure,
                min_pressure=min_pressure,
                max_pressure=max_pressure,
                pressure_range=actual_range,
                stability_score=1.0 / (1.0 + weighted_std / weighted_mean),
                confidence_interval=(optimal_pressure - weighted_std, optimal_pressure + weighted_std),
                success_rate=self._evaluate_pressure_config(channel_id, optimal_pressure, actual_range),
                variability_threshold=weighted_std * 2
            )
        
        return self._create_optimization_result(channel_results)
    
    def _evaluate_pressure_config(self, channel_id: int, pressure: float, range_size: float) -> float:
        """Belirli bir basınç konfigürasyonunun başarı oranını değerlendir"""
        channel_col = f'channel_{channel_id}' if f'channel_{channel_id}' in self.data.columns else f'deger_{channel_id}_parsed'
        if channel_col not in self.data.columns:
            return 0.5  # Veri yoksa varsayılan değer
        
        # Test sonuçlarını al
        test_results = self.data['test_result'] if 'test_result' in self.data.columns else None
        
        # Basınç aralığını belirle
        min_pressure = max(0, pressure - range_size / 2)
        max_pressure = min(3, pressure + range_size / 2)
        
        if test_results is not None:
            # Channel data ve test results'ı birlikte filtrele
            mask = self.data[channel_col].notna()
            filtered_data = self.data[mask].copy()
            
            if len(filtered_data) > 0:
                # Aralıkta olan testleri bul
                in_range = (filtered_data[channel_col] >= min_pressure) & (filtered_data[channel_col] <= max_pressure)
                
                if in_range.any():
                    # OK olanların oranını hesapla
                    ok_count = (filtered_data.loc[in_range, 'test_result'] == 'OK').sum()
                    total_count = in_range.sum()
                    success_rate = ok_count / total_count if total_count > 0 else 0
                else:
                    success_rate = 0
            else:
                success_rate = 0.5
        else:
            # Test sonucu yoksa istatistiksel tahmin yap
            stats_info = self.channel_stats.get(channel_id, {})
            if 'mean' in stats_info and 'std' in stats_info:
                # Optimal değere yakınlık ve dar aralık avantajı
                mean_distance = abs(pressure - stats_info['mean']) / (stats_info['std'] + 0.1)
                range_penalty = range_size / 0.5  # 0.5 bar'dan büyük aralıklar cezalandırılır
                
                # Başarı oranı: optimal değere yakın ve dar aralık = yüksek başarı
                success_rate = max(0, min(1, np.exp(-mean_distance * 0.5) * (1 - range_penalty * 0.3)))
            else:
                success_rate = 0.5
        
        return max(0.0, min(1.0, success_rate))  # 0-1 arasında sınırla
    
    def _evaluate_bimal_compatibility(self, channel_id: int, pressure: float) -> float:
        """Bimal testi ile uyumluluk skorunu hesapla - BİMAL REFERANS TEST"""
        if self.bimal_data is None:
            return 0.5  # Bimal verisi yoksa nötr skor
        
        channel_col = f'channel_{channel_id}' if f'channel_{channel_id}' in self.bimal_data.columns else f'deger_{channel_id}_parsed'
        if channel_col not in self.bimal_data.columns:
            return 0.5
        
        # Hem Bimal hem de Airleak test sonuçlarını al
        if 'bimal_result' in self.bimal_data.columns and 'test_result' in self.bimal_data.columns:
            # BİMAL REFERANS olduğu için öncelikler değişti!
            bimal_ok_all = self.bimal_data[self.bimal_data['bimal_result'] == 'OK']
            bimal_nok_all = self.bimal_data[self.bimal_data['bimal_result'] == 'NOK']
            
            # Çapraz analiz
            bimal_ok_airleak_ok = self.bimal_data[
                (self.bimal_data['bimal_result'] == 'OK') & 
                (self.bimal_data['test_result'] == 'OK')
            ]
            bimal_ok_airleak_nok = self.bimal_data[
                (self.bimal_data['bimal_result'] == 'OK') & 
                (self.bimal_data['test_result'] == 'NOK')
            ]
            
            scores = []
            
            # 1. BİMAL OK olan TÜM basınçlara yakınlık (EN YÜKSEK AĞIRLIK - Bimal referans!)
            if len(bimal_ok_all) > 0 and channel_col in bimal_ok_all.columns:
                pressures = bimal_ok_all[channel_col].dropna()
                if len(pressures) > 0:
                    # Bimal OK basınçlarına yakın olması ÇOK ÖNEMLİ
                    distance = np.min(np.abs(pressures - pressure))
                    scores.append(np.exp(-distance * 0.3) * 1.5)  # Yüksek ağırlık (1.5x)
            
            # 2. Bimal OK ama Airleak NOK olanlar - ANORMALLİK (bu basınçlar aslında İYİ!)
            if len(bimal_ok_airleak_nok) > 0 and channel_col in bimal_ok_airleak_nok.columns:
                pressures = bimal_ok_airleak_nok[channel_col].dropna()
                if len(pressures) > 0:
                    # Bu basınçlara YAKIN olması İYİ (Bimal OK demiş!)
                    distance = np.min(np.abs(pressures - pressure))
                    scores.append(np.exp(-distance * 0.5))  # Yakın olması iyi
                    
                    # Airleak'in yanlış NOK dediği bir aralık - hassasiyet ayarı önerisi
                    if distance < 0.1:  # Çok yakınsa
                        scores.append(1.2)  # Bonus puan
            
            # 3. Bimal NOK olan basınçlardan UZAKLIK (önemli)
            if len(bimal_nok_all) > 0 and channel_col in bimal_nok_all.columns:
                pressures = bimal_nok_all[channel_col].dropna()
                if len(pressures) > 0:
                    distance = np.min(np.abs(pressures - pressure))
                    # Bimal NOK basınçlarından uzak olması İYİ
                    scores.append(1 - np.exp(-distance * 2))
            
            # NOK kanal bilgisi varsa değerlendir
            if 'nok_channels' in self.bimal_data.columns:
                nok_records = self.bimal_data[self.bimal_data['nok_channels'].notna()]
                channel_nok_count = 0
                for channels in nok_records['nok_channels']:
                    if isinstance(channels, list) and channel_id in channels:
                        channel_nok_count += 1
                
                # NOK oranı düşük kanallar daha iyi
                if len(nok_records) > 0:
                    nok_ratio = channel_nok_count / len(nok_records)
                    scores.append(1 - nok_ratio)
            
            return np.mean(scores) if scores else 0.5
        
        # Sadece Bimal sonucu varsa (eski yöntem)
        elif 'bimal_result' in self.bimal_data.columns:
            bimal_ok = self.bimal_data[self.bimal_data['bimal_result'] == 'OK']
            if len(bimal_ok) > 0 and channel_col in bimal_ok.columns:
                pressures = bimal_ok[channel_col].dropna()
                if len(pressures) > 0:
                    distance = np.min(np.abs(pressures - pressure))
                    return np.exp(-distance)
        
        return 0.5
    
    def _calculate_bimal_correlation(self, channel_results: Dict[int, PressureRangeResult]) -> float:
        """Optimizasyon sonuçlarının Bimal testi ile korelasyonunu hesapla (r² değeri)"""
        if self.bimal_data is None or 'bimal_result' not in self.bimal_data.columns:
            return 0.0
        
        # Airleak ve Bimal test sonuçlarını karşılaştır
        if 'test_result' not in self.bimal_data.columns:
            return 0.0
            
        # Her iki testin de sonucu olan kayıtları al
        valid_data = self.bimal_data.dropna(subset=['test_result', 'bimal_result'])
        
        if len(valid_data) < 10:  # Minimum veri gereksinimi
            return 0.0
        
        # Test sonuçlarını sayısal değerlere çevir (OK=1, NOK=0)
        airleak_values = (valid_data['test_result'].str.upper() == 'OK').astype(int)
        bimal_values = (valid_data['bimal_result'].str.upper() == 'OK').astype(int)
        
        # Pearson korelasyon katsayısını hesapla
        if len(airleak_values) > 0 and len(bimal_values) > 0:
            # Varyans kontrolü
            if np.var(airleak_values) == 0 or np.var(bimal_values) == 0:
                return 0.0
            
            # Korelasyon hesapla
            correlation = np.corrcoef(airleak_values, bimal_values)[0, 1]
            
            # r² değerini döndür
            r_squared = correlation ** 2 if not np.isnan(correlation) else 0.0
            return r_squared
        
        return 0.0
    
    def _calculate_risk_score(self, channel_results: Dict[int, PressureRangeResult]) -> float:
        """Risk skorunu hesapla (0: düşük risk, 1: yüksek risk)"""
        risks = []
        
        for channel_id, result in channel_results.items():
            # Risk faktörleri
            # 1. Düşük başarı oranı riski
            success_risk = 1 - result.success_rate
            
            # 2. Yüksek varyans riski
            variance_risk = 1 - result.stability_score
            
            # 3. Dar güven aralığı riski
            confidence_width = result.confidence_interval[1] - result.confidence_interval[0]
            confidence_risk = 1 / (1 + confidence_width)
            
            # Toplam risk
            channel_risk = 0.4 * success_risk + 0.4 * variance_risk + 0.2 * confidence_risk
            risks.append(channel_risk)
        
        return np.mean(risks) if risks else 0.5
    
    def _generate_recommendations(self, channel_results: Dict[int, PressureRangeResult]) -> List[str]:
        """Optimizasyon sonuçlarına göre öneriler oluştur"""
        recommendations = []
        
        for channel_id, result in channel_results.items():
            # Düşük başarı oranı uyarısı
            if result.success_rate < 0.7:
                recommendations.append(
                    f"Kanal {channel_id}: Başarı oranı düşük ({result.success_rate:.1%}). "
                    f"Basınç aralığını daraltmayı düşünün."
                )
            
            # Yüksek varyans uyarısı
            if result.stability_score < 0.6:
                recommendations.append(
                    f"Kanal {channel_id}: Stabilite düşük. "
                    f"Test ekipmanını kalibre edin veya proses kontrolünü artırın."
                )
            
            # Geniş aralık uyarısı
            if result.pressure_range > result.variability_threshold:
                recommendations.append(
                    f"Kanal {channel_id}: Basınç aralığı çok geniş ({result.pressure_range:.3f} bar). "
                    f"Maksimum {result.variability_threshold:.3f} bar önerilir."
                )
            
            # Optimal basınç önerisi
            recommendations.append(
                f"Kanal {channel_id}: Optimal basınç {result.optimal_pressure:.3f} bar "
                f"({result.min_pressure:.3f} - {result.max_pressure:.3f} bar aralığında)"
            )
        
        # Bimal korelasyon analizi
        if self.bimal_data is not None and 'bimal_result' in self.bimal_data.columns:
            bimal_correlation = self._calculate_bimal_correlation(channel_results)
            
            if bimal_correlation < 0.5:
                recommendations.append(
                    "⚠️ Bimal testi ile düşük korelasyon. Optimizasyon parametreleri Bimal gereksinimlerine uygun değil."
                )
            elif bimal_correlation > 0.8:
                recommendations.append(
                    "✅ Bimal testi ile yüksek korelasyon. Parametreler her iki test için de uygun."
                )
            
            # Çapraz analiz önerileri
            if 'test_result' in self.bimal_data.columns:
                bimal_ok_airleak_nok = self.bimal_data[
                    (self.bimal_data['bimal_result'] == 'OK') & 
                    (self.bimal_data['test_result'] == 'NOK')
                ]
                bimal_nok_airleak_ok = self.bimal_data[
                    (self.bimal_data['bimal_result'] == 'NOK') & 
                    (self.bimal_data['test_result'] == 'OK')
                ]
                
                if len(bimal_ok_airleak_nok) > 5:
                    recommendations.append(
                        f"⚠️ DİKKAT: {len(bimal_ok_airleak_nok)} üründe Bimal OK ama Airleak NOK (YANLIŞ RET!). "
                        "Bimal referans test olduğu için bu ürünler aslında SAĞLAM. "
                        "Airleak testinin hassasiyet ayarı düşürülmeli veya basınç aralığı genişletilmeli."
                    )
                    
                    # Hangi kanallarda sorun var?
                    for ch_id in range(1, 5):
                        ch_col = f'channel_{ch_id}' if f'channel_{ch_id}' in bimal_ok_airleak_nok.columns else f'deger_{ch_id}_parsed'
                        if ch_col in bimal_ok_airleak_nok.columns:
                            problem_pressures = bimal_ok_airleak_nok[ch_col].dropna()
                            if len(problem_pressures) > 0:
                                recommendations.append(
                                    f"   → Kanal {ch_id}: {problem_pressures.mean():.3f} bar civarı Bimal OK veriyor, "
                                    f"aralığı {problem_pressures.min():.3f}-{problem_pressures.max():.3f} bar olarak ayarlayın."
                                )
                
                if len(bimal_nok_airleak_ok) > 5:
                    recommendations.append(
                        f"✅ {len(bimal_nok_airleak_ok)} üründe Bimal NOK ama Airleak OK. "
                        "Bu ürünlerin Bimal'da NOK çıkma sebebi incelenmeli (farklı parametre hassasiyeti)."
                    )
        
        # Genel öneriler
        overall_success = np.mean([r.success_rate for r in channel_results.values()])
        if overall_success > 0.85:
            recommendations.append("✅ Genel başarı oranı yüksek. Mevcut parametreler iyi optimize edilmiş.")
        elif overall_success > 0.7:
            recommendations.append("⚠️ Genel başarı oranı orta seviyede. İyileştirme potansiyeli var.")
        else:
            recommendations.append("❌ Genel başarı oranı düşük. Acil iyileştirme gerekli.")
        
        return recommendations
    
    def _sensitivity_analysis(self, channel_results: Dict[int, PressureRangeResult]) -> Dict[int, Dict[str, float]]:
        """Duyarlılık analizi: Her kanalın diğer kanallara etkisi"""
        sensitivity = {}
        
        for channel_id in channel_results:
            sensitivity[channel_id] = {}
            
            # Basınç değişiminin etkisi
            base_pressure = channel_results[channel_id].optimal_pressure
            delta = 0.1  # 0.1 bar değişim
            
            # Yukarı değişim
            up_success = self._evaluate_pressure_config(
                channel_id, 
                base_pressure + delta,
                channel_results[channel_id].pressure_range
            )
            
            # Aşağı değişim
            down_success = self._evaluate_pressure_config(
                channel_id,
                base_pressure - delta,
                channel_results[channel_id].pressure_range
            )
            
            # Normalize sensitivity: success change per unit change (0-1 scale)
            pressure_sens_raw = abs(up_success - down_success) / (2 * delta)
            sensitivity[channel_id]['pressure_sensitivity'] = min(1.0, pressure_sens_raw / 10)  # Normalize to 0-1
            
            # Aralık değişiminin etkisi
            base_range = channel_results[channel_id].pressure_range
            range_delta = 0.05
            
            narrow_success = self._evaluate_pressure_config(
                channel_id,
                base_pressure,
                max(0, base_range - range_delta)
            )
            
            wide_success = self._evaluate_pressure_config(
                channel_id,
                base_pressure,
                base_range + range_delta
            )
            
            # Normalize range sensitivity
            range_sens_raw = abs(wide_success - narrow_success) / (2 * range_delta)
            sensitivity[channel_id]['range_sensitivity'] = min(1.0, range_sens_raw / 10)  # Normalize to 0-1
            
            # Diğer kanallarla etkileşim (korelasyon matrisi kullanarak)
            if channel_id <= len(self.correlation_matrix):
                correlations = self.correlation_matrix[channel_id - 1]
                for other_channel in range(1, 5):
                    if other_channel != channel_id and other_channel <= len(correlations):
                        sensitivity[channel_id][f'correlation_ch{other_channel}'] = abs(correlations[other_channel - 1])
        
        return sensitivity
    
    def _create_optimization_result(self, channel_results: Dict[int, PressureRangeResult]) -> ChannelOptimizationResult:
        """Optimizasyon sonuç nesnesini oluştur"""
        overall_success = np.mean([r.success_rate for r in channel_results.values()])
        stability_index = np.mean([r.stability_score for r in channel_results.values()])
        bimal_correlation = self._calculate_bimal_correlation(channel_results)
        risk_score = self._calculate_risk_score(channel_results)
        recommendations = self._generate_recommendations(channel_results)
        sensitivity = self._sensitivity_analysis(channel_results)
        
        convergence_history = {
            'success_rate': [r.success_rate for r in channel_results.values()],
            'stability': [r.stability_score for r in channel_results.values()],
            'risk': [risk_score] * len(channel_results)
        }
        
        return ChannelOptimizationResult(
            channel_results=channel_results,
            overall_success_rate=overall_success,
            bimal_correlation=bimal_correlation,
            stability_index=stability_index,
            risk_score=risk_score,
            recommendations=recommendations,
            convergence_history=convergence_history,
            sensitivity_analysis=sensitivity
        )

def analyze_pressure_patterns(data: pd.DataFrame) -> Dict[str, Any]:
    """Basınç paternlerini analiz et"""
    patterns = {}
    
    for channel_id in range(1, 5):
        channel_col = f'channel_{channel_id}' if f'channel_{channel_id}' in data.columns else f'deger_{channel_id}_parsed'
        if channel_col not in data.columns:
            continue
        
        channel_data = data[channel_col].dropna()
        
        # Pattern analizi
        patterns[channel_id] = {
            'trend': 'stable',  # stable, increasing, decreasing
            'seasonality': False,
            'outliers': [],
            'clusters': []
        }
        
        # Trend analizi (basit lineer regresyon)
        if len(channel_data) > 10:
            x = np.arange(len(channel_data))
            slope, intercept = np.polyfit(x, channel_data, 1)
            
            if abs(slope) < 0.001:
                patterns[channel_id]['trend'] = 'stable'
            elif slope > 0:
                patterns[channel_id]['trend'] = 'increasing'
            else:
                patterns[channel_id]['trend'] = 'decreasing'
        
        # Outlier tespiti (IQR yöntemi)
        q1 = channel_data.quantile(0.25)
        q3 = channel_data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = channel_data[(channel_data < lower_bound) | (channel_data > upper_bound)]
        patterns[channel_id]['outliers'] = outliers.tolist()
        
        # Kümeleme analizi (basit K-means benzeri)
        if len(channel_data) > 20:
            # 3 küme varsayımı
            percentiles = [channel_data.quantile(p) for p in [0.25, 0.5, 0.75]]
            patterns[channel_id]['clusters'] = percentiles
    
    return patterns