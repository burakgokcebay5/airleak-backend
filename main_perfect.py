"""
SANZIMAN LOB TEST ANALIZ PLATFORMU - PERFECT EDITION
Hatasiz, tam ozellikli, production-ready versiyon
"""

import warnings
warnings.filterwarnings('ignore')

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

import pandas as pd
import numpy as np

# Import advanced optimization modules
OPTIMIZATION_AVAILABLE = False
PRESSURE_OPTIMIZATION_AVAILABLE = False

try:
    from optimization_advanced import (
        AdvancedOptimizationEngine,
        format_optimization_results,
        get_optimization_recommendations
    )
    OPTIMIZATION_AVAILABLE = True
    print("Advanced optimization module loaded")
except ImportError:
    print("Warning: Advanced optimization module not found")

try:
    from optimization_pressure_range import (
        PressureRangeOptimizer,
        analyze_pressure_patterns
    )
    PRESSURE_OPTIMIZATION_AVAILABLE = True
    print("Pressure range optimization module loaded")
except ImportError:
    print("Warning: Pressure range optimization not found")
from scipy import stats, signal
from scipy.stats import gaussian_kde, shapiro, chi2_contingency
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                            IsolationForest, RandomForestRegressor)
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
import traceback
from typing import Optional, Dict, Any, List, Tuple
from io import BytesIO
from datetime import datetime, timedelta
import re
from collections import Counter, defaultdict
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import math

# Helper function to safely serialize data
def safe_serialize(obj):
    """Safely serialize data by handling NaN, inf, and other non-JSON values"""
    if isinstance(obj, dict):
        return {k: safe_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_serialize(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return safe_serialize(obj.tolist())
    elif pd.isna(obj):
        return None
    elif isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    else:
        return obj

# Initialize FastAPI
app = FastAPI(
    title="LOB Test Analiz Platformu - Perfect Edition", 
    version="6.0.0",
    description="Hatasiz ve tam ozellikli uretim versiyonu"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup event - Initialize backend
@app.on_event("startup")
async def startup_event():
    """Initialize backend on startup"""
    print("=" * 60)
    print("Backend starting up...")
    print("=" * 60)
    print("Waiting for Firebase data sync...")
    print("Frontend will automatically sync Firebase data")
    print("=" * 60)
    
    # Initialize empty dataframe structure to prevent errors
    data_store.raw_data = pd.DataFrame()
    data_store.processed_data = pd.DataFrame()
    data_store.patterns = {}
    data_store.ml_models = {}
    data_store.detailed_channel_analysis = {}
    data_store.sheet_based_analysis = {}
    data_store.pressure_analysis = {}
    data_store.bimal_analysis = {}
    
    print("Backend ready to receive data")

# Global storage
class DataStore:
    def __init__(self):
        self.raw_data = None
        self.processed_data = None
        self.patterns = {}
        self.ml_models = {}
        self.optimization_results = {}
        self.real_time_predictions = []
        self.metadata = {}
        self.detailed_channel_analysis = {}  # Yeni: Her kanal icin detayli analiz
        self.sheet_based_analysis = {}      # Yeni: Sheet bazli analiz
        self.pressure_analysis = {}         # Yeni: Basinc degerleri analizi
        self.bimal_analysis = {}            # Yeni: Bimal test karşılaştırma analizi
        self.bimal_data = None              # Bimal data for correlation calculation
        
data_store = DataStore()

# Thread pool for parallel processing
executor = ThreadPoolExecutor(max_workers=4)

# Data Models
class ChannelValue(BaseModel):
    channel_1: Optional[float] = Field(None, ge=0, le=10)
    channel_2: Optional[float] = Field(None, ge=0, le=10)
    channel_3: Optional[float] = Field(None, ge=0, le=10)
    channel_4: Optional[float] = Field(None, ge=0, le=10)

class PredictionRequest(BaseModel):
    values: ChannelValue
    include_confidence: bool = True
    include_explanation: bool = False

class OptimizationRequest(BaseModel):
    method: str = Field("genetic", pattern="^(genetic|bayesian|pso|annealing|ensemble|grid_search|advanced_bayesian|multi_objective|robust|adaptive)$")
    iterations: Optional[int] = Field(100, ge=10, le=1000)
    target_metric: Optional[str] = Field("success_rate", pattern="^(success_rate|stability|efficiency)$")
    parameters: Optional[Dict[str, Any]] = None
    test_filters: Optional[Dict[str, bool]] = None  # Test combination filters

# ==================== PERFECT EXCEL PARSER ====================
class PerfectExcelParser:
    """Mukemmel Excel parser - tum hatalari handle eder"""
    
    def __init__(self):
        self.encoding_fixes = {
            '�': 'ğ', 'ı': 'i', 'ğ': 'g', 'ü': 'u', 'ş': 's', 'ö': 'o', 'ç': 'c',
            'İ': 'i', 'Ğ': 'g', 'Ü': 'u', 'Ş': 's', 'Ö': 'o', 'Ç': 'c',
            'ý': 'i', 'þ': 's', 'ð': 'g'
        }
        
    def clean_column_name(self, col: str) -> str:
        """Kolon isimlerini temizle ve standardize et"""
        col_str = str(col).strip()
        
        # Encoding duzeltmeleri
        for bad, good in self.encoding_fixes.items():
            col_str = col_str.replace(bad, good)
        
        # Normalize
        col_str = re.sub(r'[^\w\s]', '', col_str)
        col_str = col_str.lower().replace(' ', '_')
        
        return col_str
    
    def parse_value_field(self, value: Any) -> Dict[str, Any]:
        """Deger alanlarini parse et (0,757-0,783 formati dahil)"""
        if pd.isna(value):
            return {'value': None, 'min': None, 'max': None, 'is_range': False}
        
        val_str = str(value).strip().replace(',', '.')
        
        # Çok uzun string ise (birleşik değerler) ilk değeri al
        if len(val_str) > 20:
            # Birleşik değerler - ilk değeri parse et
            parts = val_str.split('0.')
            if len(parts) > 1 and parts[0]:
                val_str = parts[0]
            else:
                # İlk range'i bulmaya çalış
                first_range = val_str[:20]
                val_str = first_range
        
        # Range format: "0.757-0.783" veya "0.757 - 0.783"
        range_pattern = r'^\s*([+-]?\d*\.?\d+)\s*[-–]\s*([+-]?\d*\.?\d+)\s*'
        match = re.match(range_pattern, val_str)
        
        if match:
            try:
                min_val = float(match.group(1))
                max_val = float(match.group(2))
                
                # Mantıklı aralık kontrolü (3 bar max basınç)
                if max_val > 5:  # 5 bar üstü değerler muhtemelen hatalı
                    # Ondalık hata olabilir
                    if max_val > 100:
                        max_val = max_val / 1000
                        min_val = min_val / 1000
                
                return {
                    'value': (min_val + max_val) / 2,
                    'min': min_val,
                    'max': max_val,
                    'is_range': True,
                    'range_width': max_val - min_val
                }
            except:
                pass
        
        # Single value
        try:
            val = float(val_str)
            # Mantıklı değer kontrolü
            if val > 5:  # 5 bar üstü kontrol
                if val > 100:
                    val = val / 1000  # Muhtemelen ondalık hatası
            return {'value': val, 'min': val, 'max': val, 'is_range': False}
        except:
            return {'value': None, 'min': None, 'max': None, 'is_range': False}
    
    def parse_nok_channels(self, value: Any) -> List[int]:
        """NOK kanallarini parse et"""
        if pd.isna(value):
            return []
        
        val_str = str(value).strip()
        channels = []
        
        # "1,2,3" veya "1, 2, 3" formati
        if ',' in val_str:
            for part in val_str.split(','):
                part = part.strip()
                if part.isdigit():
                    channels.append(int(part))
        # Tek kanal
        elif val_str.isdigit():
            channels.append(int(val_str))
        # "1 2 3" formati
        elif ' ' in val_str:
            for part in val_str.split():
                if part.isdigit():
                    channels.append(int(part))
        
        return channels
    
    def parse_excel_file(self, file_content: bytes) -> pd.DataFrame:
        """Excel dosyasini komple parse et"""
        try:
            all_data = []
            xl = pd.ExcelFile(BytesIO(file_content))
            
            print(f"Found sheets: {xl.sheet_names}")
            
            for sheet_idx, sheet_name in enumerate(xl.sheet_names):
                df = pd.read_excel(BytesIO(file_content), sheet_name=sheet_name)
                print(f"\nProcessing sheet: {sheet_name}")
                print(f"Original columns: {list(df.columns)[:10]}")
                
                # Kolon mapping - collision detection ile
                column_mapping = {}
                value_columns = []
                test_col_count = 0
                
                for i, col in enumerate(df.columns):
                    col_clean = self.clean_column_name(col)
                    
                    # Unnamed kolonlari handle et
                    if 'unnamed' in col_clean or col_clean == '' or pd.isna(col):
                        column_mapping[col] = f'col_{i}'
                    # Sira no
                    elif 'sira' in col_clean and 'no' in col_clean:
                        column_mapping[col] = 'sira_no'
                    # Tarih
                    elif 'tarih' in col_clean:
                        column_mapping[col] = 'tarih'
                    # Vardiya
                    elif 'vardiya' in col_clean:
                        column_mapping[col] = 'vardiya'
                    # Order
                    elif 'order' in col_clean:
                        column_mapping[col] = 'order'
                    # Test result - collision handling
                    elif 'test' in col_clean and ('ok' in col_clean or 'nok' in col_clean):
                        if test_col_count == 0:
                            column_mapping[col] = 'test_result'
                        else:
                            column_mapping[col] = f'test_result_{test_col_count}'
                        test_col_count += 1
                    # Bimal ok/nok
                    elif 'bimal' in col_clean and ('ok' in col_clean or 'nok' in col_clean):
                        column_mapping[col] = 'bimal_result'
                    # Aciklama
                    elif 'aciklama' in col_clean or 'description' in col_clean:
                        column_mapping[col] = 'aciklama'
                    # Deger kolonlari
                    elif 'deger' in col_clean or 'value' in col_clean or 'degeri' in col_clean:
                        # Hangi kanal oldugunu bul
                        found_channel = False
                        for j in range(1, 5):
                            if str(j) in col_clean:
                                column_mapping[col] = f'deger_{j}'
                                value_columns.append(f'deger_{j}')
                                found_channel = True
                                break
                        
                        if not found_channel:
                            # Kanal numarasi yoksa sirayla ata
                            val_idx = len(value_columns) + 1
                            if val_idx <= 4:
                                column_mapping[col] = f'deger_{val_idx}'
                                value_columns.append(f'deger_{val_idx}')
                            else:
                                column_mapping[col] = f'col_{i}'
                    else:
                        column_mapping[col] = f'col_{i}'
                
                # Rename columns
                df.rename(columns=column_mapping, inplace=True)
                print(f"Mapped columns: {list(df.columns)[:15]}")
                
                # Parse test results - main test column
                if 'test_result' in df.columns:
                    df['test_result_raw'] = df['test_result'].copy()
                    df['test_result'] = df['test_result_raw'].apply(lambda x: 
                        'OK' if str(x).upper().strip() in ['OK', 'TAMAM', 'BASARILI', 'GOOD'] 
                        else 'NOK' if str(x).upper().strip() in ['NOK', 'HATA', 'BASARISIZ', 'BAD', 'FAIL']
                        else 'UNKNOWN'
                    )
                else:
                    df['test_result'] = 'UNKNOWN'
                
                # Parse bimal results if exists
                if 'bimal_result' in df.columns:
                    df['bimal_result_parsed'] = df['bimal_result'].apply(lambda x:
                        'OK' if str(x).upper().strip() in ['OK', 'TAMAM']
                        else 'NOK' if str(x).upper().strip() in ['NOK', 'HATA']
                        else 'UNKNOWN'
                    )
                
                # Bimal sheet için özel işlem - col_10'dan (K kolonu) NOK durumunu kontrol et
                if 'bimal' in sheet_name.lower():
                    # col_10 (yeni K kolonu) NOK içeriyorsa bimal_result NOK yap
                    if 'col_10' in df.columns:
                        print(f"Bimal sheet - checking col_10 for NOK markers")
                        nok_mask = df['col_10'].notna() & df['col_10'].astype(str).str.strip().str.upper().str.contains('NOK', na=False)
                        df['bimal_result'] = 'OK'  # Varsayılan OK
                        df.loc[nok_mask, 'bimal_result'] = 'NOK'
                        print(f"Bimal NOK count from col_10: {nok_mask.sum()}")
                    elif 'bimal_result' not in df.columns:
                        df['bimal_result'] = 'OK'  # Varsayılan OK
                
                # Parse value columns
                for val_col in value_columns:
                    if val_col in df.columns:
                        parsed_values = df[val_col].apply(self.parse_value_field)
                        df[f'{val_col}_parsed'] = parsed_values.apply(lambda x: x['value'])
                        df[f'{val_col}_min'] = parsed_values.apply(lambda x: x['min'])
                        df[f'{val_col}_max'] = parsed_values.apply(lambda x: x['max'])
                        df[f'{val_col}_is_range'] = parsed_values.apply(lambda x: x['is_range'])
                
                # Parse NOK channels - Unnamed kolonlarindan veya col_ kolonlarindan
                nok_channel_candidates = []
                
                # Oncelik: col_ kolonlari (Unnamed kolonlar col_ olarak rename edilmis)
                for col in df.columns:
                    if 'col_' in col or 'test_result_' in col:
                        # Bu kolonda kanal numaralari var mi kontrol et
                        sample_values = df[col].dropna().head(20)
                        has_channel_numbers = False
                        for val in sample_values:
                            val_str = str(val).strip()
                            # "1, 2, 3, 4" veya "3, 4" gibi formatlar
                            if ',' in val_str:
                                parts = val_str.split(',')
                                # En az bir parca rakam ise
                                if any(p.strip().isdigit() and p.strip() in ['1','2','3','4'] for p in parts):
                                    has_channel_numbers = True
                                    break
                            elif val_str.isdigit() and val_str in ['1','2','3','4']:
                                has_channel_numbers = True
                                break
                        
                        if has_channel_numbers:
                            nok_channel_candidates.append(col)
                            print(f"Found potential NOK channel column: {col}")
                
                # En uygun NOK kanal kolonunu sec
                if nok_channel_candidates:
                    # col_10 genellikle NOK kanal bilgisi (Unnamed: 10)
                    best_col = None
                    if 'col_10' in nok_channel_candidates:
                        best_col = 'col_10'
                    else:
                        # Test result NOK olan satirlarda en cok veri olan kolonu sec
                        max_count = 0
                        for col in nok_channel_candidates:
                            valid_count = df[col].notna().sum()
                            if valid_count > max_count:
                                max_count = valid_count
                                best_col = col
                    
                    if best_col:
                        print(f"Using column '{best_col}' for NOK channel information")
                        df['nok_channels'] = df[best_col].apply(self.parse_nok_channels)
                    else:
                        df['nok_channels'] = [[] for _ in range(len(df))]
                else:
                    print("No NOK channel column found, using empty list")
                    df['nok_channels'] = [[] for _ in range(len(df))]
                
                # Add metadata
                df['sheet_name'] = sheet_name
                df['sheet_index'] = sheet_idx
                df['row_index'] = range(len(df))
                
                # Parse dates if exists
                if 'tarih' in df.columns:
                    df['tarih'] = pd.to_datetime(df['tarih'], errors='coerce', dayfirst=True)
                
                # Clean up any infinity values
                numeric_columns = df.select_dtypes(include=[np.number]).columns
                df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], np.nan)
                
                # ADD SHEET NAME AND INDEX TO DATAFRAME
                df['sheet_name'] = sheet_name
                df['sheet_index'] = sheet_idx
                df['row_index'] = range(len(df))  # Add row index within sheet
                
                print(f"Sheet {sheet_name} has {len(df)} rows before appending")
                all_data.append(df)
            
            # Combine all sheets
            if all_data:
                print(f"\nCombining {len(all_data)} sheets...")
                for i, df in enumerate(all_data):
                    print(f"  Sheet {i}: {len(df)} rows")
                
                combined_df = pd.concat(all_data, ignore_index=True, sort=False)
                print(f"Combined dataframe has {len(combined_df)} rows before cleaning")
                
                # Remove completely empty rows
                rows_before = len(combined_df)
                combined_df = combined_df.dropna(how='all')
                rows_after = len(combined_df)
                print(f"Dropped {rows_before - rows_after} completely empty rows")
                
                # Check sheet distribution
                if 'sheet_name' in combined_df.columns:
                    sheet_counts = combined_df['sheet_name'].value_counts()
                    print("\nSheet distribution in final data:")
                    for sheet, count in sheet_counts.items():
                        print(f"  {sheet}: {count} rows")
                
                # Final statistics
                print(f"\nTotal rows parsed: {len(combined_df)}")
                if 'test_result' in combined_df.columns:
                    print(f"Test distribution: {combined_df['test_result'].value_counts().to_dict()}")
                
                return combined_df
            
            return pd.DataFrame()
            
        except Exception as e:
            print(f"Excel parse error: {str(e)}")
            print(traceback.format_exc())
            raise

# ==================== DETAILED CHANNEL ANALYZER ====================
class DetailedChannelAnalyzer:
    """Her kanal icin detayli OK/NOK analizi"""
    
    def analyze_channel_details(self, df: pd.DataFrame, channel_num: int) -> Dict[str, Any]:
        """Tek bir kanalin detayli analizi - GERCEK NOK kanallari ile"""
        channel_col = f'deger_{channel_num}_parsed'
        channel_min_col = f'deger_{channel_num}_min'
        channel_max_col = f'deger_{channel_num}_max'
        
        if channel_col not in df.columns:
            return {}
        
        analysis = {
            'channel_number': channel_num,
            'total_measurements': 0,
            'ok_count': 0,
            'nok_count': 0,
            'unknown_count': 0,
            'ok_percentage': 0,
            'nok_percentage': 0,
            'ok_statistics': {},
            'nok_statistics': {},
            'pressure_ranges': {},
            'anomalies': []
        }
        
        # NOK kanallarini parse et - Unnamed kolonlarindan veya nok_channels'dan
        nok_channel_mask = pd.Series([False] * len(df), index=df.index)
        ok_channel_mask = pd.Series([False] * len(df), index=df.index)
        
        # nok_channels kolonu varsa kullan
        if 'nok_channels' in df.columns:
            for idx, row in df.iterrows():
                if df[channel_col].notna().loc[idx]:  # Bu kanalda deger varsa
                    nok_channels = row['nok_channels']
                    if isinstance(nok_channels, list) and channel_num in nok_channels:
                        nok_channel_mask.loc[idx] = True
                    elif row['test_result'] == 'OK':
                        ok_channel_mask.loc[idx] = True
                    elif row['test_result'] == 'NOK' and not (isinstance(nok_channels, list) and channel_num in nok_channels):
                        # Test NOK ama bu kanal NOK listesinde degilse, kanal OK
                        ok_channel_mask.loc[idx] = True
        else:
            # Eski yontem - sadece test_result'a bak
            ok_channel_mask = (df['test_result'] == 'OK') & df[channel_col].notna()
            nok_channel_mask = (df['test_result'] == 'NOK') & df[channel_col].notna()
        
        # Degerler - Outlier filtreleme ile (3 bar max)
        all_values = df[channel_col].dropna()
        
        # Mantıklı değer aralığında filtrele (0-3.5 bar arası)
        MAX_PRESSURE = 3.5  # Maximum basınç limiti
        MIN_PRESSURE = 0    # Minimum basınç limiti
        
        # Outlier'ları tespit et
        outliers_mask = (all_values > MAX_PRESSURE) | (all_values < MIN_PRESSURE)
        outlier_values = all_values[outliers_mask]
        
        # Temiz değerleri al
        clean_mask = (all_values <= MAX_PRESSURE) & (all_values >= MIN_PRESSURE)
        all_values_clean = all_values[clean_mask]
        
        # OK ve NOK değerlerini temiz maskelerle al
        ok_values = df.loc[ok_channel_mask, channel_col].dropna()
        ok_values = ok_values[(ok_values <= MAX_PRESSURE) & (ok_values >= MIN_PRESSURE)]
        
        nok_values = df.loc[nok_channel_mask, channel_col].dropna()
        nok_values = nok_values[(nok_values <= MAX_PRESSURE) & (nok_values >= MIN_PRESSURE)]
        
        analysis['total_measurements'] = len(all_values)
        analysis['ok_count'] = len(ok_values)
        analysis['nok_count'] = len(nok_values)
        analysis['unknown_count'] = len(all_values_clean) - len(ok_values) - len(nok_values)
        analysis['outlier_count'] = len(outlier_values)
        
        # Outlier detayları
        if len(outlier_values) > 0:
            analysis['outliers'] = {
                'count': len(outlier_values),
                'values': outlier_values.head(10).tolist(),  # İlk 10 outlier
                'indices': outlier_values.index[:10].tolist(),
                'max_outlier': float(outlier_values.max()),
                'min_outlier': float(outlier_values.min())
            }
        
        # Yuzdelik hesapla
        if analysis['total_measurements'] > 0:
            analysis['ok_percentage'] = (analysis['ok_count'] / analysis['total_measurements']) * 100
            analysis['nok_percentage'] = (analysis['nok_count'] / analysis['total_measurements']) * 100
        
        # OK degerlerinin detayli istatistikleri
        if len(ok_values) > 0:
            ok_min_vals = df.loc[ok_channel_mask, channel_min_col] if channel_min_col in df.columns else ok_values
            ok_max_vals = df.loc[ok_channel_mask, channel_max_col] if channel_max_col in df.columns else ok_values
            
            analysis['ok_statistics'] = {
                'mean': float(ok_values.mean()),
                'median': float(ok_values.median()),
                'std': float(ok_values.std()),
                'min': float(ok_values.min()),
                'max': float(ok_values.max()),
                'q1': float(ok_values.quantile(0.25)),
                'q3': float(ok_values.quantile(0.75)),
                'iqr': float(ok_values.quantile(0.75) - ok_values.quantile(0.25)),
                'actual_min': float(ok_min_vals.min()) if len(ok_min_vals) > 0 else float(ok_values.min()),
                'actual_max': float(ok_max_vals.max()) if len(ok_max_vals) > 0 else float(ok_values.max()),
                'optimal_range': {
                    'lower': float(ok_values.quantile(0.025)),
                    'upper': float(ok_values.quantile(0.975))
                }
            }
        
        # NOK degerlerinin detayli istatistikleri
        if len(nok_values) > 0:
            nok_min_vals = df.loc[nok_channel_mask, channel_min_col] if channel_min_col in df.columns else nok_values
            nok_max_vals = df.loc[nok_channel_mask, channel_max_col] if channel_max_col in df.columns else nok_values
            
            analysis['nok_statistics'] = {
                'mean': float(nok_values.mean()),
                'median': float(nok_values.median()),
                'std': float(nok_values.std()),
                'min': float(nok_values.min()),
                'max': float(nok_values.max()),
                'q1': float(nok_values.quantile(0.25)),
                'q3': float(nok_values.quantile(0.75)),
                'actual_min': float(nok_min_vals.min()) if len(nok_min_vals) > 0 else float(nok_values.min()),
                'actual_max': float(nok_max_vals.max()) if len(nok_max_vals) > 0 else float(nok_values.max())
            }
            
            # OK ve NOK karsilastirmasi
            if len(ok_values) > 0:
                analysis['nok_statistics']['deviation_from_ok'] = {
                    'mean_difference': float(nok_values.mean() - ok_values.mean()),
                    'mean_diff_percent': float((nok_values.mean() - ok_values.mean()) / ok_values.mean() * 100) if ok_values.mean() != 0 else 0
                }
        
        # Basinc araliklari analizi
        if channel_min_col in df.columns and channel_max_col in df.columns:
            pressure_data = df[[channel_min_col, channel_max_col, 'test_result', channel_col]].dropna()
            
            if len(pressure_data) > 0:
                analysis['pressure_ranges'] = {
                    'overall': {
                        'min_pressure': float(pressure_data[channel_min_col].min()),
                        'max_pressure': float(pressure_data[channel_max_col].max()),
                        'avg_range_width': float((pressure_data[channel_max_col] - pressure_data[channel_min_col]).mean())
                    }
                }
                
                # OK basinc araliklari
                ok_pressure = pressure_data[pressure_data['test_result'] == 'OK']
                if len(ok_pressure) > 0:
                    analysis['pressure_ranges']['ok_ranges'] = {
                        'min_avg': float(ok_pressure[channel_min_col].mean()),
                        'max_avg': float(ok_pressure[channel_max_col].mean()),
                        'typical_range': {
                            'lower': float(ok_pressure[channel_min_col].quantile(0.25)),
                            'upper': float(ok_pressure[channel_max_col].quantile(0.75))
                        }
                    }
                
                # NOK basinc araliklari
                nok_pressure = pressure_data[pressure_data['test_result'] == 'NOK']
                if len(nok_pressure) > 0:
                    analysis['pressure_ranges']['nok_ranges'] = {
                        'min_avg': float(nok_pressure[channel_min_col].mean()),
                        'max_avg': float(nok_pressure[channel_max_col].mean())
                    }
        
        # Anomali tespiti - sadece mantıklı değerler üzerinde
        if len(ok_values) > 10:
            mean_val = ok_values.mean()
            std_val = ok_values.std()
            
            # Z-score anomalileri - sadece temiz değerler üzerinde
            for idx, val in all_values_clean.items():
                z_score = abs((val - mean_val) / std_val) if std_val > 0 else 0
                if z_score > 3:
                    analysis['anomalies'].append({
                        'index': int(idx),
                        'value': float(val),
                        'z_score': float(z_score),
                        'test_result': df.loc[idx, 'test_result'] if idx in df.index else 'UNKNOWN',
                        'type': 'statistical_outlier'
                    })
            
            # Sadece ilk 10 anomaliyi al
            analysis['anomalies'] = sorted(analysis['anomalies'], key=lambda x: x['z_score'], reverse=True)[:10]
        
        return analysis
    
    def analyze_all_channels(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Tum kanallarin detayli analizi"""
        all_analysis = {}
        summary = {
            'total_channels': 0,
            'best_channel': None,
            'worst_channel': None,
            'channel_comparison': {}
        }
        
        channel_success_rates = {}
        
        for i in range(1, 5):
            channel_analysis = self.analyze_channel_details(df, i)
            if channel_analysis and channel_analysis['total_measurements'] > 0:
                all_analysis[f'channel_{i}'] = channel_analysis
                summary['total_channels'] += 1
                
                # Basari orani hesapla
                success_rate = channel_analysis['ok_percentage']
                channel_success_rates[f'channel_{i}'] = success_rate
        
        # En iyi ve en kotu kanali bul
        if channel_success_rates:
            summary['best_channel'] = max(channel_success_rates, key=channel_success_rates.get)
            summary['worst_channel'] = min(channel_success_rates, key=channel_success_rates.get)
            summary['channel_comparison'] = channel_success_rates
        
        return {
            'channels': all_analysis,
            'summary': summary
        }

# ==================== SHEET BASED ANALYZER ====================
class SheetBasedAnalyzer:
    """Sheet bazli detayli analiz"""
    
    def analyze_sheet(self, df: pd.DataFrame, sheet_name: str) -> Dict[str, Any]:
        """Tek bir sheet'in analizi"""
        sheet_data = df[df['sheet_name'] == sheet_name] if 'sheet_name' in df.columns else df
        
        if len(sheet_data) == 0:
            return {}
        
        channel_analyzer = DetailedChannelAnalyzer()
        
        analysis = {
            'sheet_name': sheet_name,
            'total_tests': len(sheet_data),
            'test_distribution': {},
            'success_rate': 0,
            'channel_analysis': {},
            'time_analysis': {},
            'shift_analysis': {},
            'nok_channel_frequency': {}
        }
        
        # Test dagilimi
        if 'test_result' in sheet_data.columns:
            test_counts = sheet_data['test_result'].value_counts().to_dict()
            analysis['test_distribution'] = test_counts
            
            ok_count = test_counts.get('OK', 0)
            nok_count = test_counts.get('NOK', 0)
            total = ok_count + nok_count
            
            if total > 0:
                analysis['success_rate'] = (ok_count / total) * 100
        
        # Her kanal icin detayli analiz
        for i in range(1, 5):
            channel_analysis = channel_analyzer.analyze_channel_details(sheet_data, i)
            if channel_analysis and channel_analysis['total_measurements'] > 0:
                analysis['channel_analysis'][f'channel_{i}'] = channel_analysis
        
        # Zaman analizi
        if 'tarih' in sheet_data.columns:
            valid_dates = sheet_data['tarih'].dropna()
            if len(valid_dates) > 0:
                analysis['time_analysis'] = {
                    'start_date': valid_dates.min().isoformat(),
                    'end_date': valid_dates.max().isoformat(),
                    'days_covered': (valid_dates.max() - valid_dates.min()).days
                }
        
        # Vardiya analizi
        if 'vardiya' in sheet_data.columns:
            shift_counts = sheet_data['vardiya'].value_counts().to_dict()
            analysis['shift_analysis']['distribution'] = shift_counts
            
            # Her vardiya icin basari orani
            shift_success = {}
            for shift in sheet_data['vardiya'].unique():
                if pd.notna(shift):
                    shift_data = sheet_data[sheet_data['vardiya'] == shift]
                    shift_test = shift_data['test_result'].value_counts().to_dict()
                    shift_ok = shift_test.get('OK', 0)
                    shift_nok = shift_test.get('NOK', 0)
                    shift_total = shift_ok + shift_nok
                    
                    if shift_total > 0:
                        shift_success[str(shift)] = {
                            'success_rate': (shift_ok / shift_total) * 100,
                            'ok_count': shift_ok,
                            'nok_count': shift_nok
                        }
            
            analysis['shift_analysis']['success_by_shift'] = shift_success
        
        # NOK kanal frekansi
        if 'nok_channels' in sheet_data.columns:
            nok_data = sheet_data[sheet_data['test_result'] == 'NOK']
            channel_counts = {1: 0, 2: 0, 3: 0, 4: 0}
            
            for channels in nok_data['nok_channels']:
                if channels:
                    for ch in channels:
                        if ch in channel_counts:
                            channel_counts[ch] += 1
            
            analysis['nok_channel_frequency'] = channel_counts
        
        return analysis
    
    def analyze_all_sheets(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Tum sheet'lerin analizi"""
        all_sheets = {}
        
        if 'sheet_name' in df.columns:
            for sheet_name in df['sheet_name'].unique():
                sheet_analysis = self.analyze_sheet(df, sheet_name)
                if sheet_analysis:
                    all_sheets[sheet_name] = sheet_analysis
        else:
            sheet_analysis = self.analyze_sheet(df, 'Main')
            if sheet_analysis:
                all_sheets['Main'] = sheet_analysis
        
        return all_sheets

# ==================== BIMAL TEST ANALYZER ====================
class BimalTestAnalyzer:
    """Bimal ve Airleak test karşılaştırma analizi"""
    
    def parse_nok_channels_from_value(self, value):
        """Parse NOK channels from a value like '3, 4' or '2'"""
        if pd.isna(value):
            return []
        
        channels = []
        val_str = str(value).strip()
        
        if ',' in val_str:
            # Multiple channels
            parts = val_str.split(',')
            for part in parts:
                part = part.strip()
                if part.isdigit():
                    channels.append(int(part))
        elif val_str.isdigit():
            # Single channel
            channels.append(int(val_str))
        
        return channels
    
    def analyze_bimal_comparison(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Bimal ve Airleak testlerinin karşılaştırmalı analizi"""
        
        # Her iki sheet'i de al
        sayfa1_data = df[df['sheet_name'] == 'Sayfa1'] if 'sheet_name' in df.columns else pd.DataFrame()
        bimal_data = df[df['sheet_name'] == 'bimal değerleri'] if 'sheet_name' in df.columns else pd.DataFrame()
        
        # Bimal sheet'ini kullan ve NOK kanalları ekle
        if not bimal_data.empty:
            primary_data = bimal_data.copy()
            
            # R kolonu (Unnamed: 17) NOK kanalları içeriyor
            if 'Unnamed: 17' in primary_data.columns:
                print("Parsing NOK channels from Unnamed: 17 (R column)")
                primary_data['nok_channels_from_r'] = primary_data['Unnamed: 17'].apply(self.parse_nok_channels_from_value)
                
                # Eğer nok_channels kolonu yoksa veya boşsa, R kolonundan al
                if 'nok_channels' not in primary_data.columns:
                    primary_data['nok_channels'] = primary_data['nok_channels_from_r']
                else:
                    # Boş olanları doldur
                    mask = primary_data['nok_channels'].isna() | (primary_data['nok_channels'].apply(lambda x: len(x) if isinstance(x, list) else 0) == 0)
                    primary_data.loc[mask, 'nok_channels'] = primary_data.loc[mask, 'nok_channels_from_r']
                
                print(f"NOK channels parsed from R column")
            
            print("Using bimal değerleri data with NOK channels")
        elif not sayfa1_data.empty:
            # Fallback olarak Sayfa1'i kullan
            primary_data = sayfa1_data
            print("Using Sayfa1 data as fallback")
        else:
            return {'error': 'Analiz için veri bulunamadı'}
        
        # Debug: Check columns  
        print(f"\nPrimary data columns: {list(primary_data.columns)}")
        if 'nok_channels' in primary_data.columns:
            print(f"NOK channels column found!")
            # Sample NOK channels
            nok_samples = primary_data[primary_data['test_result'] == 'NOK']['nok_channels'].head(5)
            for idx, channels in enumerate(nok_samples):
                if channels:
                    print(f"  Sample {idx}: {channels}")
        
        analysis = {
            'cross_table': {},
            'test_combinations': {},
            'failure_reasons': {},
            'channel_specific_failures': {},
            'success_rates': {},
            'insights': []
        }
        
        # Test kolonlarını kontrol et
        if 'test_result' in primary_data.columns and 'bimal_result' in primary_data.columns:
            # Çapraz tablo oluştur
            valid_data = primary_data[(primary_data['test_result'].isin(['OK', 'NOK'])) & 
                                   (primary_data['bimal_result'].isin(['OK', 'NOK']))]
            
            if len(valid_data) > 0:
                # Test kombinasyonları
                combinations = {
                    'both_ok': ((valid_data['test_result'] == 'OK') & 
                               (valid_data['bimal_result'] == 'OK')).sum(),
                    'both_nok': ((valid_data['test_result'] == 'NOK') & 
                                (valid_data['bimal_result'] == 'NOK')).sum(),
                    'airleak_ok_bimal_nok': ((valid_data['test_result'] == 'OK') & 
                                            (valid_data['bimal_result'] == 'NOK')).sum(),
                    'airleak_nok_bimal_ok': ((valid_data['test_result'] == 'NOK') & 
                                            (valid_data['bimal_result'] == 'OK')).sum()
                }
                
                total = sum(combinations.values())
                
                analysis['test_combinations'] = {
                    'counts': combinations,
                    'percentages': {k: (v/total*100) if total > 0 else 0 
                                  for k, v in combinations.items()},
                    'total_tests': total
                }
                
                # Başarı oranları
                airleak_ok = combinations['both_ok'] + combinations['airleak_ok_bimal_nok']
                bimal_ok = combinations['both_ok'] + combinations['airleak_nok_bimal_ok']
                
                analysis['success_rates'] = {
                    'airleak_success_rate': (airleak_ok / total * 100) if total > 0 else 0,
                    'bimal_success_rate': (bimal_ok / total * 100) if total > 0 else 0,
                    'both_success_rate': (combinations['both_ok'] / total * 100) if total > 0 else 0,
                    'at_least_one_fail_rate': ((total - combinations['both_ok']) / total * 100) if total > 0 else 0
                }
                
                # Çapraz tablo
                cross_tab = pd.crosstab(valid_data['test_result'], 
                                       valid_data['bimal_result'], 
                                       margins=True, 
                                       margins_name='Toplam')
                
                analysis['cross_table'] = {
                    'data': cross_tab.to_dict(),
                    'matrix': [
                        ['', 'Bimal OK', 'Bimal NOK', 'Toplam'],
                        ['Airleak OK', 
                         combinations['both_ok'], 
                         combinations['airleak_ok_bimal_nok'],
                         combinations['both_ok'] + combinations['airleak_ok_bimal_nok']],
                        ['Airleak NOK', 
                         combinations['airleak_nok_bimal_ok'], 
                         combinations['both_nok'],
                         combinations['airleak_nok_bimal_ok'] + combinations['both_nok']],
                        ['Toplam',
                         bimal_ok,
                         total - bimal_ok,
                         total]
                    ]
                }
                
                # Detaylı kayıtları hazırla (modal için)
                detailed_records = {
                    'both_ok': [],
                    'both_nok': [],
                    'airleak_ok_bimal_nok': [],
                    'airleak_nok_bimal_ok': []
                }
                
                # Her kombinasyon için kayıtları topla
                both_ok_data = valid_data[(valid_data['test_result'] == 'OK') & (valid_data['bimal_result'] == 'OK')]
                both_nok_data = valid_data[(valid_data['test_result'] == 'NOK') & (valid_data['bimal_result'] == 'NOK')]
                airleak_ok_bimal_nok_data = valid_data[(valid_data['test_result'] == 'OK') & (valid_data['bimal_result'] == 'NOK')]
                airleak_nok_bimal_ok_data = valid_data[(valid_data['test_result'] == 'NOK') & (valid_data['bimal_result'] == 'OK')]
                
                # Kayıtları format et
                def format_record(row):
                    # NOK kanallarını al - row is a pandas Series from iterrows()
                    nok_ch = row['nok_channels'] if 'nok_channels' in row.index else []
                    
                    # Debug: Check what we have
                    sira_no = row['sira_no'] if 'sira_no' in row.index else 'unknown'
                    if sira_no in [1, 2, 25, 93]:  # Debug specific rows
                        print(f"\nDEBUG Row {sira_no}:")
                        print(f"  - Has 'nok_channels' in index: {'nok_channels' in row.index}")
                        if 'nok_channels' in row.index:
                            print(f"  - nok_channels value: {row['nok_channels']}")
                            print(f"  - nok_channels type: {type(row['nok_channels'])}")
                    
                    if isinstance(nok_ch, list):
                        nok_channels_list = nok_ch
                    else:
                        nok_channels_list = []
                    
                    # Debug ilk birkaç kayıt
                    if nok_channels_list:
                        print(f"Including NOK channels for sira_no {sira_no}: {nok_channels_list}")
                    
                    # Handle date formatting safely
                    tarih_value = ''
                    if 'tarih' in row.index and pd.notna(row['tarih']):
                        if isinstance(row['tarih'], str):
                            tarih_value = row['tarih']
                        elif hasattr(row['tarih'], 'strftime'):
                            tarih_value = row['tarih'].strftime('%d.%m.%Y')
                        else:
                            tarih_value = str(row['tarih'])
                    
                    record = {
                        'sira_no': row['sira_no'] if 'sira_no' in row.index else '',
                        'tarih': tarih_value,
                        'vardiya': row['vardiya'] if 'vardiya' in row.index else '',
                        'test_result': row['test_result'] if 'test_result' in row.index else '',
                        'bimal_result': row['bimal_result'] if 'bimal_result' in row.index else '',
                        'aciklama': row['aciklama'] if 'aciklama' in row.index and pd.notna(row['aciklama']) else '-',
                        'nok_channels': nok_channels_list
                    }
                    
                    # Kanal değerlerini ekle - Frontend'in beklediği format
                    for i in range(1, 5):
                        val_col = f'deger_{i}_parsed'
                        if val_col in row.index:
                            val = row[val_col]
                            if pd.notna(val):
                                record[f'deger_{i}_parsed'] = float(val)
                                record[f'deger_{i}'] = float(val)  # Frontend compatibility
                                record[f'channel_{i}'] = f"{val:.3f}"
                            else:
                                record[f'deger_{i}_parsed'] = None
                                record[f'deger_{i}'] = None
                                record[f'channel_{i}'] = '-'
                        else:
                            record[f'deger_{i}_parsed'] = None
                            record[f'deger_{i}'] = None
                            record[f'channel_{i}'] = '-'
                    
                    return record
                
                # Her kategori için kayıtları ekle (max 100 kayıt)
                for _, row in both_ok_data.head(100).iterrows():
                    detailed_records['both_ok'].append(format_record(row))
                
                for _, row in both_nok_data.head(100).iterrows():
                    detailed_records['both_nok'].append(format_record(row))
                
                for _, row in airleak_ok_bimal_nok_data.head(100).iterrows():
                    detailed_records['airleak_ok_bimal_nok'].append(format_record(row))
                
                for _, row in airleak_nok_bimal_ok_data.head(100).iterrows():
                    detailed_records['airleak_nok_bimal_ok'].append(format_record(row))
                
                analysis['detailed_records'] = detailed_records
                
                # Debug: Check if NOK channels are in detailed records
                print(f"\nDetailed records summary:")
                for key, records in detailed_records.items():
                    if records:
                        print(f"{key}: {len(records)} records")
                        if records[0].get('nok_channels'):
                            print(f"  First record has NOK channels: {records[0]['nok_channels']}")
        
        # NOK sebeplerini analiz et
        # Bimal sheet varsa ondan açıklamaları al
        if not bimal_data.empty and 'aciklama' in bimal_data.columns:
            # K sütununda (col_10) NOK olanları bul (eski J silindikten sonra yeni K)
            if 'col_10' in bimal_data.columns:
                # K sütununda NOK olanların açıklamalarını al
                k_nok_mask = bimal_data['col_10'].str.upper().str.strip() == 'NOK' if bimal_data['col_10'].dtype == 'object' else False
                nok_reasons = bimal_data[k_nok_mask]['aciklama'].dropna()
                print(f"K sütununda (col_10) NOK olan {k_nok_mask.sum()} kayıt bulundu, {len(nok_reasons)} tanesinin açıklaması var")
            elif 'col_11' in bimal_data.columns:
                # Eski yapı - L sütunu
                l_nok_mask = bimal_data['col_11'].str.upper().str.strip() == 'NOK' if bimal_data['col_11'].dtype == 'object' else False
                nok_reasons = bimal_data[l_nok_mask]['aciklama'].dropna()
                print(f"L sütununda NOK olan {l_nok_mask.sum()} kayıt bulundu, {len(nok_reasons)} tanesinin açıklaması var")
            else:
                # Fallback - eski yöntem
                nok_reasons = bimal_data[bimal_data['bimal_result'] == 'NOK']['aciklama'].dropna()
        elif 'aciklama' in primary_data.columns:
            # Primary data'dan NOK açıklamalarını al
            nok_reasons = primary_data[primary_data['bimal_result'] == 'NOK']['aciklama'].dropna()
        else:
            nok_reasons = pd.Series([])
            
        if len(nok_reasons) > 0:
            reason_counts = {}
            reason_categories = {
                'hareket_problemi': [],
                'kacak_problemi': [],
                'yazilim_problemi': [],
                'basinc_problemi': [],
                'yaglama_problemi': [],
                'diger': []
            }
            
            for reason in nok_reasons:
                reason_clean = str(reason).strip().lower()
                
                # Sebep sayımı
                if reason_clean not in reason_counts:
                    reason_counts[reason_clean] = 0
                reason_counts[reason_clean] += 1
                
                # Kategorizasyon
                if 'hareket' in reason_clean:
                    reason_categories['hareket_problemi'].append(reason)
                elif 'kaçak' in reason_clean or 'kacak' in reason_clean:
                    reason_categories['kacak_problemi'].append(reason)
                elif 'yazılım' in reason_clean or 'yazilim' in reason_clean:
                    reason_categories['yazilim_problemi'].append(reason)
                elif 'basınç' in reason_clean or 'basinc' in reason_clean or 'tbv' in reason_clean:
                    reason_categories['basinc_problemi'].append(reason)
                elif 'yağlama' in reason_clean or 'yaglama' in reason_clean:
                    reason_categories['yaglama_problemi'].append(reason)
                else:
                    reason_categories['diger'].append(reason)
            
            # En sık görülen sebepler
            sorted_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)
            
            analysis['failure_reasons'] = {
                'top_reasons': [{'reason': r[0], 'count': r[1]} for r in sorted_reasons[:10]],
                'categories': {
                    cat: {
                        'count': len(reasons),
                        'percentage': (len(reasons) / len(nok_reasons) * 100),
                        'examples': list(set([str(r) for r in reasons[:3]]))
                    }
                    for cat, reasons in reason_categories.items() if reasons
                },
                'total_failures_with_reason': len(nok_reasons)
            }
        
        # Kanal bazlı NOK analizi
        if 'nok_channels' in primary_data.columns:
            bimal_nok = primary_data[primary_data['bimal_result'] == 'NOK']
            
            if len(bimal_nok) > 0 and 'nok_channels' in bimal_nok.columns:
                channel_failures = {1: 0, 2: 0, 3: 0, 4: 0}
                
                for channels in bimal_nok['nok_channels']:
                    if channels:
                        for ch in channels:
                            if ch in channel_failures:
                                channel_failures[ch] += 1
                
                analysis['channel_specific_failures'] = {
                    f'channel_{ch}': {
                        'failure_count': count,
                        'failure_rate': (count / len(bimal_nok) * 100)
                    }
                    for ch, count in channel_failures.items()
                }
        
        # Öneriler ve içgörüler (Bimal kesin test olarak kabul edilir)
        if analysis['test_combinations']:
            insights = []
            
            # Airleak OK ama Bimal NOK durumu - Bu ciddi bir problem
            if analysis['test_combinations']['counts']['airleak_ok_bimal_nok'] > 0:
                rate = analysis['test_combinations']['percentages']['airleak_ok_bimal_nok']
                insights.append({
                    'type': 'critical',
                    'message': f'Ürünlerin %{rate:.1f}\'i Airleak testini geçmesine rağmen Bimal testinde başarısız',
                    'recommendation': 'Airleak test hassasiyeti yetersiz - daha sıkı kriterler uygulanmalı'
                })
            
            # Airleak NOK ama Bimal OK durumu - False positive
            if analysis['test_combinations']['counts']['airleak_nok_bimal_ok'] > 0:
                rate = analysis['test_combinations']['percentages']['airleak_nok_bimal_ok']
                insights.append({
                    'type': 'warning',
                    'message': f'Ürünlerin %{rate:.1f}\'i Airleak testinde başarısız olmasına rağmen Bimal testini geçiyor',
                    'recommendation': 'Airleak testi çok hassas olabilir - kalibrasyon gerekebilir'
                })
            
            # Yüksek başarısızlık oranı
            if analysis['success_rates']['bimal_success_rate'] < 70:
                insights.append({
                    'type': 'critical',
                    'message': f'Bimal test başarı oranı düşük: %{analysis["success_rates"]["bimal_success_rate"]:.1f}',
                    'recommendation': 'Üretim kalitesi iyileştirilmeli'
                })
            
            # İyi performans
            if analysis['success_rates']['both_success_rate'] > 95:
                insights.append({
                    'type': 'success',
                    'message': f'Mükemmel performans: %{analysis["success_rates"]["both_success_rate"]:.1f} ürün her iki testi de geçiyor',
                    'recommendation': 'Mevcut üretim kalitesi korunmalı'
                })
            
            analysis['insights'] = insights
        
        # Test detaylarını da ekleyelim (tıklama için)
        if 'test_result' in bimal_data.columns and 'bimal_result' in bimal_data.columns:
            # Tarih formatını düzelt
            def format_date(date_val):
                if pd.isna(date_val):
                    return None
                if isinstance(date_val, (pd.Timestamp, datetime)):
                    return date_val.strftime('%d.%m.%Y')
                return str(date_val)
            
            # Her kombinasyon için ürün detayları
            def get_records(filter_condition, include_aciklama=False):
                filtered = bimal_data[filter_condition]
                cols = ['sira_no', 'tarih', 'test_result', 'bimal_result', 
                       'deger_1_parsed', 'deger_2_parsed', 'deger_3_parsed', 'deger_4_parsed']
                if include_aciklama and 'aciklama' in bimal_data.columns:
                    cols.append('aciklama')
                    print(f"Including aciklama column for {len(filtered)} records")
                    # Debug: Print some aciklama values
                    if len(filtered) > 0:
                        print(f"Sample aciklama values: {filtered['aciklama'].dropna().head(3).tolist()}")
                
                # Check which columns actually exist
                existing_cols = [col for col in cols if col in filtered.columns]
                missing_cols = [col for col in cols if col not in filtered.columns]
                if missing_cols:
                    print(f"Warning: Missing columns: {missing_cols}")
                
                records = filtered[existing_cols].head(100).to_dict('records')
                # Tarih formatını düzelt
                for record in records:
                    if 'tarih' in record:
                        record['tarih'] = format_date(record['tarih'])
                return records
            
            # Bu kısım zaten yukarıda format_record ile doğru şekilde yapıldı
            # Burayı kaldırıyoruz çünkü NOK kanalları olan doğru detailed_records'u eziyor
            pass
        
        return analysis

# ==================== ADVANCED PATTERN DISCOVERY ====================
class AdvancedPatternDiscovery:
    """Gelismis pattern kesfi - tum algoritmalar"""
    
    def __init__(self):
        self.methods = ['kde', 'bayesian', 'iqr', 'isolation_forest', 'dbscan']
        
    def discover_optimal_ranges_kde(self, values: np.ndarray) -> Dict[str, float]:
        """Kernel Density Estimation ile optimal range bulma"""
        if len(values) < 3:
            return {}
        
        try:
            kde = gaussian_kde(values)
            x_range = np.linspace(values.min(), values.max(), 1000)
            density = kde(x_range)
            
            # Peak noktasini bul
            peak_idx = np.argmax(density)
            peak_value = x_range[peak_idx]
            
            # Confidence interval (yuksek yogunluklu bolge)
            threshold = np.max(density) * 0.5
            high_density_indices = np.where(density > threshold)[0]
            
            if len(high_density_indices) > 0:
                ci_lower = x_range[high_density_indices[0]]
                ci_upper = x_range[high_density_indices[-1]]
            else:
                ci_lower = peak_value - np.std(values)
                ci_upper = peak_value + np.std(values)
            
            return {
                'optimal_value': float(peak_value),
                'ci_lower': float(ci_lower),
                'ci_upper': float(ci_upper),
                'density_peak': float(np.max(density))
            }
        except:
            return {}
    
    def discover_optimal_ranges_bayesian(self, ok_values: np.ndarray, nok_values: np.ndarray) -> Dict[str, float]:
        """Bayesian inference ile optimal range bulma"""
        if len(ok_values) < 3:
            return {}
        
        try:
            # Prior: uniform distribution
            # Likelihood: Gaussian
            ok_mean, ok_std = np.mean(ok_values), np.std(ok_values) + 1e-6
            
            # Posterior mean ve variance
            n = len(ok_values)
            posterior_mean = ok_mean
            posterior_std = ok_std / np.sqrt(n)
            
            # Credible interval
            ci_lower = posterior_mean - 1.96 * posterior_std
            ci_upper = posterior_mean + 1.96 * posterior_std
            
            result = {
                'posterior_mean': float(posterior_mean),
                'posterior_std': float(posterior_std),
                'credible_interval_lower': float(ci_lower),
                'credible_interval_upper': float(ci_upper)
            }
            
            # NOK degerleri varsa separation score hesapla
            if len(nok_values) > 0:
                nok_mean = np.mean(nok_values)
                nok_std = np.std(nok_values) + 1e-6
                separation = abs(ok_mean - nok_mean) / (ok_std + nok_std)
                result['separation_score'] = float(separation)
            
            return result
        except:
            return {}
    
    def text_mining_defects(self, descriptions: List[str]) -> Dict[str, Any]:
        """Aciklama metinlerinden pattern cikarma"""
        if not descriptions or len(descriptions) < 5:
            return {}
        
        try:
            # TF-IDF vektorizasyon
            vectorizer = TfidfVectorizer(max_features=20, ngram_range=(1, 2))
            tfidf_matrix = vectorizer.fit_transform(descriptions)
            
            # En onemli terimleri bul
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.sum(axis=0).A1
            term_scores = [(feature_names[i], scores[i]) for i in np.argsort(scores)[-10:][::-1]]
            
            # Pattern clustering
            n_clusters = min(3, len(descriptions) // 10)
            cluster_patterns = {}
            
            if n_clusters > 1:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(tfidf_matrix)
                
                for i in range(n_clusters):
                    cluster_docs = [descriptions[j] for j in range(len(descriptions)) if clusters[j] == i]
                    cluster_patterns[f'cluster_{i}'] = {
                        'size': len(cluster_docs),
                        'sample': cluster_docs[:3] if cluster_docs else []
                    }
            
            return {
                'top_terms': term_scores,
                'clusters': cluster_patterns,
                'vocabulary_size': len(feature_names)
            }
        except:
            return {}
    
    def anomaly_detection_ensemble(self, features: np.ndarray) -> Dict[str, Any]:
        """Ensemble anomaly detection"""
        if len(features) < 10:
            return {}
        
        results = {}
        
        try:
            # Isolation Forest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            iso_predictions = iso_forest.fit_predict(features)
            iso_scores = iso_forest.score_samples(features)
            
            results['isolation_forest'] = {
                'n_anomalies': int(np.sum(iso_predictions == -1)),
                'anomaly_ratio': float(np.mean(iso_predictions == -1)),
                'avg_anomaly_score': float(np.mean(iso_scores))
            }
            
            # DBSCAN
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            clusters = dbscan.fit_predict(features_scaled)
            
            results['dbscan'] = {
                'n_clusters': int(len(set(clusters)) - (1 if -1 in clusters else 0)),
                'n_noise_points': int(np.sum(clusters == -1)),
                'noise_ratio': float(np.mean(clusters == -1))
            }
            
            # Statistical outliers (Z-score)
            z_scores = np.abs(stats.zscore(features, axis=0))
            outliers = np.any(z_scores > 3, axis=1)
            
            results['statistical'] = {
                'n_outliers': int(np.sum(outliers)),
                'outlier_ratio': float(np.mean(outliers))
            }
            
            # Ensemble voting
            ensemble_anomalies = (
                (iso_predictions == -1).astype(int) +
                (clusters == -1).astype(int) +
                outliers.astype(int)
            ) >= 2  # At least 2 methods agree
            
            results['ensemble'] = {
                'n_anomalies': int(np.sum(ensemble_anomalies)),
                'anomaly_ratio': float(np.mean(ensemble_anomalies)),
                'anomaly_indices': np.where(ensemble_anomalies)[0].tolist()[:10]  # First 10
            }
        except Exception as e:
            print(f"Anomaly detection error: {str(e)}")
        
        return results
    
    def discover_all_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Tum pattern discovery metodlarini calistir"""
        patterns = {
            'optimal_ranges': {},
            'defect_patterns': {},
            'anomalies': {},
            'clustering': {},
            'time_series': {},
            'correlations': {}
        }
        
        try:
            # Her kanal icin analiz
            for i in range(1, 5):
                # Support both Excel (deger_) and Firebase (channel_) formats
                channel_col = f'deger_{i}_parsed' if f'deger_{i}_parsed' in df.columns else f'channel_{i}'
                if channel_col not in df.columns:
                    continue
                
                # OK ve NOK degerlerini ayir
                ok_mask = df['test_result'] == 'OK'
                nok_mask = df['test_result'] == 'NOK'
                
                ok_values = df.loc[ok_mask, channel_col].dropna().values
                nok_values = df.loc[nok_mask, channel_col].dropna().values
                all_values = df[channel_col].dropna().values
                
                if len(ok_values) > 2:
                    channel_patterns = {
                        'statistics': {
                            'mean': float(np.mean(ok_values)),
                            'std': float(np.std(ok_values)),
                            'median': float(np.median(ok_values)),
                            'min': float(np.min(ok_values)),
                            'max': float(np.max(ok_values)),
                            'q1': float(np.percentile(ok_values, 25)),
                            'q3': float(np.percentile(ok_values, 75)),
                            'iqr': float(np.percentile(ok_values, 75) - np.percentile(ok_values, 25)),
                            'sample_size': len(ok_values)
                        }
                    }
                    
                    # KDE
                    kde_results = self.discover_optimal_ranges_kde(ok_values)
                    if kde_results:
                        channel_patterns['kde'] = kde_results
                    
                    # Bayesian
                    if len(nok_values) > 0:
                        bayesian_results = self.discover_optimal_ranges_bayesian(ok_values, nok_values)
                        if bayesian_results:
                            channel_patterns['bayesian'] = bayesian_results
                    
                    # Normality test
                    if len(ok_values) > 3:
                        try:
                            stat, p_value = shapiro(ok_values)
                            channel_patterns['normality'] = {
                                'shapiro_stat': float(stat),
                                'p_value': float(p_value),
                                'is_normal': p_value > 0.05
                            }
                        except:
                            pass
                    
                    # Range information from original data
                    range_col = f'deger_{i}_is_range'
                    if range_col in df.columns:
                        range_data = df.loc[ok_mask, range_col]
                        channel_patterns['range_info'] = {
                            'has_ranges': bool(range_data.any()),
                            'range_ratio': float(range_data.mean()) if len(range_data) > 0 else 0
                        }
                    
                    patterns['optimal_ranges'][f'channel_{i}'] = channel_patterns
                
                elif len(all_values) > 2:
                    # Eger OK veri yoksa tum verileri kullan
                    channel_patterns = {
                        'statistics': {
                            'mean': float(np.mean(all_values)),
                            'std': float(np.std(all_values)),
                            'median': float(np.median(all_values)),
                            'min': float(np.min(all_values)),
                            'max': float(np.max(all_values)),
                            'q1': float(np.percentile(all_values, 25)),
                            'q3': float(np.percentile(all_values, 75)),
                            'sample_size': len(all_values),
                            'note': 'Using all data (no OK/NOK separation)'
                        }
                    }
                    patterns['optimal_ranges'][f'channel_{i}'] = channel_patterns
            
            # Feature matrix for anomaly detection
            feature_cols = [f'deger_{i}_parsed' for i in range(1, 5) if f'deger_{i}_parsed' in df.columns]
            if len(feature_cols) >= 2:
                feature_matrix = df[feature_cols].fillna(0).values
                if len(feature_matrix) > 10:
                    patterns['anomalies'] = self.anomaly_detection_ensemble(feature_matrix)
            
            # Defect pattern analysis
            if 'aciklama' in df.columns:
                nok_descriptions = df.loc[df['test_result'] == 'NOK', 'aciklama'].dropna().tolist()
                if nok_descriptions:
                    patterns['defect_patterns']['text_mining'] = self.text_mining_defects(nok_descriptions)
            
            # NOK channel analysis
            if 'nok_channels' in df.columns:
                nok_df = df[df['test_result'] == 'NOK']
                all_nok_channels = []
                for channels in nok_df['nok_channels']:
                    if channels:
                        all_nok_channels.extend(channels)
                
                if all_nok_channels:
                    channel_freq = Counter(all_nok_channels)
                    patterns['defect_patterns']['channel_failures'] = dict(channel_freq)
                    patterns['defect_patterns']['most_failed_channel'] = max(channel_freq, key=channel_freq.get)
                    patterns['defect_patterns']['total_nok_channels'] = len(all_nok_channels)
            
            # Test result summary
            test_counts = df['test_result'].value_counts().to_dict()
            patterns['defect_patterns']['test_summary'] = {
                'total_tests': len(df),
                'ok_count': test_counts.get('OK', 0),
                'nok_count': test_counts.get('NOK', 0),
                'unknown_count': test_counts.get('UNKNOWN', 0),
                'success_rate': test_counts.get('OK', 0) / len(df) * 100 if len(df) > 0 else 0
            }
            
            # Time series patterns
            if 'tarih' in df.columns and df['tarih'].notna().any():
                df_sorted = df.sort_values('tarih')
                for i in range(1, 5):
                    col = f'deger_{i}_parsed'
                    if col in df_sorted.columns:
                        values = df_sorted[col].dropna().values
                        if len(values) > 10:
                            try:
                                # Simple trend analysis
                                x = np.arange(len(values))
                                slope, intercept = np.polyfit(x, values, 1)
                                
                                patterns['time_series'][f'channel_{i}'] = {
                                    'trend_slope': float(slope),
                                    'trend_direction': 'increasing' if slope > 0 else 'decreasing',
                                    'volatility': float(np.std(np.diff(values)))
                                }
                            except:
                                pass
            
            # Correlation analysis
            if len(feature_cols) >= 2:
                try:
                    corr_matrix = df[feature_cols].corr()
                    patterns['correlations'] = corr_matrix.to_dict()
                except:
                    pass
            
        except Exception as e:
            print(f"Pattern discovery error: {str(e)}")
        
        return patterns

# ==================== ML ENGINE ====================
class MLEngine:
    """Ensemble ML modelleri"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Feature matrix hazirla"""
        features = []
        labels = []
        
        # Support both Excel (deger_) and Firebase (channel_) formats
        feature_cols = []
        for i in range(1, 5):
            if f'deger_{i}_parsed' in df.columns:
                feature_cols.append(f'deger_{i}_parsed')
            elif f'channel_{i}' in df.columns:
                feature_cols.append(f'channel_{i}')
            else:
                feature_cols.append(None)
        
        for idx, row in df.iterrows():
            feat = []
            valid_row = True
            
            for col in feature_cols:
                if col in df.columns:
                    val = row.get(col, 0)
                    if pd.notna(val):
                        feat.append(float(val))
                    else:
                        feat.append(0)
                else:
                    feat.append(0)
            
            # Additional features
            if 'vardiya' in df.columns:
                # Encode shift
                shift_map = {'GUNDUZ': 0, 'GECE': 1, 'SABAH': 0, 'AKSAM': 1}
                shift_val = shift_map.get(str(row.get('vardiya', '')).upper(), 0)
                feat.append(shift_val)
            
            # Check if we have valid features
            if len(feat) >= 4 and any(f != 0 for f in feat[:4]):
                features.append(feat)
                labels.append(1 if row['test_result'] == 'OK' else 0)
        
        return np.array(features), np.array(labels)
    
    def train_ensemble(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Ensemble model training"""
        try:
            X, y = self.prepare_features(df)
            
            if len(X) < 20:
                return {'success': False, 'error': 'Insufficient data for training'}
            
            if len(np.unique(y)) < 2:
                return {'success': False, 'error': 'All samples have the same label'}
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            self.scalers['robust'] = scaler
            
            results = {
                'success': True,
                'models': [],
                'best_model': None,
                'best_accuracy': 0
            }
            
            # 1. Random Forest
            try:
                rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
                rf.fit(X_train_scaled, y_train)
                rf_score = rf.score(X_test_scaled, y_test)
                self.models['random_forest'] = rf
                
                results['models'].append({
                    'name': 'Random Forest',
                    'accuracy': float(rf_score),
                    'feature_importance': rf.feature_importances_.tolist()
                })
            except Exception as e:
                print(f"RF training error: {str(e)}")
            
            # 2. Gradient Boosting
            try:
                gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
                gb.fit(X_train_scaled, y_train)
                gb_score = gb.score(X_test_scaled, y_test)
                self.models['gradient_boosting'] = gb
                
                results['models'].append({
                    'name': 'Gradient Boosting',
                    'accuracy': float(gb_score),
                    'feature_importance': gb.feature_importances_.tolist()
                })
            except Exception as e:
                print(f"GB training error: {str(e)}")
            
            # 3. XGBoost
            try:
                xgb_model = xgb.XGBClassifier(
                    n_estimators=100, 
                    learning_rate=0.1, 
                    max_depth=5, 
                    random_state=42,
                    use_label_encoder=False,
                    eval_metric='logloss'
                )
                xgb_model.fit(X_train_scaled, y_train)
                xgb_score = xgb_model.score(X_test_scaled, y_test)
                self.models['xgboost'] = xgb_model
                
                results['models'].append({
                    'name': 'XGBoost',
                    'accuracy': float(xgb_score),
                    'feature_importance': xgb_model.feature_importances_.tolist()
                })
            except Exception as e:
                print(f"XGBoost training error: {str(e)}")
            
            # 4. Neural Network
            try:
                mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
                mlp.fit(X_train_scaled, y_train)
                mlp_score = mlp.score(X_test_scaled, y_test)
                self.models['neural_network'] = mlp
                
                results['models'].append({
                    'name': 'Neural Network',
                    'accuracy': float(mlp_score)
                })
            except Exception as e:
                print(f"MLP training error: {str(e)}")
            
            # Find best model
            if results['models']:
                best_model_info = max(results['models'], key=lambda x: x['accuracy'])
                results['best_model'] = best_model_info['name']
                results['best_accuracy'] = best_model_info['accuracy']
                
                # Cross validation on best
                if results['best_model'] == 'Random Forest' and 'random_forest' in self.models:
                    cv_scores = cross_val_score(self.models['random_forest'], X_train_scaled, y_train, cv=5)
                    results['cross_validation'] = {
                        'mean': float(np.mean(cv_scores)),
                        'std': float(np.std(cv_scores))
                    }
            
            return results
            
        except Exception as e:
            print(f"ML training error: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def detect_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect patterns in data"""
        patterns = {'optimal_ranges': {}}
        
        # Analyze each channel
        for i in range(1, 5):
            channel_col = f'channel_{i}'
            if channel_col in df.columns:
                channel_data = df[df[channel_col].notna()][channel_col]
                if len(channel_data) > 0:
                    patterns['optimal_ranges'][channel_col] = {
                        'min': float(channel_data.min()),
                        'max': float(channel_data.max()),
                        'mean': float(channel_data.mean()),
                        'median': float(channel_data.median()),
                        'std': float(channel_data.std()),
                        'statistics': {
                            'sample_size': len(channel_data),
                            'std': float(channel_data.std())
                        }
                    }
        
        return patterns
    
    def predict_ensemble(self, features: List[float]) -> Dict[str, Any]:
        """Ensemble prediction"""
        if not self.models or 'robust' not in self.scalers:
            return {'error': 'Models not trained'}
        
        try:
            # Prepare features
            X = np.array([features])
            X_scaled = self.scalers['robust'].transform(X)
            
            predictions = {}
            probabilities = {}
            
            for name, model in self.models.items():
                try:
                    pred = model.predict(X_scaled)[0]
                    predictions[name] = 'OK' if pred == 1 else 'NOK'
                    
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X_scaled)[0]
                        probabilities[name] = float(max(proba))
                except:
                    continue
            
            # Ensemble voting
            ok_votes = sum(1 for p in predictions.values() if p == 'OK')
            nok_votes = len(predictions) - ok_votes
            
            ensemble_prediction = 'OK' if ok_votes > nok_votes else 'NOK'
            ensemble_confidence = max(ok_votes, nok_votes) / len(predictions) if predictions else 0
            
            return {
                'ensemble_prediction': ensemble_prediction,
                'ensemble_confidence': ensemble_confidence,
                'individual_predictions': predictions,
                'probabilities': probabilities
            }
        except Exception as e:
            return {'error': str(e)}

# ==================== GENETIC OPTIMIZATION ====================
class GeneticOptimizer:
    """Genetic algorithm optimization"""
    
    def __init__(self, population_size=50, generations=100):
        self.population_size = population_size
        self.generations = generations
        
    def fitness_function(self, params: np.ndarray, target_data: Dict) -> float:
        """Fitness hesaplama"""
        fitness = 0
        for i, val in enumerate(params):
            if f'channel_{i+1}' in target_data:
                optimal = target_data[f'channel_{i+1}'].get('optimal_value', val)
                fitness -= abs(val - optimal)
        return fitness
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Crossover operation"""
        mask = np.random.rand(len(parent1)) > 0.5
        child = np.where(mask, parent1, parent2)
        return child
    
    def mutate(self, individual: np.ndarray, mutation_rate=0.1) -> np.ndarray:
        """Mutation operation"""
        mask = np.random.rand(len(individual)) < mutation_rate
        individual[mask] += np.random.randn(np.sum(mask)) * 0.1
        return individual
    
    def optimize(self, target_data: Dict, bounds: List[Tuple[float, float]]) -> Dict[str, Any]:
        """Run genetic optimization"""
        try:
            n_params = len(bounds)
            
            # Initialize population
            population = np.random.rand(self.population_size, n_params)
            for i, (low, high) in enumerate(bounds):
                population[:, i] = population[:, i] * (high - low) + low
            
            best_fitness_history = []
            
            for generation in range(self.generations):
                # Calculate fitness
                fitness_scores = np.array([self.fitness_function(ind, target_data) for ind in population])
                
                # Select best individuals
                sorted_indices = np.argsort(fitness_scores)[::-1]
                population = population[sorted_indices]
                fitness_scores = fitness_scores[sorted_indices]
                
                best_fitness_history.append(fitness_scores[0])
                
                # Create new generation
                new_population = population[:self.population_size//2].copy()
                
                # Crossover and mutation
                for _ in range(self.population_size//2):
                    parent1_idx = np.random.randint(0, self.population_size//2)
                    parent2_idx = np.random.randint(0, self.population_size//2)
                    
                    child = self.crossover(population[parent1_idx], population[parent2_idx])
                    child = self.mutate(child)
                    
                    # Ensure bounds
                    for i, (low, high) in enumerate(bounds):
                        child[i] = np.clip(child[i], low, high)
                    
                    new_population = np.vstack([new_population, child])
                
                population = new_population[:self.population_size]
            
            # Get best solution
            final_fitness = np.array([self.fitness_function(ind, target_data) for ind in population])
            best_idx = np.argmax(final_fitness)
            best_solution = population[best_idx]
            
            return {
                'optimal_parameters': {f'channel_{i+1}': float(val) for i, val in enumerate(best_solution)},
                'fitness': float(final_fitness[best_idx]),
                'convergence_history': best_fitness_history[-10:],  # Last 10 generations
                'generations': self.generations
            }
        except Exception as e:
            return {'error': str(e)}

# ==================== API ENDPOINTS ====================

@app.get("/")
async def root():
    return {
        "message": "LOB Test Analiz Platformu - Perfect Edition v6.0",
        "status": "operational",
        "features": [
            "Advanced Excel parsing with full encoding support",
            "KDE & Bayesian pattern discovery",
            "Ensemble ML models (RF, GB, XGB, NN)",
            "Genetic algorithm optimization",
            "Real-time anomaly detection",
            "Text mining for defect patterns",
            "Full error handling and edge case coverage"
        ]
    }

@app.post("/api/upload-excel")
async def upload_excel(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Excel yukle ve detayli analiz et"""
    try:
        # Read file
        contents = await file.read()
        
        # Parse Excel
        parser = PerfectExcelParser()
        df = parser.parse_excel_file(contents)
        
        if df.empty:
            raise HTTPException(status_code=400, detail="Excel dosyasi bos veya okunamadi")
        
        # Store data
        data_store.raw_data = df
        data_store.processed_data = df
        
        # Detayli kanal analizi
        channel_analyzer = DetailedChannelAnalyzer()
        data_store.detailed_channel_analysis = channel_analyzer.analyze_all_channels(df)
        
        # Sheet bazli analiz
        sheet_analyzer = SheetBasedAnalyzer()
        data_store.sheet_based_analysis = sheet_analyzer.analyze_all_sheets(df)
        
        # Bimal test karşılaştırma analizi
        bimal_analyzer = BimalTestAnalyzer()
        data_store.bimal_analysis = bimal_analyzer.analyze_bimal_comparison(df)
        
        # Calculate statistics
        test_dist = df['test_result'].value_counts().to_dict()
        ok_count = test_dist.get('OK', 0)
        nok_count = test_dist.get('NOK', 0)
        unknown_count = test_dist.get('UNKNOWN', 0)
        total = ok_count + nok_count
        
        # Pattern discovery
        pattern_discovery = AdvancedPatternDiscovery()
        patterns = pattern_discovery.discover_all_patterns(df)
        data_store.patterns = patterns
        
        # ML training
        ml_engine = MLEngine()
        ml_results = ml_engine.train_ensemble(df)
        data_store.ml_models['engine'] = ml_engine
        data_store.ml_models['results'] = ml_results
        
        # Detayli ozet - her kanal icin OK/NOK sayilari
        channel_summary = {}
        if 'channels' in data_store.detailed_channel_analysis:
            for ch_key, ch_data in data_store.detailed_channel_analysis['channels'].items():
                channel_summary[ch_key] = {
                    'ok': ch_data.get('ok_count', 0),
                    'nok': ch_data.get('nok_count', 0),
                    'total': ch_data.get('total_measurements', 0),
                    'ok_percentage': ch_data.get('ok_percentage', 0),
                    'nok_percentage': ch_data.get('nok_percentage', 0)
                }
        
        # Sheet bazli ozet
        sheet_summary = {}
        for sheet_name, sheet_data in data_store.sheet_based_analysis.items():
            sheet_summary[sheet_name] = {
                'total_tests': sheet_data.get('total_tests', 0),
                'success_rate': sheet_data.get('success_rate', 0),
                'test_distribution': sheet_data.get('test_distribution', {})
            }
        
        # Summary
        summary = {
            'total_records': len(df),
            'sheets_processed': df['sheet_name'].nunique() if 'sheet_name' in df.columns else 1,
            'test_distribution': test_dist,
            'success_rate': (ok_count / total * 100) if total > 0 else 0,
            'unknown_records': unknown_count,
            'channel_details': channel_summary,
            'sheet_details': sheet_summary,
            'patterns_discovered': len(patterns.get('optimal_ranges', {})),
            'ml_models_trained': len(ml_results.get('models', [])) if ml_results.get('success') else 0,
            'best_model': ml_results.get('best_model'),
            'best_accuracy': ml_results.get('best_accuracy', 0),
            'anomalies_detected': patterns.get('anomalies', {}).get('ensemble', {}).get('n_anomalies', 0)
        }
        
        # Store metadata
        data_store.metadata = {
            'upload_time': datetime.now().isoformat(),
            'file_name': file.filename,
            'file_size': len(contents),
            'summary': summary
        }
        
        return JSONResponse(content={
            'success': True,
            'message': 'Excel basariyla yuklendi ve detayli analiz edildi',
            'summary': summary
        })
        
    except Exception as e:
        print(f"Upload error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/discovered-patterns")
async def get_discovered_patterns():
    """Kesfedilen pattern'leri dondur"""
    if not data_store.patterns:
        return JSONResponse(content={
            'success': False,
            'error': 'No patterns discovered yet'
        })
    
    # Safely serialize the data to handle NaN and inf values
    safe_patterns = safe_serialize(data_store.patterns)
    
    return JSONResponse(content={
        'success': True,
        'patterns': safe_patterns
    })

@app.get("/api/channel-health/{channel_id}")
async def channel_health(channel_id: int):
    """Kanal saglik durumu analizi"""
    if channel_id < 1 or channel_id > 4:
        raise HTTPException(status_code=400, detail="Invalid channel ID")
    
    channel_key = f'channel_{channel_id}'
    health_data = {
        'channel_id': channel_id,
        'health_score': 0,
        'risk_level': 'unknown',
        'statistics': {},
        'recommendations': []
    }
    
    if data_store.patterns and channel_key in data_store.patterns.get('optimal_ranges', {}):
        channel_patterns = data_store.patterns['optimal_ranges'][channel_key]
        
        # Statistics
        if 'statistics' in channel_patterns:
            health_data['statistics'] = channel_patterns['statistics']
        
        # Health score calculation
        sample_size = channel_patterns.get('statistics', {}).get('sample_size', 0)
        std = channel_patterns.get('statistics', {}).get('std', 1)
        
        # Score based on sample size and stability
        size_score = min(sample_size / 100, 1.0) * 50
        stability_score = (1 / (1 + std)) * 50
        health_data['health_score'] = size_score + stability_score
        
        # Risk level
        if health_data['health_score'] > 80:
            health_data['risk_level'] = 'low'
            health_data['recommendations'].append("Kanal stabil ve saglikli")
        elif health_data['health_score'] > 60:
            health_data['risk_level'] = 'medium'
            health_data['recommendations'].append("Kanal izlenmeli, kucuk sapmalar var")
        else:
            health_data['risk_level'] = 'high'
            health_data['recommendations'].append("Kanal kritik durumda, bakim gerekli")
        
        # Add optimal ranges
        if 'kde' in channel_patterns:
            health_data['optimal_range'] = {
                'lower': channel_patterns['kde']['ci_lower'],
                'upper': channel_patterns['kde']['ci_upper'],
                'optimal': channel_patterns['kde']['optimal_value']
            }
        
        # Normality info
        if 'normality' in channel_patterns:
            health_data['distribution'] = {
                'is_normal': channel_patterns['normality']['is_normal'],
                'p_value': channel_patterns['normality']['p_value']
            }
            
            if not channel_patterns['normality']['is_normal']:
                health_data['recommendations'].append("Deger dagilimi normal degil, proses kontrolu onerilir")
    
    return JSONResponse(content=health_data)

@app.get("/api/raw-data")
async def get_raw_data():
    """Ham veriyi döndür (Firebase için tüm veri)"""
    if data_store.processed_data is None:
        return JSONResponse(content={'error': 'No data available'})
    
    df = data_store.processed_data
    
    # Sadece gerekli kolonları al
    columns_to_include = [
        'sira_no', 'test_result', 'nok_channels', 'tarih', 'vardiya', 'row_index',
        'deger_1_parsed', 'deger_2_parsed', 'deger_3_parsed', 'deger_4_parsed',
        'deger_1_min', 'deger_1_max', 'deger_2_min', 'deger_2_max',
        'deger_3_min', 'deger_3_max', 'deger_4_min', 'deger_4_max',
        'sheet_name', 'sheet_index', 'bimal_result'  # Sheet bilgisi ve bimal sonucu da ekle
    ]
    
    # Mevcut kolonları kontrol et
    available_columns = [col for col in columns_to_include if col in df.columns]
    
    # TÜM veriyi gönder (limit yok)
    raw_data = df[available_columns].to_dict('records')  # TÜM kayıtlar
    
    # NaN ve inf değerleri temizle
    safe_data = safe_serialize(raw_data)
    
    return JSONResponse(content=safe_data)

@app.get("/api/debug-bimal")
async def debug_bimal():
    """Debug endpoint for Bimal data correlation"""
    debug_info = {
        'has_bimal_data': hasattr(data_store, 'bimal_data') and data_store.bimal_data is not None,
        'bimal_data_length': len(data_store.bimal_data) if hasattr(data_store, 'bimal_data') and data_store.bimal_data is not None else 0,
        'bimal_columns': list(data_store.bimal_data.columns) if hasattr(data_store, 'bimal_data') and data_store.bimal_data is not None else [],
        'processed_data_length': len(data_store.processed_data) if data_store.processed_data is not None else 0,
        'processed_columns': list(data_store.processed_data.columns) if data_store.processed_data is not None else []
    }
    
    # Check correlation calculation
    if hasattr(data_store, 'bimal_data') and data_store.bimal_data is not None and 'bimal_result' in data_store.bimal_data.columns:
        if data_store.processed_data is not None and 'test_result' in data_store.processed_data.columns:
            # Merge data
            merged = pd.merge(
                data_store.processed_data[['sira_no', 'test_result']],
                data_store.bimal_data[['sira_no', 'bimal_result']],
                on='sira_no',
                how='inner'
            )
            
            debug_info['merged_count'] = len(merged)
            
            if len(merged) > 0:
                # Convert to numeric
                airleak_values = (merged['test_result'].str.upper() == 'OK').astype(int)
                bimal_values = (merged['bimal_result'].str.upper() == 'OK').astype(int)
                
                debug_info['airleak_ok_count'] = int(airleak_values.sum())
                debug_info['bimal_ok_count'] = int(bimal_values.sum())
                
                # Calculate correlation
                if np.var(airleak_values) > 0 and np.var(bimal_values) > 0:
                    correlation = np.corrcoef(airleak_values, bimal_values)[0, 1]
                    debug_info['correlation'] = correlation
                    debug_info['r_squared'] = correlation ** 2
                else:
                    debug_info['correlation'] = 'No variance in data'
    
    return JSONResponse(content=debug_info)

@app.get("/api/statistics")
async def get_statistics():
    """Genel istatistikler"""
    if data_store.processed_data is None:
        return JSONResponse(content={'error': 'No data available'})
    
    df = data_store.processed_data
    
    # Basic stats
    stats = {
        'total_records': len(df),
        'test_results': df['test_result'].value_counts().to_dict() if 'test_result' in df.columns else {},
        'channels_analyzed': sum(1 for i in range(1, 5) if f'deger_{i}_parsed' in df.columns or f'channel_{i}' in df.columns)
    }
    
    # Time range
    if 'tarih' in df.columns:
        valid_dates = df['tarih'].dropna()
        if len(valid_dates) > 0:
            try:
                # Convert to datetime if needed
                if not pd.api.types.is_datetime64_any_dtype(valid_dates):
                    valid_dates = pd.to_datetime(valid_dates, errors='coerce')
                    valid_dates = valid_dates.dropna()
                
                if len(valid_dates) > 0:
                    stats['date_range'] = {
                        'start': str(valid_dates.min()),
                        'end': str(valid_dates.max()),
                        'days_covered': int((valid_dates.max() - valid_dates.min()).days) if len(valid_dates) > 1 else 0
                    }
            except:
                pass  # Skip date range if conversion fails
    
    # Shift distribution
    if 'vardiya' in df.columns:
        stats['shift_distribution'] = df['vardiya'].value_counts().to_dict()
    
    # Sheet distribution
    if 'sheet_name' in df.columns:
        stats['sheet_distribution'] = df['sheet_name'].value_counts().to_dict()
    
    return JSONResponse(content=stats)

@app.get("/api/ml-insights")
async def get_ml_insights():
    """ML model sonuclari"""
    if 'results' not in data_store.ml_models:
        return JSONResponse(content={'error': 'ML models not trained yet'})
    
    results = data_store.ml_models['results']
    
    # Add more insights
    insights = {
        'models': results.get('models', []),
        'feature_importance': results.get('feature_importance', []),
        'anomalies': results.get('anomalies', []),
        'clusters': results.get('clusters', {}),
        'summary': {
            'total_models_trained': len(results.get('models', [])),
            'best_model': results.get('best_model', {}),
            'anomaly_detection_enabled': len(results.get('anomalies', [])) > 0,
            'clustering_performed': bool(results.get('clusters', {}))
        },
        'recommendations': generate_ml_recommendations(results) if 'generate_ml_recommendations' in globals() else []
    }
    
    return JSONResponse(content=insights)

@app.post("/api/ml/train-advanced")
async def train_advanced_ml_models():
    """Gelişmiş ML modellerini eğit"""
    try:
        if data_store.processed_data is None:
            return JSONResponse(content={'error': 'No data available. Please upload data first.'})
        
        df = data_store.processed_data
        
        # Prepare features
        feature_cols = []
        for i in range(1, 5):
            col_name = f'channel_{i}'
            if col_name in df.columns:
                feature_cols.append(col_name)
        
        if not feature_cols:
            return JSONResponse(content={'error': 'No channel data found'})
        
        X = df[feature_cols].fillna(0).values
        y = (df['test_result'] == 'NOK').astype(int).values if 'test_result' in df.columns else np.zeros(len(df))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        results = {
            'models': [],
            'feature_importance': {},
            'best_model': {},
            'training_metrics': {}
        }
        
        # 1. Random Forest with hyperparameter tuning
        try:
            rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
            rf.fit(X_train, y_train)
            rf_score = rf.score(X_test, y_test)
            rf_cv_scores = cross_val_score(rf, X_train, y_train, cv=3)
            
            results['models'].append({
                'name': 'Random Forest (Optimized)',
                'accuracy': float(rf_score),
                'cv_mean': float(rf_cv_scores.mean()),
                'cv_std': float(rf_cv_scores.std())
            })
            
            # Feature importance
            importance = rf.feature_importances_
            results['feature_importance']['random_forest'] = [
                {'feature': f'Channel {i+1}', 'importance': float(imp)} 
                for i, imp in enumerate(importance)
            ]
            
            data_store.ml_models['random_forest_optimized'] = rf
        except Exception as e:
            print(f"Random Forest error: {str(e)}")
        
        # 2. XGBoost
        try:
            xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='logloss')
            xgb_model.fit(X_train, y_train)
            xgb_score = xgb_model.score(X_test, y_test)
            xgb_cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=3)
            
            results['models'].append({
                'name': 'XGBoost',
                'accuracy': float(xgb_score),
                'cv_mean': float(xgb_cv_scores.mean()),
                'cv_std': float(xgb_cv_scores.std())
            })
            
            # Feature importance
            importance = xgb_model.feature_importances_
            results['feature_importance']['xgboost'] = [
                {'feature': f'Channel {i+1}', 'importance': float(imp)} 
                for i, imp in enumerate(importance)
            ]
            
            data_store.ml_models['xgboost_optimized'] = xgb_model
        except Exception as e:
            print(f"XGBoost error: {str(e)}")
        
        # 3. Gradient Boosting
        try:
            gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
            gb.fit(X_train, y_train)
            gb_score = gb.score(X_test, y_test)
            gb_cv_scores = cross_val_score(gb, X_train, y_train, cv=3)
            
            results['models'].append({
                'name': 'Gradient Boosting',
                'accuracy': float(gb_score),
                'cv_mean': float(gb_cv_scores.mean()),
                'cv_std': float(gb_cv_scores.std())
            })
            
            data_store.ml_models['gradient_boosting_optimized'] = gb
        except Exception as e:
            print(f"Gradient Boosting error: {str(e)}")
        
        # 4. Deep Neural Network
        try:
            mlp = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', max_iter=1000, random_state=42)
            mlp.fit(X_train_scaled, y_train)
            mlp_score = mlp.score(X_test_scaled, y_test)
            mlp_cv_scores = cross_val_score(mlp, X_train_scaled, y_train, cv=3)
            
            results['models'].append({
                'name': 'Deep Neural Network',
                'accuracy': float(mlp_score),
                'cv_mean': float(mlp_cv_scores.mean()),
                'cv_std': float(mlp_cv_scores.std()),
                'architecture': '100-50 neurons (2 layers)'
            })
            
            data_store.ml_models['neural_network_optimized'] = mlp
        except Exception as e:
            print(f"Neural Network error: {str(e)}")
        
        # Find best model
        if results['models']:
            best_model = max(results['models'], key=lambda x: x['accuracy'])
            results['best_model'] = best_model
        
        # Store results
        data_store.ml_models['results'] = results
        
        return JSONResponse(content={
            'success': True,
            'results': results,
            'message': f"Successfully trained {len(results['models'])} advanced ML models"
        })
        
    except Exception as e:
        print(f"Advanced ML training error: {str(e)}")
        traceback.print_exc()
        return JSONResponse(content={'error': str(e)})

@app.post("/api/ml/anomaly-detection")
async def detect_anomalies_advanced():
    """Advanced anomaly detection with multiple algorithms"""
    try:
        if data_store.processed_data is None:
            return JSONResponse(content={'error': 'No data available'})
        
        df = data_store.processed_data
        
        # Prepare features
        feature_cols = [f'channel_{i}' for i in range(1, 5) if f'channel_{i}' in df.columns]
        X = df[feature_cols].fillna(0).values
        
        results = {
            'isolation_forest': {},
            'statistical': {},
            'combined': []
        }
        
        # 1. Isolation Forest
        try:
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            iso_predictions = iso_forest.fit_predict(X)
            iso_scores = iso_forest.score_samples(X)
            
            iso_anomalies = np.where(iso_predictions == -1)[0]
            results['isolation_forest'] = {
                'n_anomalies': int(len(iso_anomalies)),
                'anomaly_indices': iso_anomalies.tolist()[:100],
                'anomaly_rate': float(len(iso_anomalies) / len(X)) if len(X) > 0 else 0
            }
        except Exception as e:
            print(f"Isolation Forest error: {str(e)}")
        
        # 2. Statistical anomalies (Z-score)
        try:
            z_scores = np.abs(stats.zscore(X, axis=0, nan_policy='omit'))
            statistical_anomalies = np.where(np.nanmax(z_scores, axis=1) > 3)[0]
            
            results['statistical'] = {
                'n_anomalies': int(len(statistical_anomalies)),
                'anomaly_indices': statistical_anomalies.tolist()[:100],
                'threshold': 3.0
            }
        except Exception as e:
            print(f"Statistical anomaly error: {str(e)}")
        
        # Combine results
        all_anomalies = set()
        if results['isolation_forest']:
            all_anomalies.update(results['isolation_forest'].get('anomaly_indices', []))
        if results['statistical']:
            all_anomalies.update(results['statistical'].get('anomaly_indices', []))
        
        # Get details for combined anomalies
        for idx in list(all_anomalies)[:50]:
            if idx < len(df):
                row = df.iloc[idx]
                results['combined'].append({
                    'index': int(idx),
                    'sira_no': int(row.get('sira_no', idx)),
                    'test_result': row.get('test_result', 'UNKNOWN'),
                    'channels': {
                        f'channel_{i}': float(row.get(f'channel_{i}', 0)) 
                        for i in range(1, 5)
                    },
                    'detected_by': []
                })
                
                # Track which methods detected this anomaly
                if idx in results['isolation_forest'].get('anomaly_indices', []):
                    results['combined'][-1]['detected_by'].append('isolation_forest')
                if idx in results['statistical'].get('anomaly_indices', []):
                    results['combined'][-1]['detected_by'].append('statistical')
        
        return JSONResponse(content={
            'success': True,
            'results': results,
            'summary': {
                'total_samples': len(X),
                'total_anomalies': len(all_anomalies),
                'anomaly_rate': float(len(all_anomalies) / len(X)) if len(X) > 0 else 0
            }
        })
        
    except Exception as e:
        print(f"Anomaly detection error: {str(e)}")
        traceback.print_exc()
        return JSONResponse(content={'error': str(e)})

@app.post("/api/predict-defect")
async def predict_defect(request: PredictionRequest):
    """Hata tahmini yap"""
    if 'engine' not in data_store.ml_models:
        return JSONResponse(content={'error': 'ML models not ready'})
    
    try:
        # Prepare features
        features = [
            request.values.channel_1 or 0,
            request.values.channel_2 or 0,
            request.values.channel_3 or 0,
            request.values.channel_4 or 0
        ]
        
        # Get prediction
        ml_engine = data_store.ml_models['engine']
        result = ml_engine.predict_ensemble(features)
        
        # Add explanation if requested
        if request.include_explanation and 'ensemble_prediction' in result:
            explanation = []
            
            # Check against optimal ranges
            for i, val in enumerate(features, 1):
                channel_key = f'channel_{i}'
                if channel_key in data_store.patterns.get('optimal_ranges', {}):
                    opt_range = data_store.patterns['optimal_ranges'][channel_key]
                    if 'kde' in opt_range:
                        lower = opt_range['kde']['ci_lower']
                        upper = opt_range['kde']['ci_upper']
                        if val < lower:
                            explanation.append(f"Kanal {i}: Deger cok dusuk ({val:.3f} < {lower:.3f})")
                        elif val > upper:
                            explanation.append(f"Kanal {i}: Deger cok yuksek ({val:.3f} > {upper:.3f})")
            
            result['explanation'] = explanation if explanation else ["Tum degerler normal aralikta"]
        
        # Store for real-time tracking
        prediction_record = {
            'timestamp': datetime.now().isoformat(),
            'input': features,
            'prediction': result.get('ensemble_prediction'),
            'confidence': result.get('ensemble_confidence')
        }
        data_store.real_time_predictions.append(prediction_record)
        
        # Keep only last 100 predictions
        if len(data_store.real_time_predictions) > 100:
            data_store.real_time_predictions = data_store.real_time_predictions[-100:]
        
        return JSONResponse(content=result)
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return JSONResponse(content={'error': str(e)})

@app.get("/api/comparative-analysis")
async def comparative_analysis():
    """Sheet'ler arasi karsilastirma"""
    if data_store.processed_data is None:
        return JSONResponse(content={'error': 'No data available'})
    
    df = data_store.processed_data
    comparison = {}
    
    if 'sheet_name' in df.columns:
        for sheet in df['sheet_name'].unique():
            sheet_data = df[df['sheet_name'] == sheet]
            
            # Test distribution
            test_dist = sheet_data['test_result'].value_counts().to_dict()
            ok_count = test_dist.get('OK', 0)
            nok_count = test_dist.get('NOK', 0)
            total = ok_count + nok_count
            
            # Channel statistics
            channel_stats = {}
            for i in range(1, 5):
                # Support both formats
                col = f'deger_{i}_parsed' if f'deger_{i}_parsed' in sheet_data.columns else f'channel_{i}'
                if col in sheet_data.columns:
                    values = sheet_data[col].dropna()
                    if len(values) > 0:
                        channel_stats[f'channel_{i}'] = {
                            'mean': float(values.mean()),
                            'std': float(values.std()),
                            'min': float(values.min()),
                            'max': float(values.max()),
                            'sample_size': len(values)
                        }
            
            comparison[sheet] = {
                'total_tests': len(sheet_data),
                'ok_count': ok_count,
                'nok_count': nok_count,
                'success_rate': (ok_count / total * 100) if total > 0 else 0,
                'channel_statistics': channel_stats
            }
    
    return JSONResponse(content=comparison)

import asyncio
import json
from typing import AsyncGenerator

# Global progress store
optimization_progress = {
    'current_method': '',
    'current_iteration': 0,
    'total_iterations': 0,
    'percentage': 0,
    'status': 'idle'
}

async def optimization_progress_generator() -> AsyncGenerator[str, None]:
    """Generate SSE events for optimization progress"""
    while True:
        # Send current progress
        data = json.dumps(optimization_progress)
        yield f"data: {data}\n\n"
        await asyncio.sleep(0.5)  # Update every 500ms
        
        # Stop if optimization is complete
        if optimization_progress['status'] in ['completed', 'error']:
            break

@app.get("/api/optimization/progress")
async def get_optimization_progress():
    """SSE endpoint for real-time optimization progress"""
    return StreamingResponse(
        optimization_progress_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.post("/api/optimization/run")
async def run_optimization(request: OptimizationRequest):
    """Run advanced optimization algorithms"""
    global optimization_progress
    
    if data_store.processed_data is None or len(data_store.processed_data) == 0:
        return JSONResponse(content={'error': 'No data available for optimization'})
    
    try:
        # Reset progress
        optimization_progress = {
            'status': 'running',
            'method': request.method,
            'current_iteration': 0,
            'total_iterations': request.parameters.get('iterations', 100) if request.parameters else 100,
            'percentage': 0,
            'current_method': request.method
        }
        
        if OPTIMIZATION_AVAILABLE:
            # Progress callback to update global progress
            def update_progress(iteration, total, method=None):
                global optimization_progress
                optimization_progress['current_iteration'] = iteration
                optimization_progress['total_iterations'] = total
                optimization_progress['percentage'] = int((iteration / total) * 100)
                if method:
                    optimization_progress['current_method'] = method
            
            # Use advanced optimization engine with progress callback
            engine = AdvancedOptimizationEngine(data_store.processed_data, progress_callback=update_progress)
            
            # Run specified optimization method
            if request.method == 'ensemble':
                # Run multiple methods and return best
                results = engine.run_ensemble(['genetic', 'bayesian', 'pso', 'annealing'])
                
                # Format results
                formatted_results = {}
                for method, result in results.items():
                    if method not in ['best_method', 'best_result']:
                        formatted_results[method] = format_optimization_results(result)
                
                # Add best result summary
                if 'best_result' in results:
                    best_result = results['best_result']
                    formatted_results['best'] = format_optimization_results(best_result)
                    
                    # Get current parameters
                    current_params = {}
                    for i in range(1, 5):
                        channel_col = f'channel_{i}'
                        if channel_col in data_store.processed_data.columns:
                            current_params[channel_col] = data_store.processed_data[channel_col].mean()
                    
                    # Generate recommendations
                    recommendations = get_optimization_recommendations(best_result, current_params)
                    formatted_results['recommendations'] = recommendations
                    
                    # Sensitivity analysis
                    try:
                        sensitivity = engine.sensitivity_analysis(best_result.optimal_parameters)
                        formatted_results['best']['sensitivity_analysis'] = sensitivity
                    except Exception as e:
                        print(f"Sensitivity analysis error: {e}")
                        formatted_results['best']['sensitivity_analysis'] = None
                
                result = formatted_results
                print(f"Ensemble result keys: {list(formatted_results.keys())}")
                if 'best' in formatted_results:
                    print(f"Best result keys: {list(formatted_results['best'].keys())}")
            else:
                # Run single method with parameters
                kwargs = {}
                if request.iterations:
                    kwargs['n_iterations'] = request.iterations
                    kwargs['iterations'] = request.iterations
                    kwargs['generations'] = request.iterations
                    kwargs['max_iterations'] = request.iterations
                
                # Add any additional parameters from request
                if request.parameters:
                    kwargs.update(request.parameters)
                
                opt_result = engine.run_optimization(
                    request.method,
                    **kwargs
                )
                
                result = format_optimization_results(opt_result)
                
                # Add recommendations
                current_params = {}
                for i in range(1, 5):
                    channel_col = f'channel_{i}'
                    if channel_col in data_store.processed_data.columns:
                        current_params[channel_col] = data_store.processed_data[channel_col].mean()
                
                recommendations = get_optimization_recommendations(opt_result, current_params)
                result['recommendations'] = recommendations
                
                # Sensitivity analysis
                try:
                    sensitivity = engine.sensitivity_analysis(opt_result.optimal_parameters)
                    result['sensitivity_analysis'] = sensitivity
                except Exception as e:
                    print(f"Sensitivity analysis error: {e}")
                    result['sensitivity_analysis'] = None
        else:
            # Fallback to simple genetic algorithm
            optimizer = GeneticOptimizer(
                population_size=50,
                generations=request.iterations
            )
            
            # Determine bounds from data
            bounds = []
            for i in range(1, 5):
                channel_col = f'channel_{i}'
                if channel_col in data_store.processed_data.columns:
                    channel_data = data_store.processed_data[channel_col].dropna()
                    if len(channel_data) > 0:
                        bounds.append((channel_data.min() * 0.9, channel_data.max() * 1.1))
                    else:
                        bounds.append((0, 3))
                else:
                    bounds.append((0, 3))
            
            # Create target data
            target_data = {}
            for i in range(1, 5):
                channel_key = f'channel_{i}'
                if data_store.patterns and channel_key in data_store.patterns.get('optimal_ranges', {}):
                    channel_data = data_store.patterns['optimal_ranges'][channel_key]
                    if 'kde' in channel_data:
                        target_data[channel_key] = {'optimal_value': channel_data['kde']['optimal_value']}
                    else:
                        target_data[channel_key] = {'optimal_value': channel_data.get('statistics', {}).get('median', 0.5)}
                else:
                    target_data[channel_key] = {'optimal_value': 0.5}
            
            result = optimizer.optimize(target_data, bounds)
            result['method'] = request.method
        
        # Store results
        data_store.optimization_results[request.method] = result
        
        # Mark optimization as complete
        optimization_progress['status'] = 'completed'
        optimization_progress['percentage'] = 100
        optimization_progress['current_iteration'] = optimization_progress['total_iterations']
        
        return JSONResponse(content=result)
        
    except Exception as e:
        print(f"Optimization error: {str(e)}")
        import traceback
        traceback.print_exc()
        # Mark optimization as error
        optimization_progress['status'] = 'error'
        optimization_progress['percentage'] = 0
        return JSONResponse(content={'error': str(e)})

@app.post("/api/optimization/pressure-range")
async def optimize_pressure_ranges(request: OptimizationRequest):
    """
    Basınç aralığı optimizasyonu - Test süresi boyunca min-max değerleri ve stabiliteyi optimize eder
    """
    global optimization_progress
    
    if not PRESSURE_OPTIMIZATION_AVAILABLE:
        return JSONResponse(content={'error': 'Pressure range optimization module not available'})
    
    if data_store.processed_data is None or len(data_store.processed_data) == 0:
        return JSONResponse(content={'error': 'No data available for optimization'})
    
    try:
        # Reset progress
        optimization_progress = {
            'status': 'running',
            'method': 'pressure_range_' + request.method,
            'current_iteration': 0,
            'total_iterations': 100,
            'percentage': 0,
            'current_method': 'Basınç Aralığı Optimizasyonu'
        }
        
        # Get Bimal data if available
        bimal_data = data_store.bimal_data if hasattr(data_store, 'bimal_data') else None
        
        # Apply test filters if provided
        filtered_data = data_store.processed_data.copy()
        filtered_bimal = bimal_data.copy() if bimal_data is not None else None
        
        if request.test_filters:
            # Build filter condition based on selected combinations
            conditions = []
            
            if request.test_filters.get('both_ok', True):
                # Both tests OK
                conditions.append(
                    (filtered_data['test_result'].str.upper() == 'OK') & 
                    (filtered_data['bimal_result'].str.upper() == 'OK')
                )
            
            if request.test_filters.get('both_nok', True):
                # Both tests NOK
                conditions.append(
                    (filtered_data['test_result'].str.upper() == 'NOK') & 
                    (filtered_data['bimal_result'].str.upper() == 'NOK')
                )
            
            if request.test_filters.get('bimal_ok_airleak_nok', True):
                # Bimal OK, Airleak NOK (False rejection)
                conditions.append(
                    (filtered_data['bimal_result'].str.upper() == 'OK') & 
                    (filtered_data['test_result'].str.upper() == 'NOK')
                )
            
            if request.test_filters.get('bimal_nok_airleak_ok', True):
                # Bimal NOK, Airleak OK (Escaped defect)
                conditions.append(
                    (filtered_data['bimal_result'].str.upper() == 'NOK') & 
                    (filtered_data['test_result'].str.upper() == 'OK')
                )
            
            # Apply filter
            if conditions:
                combined_condition = conditions[0]
                for cond in conditions[1:]:
                    combined_condition = combined_condition | cond
                
                filtered_data = filtered_data[combined_condition]
                if filtered_bimal is not None:
                    filtered_bimal = filtered_bimal[combined_condition]
                
                print(f"Applied test filters - reduced from {len(data_store.processed_data)} to {len(filtered_data)} records")
                print(f"Selected combinations: {[k for k, v in request.test_filters.items() if v]}")
        
        # Create optimizer with filtered data
        optimizer = PressureRangeOptimizer(filtered_data, filtered_bimal)
        
        # Progress callback
        def update_progress(current, total, status=''):
            global optimization_progress
            optimization_progress['current_iteration'] = current
            optimization_progress['total_iterations'] = total
            optimization_progress['percentage'] = int((current / total) * 100)
            if status:
                optimization_progress['current_method'] = status
        
        # Run optimization with selected method
        method = request.method if request.method in ['advanced_bayesian', 'multi_objective', 'robust', 'adaptive'] else 'advanced_bayesian'
        result = optimizer.optimize_pressure_ranges(method=method)
        
        # Analyze patterns
        patterns = analyze_pressure_patterns(data_store.processed_data)
        
        # Format results for frontend
        formatted_result = {
            'method': method,
            'channel_results': {},
            'overall_metrics': {
                'success_rate': result.overall_success_rate,
                'stability_index': result.stability_index,
                'bimal_correlation': result.bimal_correlation,
                'risk_score': result.risk_score
            },
            'recommendations': result.recommendations,
            'patterns': patterns,
            'convergence_history': result.convergence_history,
            'sensitivity_analysis': result.sensitivity_analysis
        }
        
        # Format channel results
        for channel_id, channel_result in result.channel_results.items():
            formatted_result['channel_results'][f'channel_{channel_id}'] = {
                'optimal_pressure': channel_result.optimal_pressure,
                'min_pressure': channel_result.min_pressure,
                'max_pressure': channel_result.max_pressure,
                'pressure_range': channel_result.pressure_range,
                'stability_score': channel_result.stability_score,
                'success_rate': channel_result.success_rate,
                'confidence_interval': channel_result.confidence_interval,
                'variability_threshold': channel_result.variability_threshold
            }
        
        # Mark as complete
        optimization_progress['status'] = 'completed'
        optimization_progress['percentage'] = 100
        
        return JSONResponse(content=safe_serialize(formatted_result))
        
    except Exception as e:
        print(f"Pressure range optimization error: {str(e)}")
        traceback.print_exc()
        optimization_progress['status'] = 'error'
        return JSONResponse(content={'error': str(e)})

@app.get("/api/optimization/methods")
async def get_optimization_methods():
    """Get available optimization methods"""
    methods = {
        'genetic': {
            'name': 'Genetic Algorithm',
            'description': 'Evolutionary optimization with adaptive operators',
            'parameters': ['population_size', 'generations', 'mutation_rate'],
            'pros': ['Global optimization', 'Handles non-convex problems', 'Robust to noise'],
            'cons': ['Computationally intensive', 'Many hyperparameters']
        },
        'bayesian': {
            'name': 'Bayesian Optimization',
            'description': 'Gaussian Process-based sequential optimization',
            'parameters': ['n_iterations', 'acquisition_function'],
            'pros': ['Sample efficient', 'Uncertainty quantification', 'Good for expensive evaluations'],
            'cons': ['Struggles with high dimensions', 'Computationally expensive for large datasets']
        },
        'pso': {
            'name': 'Particle Swarm Optimization',
            'description': 'Swarm intelligence-based optimization',
            'parameters': ['n_particles', 'n_iterations', 'inertia_weight'],
            'pros': ['Simple implementation', 'Fast convergence', 'Few parameters'],
            'cons': ['Can get stuck in local optima', 'Performance depends on parameters']
        },
        'annealing': {
            'name': 'Simulated Annealing',
            'description': 'Probabilistic global optimization',
            'parameters': ['initial_temp', 'cooling_rate', 'max_iterations'],
            'pros': ['Escapes local optima', 'Simple to understand', 'Guaranteed convergence'],
            'cons': ['Slow convergence', 'Sensitive to cooling schedule']
        },
        'ensemble': {
            'name': 'Ensemble Methods',
            'description': 'Runs multiple algorithms and selects best result',
            'parameters': ['methods_to_run'],
            'pros': ['Robust results', 'Best of all methods', 'Reduces algorithm bias'],
            'cons': ['Most computationally expensive', 'Takes longer time']
        }
    }
    
    return JSONResponse(content={'methods': methods, 'available': OPTIMIZATION_AVAILABLE})

@app.get("/api/optimization/results")
async def get_optimization_results():
    """Get all optimization results"""
    if not data_store.optimization_results:
        return JSONResponse(content={'error': 'No optimization results available'})
    
    return JSONResponse(content=data_store.optimization_results)

@app.get("/api/optimization/status")
async def get_optimization_status():
    """Get optimization status and progress"""
    status = {
        'has_data': data_store.processed_data is not None,
        'data_size': len(data_store.processed_data) if data_store.processed_data is not None else 0,
        'patterns_available': bool(data_store.patterns),
        'optimization_ready': OPTIMIZATION_AVAILABLE and data_store.processed_data is not None,
        'last_optimization': list(data_store.optimization_results.keys())[-1] if data_store.optimization_results else None,
        'total_runs': len(data_store.optimization_results)
    }
    
    return JSONResponse(content=status)

@app.get("/api/real-time/predictions")
async def get_real_time_predictions():
    """Son tahminleri getir"""
    return JSONResponse(content={
        'predictions': data_store.real_time_predictions[-20:],  # Last 20
        'total_predictions': len(data_store.real_time_predictions)
    })

@app.get("/api/detailed-channel-analysis")
async def get_detailed_channel_analysis():
    """Detayli kanal analizi dondur"""
    if not data_store.detailed_channel_analysis:
        return JSONResponse(content={'error': 'No channel analysis available'})
    
    # Safely serialize the data to handle NaN and inf values
    safe_data = safe_serialize(data_store.detailed_channel_analysis)
    return JSONResponse(content=safe_data)

@app.get("/api/sheet-based-analysis")
async def get_sheet_based_analysis():
    """Sheet bazli analizi dondur"""
    if not data_store.sheet_based_analysis:
        return JSONResponse(content={'error': 'No sheet analysis available'})
    
    # Safely serialize the data to handle NaN and inf values
    safe_data = safe_serialize(data_store.sheet_based_analysis)
    return JSONResponse(content=safe_data)

@app.get("/api/channel-details/{channel_id}")
async def get_channel_details(channel_id: int):
    """Belirli bir kanalin detayli analizi"""
    if channel_id < 1 or channel_id > 4:
        raise HTTPException(status_code=400, detail="Invalid channel ID")
    
    channel_key = f'channel_{channel_id}'
    
    if not data_store.detailed_channel_analysis or 'channels' not in data_store.detailed_channel_analysis:
        return JSONResponse(content={'error': 'No analysis available'})
    
    if channel_key not in data_store.detailed_channel_analysis['channels']:
        return JSONResponse(content={'error': f'No data for {channel_key}'})
    
    channel_data = data_store.detailed_channel_analysis['channels'][channel_key]
    
    # Ek bilgiler ekle
    response = {
        **channel_data,
        'health_score': 0,
        'recommendations': []
    }
    
    # Saglik skoru hesapla
    if channel_data['total_measurements'] > 0:
        ok_ratio = channel_data['ok_percentage'] / 100
        stability = 1.0
        if 'ok_statistics' in channel_data and 'std' in channel_data['ok_statistics']:
            mean_val = channel_data['ok_statistics']['mean']
            std_val = channel_data['ok_statistics']['std']
            if mean_val > 0:
                cv = std_val / mean_val  # Coefficient of variation
                stability = 1 / (1 + cv)
        
        response['health_score'] = (ok_ratio * 70 + stability * 30)
        
        # Onerileri ekle
        if response['health_score'] > 80:
            response['recommendations'].append("Kanal mukemmel durumda")
        elif response['health_score'] > 60:
            response['recommendations'].append("Kanal stabil, izleme devam etmeli")
        else:
            response['recommendations'].append("Kanal problemli, mudahale gerekli")
        
        if channel_data['nok_percentage'] > 20:
            response['recommendations'].append(f"Yuksek hata orani: %{channel_data['nok_percentage']:.1f}")
        
        if len(channel_data.get('anomalies', [])) > 5:
            response['recommendations'].append(f"{len(channel_data['anomalies'])} anomali tespit edildi")
    
    return JSONResponse(content=response)

@app.get("/api/bimal-analysis")
async def get_bimal_analysis():
    """Bimal test karşılaştırma analizini döndür"""
    if not data_store.bimal_analysis:
        return JSONResponse(content={'error': 'No Bimal analysis available'})
    
    # Safely serialize the data
    safe_data = safe_serialize(data_store.bimal_analysis)
    return JSONResponse(content=safe_data)

@app.get("/api/pressure-analysis")
async def get_pressure_analysis():
    """Basinc degerleri analizi"""
    if data_store.processed_data is None:
        return JSONResponse(content={'error': 'No data available'})
    
    df = data_store.processed_data
    pressure_analysis = {}
    
    for i in range(1, 5):
        # Support both Excel (deger_) and Firebase (channel_) formats
        min_col = f'deger_{i}_min' if f'deger_{i}_min' in df.columns else f'channel_{i}_min'
        max_col = f'deger_{i}_max' if f'deger_{i}_max' in df.columns else f'channel_{i}_max'
        parsed_col = f'deger_{i}_parsed' if f'deger_{i}_parsed' in df.columns else f'channel_{i}'
        
        if min_col in df.columns and max_col in df.columns:
            # Genel basinc istatistikleri
            min_vals = df[min_col].dropna()
            max_vals = df[max_col].dropna()
            
            if len(min_vals) > 0 and len(max_vals) > 0:
                pressure_analysis[f'channel_{i}'] = {
                    'pressure_range': {
                        'absolute_min': float(min_vals.min()),
                        'absolute_max': float(max_vals.max()),
                        'typical_min': float(min_vals.quantile(0.25)),
                        'typical_max': float(max_vals.quantile(0.75))
                    },
                    'pressure_statistics': {
                        'avg_min': float(min_vals.mean()),
                        'avg_max': float(max_vals.mean()),
                        'std_min': float(min_vals.std()),
                        'std_max': float(max_vals.std())
                    }
                }
                
                # OK ve NOK icin ayri basinc analizi
                ok_data = df[df['test_result'] == 'OK']
                nok_data = df[df['test_result'] == 'NOK']
                
                if len(ok_data) > 0:
                    ok_min = ok_data[min_col].dropna()
                    ok_max = ok_data[max_col].dropna()
                    if len(ok_min) > 0 and len(ok_max) > 0:
                        pressure_analysis[f'channel_{i}']['ok_pressure'] = {
                            'min_avg': float(ok_min.mean()),
                            'max_avg': float(ok_max.mean()),
                            'optimal_range': {
                                'lower': float(ok_min.quantile(0.05)),
                                'upper': float(ok_max.quantile(0.95))
                            }
                        }
                
                if len(nok_data) > 0:
                    nok_min = nok_data[min_col].dropna()
                    nok_max = nok_data[max_col].dropna()
                    if len(nok_min) > 0 and len(nok_max) > 0:
                        pressure_analysis[f'channel_{i}']['nok_pressure'] = {
                            'min_avg': float(nok_min.mean()),
                            'max_avg': float(nok_max.mean())
                        }
    
    data_store.pressure_analysis = pressure_analysis
    return JSONResponse(content=pressure_analysis)

@app.get("/api/export/report")
async def export_report():
    """Detayli rapor olustur"""
    if not data_store.metadata:
        return JSONResponse(content={'error': 'No data to export'})
    
    report = {
        'metadata': data_store.metadata,
        'patterns_summary': {},
        'ml_performance': data_store.ml_models.get('results', {}),
        'optimization_results': data_store.optimization_results,
        'recommendations': []
    }
    
    # Patterns summary
    if data_store.patterns:
        for channel_key in data_store.patterns.get('optimal_ranges', {}).keys():
            channel_data = data_store.patterns['optimal_ranges'][channel_key]
            report['patterns_summary'][channel_key] = {
                'optimal_range': channel_data.get('kde', {}),
                'statistics': channel_data.get('statistics', {}),
                'normality': channel_data.get('normality', {})
            }
        
        # Anomalies
        if 'anomalies' in data_store.patterns:
            report['anomalies'] = data_store.patterns['anomalies'].get('ensemble', {})
    
    # Generate recommendations
    if data_store.patterns:
        # Check for high failure channels
        if 'defect_patterns' in data_store.patterns:
            channel_failures = data_store.patterns['defect_patterns'].get('channel_failures', {})
            if channel_failures:
                worst_channel = max(channel_failures, key=channel_failures.get)
                report['recommendations'].append(
                    f"Kanal {worst_channel} en cok hata veriyor ({channel_failures[worst_channel]} hata). Oncelikli bakim onerilir."
                )
        
        # Check for anomalies
        anomaly_ratio = data_store.patterns.get('anomalies', {}).get('ensemble', {}).get('anomaly_ratio', 0)
        if anomaly_ratio > 0.15:
            report['recommendations'].append(
                f"Yuksek anomali orani tespit edildi (%{anomaly_ratio*100:.1f}). Surec stabilitesi kontrol edilmeli."
            )
        
        # ML model performance
        best_accuracy = data_store.ml_models.get('results', {}).get('best_accuracy', 0)
        if best_accuracy < 0.85:
            report['recommendations'].append(
                f"ML model dogrulugu dusuk (%{best_accuracy*100:.1f}). Daha fazla veri toplanmasi onerilir."
            )
        elif best_accuracy > 0.95:
            report['recommendations'].append(
                f"ML model cok basarili (%{best_accuracy*100:.1f}). Tahminlere guvenle basvurabilirsiniz."
            )
    
    return JSONResponse(content=report)

@app.post("/api/load-firebase-data")
async def load_firebase_data(request: Dict[str, Any]):
    """Load data from Firebase into backend memory - Enhanced version"""
    try:
        firebase_data = request.get('data', [])
        if not firebase_data:
            return JSONResponse(content={'error': 'No data provided', 'success': False})
        
        print(f"Received {len(firebase_data)} records from Firebase")
        print(f"First record keys: {list(firebase_data[0].keys()) if firebase_data else []}")
        
        # Reconstruct data from Firebase format
        reconstructed_data = []
        for idx, record in enumerate(firebase_data):
            # Try to parse original_data if exists
            if 'original_data' in record and record['original_data']:
                try:
                    original = json.loads(record['original_data'])
                    reconstructed_data.append(original)
                    continue
                except:
                    pass
            
            # Reconstruct from Firebase fields
            reconstructed = {
                'sira_no': record.get('sira_no', idx),
                'test_result': record.get('test_result', 'UNKNOWN'),
                'sheet_name': record.get('sheet_name', 'Sayfa1'),
                'sheet_index': record.get('sheet_index', 0),
                'row_index': record.get('row_index', idx),
                'vardiya': record.get('vardiya', ''),
                'tarih': record.get('tarih', ''),
                'bimal_result': record.get('bimal_result', ''),
            }
            
            # Reconstruct channel data with multiple format support
            for i in range(1, 5):
                # Try different field names
                channel_value = (
                    record.get(f'channel_{i}') or 
                    record.get(f'channel_{i}_min') or 
                    record.get(f'deger_{i}') or 
                    record.get(f'deger_{i}_parsed') or 
                    0
                )
                reconstructed[f'channel_{i}'] = float(channel_value) if channel_value else 0
                reconstructed[f'deger_{i}_parsed'] = reconstructed[f'channel_{i}']
                
                # Min/max values
                reconstructed[f'deger_{i}_min'] = float(record.get(f'channel_{i}_min', 0) or 0)
                reconstructed[f'deger_{i}_max'] = float(record.get(f'channel_{i}_max', 0) or 0)
            
            # NOK channels
            nok_channels_raw = record.get('nok_channels', '')
            if isinstance(nok_channels_raw, str) and nok_channels_raw:
                try:
                    reconstructed['nok_channels'] = [int(ch) for ch in nok_channels_raw.split(',') if ch.strip().isdigit()]
                except:
                    reconstructed['nok_channels'] = []
            elif isinstance(nok_channels_raw, list):
                reconstructed['nok_channels'] = nok_channels_raw
            else:
                reconstructed['nok_channels'] = []
            
            reconstructed_data.append(reconstructed)
        
        # Convert to DataFrame
        df = pd.DataFrame(reconstructed_data)
        
        print(f"Reconstructed {len(df)} records")
        print(f"Columns: {list(df.columns)[:15]}...")
        
        # Ensure all channel columns exist
        for i in range(1, 5):
            if f'channel_{i}' not in df.columns:
                df[f'channel_{i}'] = df.get(f'deger_{i}_parsed', 0)
        
        # Store in backend
        data_store.raw_data = df
        data_store.processed_data = df
        data_store.bimal_data = df  # Also store as bimal_data for correlation
        
        # Run comprehensive analysis
        if len(df) > 0:
            # Channel Analysis - Correct Logic
            try:
                channel_analysis = {'channels': {}}
                
                # Analyze each channel
                for i in range(1, 5):
                    channel_col = f'channel_{i}'
                    
                    # Initialize counters
                    ok_count = 0
                    nok_count = 0
                    
                    # Go through all rows
                    for idx, row in df.iterrows():
                        # Check if this channel has data
                        if channel_col in df.columns and pd.notna(row[channel_col]):
                            test_result = row.get('test_result', 'UNKNOWN')
                            nok_channels = row.get('nok_channels', [])
                            
                            # Parse nok_channels if it's a list
                            if isinstance(nok_channels, list) and len(nok_channels) > 0:
                                # Check if this channel number is in NOK list
                                if i in nok_channels:
                                    nok_count += 1
                                else:
                                    ok_count += 1
                            elif test_result == 'OK':
                                # If test is OK, all channels are OK
                                ok_count += 1
                            elif test_result == 'NOK':
                                # If test is NOK but no specific channels listed, count as NOK
                                nok_count += 1
                    
                    total = ok_count + nok_count
                    if total > 0:
                        channel_analysis['channels'][channel_col] = {
                            'ok_count': ok_count,
                            'nok_count': nok_count,
                            'total_measurements': total,
                            'ok_percentage': (ok_count / total * 100) if total > 0 else 0,
                            'nok_percentage': (nok_count / total * 100) if total > 0 else 0
                        }
                
                # Add summary with best and worst channels
                if channel_analysis.get('channels'):
                    # Find best and worst channels
                    best_channel = None
                    worst_channel = None
                    best_ok_pct = 0
                    worst_ok_pct = 100
                    
                    for channel_name, stats in channel_analysis['channels'].items():
                        ok_pct = stats.get('ok_percentage', 0)
                        if ok_pct > best_ok_pct:
                            best_ok_pct = ok_pct
                            # Extract channel number from channel_name (e.g., 'channel_1' -> 1)
                            best_channel = int(channel_name.split('_')[-1]) if '_' in channel_name else 1
                        if ok_pct < worst_ok_pct:
                            worst_ok_pct = ok_pct
                            # Extract channel number from channel_name
                            worst_channel = int(channel_name.split('_')[-1]) if '_' in channel_name else 1
                    
                    channel_analysis['summary'] = {
                        'best_channel': f'channel_{best_channel}' if best_channel else None,
                        'worst_channel': f'channel_{worst_channel}' if worst_channel else None,
                        'best_channel_ok_percentage': best_ok_pct,
                        'worst_channel_ok_percentage': worst_ok_pct
                    }
                
                data_store.detailed_channel_analysis = channel_analysis
                print(f"Channel analysis completed: {channel_analysis}")
            except Exception as e:
                print(f"Channel analysis error: {str(e)}")
                data_store.detailed_channel_analysis = {}
            
            # Sheet Analysis
            try:
                sheet_analysis = {}
                if 'sheet_name' in df.columns:
                    for sheet_name in df['sheet_name'].unique():
                        sheet_data = df[df['sheet_name'] == sheet_name]
                        test_dist = sheet_data['test_result'].value_counts().to_dict() if 'test_result' in sheet_data.columns else {}
                        sheet_analysis[sheet_name] = {
                            'total_tests': len(sheet_data),
                            'test_distribution': test_dist,
                            'success_rate': (test_dist.get('OK', 0) / len(sheet_data) * 100) if len(sheet_data) > 0 else 0
                        }
                    data_store.sheet_based_analysis = sheet_analysis
                    print(f"Sheet analysis completed")
            except Exception as e:
                print(f"Sheet analysis error: {str(e)}")
                data_store.sheet_based_analysis = {}
            
            # Detailed Channel Analysis (for channel details)
            try:
                channel_analyzer = DetailedChannelAnalyzer()
                detailed_analysis = channel_analyzer.analyze_all_channels(df)
                # Merge with existing channel analysis
                if data_store.detailed_channel_analysis:
                    data_store.detailed_channel_analysis.update(detailed_analysis)
                else:
                    data_store.detailed_channel_analysis = detailed_analysis
                print(f"Detailed channel analysis completed")
            except Exception as e:
                print(f"Detailed channel analysis error: {str(e)}")
            
            # Bimal Test Analysis
            try:
                bimal_analyzer = BimalTestAnalyzer()
                data_store.bimal_analysis = bimal_analyzer.analyze_bimal_comparison(df)
                print(f"Bimal analysis completed")
            except Exception as e:
                print(f"Bimal analysis error: {str(e)}")
                data_store.bimal_analysis = {}
            
            # Pattern Discovery and Anomaly Detection
            try:
                pattern_discovery = AdvancedPatternDiscovery()
                patterns = pattern_discovery.discover_all_patterns(df)
                data_store.patterns = patterns
                print(f"Pattern discovery completed - {patterns.get('anomalies', {}).get('ensemble', {}).get('n_anomalies', 0)} anomalies found")
            except Exception as e:
                print(f"Pattern discovery error: {str(e)}")
                data_store.patterns = {}
            
            # ML Training
            try:
                # Pattern detection
                ml_engine = MLEngine()
                ml_results = ml_engine.train_ensemble(df)
                data_store.ml_models['engine'] = ml_engine
                data_store.ml_models['results'] = ml_results
                print(f"ML Engine trained successfully")
            except Exception as e:
                print(f"ML training skipped: {str(e)}")
                data_store.ml_models = {}
            
            # Basic statistics
            try:
                ok_count = len(df[df['test_result'] == 'OK']) if 'test_result' in df.columns else 0
                nok_count = len(df[df['test_result'] == 'NOK']) if 'test_result' in df.columns else 0
                
                data_store.metadata = {
                    'total_records': len(df),
                    'ok_count': ok_count,
                    'nok_count': nok_count,
                    'columns': list(df.columns),
                    'last_updated': datetime.now().isoformat()
                }
                
                print(f"Loaded {len(df)} records from Firebase into backend")
                print(f"Columns available: {list(df.columns)[:10]}...")
                print(f"Test distribution: OK={ok_count}, NOK={nok_count}")
            except Exception as e:
                print(f"Statistics calculation error: {str(e)}")
            
            return JSONResponse(content={
                'success': True,
                'records_loaded': len(df),
                'message': 'Firebase data loaded and analyzed',
                'stats': {
                    'total': len(df),
                    'ok_count': int(len(df[df['test_result'] == 'OK'])) if 'test_result' in df.columns else 0,
                    'nok_count': int(len(df[df['test_result'] == 'NOK'])) if 'test_result' in df.columns else 0
                }
            })
        
        return JSONResponse(content={'error': 'No valid data to process'})
        
    except Exception as e:
        print(f"Error loading Firebase data: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(content={'error': str(e)})

if __name__ == "__main__":
    print("=" * 60)
    print("LOB TEST ANALIZ PLATFORMU - PERFECT EDITION v6.0")
    print("=" * 60)
    print("Server starting at http://localhost:8000")
    print("Features: Full Excel parsing, KDE, Bayesian, ML Ensemble")
    print("Ready for production use - All edge cases handled")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)