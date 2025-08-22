# Başlangıç kısmını kopyalıyorum ve xgboost'u opsiyonel yapıyorum
# Bu dosyanın tamamını oluşturmak yerine sadece xgboost import kısmını düzeltelim

import sys
import os

# XGBoost'u opsiyonel yap
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not available. Some features will be disabled.")
    xgb = None