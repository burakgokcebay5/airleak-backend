@echo off
echo ========================================
echo   GITHUB'A PUSH ETME ADIMLARI
echo ========================================
echo.
echo 1. GitHub.com'a gidin ve login olun (burakgokcebay5@gmail.com)
echo.
echo 2. Yeni repository olusturun:
echo    - Repository name: airleak-backend
echo    - Description: Airleak LOB Test Analysis Backend API
echo    - Public repository secin
echo    - DO NOT initialize with README (zaten var)
echo    - Create repository tiklayin
echo.
echo 3. Repository olusturduktan sonra, asagidaki komutlari sirayla calistirin:
echo.
echo    git remote add origin https://github.com/burakgokcebay5/airleak-backend.git
echo    git branch -M main
echo    git push -u origin main
echo.
echo 4. GitHub username ve password/token isteyecek
echo    - Username: burakgokcebay5
echo    - Password: GitHub Personal Access Token kullanin
echo.
echo NOT: Eger token'iniz yoksa:
echo    GitHub Settings - Developer settings - Personal access tokens - Generate new token
echo.
pause