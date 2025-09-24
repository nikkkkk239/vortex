@echo off
REM Quantum-Enhanced Medical Imaging AI System Deployment Script for Windows
REM This script sets up the complete production environment on Windows

echo Starting Quantum-Enhanced Medical Imaging AI System Deployment...

REM Configuration
set PROJECT_NAME=quantum-medical-llava
set DOCKER_COMPOSE_FILE=docker-compose.yml
set ENV_FILE=.env

REM Function to check if Docker is installed
where docker >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Docker is not installed. Please install Docker Desktop for Windows first.
    pause
    exit /b 1
)

where docker-compose >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Docker Compose is not installed. Please install Docker Compose first.
    pause
    exit /b 1
)

echo [SUCCESS] Docker and Docker Compose are installed

REM Create necessary directories
echo [INFO] Creating necessary directories...
if not exist "uploads" mkdir uploads
if not exist "temp" mkdir temp
if not exist "temp\dicom" mkdir temp\dicom
if not exist "temp\nifti" mkdir temp\nifti
if not exist "reports" mkdir reports
if not exist "logs" mkdir logs
if not exist "models" mkdir models
if not exist "data" mkdir data
if not exist "ssl" mkdir ssl
if not exist "backups" mkdir backups
echo [SUCCESS] All directories created

REM Set up environment configuration
echo [INFO] Setting up environment configuration...
if not exist "%ENV_FILE%" (
    if exist ".env.example" (
        copy .env.example "%ENV_FILE%"
        echo [WARNING] Copied .env.example to .env. Please update with your actual values.
    ) else (
        echo [ERROR] .env.example file not found
        pause
        exit /b 1
    )
) else (
    echo [INFO] .env file already exists
)

REM Generate SSL certificates for development (requires OpenSSL)
if not exist "ssl\cert.pem" (
    echo [INFO] Generating self-signed SSL certificates for development...
    REM Note: This requires OpenSSL to be installed on Windows
    openssl req -x509 -newkey rsa:4096 -keyout ssl\key.pem -out ssl\cert.pem -days 365 -nodes -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost" 2>nul
    if %errorlevel% neq 0 (
        echo [WARNING] OpenSSL not found. SSL certificates not generated.
        echo [INFO] You can generate them manually or use a certificate authority.
    ) else (
        echo [SUCCESS] SSL certificates generated
    )
)

REM Handle command line arguments
if "%1"=="stop" goto stop
if "%1"=="restart" goto restart
if "%1"=="status" goto status
if "%1"=="logs" goto logs
if "%1"=="clean" goto clean
if "%1"=="help" goto help

REM Default: deploy
:deploy
echo [INFO] Pulling Docker images...
docker-compose pull
if %errorlevel% neq 0 (
    echo [ERROR] Failed to pull Docker images
    pause
    exit /b 1
)
echo [SUCCESS] Docker images pulled

echo [INFO] Building application images...
docker-compose build --no-cache
if %errorlevel% neq 0 (
    echo [ERROR] Failed to build application images
    pause
    exit /b 1
)
echo [SUCCESS] Application images built

echo [INFO] Starting services...
docker-compose up -d
if %errorlevel% neq 0 (
    echo [ERROR] Failed to start services
    pause
    exit /b 1
)
echo [SUCCESS] Services started

REM Wait for services to be healthy
echo [INFO] Waiting for services to be ready...
timeout /t 30 /nobreak >nul
echo [SUCCESS] Services should be ready

goto show_status

:stop
echo [INFO] Stopping services...
docker-compose down
echo [SUCCESS] Services stopped
goto end

:restart
echo [INFO] Restarting services...
docker-compose restart
echo [SUCCESS] Services restarted
goto show_status

:status
docker-compose ps
goto end

:logs
docker-compose logs -f
goto end

:clean
echo [WARNING] This will remove all containers, images, and volumes. Are you sure? (Y/N)
set /p response=
if /i "%response%"=="y" (
    echo [INFO] Cleaning up...
    docker-compose down -v --rmi all
    docker system prune -f
    echo [SUCCESS] Complete cleanup performed
) else (
    echo [INFO] Cleanup cancelled
)
goto end

:help
echo Usage: %0 [command]
echo.
echo Commands:
echo   deploy   - Full deployment (default^)
echo   start    - Start services
echo   stop     - Stop services
echo   restart  - Restart services
echo   status   - Show service status
echo   logs     - Show service logs
echo   clean    - Clean up all containers and images
echo   help     - Show this help message
goto end

:show_status
echo.
echo [SUCCESS] Deployment completed successfully!
echo.
echo [INFO] Access the application at:
echo   • HTTP:  http://localhost
echo   • HTTPS: https://localhost
echo   • API:   http://localhost/api
echo.
echo [INFO] Monitoring dashboards:
echo   • Prometheus: http://localhost:9090
echo   • Grafana:    http://localhost:3000 (admin/admin123^)
echo.
echo [WARNING] Please update the .env file with your actual configuration values!
echo.
echo Service Status:
docker-compose ps

:end
pause