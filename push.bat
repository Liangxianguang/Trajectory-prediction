@echo off
echo [1/3] Staging all changes...
git add .

echo [2/3] Committing changes...
git commit -m "V4_VS_v1_V2_V3_COMPARISON"

echo [3/3] Pushing to GitHub...
REM 如果你频繁遇到连接超时，可以尝试设置代理
REM git config --global http.proxy http://127.0.0.1:7890

git push origin main

if errorlevel 1 (
    echo.
    echo [错误] 推送失败，请检查网络连接或代理设置。
    pause
) else (
    echo [完成] 代码已成功推送到 GitHub。
)