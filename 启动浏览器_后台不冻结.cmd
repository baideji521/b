@echo off
chcp 65001 >nul
setlocal

rem ---------------------------------------------------------------------------
rem 用这个脚本开浏览器，AI_剪辑师 才能真正在后台干活。
rem
rem Windows 上 Chrome 有个「窗口被盖住就当它没显示」的优化（原生遮挡检测），
rem 一旦 Gemini 那个窗口被别的窗口盖住，页面就被冻结：不排版、不跑定时器，
rem 于是拖进去的文件发不出去、回答也读不出来。这里两个开关就是把它关掉。
rem
rem 注意：Chrome 必须先完全退出（托盘里也不能留后台进程），否则新进程会挂到
rem 老进程上，开关不生效。
rem ---------------------------------------------------------------------------

set "CHROME="
for %%P in (
  "%ProgramFiles%\Google\Chrome\Application\chrome.exe"
  "%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"
  "%LocalAppData%\Google\Chrome\Application\chrome.exe"
  "%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"
  "%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"
) do if not defined CHROME if exist %%P set "CHROME=%%~P"

if not defined CHROME (
  echo 没找到 Chrome / Edge，请手动把浏览器路径填进这个脚本。
  pause
  exit /b 1
)

tasklist /fi "imagename eq chrome.exe" | find /i "chrome.exe" >nul
if not errorlevel 1 (
  echo Chrome 现在是开着的，开关不会生效。
  echo 请先把 Chrome 完全退出（含托盘后台进程），再跑这个脚本。
  pause
  exit /b 1
)

echo 启动 %CHROME%
start "" "%CHROME%" --disable-features=CalculateNativeWinOcclusion --disable-backgrounding-occluded-windows --disable-renderer-backgrounding "https://gemini.google.com/app"
endlocal
