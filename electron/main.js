const { app, BrowserWindow, Menu, dialog, shell, ipcMain, systemPreferences, clipboard } = require('electron');
const path = require('path');
const fs = require('fs');
const { promisify } = require('util');
const { exec: execCallback } = require('child_process');
const exec = promisify(execCallback);
const isDev = process.env.ELECTRON_IS_DEV === 'true' || require('electron-is-dev');

console.log('=== Electron Main Process Starting ===');
console.log('isDev:', isDev);
console.log('__dirname:', __dirname);

let mainWindow;
let translationWindow;
let backendUrl = null;

// InfiniSST服务器配置（连接到ngrok tunnel）
const INFINISST_SERVER = {
  protocol: 'https',
  host: 'amused-fleet-aardvark.ngrok-free.app',
  port: null // ngrok doesn't use port numbers in URL
};

console.log('InfiniSST Server:', INFINISST_SERVER);


// 创建主窗口
function createWindow() {
  console.log('Creating main window...');
  
  // 获取屏幕尺寸来计算居中位置
  const { screen } = require('electron');
  const primaryDisplay = screen.getPrimaryDisplay();
  const { width: screenWidth, height: screenHeight } = primaryDisplay.workAreaSize;
  
  // 设置窗口尺寸（比之前小一些）
  const windowWidth = 1200;
  const windowHeight = 680;
  
  // 计算位置 - 水平居中，垂直方向留出更多顶部空间
  const x = Math.round((screenWidth - windowWidth) / 2);
  const y = Math.round(screenHeight * 0.15); // 从屏幕顶部15%的位置开始，而不是居中
  
  mainWindow = new BrowserWindow({
    width: windowWidth,
    height: windowHeight,
    x: x,
    y: y,
    minWidth: 900,
    minHeight: 600,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      enableRemoteModule: false,
      preload: path.join(__dirname, 'preload.js'),
      webSecurity: false // 允许跨域请求到远程服务器
    },
    icon: path.join(__dirname, 'assets', 'icon.png'),
    titleBarStyle: process.platform === 'darwin' ? 'hiddenInset' : 'default',
    show: false // 先不显示，等加载完成后再显示
  });

  console.log('Main window created');

  // 设置应用菜单
  createMenu();

  // 窗口准备好后显示
  mainWindow.once('ready-to-show', () => {
    console.log('Main window ready to show');
    mainWindow.show();
    
    // 开发模式下打开开发者工具 (已禁用)
    // if (isDev) {
    //   console.log('Opening DevTools in development mode');
    //   mainWindow.webContents.openDevTools();
    // }
  });

  // 窗口关闭时的处理
  mainWindow.on('closed', () => {
    console.log('Main window closed');
    mainWindow = null;
  });

  // 监听主窗口移动，让翻译窗口跟随
  mainWindow.on('move', () => {
    if (translationWindow && !translationWindow.isDestroyed() && translationWindow.isVisible()) {
      const mainBounds = mainWindow.getBounds();
      const translationBounds = translationWindow.getBounds();
      
      // 重新计算翻译窗口位置：翻译窗口下边框距主窗口底部20px
      const x = mainBounds.x + Math.floor((mainBounds.width - translationBounds.width) / 2);
      const y = mainBounds.y + mainBounds.height - translationBounds.height - 20;
      
      // 确保窗口不会超出屏幕范围
      const { screen } = require('electron');
      const primaryDisplay = screen.getPrimaryDisplay();
      const { width: screenWidth, height: screenHeight } = primaryDisplay.workAreaSize;
      
      const finalX = Math.max(0, Math.min(x, screenWidth - translationBounds.width));
      const finalY = Math.max(0, Math.min(y, screenHeight - translationBounds.height));
      
      translationWindow.setPosition(finalX, finalY);
    }
  });

  // 监听主窗口大小改变，重新调整翻译窗口位置
  mainWindow.on('resize', () => {
    if (translationWindow && !translationWindow.isDestroyed() && translationWindow.isVisible()) {
      const mainBounds = mainWindow.getBounds();
      const translationBounds = translationWindow.getBounds();
      
      // 重新计算翻译窗口位置：翻译窗口下边框距主窗口底部20px
      const x = mainBounds.x + Math.floor((mainBounds.width - translationBounds.width) / 2);
      const y = mainBounds.y + mainBounds.height - translationBounds.height - 20;
      
      // 确保窗口不会超出屏幕范围
      const { screen } = require('electron');
      const primaryDisplay = screen.getPrimaryDisplay();
      const { width: screenWidth, height: screenHeight } = primaryDisplay.workAreaSize;
      
      const finalX = Math.max(0, Math.min(x, screenWidth - translationBounds.width));
      const finalY = Math.max(0, Math.min(y, screenHeight - translationBounds.height));
      
      translationWindow.setPosition(finalX, finalY);
    }
  });

  // 处理外部链接
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    console.log('Opening external URL:', url);
    shell.openExternal(url);
    return { action: 'deny' };
  });

  // 监听页面加载事件
  mainWindow.webContents.on('did-start-loading', () => {
    console.log('Page started loading');
  });

  mainWindow.webContents.on('did-finish-load', () => {
    console.log('Page finished loading');
    
    // 测试IPC连接
    setTimeout(() => {
      console.log('🧪 Testing IPC connection from main process...');
      mainWindow.webContents.executeJavaScript(`
        console.log('🧪 Executing JavaScript in renderer process...');
        if (window.electronAPI) {
          console.log('🧪 electronAPI is available');
          if (window.electronAPI.testIPC) {
            console.log('🧪 testIPC method found, calling it...');
            window.electronAPI.testIPC().then(result => {
              console.log('🧪 Test IPC result:', result);
            }).catch(error => {
              console.error('🧪 Test IPC error:', error);
            });
          } else {
            console.error('🧪 testIPC method not found in electronAPI');
            console.log('🧪 Available methods:', Object.keys(window.electronAPI || {}));
          }
        } else {
          console.error('🧪 electronAPI is not available');
        }
      `);
    }, 2000);
  });

  mainWindow.webContents.on('did-fail-load', (event, errorCode, errorDescription, validatedURL) => {
    console.error('Page failed to load:', {
      errorCode,
      errorDescription,
      validatedURL
    });
  });

  // 不在这里连接后端，等创建窗口后再连接
}

// 连接到后端服务 - 直接使用ngrok tunnel
async function connectToBackend() {
  console.log('Starting backend connection...');
  
  try {
    // 直接使用ngrok tunnel，无需配置对话框
    const url = INFINISST_SERVER.port 
      ? `${INFINISST_SERVER.protocol}://${INFINISST_SERVER.host}:${INFINISST_SERVER.port}`
      : `${INFINISST_SERVER.protocol}://${INFINISST_SERVER.host}`;
    console.log(`Connecting to InfiniSST server at: ${url}`);
    
    // 测试连接
    console.log('Testing backend connection...');
    const isConnected = await testBackendConnection(url);
    
    if (!isConnected) {
      console.error('Backend connection failed');
      // 如果主窗口存在，显示错误对话框，否则记录错误并返回URL让用户尝试
      if (mainWindow) {
        const retry = await dialog.showMessageBox(mainWindow, {
          type: 'error',
          title: 'Connection Failed',
          message: 'Failed to connect to the InfiniSST server',
          detail: `Could not connect to ${url}. Please check your internet connection and try again.`,
          buttons: ['Retry', 'Continue Anyway', 'Quit'],
          defaultId: 0
        });
        
        if (retry.response === 0) {
          return await connectToBackend();
        } else if (retry.response === 2) {
          app.quit();
          return null;
        }
        // Continue anyway (response === 1)
      } else {
        console.warn('Backend connection failed, but continuing with URL anyway');
      }
    }
    
    console.log('Backend connection completed, returning URL:', url);
    return url;
    
  } catch (error) {
    console.error('Error connecting to backend server:', error);
    if (mainWindow) {
      dialog.showErrorBox('Connection Error', `Failed to connect to InfiniSST server: ${error.message}`);
      app.quit();
    }
    return null;
  }
}



// 请求麦克风权限 (macOS)
async function requestMicrophonePermission() {
  if (process.platform === 'darwin') {
    try {
      console.log('Requesting microphone permission on macOS...');
      const microphonePermission = await systemPreferences.askForMediaAccess('microphone');
      console.log('Microphone permission granted:', microphonePermission);
      
      if (!microphonePermission) {
        console.warn('Microphone permission denied by user');
        const result = await dialog.showMessageBox(mainWindow, {
          type: 'warning',
          title: 'Microphone Permission Required',
          message: 'InfiniSST needs microphone access to provide real-time translation.',
          detail: 'Please grant microphone permission in System Preferences > Security & Privacy > Privacy > Microphone, then restart the application.',
          buttons: ['Open System Preferences', 'Continue Without Microphone', 'Quit'],
          defaultId: 0
        });
        
        if (result.response === 0) {
          // 打开系统偏好设置
          shell.openExternal('x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone');
        } else if (result.response === 2) {
          app.quit();
          return false;
        }
      }
      
      return microphonePermission;
    } catch (error) {
      console.error('Error requesting microphone permission:', error);
      return false;
    }
  } else {
    // 非macOS平台，假设权限已授予
    console.log('Non-macOS platform, assuming microphone permission is available');
    return true;
  }
}

// 检查完整的系统音频配置
async function checkSystemAudioConfiguration() {
  console.log('🔍 Checking complete system audio configuration...');
  
  if (process.platform !== 'darwin') {
    console.log('System audio configuration check is only available on macOS');
    return { configured: false, reason: 'macOS only', hasBlackHole: false };
  }

  try {
    const { execSync } = require('child_process');
    
    // Step 1: 检查 BlackHole 是否安装
    console.log('📋 Step 1: Checking if BlackHole is installed...');
    const audioData = await getAudioDevices();
    const hasBlackHole = audioData.devices.some(device => 
      device.name.toLowerCase().includes('blackhole')
    );
    
    console.log('BlackHole installed:', hasBlackHole);
    
    if (!hasBlackHole) {
      return { 
        configured: false, 
        reason: 'BlackHole not installed', 
        hasBlackHole: false,
        details: 'BlackHole virtual audio device is not installed'
      };
    }
    
    // Step 2: 检查是否存在包含 BlackHole 的 Multi-Output Device
    console.log('📋 Step 2: Checking for Multi-Output Device with BlackHole...');
    const multiOutputDevices = audioData.devices.filter(device => 
      device.name.toLowerCase().includes('multi-output') || 
      device.name.toLowerCase().includes('multi output') ||
      device.name.toLowerCase().includes('aggregate')
    );
    
    console.log('Found Multi-Output devices:', multiOutputDevices.map(d => d.name));
    
    // Step 3: 检查当前系统音频输出设备
    console.log('📋 Step 3: Checking current system audio output...');
    let currentOutputDevice = '';
    try {
      // 使用 SwitchAudioSource 工具检查当前输出设备（更可靠）
      currentOutputDevice = execSync('SwitchAudioSource -c', {
        encoding: 'utf8'
      }).trim();
      console.log('Current output device:', currentOutputDevice);
    } catch (error) {
      console.warn('⚠️ Failed to detect current output device using SwitchAudioSource:', error.message);
      currentOutputDevice = '';
    }
    
    // Step 4: 综合判断配置状态
    const hasValidMultiOutput = multiOutputDevices.length > 0;

    if (!hasValidMultiOutput) {
      return {
        configured: false,
        reason: 'No Multi-Output Device found',
        hasBlackHole: true,
        details: 'BlackHole is installed but no Multi-Output Device has been created. You need to create a Multi-Output Device in Audio MIDI Setup.',
        instructions: [
          '1. Open Audio MIDI Setup (Applications > Utilities)',
          '2. Click the "+" button and select "Create Multi-Output Device"',
          '3. Check both your speakers and BlackHole in the device list',
          '4. Set the Multi-Output Device as your system output in System Preferences > Sound'
        ]
      };
    }

    // 检查 Multi-Output Device 是否被设置为当前输出
    const isMultiOutputActive = multiOutputDevices.some(device =>
      currentOutputDevice.toLowerCase().includes(device.name.toLowerCase())
    );

    if (!isMultiOutputActive) {
      return {
        configured: false,
        reason: 'Multi-Output not active',
        hasBlackHole: true,
        details: `Multi-Output Device exists but is not the current output device (Current: ${currentOutputDevice})`,
        instructions: [
          '1. Open Audio MIDI Setup (Applications > Utilities)',
          '2. Select the Multi-Output Device that includes BlackHole',
          '3. Right-click and choose "Use this device for sound output"'
        ]
      };
    }

    console.log('📋 Step 4: Verifying Multi-Output Device is current output...');

    return {
      configured: true,
      reason: 'Properly configured',
      hasBlackHole: true,
      multiOutputDevices: multiOutputDevices.map(d => d.name),
      details: 'System audio configuration appears to be correct for capture'
    };
    
  } catch (error) {
    console.error('Error checking system audio configuration:', error);
    return {
      configured: false,
      reason: 'Check failed',
      hasBlackHole: false,
      details: `Configuration check failed: ${error.message}`,
      error: error.message
    };
  }
}

// 获取音频设备信息的辅助函数
async function getAudioDevices() {
  const { execSync } = require('child_process');
  
  try {
    const output = execSync('system_profiler SPAudioDataType -json', { 
      encoding: 'utf8',
      timeout: 10000
    });
    
    const audioData = JSON.parse(output);
    const devices = [];
    
    // 遍历所有音频设备
    for (const deviceGroup of audioData.SPAudioDataType || []) {
      for (const device of deviceGroup._items || []) {
        if (device._name) {
          devices.push({
            name: device._name,
            type: deviceGroup._name || 'Unknown',
            raw: device
          });
        }
      }
    }
    
    return { devices, raw: audioData };
  } catch (error) {
    console.error('Error getting audio devices:', error);
    return { devices: [], raw: null };
  }
}



// 运行BlackHole设置脚本
async function installBlackHole() {
  if (process.platform !== 'darwin') {
    console.log('BlackHole setup is only supported on macOS');
    return { success: false, message: 'BlackHole is only available on macOS' };
  }

  try {
    // 使用打包在应用中的 setup_blackhole.command 脚本
    let scriptPath;
    
    if (isDev) {
      // 开发环境：从项目目录加载
      scriptPath = path.join(__dirname, 'scripts', 'setup_blackhole.command');
    } else {
      // 生产环境：从extraResources加载
      scriptPath = path.join(process.resourcesPath, 'scripts', 'setup_blackhole.command');
      
      // 如果extraResources路径不存在，尝试备用路径
      if (!fs.existsSync(scriptPath)) {
        scriptPath = path.join(process.resourcesPath, 'app', 'electron', 'scripts', 'setup_blackhole.command');
        
        // 如果还是不存在，尝试最后的备用路径
        if (!fs.existsSync(scriptPath)) {
          scriptPath = path.join(__dirname, 'scripts', 'setup_blackhole.command');
        }
      }
    }
    
    console.log('Running BlackHole setup script:', scriptPath);
    console.log('Script exists:', fs.existsSync(scriptPath));
    
    // 检查脚本是否存在
    if (!fs.existsSync(scriptPath)) {
      console.error('Setup script not found at any of the expected paths');
      const result = await dialog.showMessageBox(mainWindow, {
        type: 'error',
        title: 'Setup Script Not Found',
        message: 'The BlackHole setup script is missing from the application.',
        detail: 'Please reinstall the application or contact support.',
        buttons: ['OK']
      });
      return { success: false, message: 'Setup script not found' };
    }
    
    try {
      // 检查当前脚本权限
      const stats = fs.statSync(scriptPath);
      console.log('Script permissions before chmod:', stats.mode.toString(8));
      
      // 尝试设置执行权限，但不让它阻止脚本运行
      try {
        await exec(`chmod +x "${scriptPath}"`);
        console.log('Successfully set execute permission for script');
        
        // 再次检查权限
        const newStats = fs.statSync(scriptPath);
        console.log('Script permissions after chmod:', newStats.mode.toString(8));
      } catch (chmodError) {
        console.warn('chmod failed, but continuing with script execution:', chmodError.message);
      }
      
      // 使用AppleScript在Terminal中运行脚本
      const scriptDir = path.dirname(scriptPath);
      const scriptName = path.basename(scriptPath);
      const runCommand = `cd "${scriptDir}" && bash "${scriptName}"`;
      console.log('Running script with command:', runCommand);
      
      // 使用AppleScript在新的Terminal窗口中运行脚本
      const appleScript = `
        tell application "Terminal"
          activate
          do script "cd \\"${scriptDir}\\" && echo \\"Starting BlackHole setup...\\" && bash \\"${scriptName}\\""
        end tell
      `;
      
      await exec(`osascript -e '${appleScript}'`);
      
      return { 
        success: true, 
        message: 'BlackHole setup script started in Terminal. Please follow the instructions in the Terminal window.' 
      };
    } catch (execError) {
      console.error('Error running script in Terminal:', execError);
      
      // 如果Terminal运行失败，提供手动方案
      const result = await dialog.showMessageBox(mainWindow, {
        type: 'warning',
        title: 'Unable to Start Terminal',
        message: 'Could not automatically run the setup script in Terminal.',
        detail: `Please manually run the script:\n\n1. Open Terminal\n2. Navigate to: ${path.dirname(scriptPath)}\n3. Run: bash setup_blackhole.command\n\nOr copy the command below:`,
        buttons: ['Open Terminal', 'Copy Full Command', 'Show Script Location', 'Cancel'],
        defaultId: 0
      });
      
      try {
        if (result.response === 0) {
          // 使用AppleScript打开Terminal到脚本目录
          const scriptDir = path.dirname(scriptPath);
          const openTerminalScript = `
            tell application "Terminal"
              activate
              do script "cd \\"${scriptDir}\\" && echo \\"Navigate to BlackHole setup directory. Run: bash setup_blackhole.command\\""
            end tell
          `;
          await exec(`osascript -e '${openTerminalScript}'`);
        } else if (result.response === 1) {
          // 复制完整命令到剪贴板
          const fullCommand = `cd "${path.dirname(scriptPath)}" && bash setup_blackhole.command`;
          clipboard.writeText(fullCommand);
          
          // 显示已复制提示
          dialog.showMessageBox(mainWindow, {
            type: 'info',
            title: 'Command Copied',
            message: 'The command has been copied to your clipboard.',
            detail: 'Open Terminal and paste (Cmd+V) to run the script.',
            buttons: ['OK']
          });
        } else if (result.response === 2) {
          // 在Finder中显示脚本位置
          await exec(`open -R "${scriptPath}"`);
        }
      } catch (fallbackError) {
        console.error('Error in fallback operations:', fallbackError);
      }
      
      return { 
        success: false, 
        message: 'Please run the setup script manually in Terminal.' 
      };
    }
  } catch (error) {
    console.error('Error running BlackHole setup:', error);
    return { 
      success: false, 
      message: `Failed to run BlackHole setup: ${error.message}` 
    };
  }
}

// 测试后端连接
function testBackendConnection(url, maxAttempts = 3) {
  return new Promise((resolve) => {
    let attempts = 0;
    
    const testConnection = () => {
      const http = require('http');
      const https = require('https');
      const urlObj = new URL(url);
      const client = urlObj.protocol === 'https:' ? https : http;
      
      const req = client.get(url, (res) => {
        resolve(true);
      });
      
      req.on('error', () => {
        attempts++;
        if (attempts >= maxAttempts) {
          resolve(false);
        } else {
          setTimeout(testConnection, 1000);
        }
      });
      
      req.setTimeout(5000, () => {
        req.destroy();
        attempts++;
        if (attempts >= maxAttempts) {
          resolve(false);
        } else {
          setTimeout(testConnection, 1000);
        }
      });
    };
    
    testConnection();
  });
}

// 创建应用菜单
function createMenu() {
  const template = [
    {
      label: 'File',
      submenu: [
        {
          label: 'Open Audio/Video File...',
          accelerator: 'CmdOrCtrl+O',
          click: () => {
            if (mainWindow) {
              mainWindow.webContents.send('menu-open-file');
            }
          }
        },
        { type: 'separator' },
        {
          label: 'Debug Microphone',
          click: () => {
            if (mainWindow && backendUrl) {
              mainWindow.loadURL(`${backendUrl}/debug-microphone.html`);
            }
          }
        },
        {
          label: 'Back to Main App',
          click: () => {
            if (mainWindow && backendUrl) {
              mainWindow.loadURL(backendUrl);
            }
          }
        },
        { type: 'separator' },
        {
          label: 'Quit',
          accelerator: process.platform === 'darwin' ? 'Cmd+Q' : 'Ctrl+Q',
          click: () => {
            app.quit();
          }
        }
      ]
    },
    {
      label: 'Edit',
      submenu: [
        { role: 'undo' },
        { role: 'redo' },
        { type: 'separator' },
        { role: 'cut' },
        { role: 'copy' },
        { role: 'paste' }
      ]
    },
    {
      label: 'View',
      submenu: [
        { role: 'reload' },
        { role: 'forceReload' },
        { role: 'toggleDevTools' },
        { type: 'separator' },
        { role: 'resetZoom' },
        { role: 'zoomIn' },
        { role: 'zoomOut' },
        { type: 'separator' },
        { role: 'togglefullscreen' }
      ]
    },
    {
      label: 'Translation',
      submenu: [
        {
          label: 'Show Translation Window',
          accelerator: 'CmdOrCtrl+T',
          click: () => {
            createTranslationWindow();
          }
        },
        {
          label: 'Hide Translation Window',
          accelerator: 'CmdOrCtrl+Shift+T',
          click: () => {
            if (translationWindow) {
              translationWindow.hide();
            }
          }
        },
        {
          label: 'Close Translation Window',
          click: () => {
            if (translationWindow) {
              translationWindow.close();
              translationWindow = null;
            }
          }
        }
      ]
    },
    {
      label: 'Window',
      submenu: [
        { role: 'minimize' },
        { role: 'close' }
      ]
    },
    {
      label: 'Help',
      submenu: [
        {
          label: 'About InfiniSST',
          click: () => {
            dialog.showMessageBox(mainWindow, {
              type: 'info',
              title: 'About InfiniSST',
              message: 'InfiniSST Translation Desktop',
              detail: 'Simultaneous end-to-end speech translation powered by LLM\nVersion 1.0.0'
            });
          }
        },
        {
          label: 'Learn More',
          click: () => {
            shell.openExternal('https://github.com/your-repo/InfiniSST');
          }
        }
      ]
    }
  ];

  // macOS 特殊处理
  if (process.platform === 'darwin') {
    template.unshift({
      label: app.getName(),
      submenu: [
        { role: 'about' },
        { type: 'separator' },
        { role: 'services' },
        { type: 'separator' },
        { role: 'hide' },
        { role: 'hideOthers' },
        { role: 'unhide' },
        { type: 'separator' },
        { role: 'quit' }
      ]
    });

    // Window menu
    template[4].submenu = [
      { role: 'close' },
      { role: 'minimize' },
      { role: 'zoom' },
      { type: 'separator' },
      { role: 'front' }
    ];
  }

  const menu = Menu.buildFromTemplate(template);
  Menu.setApplicationMenu(menu);
}

// 应用准备就绪
app.whenReady().then(async () => {
  console.log('🚀 App is ready, setting up backend and permissions...');
  
  // 检查IPC处理器是否已注册
  console.log('🔧 IPC Handlers registered:');
  console.log('- get-backend-url: ✅');
  console.log('- test-ipc: ✅');
  console.log('- check-microphone-permission: ✅');
  console.log('- request-microphone-permission: ✅');
  console.log('- check-system-audio-config: ✅');
  console.log('- test-system-profiler: ✅');
  console.log('- install-blackhole: ✅');
  
  let canProceed = false;
  
  try {
    // 首先创建主窗口
    console.log('Creating main window...');
    createWindow();
    
    // 在创建窗口之后请求麦克风权限
    const micPermission = await requestMicrophonePermission();
    console.log('Microphone permission result:', micPermission);
    
    // 连接到后端并加载URL
    console.log('Connecting to backend...');
    backendUrl = await connectToBackend();
    console.log('Backend connected, URL:', backendUrl);
    
    if (backendUrl && mainWindow) {
      console.log('Loading main page...');
      mainWindow.loadURL(backendUrl);
    } else {
      console.error('Failed to connect to backend or main window not available');
    }
    
    canProceed = true;
  } catch (error) {
    console.error('Error during setup:', error);
    // 发生错误时确保窗口仍然存在
    if (!mainWindow) {
      createWindow();
    }
  }

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0 && canProceed) {
      createWindow();
    }
  });
});

// 所有窗口关闭时
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

// IPC 处理
ipcMain.handle('get-backend-url', () => {
  return backendUrl;
});

// 添加一个简单的测试IPC处理器
ipcMain.handle('test-ipc', () => {
  console.log('🧪 Test IPC handler called successfully!');
  return { success: true, message: 'IPC is working!' };
});

// 检查麦克风权限状态
ipcMain.handle('check-microphone-permission', async () => {
  if (process.platform === 'darwin') {
    try {
      const status = systemPreferences.getMediaAccessStatus('microphone');
      console.log('Current microphone permission status:', status);
      return status;
    } catch (error) {
      console.error('Error checking microphone permission:', error);
      return 'unknown';
    }
  } else {
    return 'granted'; // 非macOS平台假设已授权
  }
});

// 请求麦克风权限
ipcMain.handle('request-microphone-permission', async () => {
  return await requestMicrophonePermission();
});



// 检查完整的系统音频配置（详细检查）
ipcMain.handle('check-system-audio-config', async () => {
  console.log('🔍 IPC: check-system-audio-config handler called');
  try {
    const result = await checkSystemAudioConfiguration();
    console.log('✅ IPC: checkSystemAudioConfiguration completed, result:', result);
    
    // 如果检查失败，自动提供安装/配置选项
    if (!result.configured) {
      console.log('💡 System audio not configured properly, offering installation/setup...');
      
      // 询问用户是否要运行安装/配置脚本
      if (mainWindow) {
        const userResponse = await dialog.showMessageBox(mainWindow, {
          type: 'question',
          title: 'System Audio Configuration Required',
          message: 'System audio capture requires proper configuration.',
          detail: `Issue: ${result.reason}\n${result.details || ''}\n\nWould you like to run the setup script? This will guide you through installation and configuration.`,
          buttons: ['Run Setup Script', 'Cancel'],
          defaultId: 0,
          cancelId: 1
        });
        
        if (userResponse.response === 0) {
          console.log('🚀 User chose to run setup script, starting...');
          try {
            const installResult = await installBlackHole();
            console.log('📦 Setup script result:', installResult);
            
            // 返回带有安装结果的配置检查结果
            return {
              ...result,
              setupAttempted: true,
              setupResult: installResult
            };
          } catch (setupError) {
            console.error('❌ Setup script failed:', setupError);
            return {
              ...result,
              setupAttempted: true,
              setupResult: { success: false, message: setupError.message }
            };
          }
        } else {
          console.log('❌ User cancelled the setup');
          return { ...result, setupCancelled: true };
        }
      }
    }
    
    return result;
  } catch (error) {
    console.error('❌ IPC: checkSystemAudioConfiguration error:', error);
    return { configured: false, reason: 'IPC error', hasBlackHole: false, error: error.message };
  }
});

// 添加一个测试IPC处理器来手动运行system_profiler
ipcMain.handle('test-system-profiler', async () => {
  console.log('🧪 Test: system_profiler command');
  try {
    const { execSync } = require('child_process');
    const output = execSync('system_profiler SPAudioDataType -json', { 
      encoding: 'utf8',
      timeout: 10000
    });
    console.log('Raw system_profiler output:', output.substring(0, 2000) + '...');
    const audioData = JSON.parse(output);
    console.log('Parsed audio data:', JSON.stringify(audioData, null, 2));
    return audioData;
  } catch (error) {
    console.error('Test system_profiler error:', error);
    throw error;
  }
});

// 安装BlackHole
ipcMain.handle('install-blackhole', async () => {
  return await installBlackHole();
});

ipcMain.handle('show-open-dialog', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile'],
    filters: [
      { name: 'Audio Files', extensions: ['mp3', 'wav', 'flac', 'm4a', 'aac'] },
      { name: 'Video Files', extensions: ['mp4', 'avi', 'mov', 'mkv', 'webm'] },
      { name: 'All Files', extensions: ['*'] }
    ]
  });
  return result;
});

// 创建翻译窗口
function createTranslationWindow() {
  if (translationWindow) {
    // 如果窗口存在但被隐藏，显示它
    if (!translationWindow.isVisible()) {
      translationWindow.show();
    }
    translationWindow.focus();
    return;
  }

  translationWindow = new BrowserWindow({
    width: 600,
    height: 120, // 缩短默认高度，约两行内容
    minWidth: 300,
    minHeight: 120, // 减小最小高度
    maxWidth: 1200,
    maxHeight: 800,
    alwaysOnTop: true,
    skipTaskbar: true,
    resizable: true,
    frame: false, // 完全无边框
    transparent: true, // 透明背景
    hasShadow: true, // 保留窗口阴影以便更好地识别窗口边界
    autoHideMenuBar: true, // 自动隐藏菜单栏
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js'),
      backgroundThrottling: false // 防止后台时停止渲染
    },
    title: 'InfiniSST Translation',
    show: false,
    focusable: true,
    fullscreenable: false,
    maximizable: false,
    movable: true
  });
  
  // 隐藏菜单栏
  translationWindow.setMenuBarVisibility(false);

  // 加载翻译窗口页面
  translationWindow.loadFile(path.join(__dirname, 'translation-window.html'));

  // 窗口准备好后显示
  translationWindow.once('ready-to-show', () => {
    translationWindow.show();
    
    // 设置窗口位置相对于主窗口底部20px
    if (mainWindow && !mainWindow.isDestroyed()) {
      const mainBounds = mainWindow.getBounds();
      const translationBounds = translationWindow.getBounds();
      
      // 水平居中对齐主窗口，垂直位置让翻译窗口下边框距主窗口底部20px
      const x = mainBounds.x + Math.floor((mainBounds.width - translationBounds.width) / 2);
      const y = mainBounds.y + mainBounds.height - translationBounds.height - 20;
      
      // 确保窗口不会超出屏幕范围
      const { screen } = require('electron');
      const primaryDisplay = screen.getPrimaryDisplay();
      const { width: screenWidth, height: screenHeight } = primaryDisplay.workAreaSize;
      
      const finalX = Math.max(0, Math.min(x, screenWidth - translationBounds.width));
      const finalY = Math.max(0, Math.min(y, screenHeight - translationBounds.height));
      
      translationWindow.setPosition(finalX, finalY);
    } else {
      // 备用方案：如果主窗口不可用，使用屏幕中央
      const { screen } = require('electron');
      const primaryDisplay = screen.getPrimaryDisplay();
      const { width, height } = primaryDisplay.workAreaSize;
      const windowBounds = translationWindow.getBounds();
      
      const x = Math.floor((width - windowBounds.width) / 2);
      const y = Math.floor(height * 0.70 - windowBounds.height / 2);
      
      translationWindow.setPosition(x, y);
    }
    
    // 窗口显示后发送初始状态
    setTimeout(() => {
      console.log('Sending initial status to translation window');
      translationWindow.webContents.send('status-update', {
        text: 'Ready - Load model',
        type: 'ready'
      });
      
      // 发送初始样式设置
      translationWindow.webContents.send('translation-style-update', {
        fontSize: 14,
        backgroundOpacity: 95,
        textColor: '#ffffff'
      });
    }, 200);
  });

  // 窗口关闭时的处理
  translationWindow.on('closed', () => {
    translationWindow = null;
  });
  
  // 防止窗口失去焦点时隐藏
  translationWindow.on('blur', () => {
    // 不自动隐藏，保持显示状态
  });
}

// IPC 处理器
ipcMain.handle('show-translation-window', () => {
  if (translationWindow) {
    // 如果窗口存在但被隐藏，显示它
    if (!translationWindow.isVisible()) {
      translationWindow.show();
    }
    translationWindow.focus();
  } else {
    // 如果窗口不存在，创建新窗口
    createTranslationWindow();
  }
});

ipcMain.handle('hide-translation-window', () => {
  if (translationWindow) {
    translationWindow.hide();
  }
});

ipcMain.handle('close-translation-window', () => {
  if (translationWindow) {
    translationWindow.close();
    translationWindow = null;
  }
});

ipcMain.handle('minimize-translation-window', () => {
  if (translationWindow) {
    translationWindow.minimize();
  }
});

ipcMain.handle('update-translation', (event, translationData) => {
  console.log('Main process received translation update:', translationData?.text?.substring(0, 50) + '...');
  if (translationWindow && translationWindow.webContents) {
    // 确保窗口已完全加载
    if (translationWindow.webContents.isLoading()) {
      console.log('Translation window still loading, waiting...');
      translationWindow.webContents.once('did-finish-load', () => {
        console.log('Translation window loaded, sending translation update');
        translationWindow.webContents.send('translation-update', translationData);
      });
    } else {
      console.log('Sending translation update to translation window');
      translationWindow.webContents.send('translation-update', translationData);
    }
  } else {
    console.log('Translation window not available for translation update');
  }
});

ipcMain.handle('update-translation-status', (event, statusData) => {
  console.log('Main process received status update:', statusData);
  if (translationWindow && translationWindow.webContents) {
    // 确保窗口已完全加载
    if (translationWindow.webContents.isLoading()) {
      console.log('Translation window still loading, waiting...');
      translationWindow.webContents.once('did-finish-load', () => {
        console.log('Translation window loaded, sending status update');
        translationWindow.webContents.send('status-update', statusData);
      });
    } else {
      console.log('Sending status update to translation window');
      translationWindow.webContents.send('status-update', statusData);
    }
  } else {
    console.log('Translation window not available for status update');
  }
});

ipcMain.handle('update-translation-style', (event, styleData) => {
  console.log('Main process received style update:', styleData);
  if (translationWindow && translationWindow.webContents) {
    // 确保窗口已完全加载
    if (translationWindow.webContents.isLoading()) {
      console.log('Translation window still loading, waiting...');
      translationWindow.webContents.once('did-finish-load', () => {
        console.log('Translation window loaded, sending style update');
        translationWindow.webContents.send('translation-style-update', styleData);
      });
    } else {
      console.log('Sending style update to translation window');
      translationWindow.webContents.send('translation-style-update', styleData);
    }
  } else {
    console.log('Translation window not available for style update');
  }
});

ipcMain.handle('reset-translation-from-window', (event) => {
  // 向主窗口发送重置翻译的请求
  if (mainWindow && mainWindow.webContents) {
    mainWindow.webContents.executeJavaScript(`
      if (typeof resetTranslationFromElectron === 'function') {
        resetTranslationFromElectron();
      }
    `);
  }
  
  // 向翻译窗口发送重置确认
  if (translationWindow && translationWindow.webContents) {
    translationWindow.webContents.send('reset-translation');
  }
});

// 设置翻译窗口大小
ipcMain.handle('set-translation-window-size', (event, width, height) => {
  console.log(`Setting translation window size to: ${width}x${height}`);
  if (translationWindow) {
    translationWindow.setSize(width, height);
    
    // 重新定位窗口相对于主窗口底部20px
    if (mainWindow && !mainWindow.isDestroyed()) {
      const mainBounds = mainWindow.getBounds();
      
      // 水平居中对齐主窗口，垂直位置让翻译窗口下边框距主窗口底部20px
      const x = mainBounds.x + Math.floor((mainBounds.width - width) / 2);
      const y = mainBounds.y + mainBounds.height - height - 20;
      
      // 确保窗口不会超出屏幕范围
      const { screen } = require('electron');
      const primaryDisplay = screen.getPrimaryDisplay();
      const { width: screenWidth, height: screenHeight } = primaryDisplay.workAreaSize;
      
      const finalX = Math.max(0, Math.min(x, screenWidth - width));
      const finalY = Math.max(0, Math.min(y, screenHeight - height));
      
      translationWindow.setPosition(finalX, finalY);
    } else {
      // 备用方案：如果主窗口不可用，使用屏幕中央
      const { screen } = require('electron');
      const primaryDisplay = screen.getPrimaryDisplay();
      const { width: screenWidth, height: screenHeight } = primaryDisplay.workAreaSize;
      
      const x = Math.floor((screenWidth - width) / 2);
      const y = Math.floor(screenHeight * 0.70 - height / 2);
      
      translationWindow.setPosition(x, y);
    }
    
    console.log(`Translation window resized and repositioned relative to main window`);
  } else {
    console.warn('Translation window not available for resizing');
  }
});

// 获取翻译窗口边界
ipcMain.handle('get-translation-window-bounds', (event) => {
  if (translationWindow) {
    return translationWindow.getBounds();
  }
  return null;
});

// 设置翻译窗口边界（位置和大小）
ipcMain.handle('set-translation-window-bounds', (event, bounds) => {
  console.log(`Setting translation window bounds to:`, bounds);
  if (translationWindow && bounds) {
    // 确保尺寸不小于最小值
    const width = Math.max(300, bounds.width);
    const height = Math.max(120, bounds.height);
    
    // 确保位置在屏幕范围内
    const { screen } = require('electron');
    const primaryDisplay = screen.getPrimaryDisplay();
    const { width: screenWidth, height: screenHeight } = primaryDisplay.workAreaSize;
    
    const x = Math.max(0, Math.min(bounds.x, screenWidth - width));
    const y = Math.max(0, Math.min(bounds.y, screenHeight - height));
    
    translationWindow.setBounds({
      x: Math.round(x),
      y: Math.round(y),
      width: Math.round(width),
      height: Math.round(height)
    });
    
    console.log(`Translation window bounds updated to: ${x}, ${y}, ${width}x${height}`);
  } else {
    console.warn('Translation window not available for setting bounds');
  }
});

// 处理未捕获的异常
process.on('uncaughtException', (error) => {
  console.error('Uncaught Exception:', error);
  dialog.showErrorBox('Unexpected Error', error.message);
});

process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
}); 