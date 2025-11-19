# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

a = Analysis(['squid_angle_gui_enV1.1.py'],
             pathex=['.'],
             binaries=[],
             # collect_data_files('matplotlib') 会自动收集 matplotlib 运行所需的数据文件，
             # 这是 matplotlib 打包时常见的需求。
             datas=collect_data_files('matplotlib') + [('icon.ico', '.')],
             hiddenimports=['matplotlib.backends.backend_tkagg'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)

pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='squid_angle_gui',
          icon='icon.ico', # 添加图标文件
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=False, # **临时调试设置：True 会在运行时打开一个命令行窗口以显示错误信息**
          disable_windowed_traceback=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None )

coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='squid_angle_gui')
