# -*- mode: python -*-

block_cipher = None


a = Analysis(['cnmApp.py'],
             pathex=['C:\\Users\\Black\\AppData\\Local\\Programs\\Python\\Python36\\python\\thesis\\pyqt5'],
             binaries=[],
             datas=[],
             hiddenimports=[],
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
          name='cnmApp',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=False , icon='touch.ico')
coll = COLLECT(exe,
               Tree('C:\\Users\\Black\\AppData\\Local\\Programs\\Python\\Python36\\python\\thesis\\pyqt5\\fig'),
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='cnmApp')
