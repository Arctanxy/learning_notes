如果在windows系统下输入
```python
>>> import torchvision
```
出现如下报错的话：
```python
----> 2 from PIL import Image
      3 import os
      4 import os.path
      5 import six

D:\ProgramData\Anaconda3\lib\site-packages\PIL\Image.py in <module>()
     56     # Also note that Image.core is not a publicly documented interface,
     57     # and should be considered private and subject to change.
---> 58     from . import _imaging as core
     59     if PILLOW_VERSION != getattr(core, 'PILLOW_VERSION', None):
     60         raise ImportError("The _imaging extension was built for another
"

ImportError: DLL load failed: 找不到指定的模块。
```
可以尝试卸载当前版本的Pillow，重新安装Pillow4.0.0

```python
> pip uninstall Pillow
> pip install Pillow == 4.0.0
```

安装完成之后即可正常使用。