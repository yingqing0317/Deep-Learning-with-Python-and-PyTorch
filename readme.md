# Anaconda使用

## Anaconda安装

b站up：我是土堆

## conda环境配置

* 创建环境```  conda create -n py380 python=3.8 ```
* 激活环境``` conda activate py380 ```
* 安装包``` pip install numpy -i http... ```
* 卸载包``` pip uninstall numpy ```
* 更新包``` pip install --upgrade numpy ```
* 安装文件中的很多包``` pip install -r (文件路径+文件名+文件后缀) -i http... ```
* 安装pytorch包``` conda install pytorch torchvision torchaudio cpuonly -c pytorch ```

## conda常用指令

* 查询安装列表``` pip list ```

* 查询安装地址``` where numpy ```

## Jupyter在特定盘打开

* 打开Anaconda Prompt
* 激活环境
* 输入``` d: ```（即自己想进入的文件夹）
* 输入``` jupyter notebook ```


## pycharm找到新添加的环境

file-new project setup-set for new project-python interpreter-add-conda environment-existing environment-定位到想添加的环境

## pycharm设置项目默认环境

file-settings-project:(该项目名)-project interpreter-选择环境

## GPU转CPU

 原因：系统无GPU

``` device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') ```

``` 代码其余地方出现.cuda()的地方改成.to(device) ```

## pycharm快捷键

* ctrl+q：查询函数定义
* 选定+ctrl+/：注释、取消注释
* ctrl+shift+r：查找&替换代码块
* ’‘’‘’‘（三对单引号）+enter：注释函数输入

## pycharm美观性

* 运算符前后、逗号后需要加空格