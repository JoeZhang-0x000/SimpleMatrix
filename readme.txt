# 矩阵分析大作业说明
## 项目结构
..
.
├── matrix.py # 源代码
├── readme.txt
└── test.ipynb # 包含大部分功能的案例以及说明

## 运行环境
1. python 基础环境
2. numpy                     >=1.26.0
3. jupyter运行环境
numpy 版本影响不大，主要是调用了其中的sqrt()开方函数以及isClose()函数判断是否接近零，低于1.26.0应该也不会有太大影响。
jupyter运行较为关键，若无jupyter运行环境可以访问该链接，这里是运行好的案例。

## 功能包含
- 详细见test.ipynb
- 基础矩阵表示，同时支持实矩阵和复矩阵，支持从numpy格式的矩阵；
- 矩阵单目运算，求逆，转置，求1模，2模，Forbenius模，无穷模，行列式；
- 矩阵双目运算，加减乘除都已经重构运算符号，可直接使用A+B,A-B,A@B,A*B等；
- 矩阵分解，LU分解，QR分解(schmidt, modified schmidt, givens, householder)，URV分解；
- 线性方程组求解，支持LU分解求解，QR分解求解，Cramer法则求解，并支持最小二乘解；



