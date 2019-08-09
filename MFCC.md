# MFCC

https://blog.csdn.net/richard2357/article/details/17147249

1.做FFT，目的：获得语谱图
2.做Mel频域变换，目的：复合人耳对频率的感知
3.取对数，log，目的：将基频*声道转换为，基频+声道
4.做DCT或逆FFT，目的：将基频和声道分离，高频表示基音信息可以去掉，低频表示声道特征。

FFT的第一个维度代表什么？
MFCC的第一个维度代表什么？
