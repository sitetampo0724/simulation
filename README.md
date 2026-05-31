:smile:
for competition use


# infant-guide：
> 较为简略的介绍文件结构,请务必阅读


`req:`


首先我们需要一个数据库对吧，noisedatageneration是用于生成带噪声的数据集的

于是聪明的你就会想到我们需要jpg和csv两种文件格式保存相关数据（均在noisedatagdataset下）

然后我们需要各种各样的模型，请在differentAImodel下新建文件夹训练你的ai(请仔细参考我的gradient模型的格式,不然到时候要调用predict又会造成负担!!)，
如果搞串文件夹我要给你大逼都（不要造成我的merge负担）

最后我们需要一个评价系统和一个预测的程序，放在prediction&visualization下

什么？你说ailanguagemodel是啥，这个是交大api调用，以备不时之需。






# details:

noise的dataset中的参数均可以调整，jpg生成了1000张分别按照序号对应不同的信噪比，
有两个csv文件均按照行来记录相关数据，signal记录了采样频率下的横纵坐标，statics记录了所有的参数和对应的信噪比
！！！我的建议是：不要轻易去重启我的datageneration代码，不要造成我的merge负担
