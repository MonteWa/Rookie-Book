
## 绘图篇

####plt中把横纵轴的坐标换成文字(plt.xticks)

    plt.figure()
    xlabels = ['A','B','C'] # 给定想显示在轴上的文字
    x = [1,2,3] # 给定字符串个数，一一对应
    plt.xticks(x,xlabels) # 用xlabels中的字符串替代x中的数字


####plt中显示图例（legend）

    plt.(label='xxx') # 先给所画的线或者别的什么设置label
    plt.legend() # 然后调用legend()

## 模型篇

#### keras建模的流程（顺序模型）
    model = Sequential() # 声明使用顺序模型
    model.add() # 给模型添加层，给定输出节点个数，初始化方式，激活函数等参数
    ...
    model.compile() # 给定优化器，lossfunction等参数
    model.fit() # 开始训练，给定训练周期数，batch_size，validation_data，callbacks等参数
    model.save()

#### 在keras中使用多GPU训练
    # 在开头导入库：
    from keras.utils import multi_gpu_model
    model = ... # 建立完模型后
    model_parallel = multi_gpu_model(model, gpu=n) # gpu数为n
    # 然后使用model_parallel完成剩下的操作

#### 用tensorboard记录keras训练过程
    # 先设置回调函数（callback），然后再训练时（.fit()）调用设置的callback
    tbCallBack = Tensorboard(logdir = './logs', #给定保存训练过程信息的地址
						 histogram_freq = 0,
						 batch_size = 32,
						 write_graph = True,
			       write_grads = True,
						 write_images = True,
						 embeddings_freq = 0,
						 embeddings_layer_names = None,
						 embeddings_metadata = None,
 						)
    #在训练时调用callback
    postFilterModel.fit(input_x, y_,
                   epochs = 2,
                   batch_size=32,
                   validation_data = (val_x, val_y),
                   verbose = 1, # 选择训练时的反馈形式，1为进度条形式
                   callbacks=[tbCallBack])

#### 用tensorboard展示log内容

    #打开conda prompt，在logs的目录里直接输入
    tensorboard --logdir = .\logs
    # 当tensorboard命令找不到时
    # 先找到tensorboard的具体地址：path
    pip show tensorboard
    # 用如下命令使用tensorboard：
    python path\tensorboard\main.py --logdir=.\logs
    # 在chrome中输入生成的网址

#### keras中对tensor进行操作
    # 使用lamda函数包装操作，如：
    input = ..
    h1 = ..
    def mutply(in): # 定义lamda函数
	     return in * input
    output = lambda(mutply)(h1) # 第一个括号是函数名，第二个是输入值
    model = keras.models.Model(input = input, output = output) # 这样model就能接受output

#### 用keras保存训练过程和训练结果
    # 设置callback
    save_path = 'output/epochs_{epoch:02d}_valLoss_{val_loss:.2f}'+'.h5' # savepath的建议格式,每次保存的还有val_loss的信息
    model_save = keras.callbacks.ModelCheckpoint(save_path,# 保存路径
											 monitor='loss',
											 save_best_only=False,
											 mode='min', # 求最小化模式
											 save_weights_only = True,
											 period = 1) # 保存周期
    # 在fit时调用callbacks
    hist = model.fit(...,
				 callbacks = [model_save],
				 ...)
    # 训练过程中的历史被保存在了hist中，从hist中取出loss历史，备用作图
    train_loss = np.array(hist.history['loss'])
    train_xxx = np.array(hist.history['xxx'])
    train_val = {} # 用一个dict来保存所有数据
    train_val['train_loss'] = train_loss
    train_val['train_xxx'] = train_xxx
    # 将数据统统保存下来
    sio.savemat('save_path',train_val) # 以mat的形式被保存

#### 用keras当loss不在下降时提前终止训练过程
    # 调用keras.callbacks.EarlyStopping()
    stop_str = keras.callbacks.EarlyStopping(monitor='val', # 关注变量
										 patience = x, # 等待次数
										 verbose = 1,
										 mode = 'min')
    # 在fit()中调用callbacks
    hist = model.fit(...,
				 callbacks=[stop_str],
				 ...)

#### 用keras当loss不在下降时减少学习率
    # 一般使用SGD时使用这个减少学习率的方法
    # 调用keras.callbacks.ReduceLROnPlateau()
    reduce_LR = keras.callbacks.ReduceLROnPlateau(monnitor='val_loss',
											  factor = 0.1,
											  patience = 2,
											  verbose = 1,
											  mode = 'min')
    # 在fit()中调用callbacks
    hist = model.fit(...,
				 callbacks=[reduce_LR],
				 ...)

#### h5py中文件的操作
    # h5py文件的结构：file->group->dataset
    # h5py的读写
    f = h5py.File(filepath,'r') # r表示只读, w表示写并会立即创建一个文件，r+表示读写
    # h5py内文件的遍历
    def printname():
	     print name
    f.visit(printname)
    # h5py内文件的删除
    del f['path']
    # h5py文件内的复制
    f_source.copy('sourcePath',f_destination('destinationPath'))

#### 获得神经网络中间层的输出
    new_model_structure = K.function([model.layers[0].input],
								 [model.layers[i].output]) # i代表你想获得的那一层输出的层数
    hidden_output = new_model_structure([X])[0]

#### 获得Keras的backend计算的值
    K.eval()

#### Keras训练数据太大不能一次性载入内存怎么办
    # 使用 fit.generator()
    def generator():
    	while 1:
    		for i in range(datasize):
    			x = get(x) # 可以每次从硬盘中读取一batch，也可以先把数据都读入硬盘，然后用generator做类似于data augmentation的操作
    			y = get(y)
    			yield ({'input_name':x},{'output_name':y}) # yield类似于return，只是yield会在每个循环输出

    model.fit_generator(generator = generator(),# 用generator获得input和target
    					steps_per_epoch = steps,
    					epochs = epochs,
    					validation_data = generator(),
    					validation_steps = steps,
    					shuffel = True,
    					initial_epoch = 0,
    					callbacks = [])

#### kers中bi-lstm的搭建方法
    def merge(left_right):
        return left_right[0]+left_right[1]

    input = input
    left = keras.layers.LSTM(...)(input)
    right = keras.layers.LSTM(...,go_backwards = True) #将序列逆序输入，把得到的输出逆序输出
    bi_lstm_output = keras.layers.lambda(merge)([left,right]) #用自建merge函数把left和right的输出值加起来
