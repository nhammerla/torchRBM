require 'torch'
require 'image'
require 'nn'
local Rbm = require 'rbm'

-- use 4 threads
torch.setnumthreads(4)

-- Use floats throughout.
-- We could use torch.CudaTensor here and have it all run on the GPU
torch.setdefaulttensortype('torch.FloatTensor')

-- load mnist training set
train_file = 'mnist.t7/train_32x32.t7'
trainData = torch.load(train_file,'ascii')
data = trainData.data:float():div(255):view(trainData.data:size(1), 1024)
trainData = nil
-- we just want to reconstruct some data, no need for the test-set etc.

-- Create RBM with 1024 visible and 64 hidden units
-- Here we use rectified linear units and 1-step contrastive divergence
rbm = Rbm.new{n_visible=1024, n_hidden=64, CDsteps=1, momentum={0.5, 0.9},
              momentumAfter={5}, v_activation='binary', h_activation='relu',
              learningRate=0.01}

-- train for 30 epochs
--
rbm:train(data, 10)

-- construct autoencoder
-- we could further fine-tune this network using regular backprop
mlp = nn.Sequential()
mlp:add(rbm.encoder)
mlp:add(rbm.decoder)

-- reconstruct output using initialised autoencoder
v = mlp:forward(data[{{1,36}, {}}])

-- rehspae for visualisation
D = data[{{1,36},{}}]:view(torch.LongStorage{36,32,32})
V = v:view(torch.LongStorage{36,32,32})

-- show images
d = image.display{image=D, win=d, zoom=3}
d2 = image.display{image=V, win=d2, zoom=3}
