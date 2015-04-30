require 'cutorch'
require 'image'
require 'cunn'

torch.setnumthreads(8)

train_file = 'mnist.t7/train_32x32.t7'
trainData = torch.load(train_file,'ascii')
data = trainData.data:float():div(255):view(trainData.data:size(1), 1024)
trainData = nil

torch.setdefaulttensortype('torch.CudaTensor')
local Rbm = require 'rbm'
rbm = Rbm.new{n_visible=1024, n_hidden=2048, CDsteps=5, momentum={0.5, 0.9},
              momentumAfter={5}, v_activation=nn.Sigmoid(), h_activation=nn.Sigmoid(),
              learningRate=0.1, useStates=true}--false, weightCost=0}

local chunkSize = 5000
for e=1, 30 do
    sys.tic()
    rbm.epoch = e
    for i=1,data:size(1),chunkSize do
        -- copy some data to gpu
        tmp = data[{{i,i+chunkSize-1},{}}]:clone():cuda()
        for pos=1,chunkSize,rbm.minibatchSize do
            rbm:updateParameters(tmp[{{pos,pos+rbm.minibatchSize-1},{}}])
        end
        io.write('.')
        io.flush()

        collectgarbage()
    end

    if e % 10 == 0 then
        torch.save('rbm_' .. e .. '.dat', rbm)
    end
    t = sys.toc()
    print('\t' .. e .. '\t' .. t)
end

-- _, h2 = rbm:HgivenV(data[{{1,10}, {}}]:cuda())
-- _, v2 = rbm:VgivenH(h2)

-- construct autoencoder
mlp = nn.Sequential()
mlp:add(rbm.encoder)
mlp:add(rbm.decoder)
mlp:cuda()

v = mlp:forward(data[{{1,36}, {}}]:cuda())

D = data[{{1,36},{}}]:view(torch.LongStorage{36,32,32}):double()
V = v:view(torch.LongStorage{36,32,32}):double()

torch.setdefaulttensortype('torch.DoubleTensor')
image.save('D.png', D)
image.save('V.png', V)

--torch.setdefaulttensortype('torch.DoubleTensor')
--d = image.display{image=D, win=d, zoom=3}
--d2 = image.display{image=V, win=d2, zoom=3}

-- d = image.display{image=torch.reshape(data[{{1}, {}}]:double(), 32, 32), win=d, zoom=5}
-- d2 = image.display{image=torch.reshape(v:double(), 32, 32), win=d2, zoom=5}
-- d3 = image.display{image=torch.reshape(v2:double(), 32, 32), win=d3, zoom=5}
--d2 = image.display{image=torch.reshape(v1_mean[{1}],32,32), win=d2, zoom=5}
--d3 = image.display{image=torch.reshape(rbm.v1_mean[{2}],32,32), win=d3, zoom=5}
