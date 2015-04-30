require 'cutorch'
require 'image'
require 'cunn'
require 'optim'

-- config
chunkSize = 1000
batchSize = 100
learningRate = 0.01
momentum = 0.9
rbmEpochs = 100
sgdEpochs = 250

-- load mnist
train_file = 'mnist.t7/train_32x32.t7'
test_file = 'mnist.t7/test_32x32.t7'
trainData = torch.load(train_file, 'ascii')
testData = torch.load(test_file, 'ascii')
data = trainData.data:float():div(255):view(trainData.data:size(1), 1024)
tData = testData.data:float():div(255):view(testData.data:size(1), 1024)
labels = trainData.labels
tLabels = testData.labels

classes = {'0','1','2','3','4','5','6','7','8','9'}
confusion = optim.ConfusionMatrix(classes)

-- function for training
trainLayer = function(rbm, data, chunkSize, epochs)
    for e=1, epochs do
        xlua.progress(e, epochs)
        rbm.epoch = e
        for i=1,data:size(1),chunkSize do
            -- copy some data to gpu
            tmp = data[{{i,i+chunkSize-1},{}}]:clone():cuda()
            for pos=1,chunkSize,rbm.minibatchSize do
                rbm:updateParameters(tmp[{{pos,pos+rbm.minibatchSize-1},{}}])
            end

            collectgarbage()
        end
    end
end

function transformData(net, data, newData, chunkSize)
    for i=1,data:size(1),chunkSize do
        newData[{{i,i+chunkSize-1},{}}] = net:forward(data[{{i,i+chunkSize-1},{}}]:cuda()):float()
    end
end

-- train layers of model
model = nn.Sequential()
layers = torch.IntTensor({{1024, 1024}, {1024, 1024}})
act = {{'binary','relu', 0.1},{'relu', 'relu',0.01}}
-- 1024 -> 512 -> 512 -> 10

torch.setdefaulttensortype('torch.CudaTensor') -- run on GPU
local Rbm = require 'rbm'

D = data:clone()
for id = 1, layers:size(1) do
    print('training layer ' .. id .. ':')
    local rbm = Rbm.new{n_visible=layers[id][1], n_hidden=layers[id][2], CDsteps=1, momentum={0.5, 0.9},
                  momentumAfter={5}, v_activation=act[id][1], h_activation=act[id][2],
                  learningRate=act[id][3], useStates=true}--false, weightCost=0}

    -- train first layer
    trainLayer(rbm, D, chunkSize, rbmEpochs)
    model:add(rbm.encoder) -- "grow" model

    -- transform data
    local newData = torch.FloatTensor(D:size(1), rbm.n_hidden)
    transformData(model, data, newData, chunkSize)

    -- reset D for next layer
    D = newData
end
torch.setdefaulttensortype('torch.FloatTensor')
D = nil
collectgarbage()

-- complete network for prediction
model:add(nn.Linear(layers[-1][2],10))
model:add(nn.LogSoftMax())
model:cuda()
parameters,gradParameters = model:getParameters()
criterion = nn.ClassNLLCriterion():cuda()
--criterion = nn.CrossEntropyCriterion():cuda()

-- training function
function train(data, labels)
   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   -- do one epoch
   print('<trainer> on training set:')
   print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. batchSize .. ']')
   for t = 1,data:size(1),batchSize do
      -- create mini batch

      local inputs = data[{{t,t+batchSize-1},{}}]:cuda()
      local targets = labels[{{t,t+batchSize-1}}]:cuda()

      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
         -- just in case:
         collectgarbage()

         -- get new parameters
         if x ~= parameters then
            parameters:copy(x)
         end

         -- reset gradients
         gradParameters:zero()

         -- evaluate function for complete mini batch
         local outputs = model:forward(inputs)
         local f = criterion:forward(outputs, targets)

         -- estimate df/dW
         local df_do = criterion:backward(outputs, targets)
         model:backward(inputs, df_do)

         --[[
         -- penalties (L1 and L2):
         if opt.coefL1 ~= 0 or opt.coefL2 ~= 0 then
            -- locals:
            local norm,sign= torch.norm,torch.sign

            -- Loss:
            f = f + opt.coefL1 * norm(parameters,1)
            f = f + opt.coefL2 * norm(parameters,2)^2/2

            -- Gradients:
            gradParameters:add( sign(parameters):mul(opt.coefL1) + parameters:clone():mul(opt.coefL2) )
         end
         --]]

         -- update confusion
         for i = 1,batchSize do
            confusion:add(outputs[i], targets[i])
         end

         -- return f and df/dX
         return f,gradParameters
      end

      -- optimize on current mini-batch
      --[[
      if opt.optimization == 'LBFGS' then

         -- Perform LBFGS step:
         lbfgsState = lbfgsState or {
            maxIter = opt.maxIter,
            lineSearch = optim.lswolfe
         }
         optim.lbfgs(feval, parameters, lbfgsState)

         -- disp report:
         print('LBFGS step')
         print(' - progress in batch: ' .. t .. '/' .. data:size())
         print(' - nb of iterations: ' .. lbfgsState.nIter)
         print(' - nb of function evalutions: ' .. lbfgsState.funcEval)

      elseif opt.optimization == 'SGD' then
      --]]
         -- Perform SGD step:
     sgdState = sgdState or {
        learningRate = learningRate,
        momentum = momentum,
        learningRateDecay = 5e-7
     }
     optim.sgd(feval, parameters, sgdState)

     xlua.progress(t, data:size(1))
     -- disp progress
     --xlua.progress(t, data:size())
      --[[
      else
         error('unknown optimization method')
      end
      --]]
   end

   -- time taken
   time = sys.clock() - time
   time = time / data:size(1)

   print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   confusion:zero()

   -- next epoch
   epoch = epoch + 1
end

-- test function
-- test function
function test(data, labels)
   -- local vars
   local time = sys.clock()

   -- test over given data
   print('<trainer> on testing Set:')
   for t = 1,data:size(1),batchSize do
      -- disp progress
      xlua.progress(t, data:size(1))

      -- create mini batch
      local inputs = data[{{t,t+batchSize-1},{}}]:cuda()
      local targets = labels[{{t,t+batchSize-1}}]:cuda()

      -- test samples
      local preds = model:forward(inputs)

      -- confusion:
      for i = 1,batchSize do
         confusion:add(preds[i], targets[i])
      end
   end

   -- timing
   time = sys.clock() - time
   time = time / data:size(1)
   print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)
   --testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
   confusion:zero()
end


for e=1,sgdEpochs do
    train(data, labels)
    test(tData, tLabels)
end
torch.save('model250.dat',model)
