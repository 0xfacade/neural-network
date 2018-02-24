# Already trained networks
You can save trained networks with 
`NeuralNetwork.save(filename)` so that you don't
always have to retrain the network when you start up
your program.

I use filenames like `trained_networks/softmax-regressor-100-600-0.1.trained`,
where 100 is the number of epochs, 600 is the batchsize
and 0.1 the learning rate.