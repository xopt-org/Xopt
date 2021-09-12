% Setup a Matlab engine on each local worker node
spmd
  matlab.engine.shareEngine(sprintf('Engine_%d',labindex-1))
end