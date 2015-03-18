local Module = nn.Module

Module.__parameters__ = {'weight', 'bias'}
Module.__gradParameters__ = {'gradWeight', 'gradBias'}

function Module:sharedClone(shareParams, shareGradParams)
   local moduleClones, modules
   if self.modules then
      moduleClones = {}
      for i,module in ipairs(self.modules) do
         moduleClones[i] = module:sharedClone(shareParams, shareGradParams)
      end
      modules = self.modules
      self.modules = nil
   end
   
   local params = {}
   if shareParams then
      for i,paramName in ipairs(self.__parameters__) do
         local param = self[paramName]
         if param then
            params[paramName] = param
            self[paramName] = nil
         end
      end
   end
   if shareGradParams then
      for i,paramName in ipairs(self.__gradParameters__) do
         local gradparam = self[paramName]
         if gradParam then
            params[paramName] = param
            self[paramName] = nil
         end
      end
   end
   
   -- clone everything but parameters and/or gradients
   local clone = self:clone()
   
   for paramName, param in pairs(params) do
      self[paramName] = param
      -- TODO make this recursive (for table params)
      clone[paramName] = param.new():set(param)
   end
   
   if moduleClones then
      self.modules = modules
      clone.modules = moduleClones
   end
   return clone
end
